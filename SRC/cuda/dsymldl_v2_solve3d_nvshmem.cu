#include "superlu_ddefs.h"
#include "dsymldl_v2_solve3d.h"

#include <cuda_runtime.h>
#include <limits>
#include <stdint.h>
#include <vector>

#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

static void symldl_cuda_check(cudaError_t status, const char *file, int line)
{
    if (status == cudaSuccess)
        return;
    fprintf(stderr, "SymLDL CUDA failure at %s:%d: %s\n",
            file, line, cudaGetErrorString(status));
    fflush(stderr);
    ABORT("SymLDL CUDA operation failed.");
}

#define gpuErrchk(call) symldl_cuda_check((call), __FILE__, __LINE__)

namespace {

struct dSymLDL3DSolveGPUState {
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t nlevels;
    int_t panel_count;
    int_t block_count;
    int_t row_count;
    int mype;
    int npes;
    MPI_Comm comm;
    int x_layout_validated;
    int_t x_count;
    int_t pivot_count;
    int_t row_work_count;
    dSymLDL3DPanelDesc *d_panels;
    dSymLDL3DBlockDesc *d_blocks;
    int_t *d_rows;
    double *x;
    double *diag_work;
    double *pivot_work;
    double *row_work;
    cudaStream_t stream;
    std::vector<int_t> level_panel_ptr;
    std::vector<int_t> level_block_ptr;
    double t_h2d;
    double t_forward;
    double t_diagonal;
    double t_backward;
    double t_d2h;
};

static size_t checked_bytes(int_t count, size_t item_size, const char *what)
{
    if (count < 0 || (size_t) count > SIZE_MAX / item_size)
        ABORT(what);
    return (size_t) count * item_size;
}

static int_t checked_product(int_t left, int_t right, const char *what)
{
    if (left < 0 || right < 0 ||
        (left != 0 &&
         right > std::numeric_limits<int_t>::max() / left))
        ABORT(what);
    return left * right;
}

static int_t checked_sum(int_t total, int_t add, const char *what)
{
    if (total < 0 || add < 0 ||
        add > std::numeric_limits<int_t>::max() - total)
        ABORT(what);
    return total + add;
}

#ifdef HAVE_NVSHMEM

__global__ void symldl_fetch_forward_x(
    const dSymLDL3DPanelDesc *panels, int_t panel_begin, int_t panel_count,
    const double *x, double *pivot_work, int_t n, int nrhs, int mype)
{
    int_t local = (int_t) blockIdx.x;
    if (local >= panel_count)
        return;
    const dSymLDL3DPanelDesc panel = panels[panel_begin + local];
    int_t count = panel.width * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / panel.width);
        int_t col = idx - (int_t) rhs * panel.width;
        const double *source = x + (int_t) rhs * n + panel.fst_row + col;
        double value = panel.owner == mype
                           ? *source
                           : nvshmem_double_g(source, panel.owner);
        pivot_work[panel.pivot_begin + idx] = value;
    }
}

__global__ void symldl_forward_update(
    const dSymLDL3DPanelDesc *panels,
    const dSymLDL3DBlockDesc *blocks, const int_t *rows,
    int_t block_begin, int_t block_count, double *x,
    const double *pivot_work, int_t n, int nrhs)
{
    int_t local = (int_t) blockIdx.x;
    if (local >= block_count)
        return;
    const dSymLDL3DBlockDesc block = blocks[block_begin + local];
    const dSymLDL3DPanelDesc panel = panels[block.panel];
    int_t count = block.nbrow * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / block.nbrow);
        int_t row = idx - (int_t) rhs * block.nbrow;
        double sum = 0.0;
        const double *values = panel.values + block.luptr + row;
        const double *xk = pivot_work + panel.pivot_begin +
                           (int_t) rhs * panel.width;
        for (int_t col = 0; col < panel.width; ++col)
            sum += values[col * panel.nsupr] * xk[col];
        int_t grow = rows[block.row_begin + row];
        nvshmem_double_atomic_add(x + (int_t) rhs * n + grow, -sum,
                                  (int) block.target_owner);
    }
}

__global__ void symldl_apply_diagonal(
    const dSymLDL3DPanelDesc *panels, int_t panel_count,
    const double *x, double *diag_work, int_t n, int nrhs, int mype)
{
    int_t panel_id = (int_t) blockIdx.x;
    if (panel_id >= panel_count)
        return;
    const dSymLDL3DPanelDesc panel = panels[panel_id];
    if (panel.owner != mype)
        return;
    int_t count = panel.width * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / panel.width);
        int_t row = idx - (int_t) rhs * panel.width;
        double sum = 0.0;
        const double *dinv = panel.values + panel.diag_luptr + row;
        const double *rhs_block = x + (int_t) rhs * n + panel.fst_row;
        for (int_t col = 0; col < panel.width; ++col)
            sum += dinv[col * panel.nsupr] * rhs_block[col];
        diag_work[(int_t) rhs * n + panel.fst_row + row] = sum;
    }
}

__global__ void symldl_store_diagonal(
    const dSymLDL3DPanelDesc *panels, int_t panel_count,
    double *x, const double *diag_work, int_t n, int nrhs, int mype)
{
    int_t panel_id = (int_t) blockIdx.x;
    if (panel_id >= panel_count)
        return;
    const dSymLDL3DPanelDesc panel = panels[panel_id];
    if (panel.owner != mype)
        return;
    int_t count = panel.width * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / panel.width);
        int_t row = idx - (int_t) rhs * panel.width;
        x[(int_t) rhs * n + panel.fst_row + row] =
            diag_work[(int_t) rhs * n + panel.fst_row + row];
    }
}

__global__ void symldl_fetch_backward_x(
    const dSymLDL3DBlockDesc *blocks, const int_t *rows,
    int_t block_begin, int_t block_count, const double *x,
    double *row_work, int_t n, int nrhs, int mype)
{
    int_t local = (int_t) blockIdx.x;
    if (local >= block_count)
        return;
    const dSymLDL3DBlockDesc block = blocks[block_begin + local];
    int_t count = block.nbrow * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / block.nbrow);
        int_t row = idx - (int_t) rhs * block.nbrow;
        int_t grow = rows[block.row_begin + row];
        const double *source = x + (int_t) rhs * n + grow;
        double value = block.target_owner == mype
                           ? *source
                           : nvshmem_double_g(source,
                                              (int) block.target_owner);
        row_work[(block.row_begin + row) * (int_t) nrhs + rhs] = value;
    }
}

__global__ void symldl_backward_update(
    const dSymLDL3DPanelDesc *panels,
    const dSymLDL3DBlockDesc *blocks, int_t block_begin,
    int_t block_count, double *x, const double *row_work,
    int_t n, int nrhs)
{
    int_t local = (int_t) blockIdx.x;
    if (local >= block_count)
        return;
    const dSymLDL3DBlockDesc block = blocks[block_begin + local];
    const dSymLDL3DPanelDesc panel = panels[block.panel];
    int_t count = panel.width * (int_t) nrhs;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int rhs = (int) (idx / panel.width);
        int_t col = idx - (int_t) rhs * panel.width;
        double sum = 0.0;
        const double *values = panel.values + block.luptr +
                               col * panel.nsupr;
        for (int_t row = 0; row < block.nbrow; ++row)
            sum += values[row] *
                   row_work[(block.row_begin + row) * (int_t) nrhs + rhs];
        nvshmem_double_atomic_add(x + (int_t) rhs * n +
                                  panel.fst_row + col,
                                  -sum, panel.owner);
    }
}

static void quiet_and_barrier(cudaStream_t stream)
{
    nvshmemx_quiet_on_stream(stream);
    nvshmemx_barrier_all_on_stream(stream);
}

#endif

} // namespace

extern "C" int dSymLDL3DSolveGPUAvailable(void)
{
#ifdef HAVE_NVSHMEM
    return 1;
#else
    return 0;
#endif
}

extern "C" dSymLDL3DSolveGPUHandle dSymLDL3DSolveGPUCreate(
    int_t n, int_t nsupers, int nrhs, int_t maxsup, int_t nlevels,
    const int_t *level_panel_ptr, const int_t *level_block_ptr,
    int_t panel_count, const dSymLDL3DPanelDesc *panels,
    int_t block_count, const dSymLDL3DBlockDesc *blocks,
    int_t row_count, const int_t *rows, MPI_Comm comm)
{
#ifndef HAVE_NVSHMEM
    (void) n; (void) nsupers; (void) nrhs; (void) maxsup;
    (void) nlevels; (void) level_panel_ptr; (void) level_block_ptr;
    (void) panel_count; (void) panels; (void) block_count; (void) blocks;
    (void) row_count; (void) rows; (void) comm;
    return NULL;
#else
    if (n < 0 || nsupers < 0 || nrhs <= 0 || maxsup < 0 || nlevels < 0 ||
        panel_count < 0 || block_count < 0 || row_count < 0 ||
        level_panel_ptr == NULL || level_block_ptr == NULL)
        ABORT("Invalid SymLDL 3D GPU solve metadata.");

    int device_count = 0;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    gpuErrchk(cudaGetDeviceCount(&device_count));
    if (device_count <= 0)
        ABORT("SymLDL 3D solve requires a CUDA device.");
    gpuErrchk(cudaSetDevice(rank % device_count));

    nv_init_wrapper(comm);
    dSymLDL3DSolveGPUState *state = new dSymLDL3DSolveGPUState();
    state->n = n;
    state->nsupers = nsupers;
    state->nrhs = nrhs;
    state->nlevels = nlevels;
    state->panel_count = panel_count;
    state->block_count = block_count;
    state->row_count = row_count;
    state->mype = nvshmem_my_pe();
    state->npes = nvshmem_n_pes();
    state->comm = comm;
    state->x_layout_validated = 0;
    state->x_count = checked_product(
        n, (int_t) nrhs, "SymLDL solution dimensions overflow.");
    state->pivot_count = 0;
    state->row_work_count = checked_product(
        row_count, (int_t) nrhs,
        "SymLDL transpose workspace dimensions overflow.");
    state->d_panels = NULL;
    state->d_blocks = NULL;
    state->d_rows = NULL;
    state->x = NULL;
    state->diag_work = NULL;
    state->pivot_work = NULL;
    state->row_work = NULL;
    state->stream = NULL;
    state->t_h2d = state->t_forward = state->t_diagonal = 0.0;
    state->t_backward = state->t_d2h = 0.0;
    if (state->mype != rank)
        ABORT("SymLDL 3D solve MPI and NVSHMEM ranks differ.");

    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (state->npes != comm_size)
        ABORT("SymLDL 3D solve MPI and NVSHMEM sizes differ.");

    if (level_panel_ptr[0] != 0 || level_block_ptr[0] != 0 ||
        level_panel_ptr[nlevels] != panel_count ||
        level_block_ptr[nlevels] != block_count)
        ABORT("SymLDL 3D solve level metadata is inconsistent.");
    for (int_t level = 0; level < nlevels; ++level) {
        if (level_panel_ptr[level] > level_panel_ptr[level + 1] ||
            level_block_ptr[level] > level_block_ptr[level + 1])
            ABORT("SymLDL 3D solve levels are not monotone.");
    }
    std::vector<int> local_diagonal((size_t) nsupers, 0);
    std::vector<int> global_diagonal((size_t) nsupers, 0);
    for (int_t p = 0; p < panel_count; ++p) {
        const dSymLDL3DPanelDesc &panel = panels[p];
        if (panel.gid < 0 || panel.gid >= nsupers || panel.fst_row < 0 ||
            panel.width <= 0 || panel.fst_row > n - panel.width ||
            panel.nsupr < panel.width || panel.block_begin < 0 ||
            panel.block_count < 0 ||
            panel.block_begin > block_count - panel.block_count ||
            panel.row_begin < 0 || panel.row_count < 0 ||
            panel.row_begin > row_count - panel.row_count ||
            panel.pivot_begin < 0 || panel.value_count <= 0 ||
            panel.owner < 0 || panel.owner >= comm_size ||
            panel.values == NULL)
            ABORT("Invalid SymLDL 3D GPU panel metadata.");
        state->pivot_count = checked_sum(
            state->pivot_count,
            checked_product(panel.width, (int_t) nrhs,
                            "SymLDL pivot dimensions overflow."),
            "SymLDL pivot workspace size overflows.");
        if (panel.owner == rank) {
            if (panel.diag_luptr < 0 ||
                panel.diag_luptr > panel.value_count - panel.width)
                ABORT("SymLDL diagonal owner is missing its inverse block.");
            ++local_diagonal[(size_t) panel.gid];
        }
    }
    for (int_t b = 0; b < block_count; ++b) {
        const dSymLDL3DBlockDesc &block = blocks[b];
        if (block.panel < 0 || block.panel >= panel_count ||
            block.target_gid < 0 || block.target_gid >= nsupers ||
            block.target_owner < 0 || block.target_owner >= comm_size ||
            block.luptr < 0 || block.nbrow <= 0 || block.row_begin < 0 ||
            block.row_begin > row_count - block.nbrow)
            ABORT("Invalid SymLDL 3D GPU block metadata.");
        const dSymLDL3DPanelDesc &panel = panels[block.panel];
        int_t last_column = checked_product(
            panel.width - 1, panel.nsupr,
            "SymLDL factor-panel stride overflows.");
        int_t block_end = checked_sum(
            checked_sum(block.luptr, last_column,
                        "SymLDL factor-panel offset overflows."),
            block.nbrow, "SymLDL factor-panel extent overflows.");
        if (block_end > panel.value_count)
            ABORT("SymLDL 3D GPU block exceeds its factor panel.");
        for (int_t row = 0; row < block.nbrow; ++row) {
            int_t grow = rows[block.row_begin + row];
            if (grow < 0 || grow >= n)
                ABORT("SymLDL 3D GPU block has an invalid row index.");
        }
    }
    if (nsupers > std::numeric_limits<int>::max())
        ABORT("SymLDL supernode count exceeds the MPI count range.");
    MPI_Allreduce(local_diagonal.data(), global_diagonal.data(),
                  (int) nsupers, MPI_INT, MPI_SUM, comm);
    for (int_t k = 0; k < nsupers; ++k)
        if (global_diagonal[(size_t) k] != 1)
            ABORT("SymLDL 3D solve requires one inverse diagonal owner per supernode.");

    state->level_panel_ptr.assign(level_panel_ptr,
                                  level_panel_ptr + nlevels + 1);
    state->level_block_ptr.assign(level_block_ptr,
                                  level_block_ptr + nlevels + 1);
    gpuErrchk(cudaStreamCreateWithFlags(&state->stream,
                                        cudaStreamNonBlocking));
    if (panel_count > 0) {
        gpuErrchk(cudaMalloc((void **) &state->d_panels,
                             checked_bytes(panel_count, sizeof(*panels),
                                           "SymLDL GPU panel metadata overflows.")));
        gpuErrchk(cudaMemcpy(state->d_panels, panels,
                             checked_bytes(panel_count, sizeof(*panels),
                                           "SymLDL GPU panel metadata overflows."),
                             cudaMemcpyHostToDevice));
    }
    if (block_count > 0) {
        gpuErrchk(cudaMalloc((void **) &state->d_blocks,
                             checked_bytes(block_count, sizeof(*blocks),
                                           "SymLDL GPU block metadata overflows.")));
        gpuErrchk(cudaMemcpy(state->d_blocks, blocks,
                             checked_bytes(block_count, sizeof(*blocks),
                                           "SymLDL GPU block metadata overflows."),
                             cudaMemcpyHostToDevice));
    }
    if (row_count > 0) {
        gpuErrchk(cudaMalloc((void **) &state->d_rows,
                             checked_bytes(row_count, sizeof(*rows),
                                           "SymLDL GPU row metadata overflows.")));
        gpuErrchk(cudaMemcpy(state->d_rows, rows,
                             checked_bytes(row_count, sizeof(*rows),
                                           "SymLDL GPU row metadata overflows."),
                             cudaMemcpyHostToDevice));
    }

    state->x = (double *) nvshmem_malloc(
        checked_bytes(SUPERLU_MAX(state->x_count, (int_t) 1), sizeof(double),
                      "SymLDL symmetric solution allocation overflows."));
    if (state->x == NULL)
        ABORT("NVSHMEM allocation fails for SymLDL solution.");
    gpuErrchk(cudaMalloc((void **) &state->diag_work,
                         checked_bytes(SUPERLU_MAX(state->x_count, (int_t) 1),
                                       sizeof(double),
                                       "SymLDL diagonal workspace overflows.")));
    gpuErrchk(cudaMalloc((void **) &state->pivot_work,
                         checked_bytes(SUPERLU_MAX(state->pivot_count,
                                                  (int_t) 1),
                                       sizeof(double),
                                       "SymLDL pivot workspace overflows.")));
    gpuErrchk(cudaMalloc((void **) &state->row_work,
                         checked_bytes(SUPERLU_MAX(state->row_work_count,
                                                  (int_t) 1),
                                       sizeof(double),
                                       "SymLDL transpose workspace overflows.")));
    nvshmem_barrier_all();
    return (dSymLDL3DSolveGPUHandle) state;
#endif
}

extern "C" int dSymLDL3DSolveGPURun(
    dSymLDL3DSolveGPUHandle handle, double *x,
    const int_t *local_x_offsets, const int_t *local_row_gids,
    const int_t *local_row_first, const int_t *local_row_width,
    const int *local_row_owner,
    int_t local_row_count, int global_rank)
{
#ifndef HAVE_NVSHMEM
    (void) handle; (void) x; (void) local_x_offsets;
    (void) local_row_gids; (void) local_row_first; (void) local_row_width;
    (void) local_row_owner; (void) local_row_count; (void) global_rank;
    return -1;
#else
    dSymLDL3DSolveGPUState *state =
        (dSymLDL3DSolveGPUState *) handle;
    if (state == NULL || x == NULL || local_x_offsets == NULL ||
        local_row_gids == NULL || local_row_first == NULL ||
        local_row_width == NULL || local_row_owner == NULL ||
        local_row_count < 0 ||
        global_rank != state->mype)
        return -1;

    if (!state->x_layout_validated) {
        int_t local_owned_rows = 0;
        for (int_t local = 0; local < local_row_count; ++local) {
            int_t gid = local_row_gids[local];
            if (gid < 0 || gid >= state->nsupers ||
                local_x_offsets[local] < 0 || local_row_first[local] < 0 ||
                local_row_width[local] <= 0 ||
                local_row_first[local] > state->n - local_row_width[local] ||
                local_row_owner[local] < 0 ||
                local_row_owner[local] >= state->npes)
                return -1;
            if (local_row_owner[local] == global_rank)
                local_owned_rows = checked_sum(
                    local_owned_rows, local_row_width[local],
                    "SymLDL local solution layout overflows.");
        }
        int_t global_owned_rows = 0;
        MPI_Allreduce(&local_owned_rows, &global_owned_rows, 1,
                      mpi_int_t, MPI_SUM, state->comm);
        if (global_owned_rows != state->n)
            return -1;
        state->x_layout_validated = 1;
    }

    double t = SuperLU_timer_();
    gpuErrchk(cudaMemsetAsync(state->x, 0,
                              checked_bytes(SUPERLU_MAX(state->x_count,
                                                       (int_t) 1),
                                            sizeof(double),
                                            "SymLDL solution reset overflows."),
                              state->stream));
    for (int_t local = 0; local < local_row_count; ++local) {
        if (local_row_owner[local] != global_rank)
            continue;
        for (int rhs = 0; rhs < state->nrhs; ++rhs)
            gpuErrchk(cudaMemcpyAsync(
                state->x + (int_t) rhs * state->n + local_row_first[local],
                x + local_x_offsets[local] +
                    (int_t) rhs * local_row_width[local],
                checked_bytes(local_row_width[local], sizeof(double),
                              "SymLDL solution upload overflows."),
                cudaMemcpyHostToDevice, state->stream));
    }
    quiet_and_barrier(state->stream);
    gpuErrchk(cudaStreamSynchronize(state->stream));
    state->t_h2d += SuperLU_timer_() - t;

    t = SuperLU_timer_();
    const int threads = 256;
    for (int_t level = 0; level < state->nlevels; ++level) {
        int_t pb = state->level_panel_ptr[(size_t) level];
        int_t pc = state->level_panel_ptr[(size_t) level + 1] - pb;
        int_t bb = state->level_block_ptr[(size_t) level];
        int_t bc = state->level_block_ptr[(size_t) level + 1] - bb;
        if (pc > 0)
            symldl_fetch_forward_x<<<pc, threads, 0, state->stream>>>(
                state->d_panels, pb, pc, state->x, state->pivot_work,
                state->n, state->nrhs, state->mype);
        if (bc > 0)
            symldl_forward_update<<<bc, threads, 0, state->stream>>>(
                state->d_panels, state->d_blocks, state->d_rows,
                bb, bc, state->x, state->pivot_work,
                state->n, state->nrhs);
        gpuErrchk(cudaGetLastError());
        quiet_and_barrier(state->stream);
    }
    gpuErrchk(cudaStreamSynchronize(state->stream));
    state->t_forward += SuperLU_timer_() - t;

    t = SuperLU_timer_();
    if (state->panel_count > 0) {
        symldl_apply_diagonal<<<state->panel_count, threads, 0,
                                state->stream>>>(
            state->d_panels, state->panel_count, state->x,
            state->diag_work, state->n, state->nrhs, state->mype);
        symldl_store_diagonal<<<state->panel_count, threads, 0,
                                state->stream>>>(
            state->d_panels, state->panel_count, state->x,
            state->diag_work, state->n, state->nrhs, state->mype);
        gpuErrchk(cudaGetLastError());
    }
    quiet_and_barrier(state->stream);
    gpuErrchk(cudaStreamSynchronize(state->stream));
    state->t_diagonal += SuperLU_timer_() - t;

    t = SuperLU_timer_();
    for (int_t level = state->nlevels; level > 0; --level) {
        int_t bb = state->level_block_ptr[(size_t) level - 1];
        int_t bc = state->level_block_ptr[(size_t) level] - bb;
        if (bc > 0) {
            symldl_fetch_backward_x<<<bc, threads, 0, state->stream>>>(
                state->d_blocks, state->d_rows, bb, bc, state->x,
                state->row_work, state->n, state->nrhs, state->mype);
            symldl_backward_update<<<bc, threads, 0, state->stream>>>(
                state->d_panels, state->d_blocks, bb, bc, state->x,
                state->row_work, state->n, state->nrhs);
            gpuErrchk(cudaGetLastError());
        }
        quiet_and_barrier(state->stream);
    }
    gpuErrchk(cudaStreamSynchronize(state->stream));
    state->t_backward += SuperLU_timer_() - t;

    t = SuperLU_timer_();
    for (int_t local = 0; local < local_row_count; ++local) {
        if (local_row_owner[local] != global_rank)
            continue;
        for (int rhs = 0; rhs < state->nrhs; ++rhs)
            gpuErrchk(cudaMemcpyAsync(
                x + local_x_offsets[local] +
                    (int_t) rhs * local_row_width[local],
                state->x + (int_t) rhs * state->n + local_row_first[local],
                checked_bytes(local_row_width[local], sizeof(double),
                              "SymLDL solution download overflows."),
                cudaMemcpyDeviceToHost, state->stream));
    }
    gpuErrchk(cudaStreamSynchronize(state->stream));
    state->t_d2h += SuperLU_timer_() - t;
    return 0;
#endif
}

extern "C" void dSymLDL3DSolveGPUTakeTimers(
    dSymLDL3DSolveGPUHandle handle, double *h2d, double *forward,
    double *diagonal, double *backward, double *d2h)
{
    dSymLDL3DSolveGPUState *state =
        (dSymLDL3DSolveGPUState *) handle;
    if (h2d) *h2d = state ? state->t_h2d : 0.0;
    if (forward) *forward = state ? state->t_forward : 0.0;
    if (diagonal) *diagonal = state ? state->t_diagonal : 0.0;
    if (backward) *backward = state ? state->t_backward : 0.0;
    if (d2h) *d2h = state ? state->t_d2h : 0.0;
    if (state) {
        state->t_h2d = state->t_forward = state->t_diagonal = 0.0;
        state->t_backward = state->t_d2h = 0.0;
    }
}

extern "C" void dSymLDL3DSolveGPUDestroy(
    dSymLDL3DSolveGPUHandle handle)
{
    dSymLDL3DSolveGPUState *state =
        (dSymLDL3DSolveGPUState *) handle;
    if (state == NULL)
        return;
#ifdef HAVE_NVSHMEM
    nvshmem_barrier_all();
    if (state->x) nvshmem_free(state->x);
#endif
    if (state->d_panels) gpuErrchk(cudaFree(state->d_panels));
    if (state->d_blocks) gpuErrchk(cudaFree(state->d_blocks));
    if (state->d_rows) gpuErrchk(cudaFree(state->d_rows));
    if (state->diag_work) gpuErrchk(cudaFree(state->diag_work));
    if (state->pivot_work) gpuErrchk(cudaFree(state->pivot_work));
    if (state->row_work) gpuErrchk(cudaFree(state->row_work));
    if (state->stream) gpuErrchk(cudaStreamDestroy(state->stream));
#ifdef HAVE_NVSHMEM
    nvshmem_finalize();
#endif
    delete state;
}
