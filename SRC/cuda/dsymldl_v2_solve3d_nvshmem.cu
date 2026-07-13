#include "superlu_ddefs.h"
#include "dsymldl_v2_solve3d.h"

#include <algorithm>
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
    double *forward_inbox;
    double *backward_inbox;
    int_t *d_forward_inbox_rows;
    int_t *d_backward_inbox_rows;
    int_t forward_inbox_count;
    int_t backward_inbox_count;
    int_t forward_inbox_capacity;
    int_t backward_inbox_capacity;
    cudaStream_t stream;
    std::vector<int_t> level_panel_ptr;
    std::vector<int_t> level_block_ptr;
    std::vector<int_t> forward_level_ptr;
    std::vector<int_t> backward_level_ptr;
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
    int_t block_begin, int_t block_count, double *forward_inbox,
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
        (void) rows;
        (void) n;
        nvshmem_double_p(
            forward_inbox +
                (block.forward_inbox_offset + row) * (int_t) nrhs + rhs,
            sum, (int) block.target_owner);
    }
}

__global__ void symldl_apply_forward_inbox(
    double *x, const double *forward_inbox,
    const int_t *forward_inbox_rows, int_t inbox_begin,
    int_t inbox_count, int_t n, int nrhs)
{
    int_t idx = (int_t) blockIdx.x * blockDim.x + threadIdx.x;
    int_t count = inbox_count * (int_t) nrhs;
    if (idx >= count)
        return;
    int rhs = (int) (idx / inbox_count);
    int_t local = idx - (int_t) rhs * inbox_count;
    int_t inbox = inbox_begin + local;
    int_t grow = forward_inbox_rows[inbox];
    atomicAdd(x + (int_t) rhs * n + grow,
              -forward_inbox[inbox * (int_t) nrhs + rhs]);
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
    const dSymLDL3DBlockDesc *blocks, int_t block_begin, int_t block_count,
    double *backward_inbox, const double *row_work,
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
        (void) n;
        nvshmem_double_p(
            backward_inbox +
                (block.backward_inbox_offset + col) * (int_t) nrhs + rhs,
            sum, panel.owner);
    }
}

__global__ void symldl_apply_backward_inbox(
    double *x, const double *backward_inbox,
    const int_t *backward_inbox_rows, int_t inbox_begin,
    int_t inbox_count, int_t n, int nrhs)
{
    int_t idx = (int_t) blockIdx.x * blockDim.x + threadIdx.x;
    int_t count = inbox_count * (int_t) nrhs;
    if (idx >= count)
        return;
    int rhs = (int) (idx / inbox_count);
    int_t local = idx - (int_t) rhs * inbox_count;
    int_t inbox = inbox_begin + local;
    int_t grow = backward_inbox_rows[inbox];
    atomicAdd(x + (int_t) rhs * n + grow,
              -backward_inbox[inbox * (int_t) nrhs + rhs]);
}

static void quiet_and_barrier(cudaStream_t stream)
{
    nvshmemx_quiet_on_stream(stream);
    nvshmemx_barrier_all_on_stream(stream);
}

static int checked_mpi_count(int_t count, const char *what)
{
    if (count < 0 || count > std::numeric_limits<int>::max())
        ABORT(what);
    return (int) count;
}

static void build_inbox_layout(
    dSymLDL3DSolveGPUState *state,
    const dSymLDL3DPanelDesc *panels,
    std::vector<dSymLDL3DBlockDesc> &blocks, const int_t *rows,
    const int_t *level_block_ptr, MPI_Comm comm,
    std::vector<int_t> &forward_rows,
    std::vector<int_t> &backward_rows)
{
    int rank = state->mype;
    int npes = state->npes;
    int_t dimensions = checked_product(
        state->nlevels, (int_t) npes,
        "SymLDL inbox layout dimensions overflow.");
    int dimensions_i = checked_mpi_count(
        dimensions, "SymLDL inbox layout exceeds the MPI count range.");
    std::vector<int_t> local_forward((size_t) dimensions, 0);
    std::vector<int_t> local_backward((size_t) dimensions, 0);
    std::vector<int_t> prefix_forward((size_t) dimensions, 0);
    std::vector<int_t> prefix_backward((size_t) dimensions, 0);
    std::vector<int_t> total_forward((size_t) dimensions, 0);
    std::vector<int_t> total_backward((size_t) dimensions, 0);

    for (int_t level = 0; level < state->nlevels; ++level) {
        for (int_t b = level_block_ptr[level];
             b < level_block_ptr[level + 1]; ++b) {
            dSymLDL3DBlockDesc &block = blocks[(size_t) b];
            const dSymLDL3DPanelDesc &panel = panels[block.panel];
            int_t fidx = level * (int_t) npes + block.target_owner;
            int_t bidx = level * (int_t) npes + panel.owner;
            local_forward[(size_t) fidx] = checked_sum(
                local_forward[(size_t) fidx], block.nbrow,
                "SymLDL forward inbox count overflows.");
            local_backward[(size_t) bidx] = checked_sum(
                local_backward[(size_t) bidx], panel.width,
                "SymLDL backward inbox count overflows.");
        }
    }

    if (dimensions_i > 0) {
        MPI_Exscan(local_forward.data(), prefix_forward.data(), dimensions_i,
                   mpi_int_t, MPI_SUM, comm);
        MPI_Exscan(local_backward.data(), prefix_backward.data(), dimensions_i,
                   mpi_int_t, MPI_SUM, comm);
        MPI_Allreduce(local_forward.data(), total_forward.data(), dimensions_i,
                      mpi_int_t, MPI_SUM, comm);
        MPI_Allreduce(local_backward.data(), total_backward.data(), dimensions_i,
                      mpi_int_t, MPI_SUM, comm);
        if (rank == 0) {
            std::fill(prefix_forward.begin(), prefix_forward.end(), 0);
            std::fill(prefix_backward.begin(), prefix_backward.end(), 0);
        }
    }

    std::vector<int_t> forward_base((size_t) dimensions, 0);
    std::vector<int_t> backward_base((size_t) dimensions, 0);
    std::vector<int_t> forward_total((size_t) npes, 0);
    std::vector<int_t> backward_total((size_t) npes, 0);
    state->forward_level_ptr.assign((size_t) state->nlevels + 1, 0);
    state->backward_level_ptr.assign((size_t) state->nlevels + 1, 0);
    for (int target = 0; target < npes; ++target) {
        int_t fcursor = 0;
        int_t bcursor = 0;
        for (int_t level = 0; level < state->nlevels; ++level) {
            int_t idx = level * (int_t) npes + target;
            forward_base[(size_t) idx] = fcursor;
            backward_base[(size_t) idx] = bcursor;
            if (target == rank) {
                state->forward_level_ptr[(size_t) level] = fcursor;
                state->backward_level_ptr[(size_t) level] = bcursor;
            }
            fcursor = checked_sum(
                fcursor, total_forward[(size_t) idx],
                "SymLDL forward inbox capacity overflows.");
            bcursor = checked_sum(
                bcursor, total_backward[(size_t) idx],
                "SymLDL backward inbox capacity overflows.");
        }
        forward_total[(size_t) target] = fcursor;
        backward_total[(size_t) target] = bcursor;
        state->forward_inbox_capacity = SUPERLU_MAX(
            state->forward_inbox_capacity, fcursor);
        state->backward_inbox_capacity = SUPERLU_MAX(
            state->backward_inbox_capacity, bcursor);
    }
    state->forward_inbox_count = forward_total[(size_t) rank];
    state->backward_inbox_count = backward_total[(size_t) rank];
    state->forward_level_ptr[(size_t) state->nlevels] =
        state->forward_inbox_count;
    state->backward_level_ptr[(size_t) state->nlevels] =
        state->backward_inbox_count;

    std::vector<int_t> forward_running((size_t) dimensions, 0);
    std::vector<int_t> backward_running((size_t) dimensions, 0);
    std::vector<std::vector<int_t> > records((size_t) npes);
    for (int_t level = 0; level < state->nlevels; ++level) {
        for (int_t b = level_block_ptr[level];
             b < level_block_ptr[level + 1]; ++b) {
            dSymLDL3DBlockDesc &block = blocks[(size_t) b];
            const dSymLDL3DPanelDesc &panel = panels[block.panel];
            int_t fidx = level * (int_t) npes + block.target_owner;
            int_t bidx = level * (int_t) npes + panel.owner;
            block.forward_inbox_offset = checked_sum(
                forward_base[(size_t) fidx],
                checked_sum(prefix_forward[(size_t) fidx],
                            forward_running[(size_t) fidx],
                            "SymLDL forward inbox offset overflows."),
                "SymLDL forward inbox offset overflows.");
            block.backward_inbox_offset = checked_sum(
                backward_base[(size_t) bidx],
                checked_sum(prefix_backward[(size_t) bidx],
                            backward_running[(size_t) bidx],
                            "SymLDL backward inbox offset overflows."),
                "SymLDL backward inbox offset overflows.");
            forward_running[(size_t) fidx] = checked_sum(
                forward_running[(size_t) fidx], block.nbrow,
                "SymLDL forward inbox offset overflows.");
            backward_running[(size_t) bidx] = checked_sum(
                backward_running[(size_t) bidx], panel.width,
                "SymLDL backward inbox offset overflows.");

            std::vector<int_t> &forward_record =
                records[(size_t) block.target_owner];
            for (int_t row = 0; row < block.nbrow; ++row) {
                forward_record.push_back(0);
                forward_record.push_back(block.forward_inbox_offset + row);
                forward_record.push_back(rows[block.row_begin + row]);
            }
            std::vector<int_t> &backward_record =
                records[(size_t) panel.owner];
            for (int_t col = 0; col < panel.width; ++col) {
                backward_record.push_back(1);
                backward_record.push_back(block.backward_inbox_offset + col);
                backward_record.push_back(panel.fst_row + col);
            }
        }
    }

    std::vector<int> send_counts((size_t) npes, 0);
    std::vector<int> recv_counts((size_t) npes, 0);
    std::vector<int> send_displs((size_t) npes, 0);
    std::vector<int> recv_displs((size_t) npes, 0);
    int_t send_total = 0;
    for (int target = 0; target < npes; ++target) {
        send_counts[(size_t) target] = checked_mpi_count(
            (int_t) records[(size_t) target].size(),
            "SymLDL inbox records exceed the MPI count range.");
        if (target > 0)
            send_displs[(size_t) target] = checked_mpi_count(
                send_total, "SymLDL inbox record displacement overflows.");
        send_total = checked_sum(
            send_total, (int_t) send_counts[(size_t) target],
            "SymLDL inbox record storage overflows.");
    }
    std::vector<int_t> send_records((size_t) send_total);
    for (int target = 0; target < npes; ++target)
        std::copy(records[(size_t) target].begin(),
                  records[(size_t) target].end(),
                  send_records.begin() + send_displs[(size_t) target]);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm);
    int_t recv_total = 0;
    for (int source = 0; source < npes; ++source) {
        if (recv_counts[(size_t) source] < 0)
            ABORT("Invalid SymLDL inbox receive count.");
        if (source > 0)
            recv_displs[(size_t) source] = checked_mpi_count(
                recv_total, "SymLDL inbox receive displacement overflows.");
        recv_total = checked_sum(
            recv_total, (int_t) recv_counts[(size_t) source],
            "SymLDL inbox receive storage overflows.");
    }
    std::vector<int_t> recv_records((size_t) recv_total);
    MPI_Alltoallv(send_records.data(), send_counts.data(), send_displs.data(),
                  mpi_int_t, recv_records.data(), recv_counts.data(),
                  recv_displs.data(), mpi_int_t, comm);
    if (recv_total % 3 != 0)
        ABORT("Invalid SymLDL inbox record payload.");

    forward_rows.assign((size_t) state->forward_inbox_count, -1);
    backward_rows.assign((size_t) state->backward_inbox_count, -1);
    for (int_t pos = 0; pos < recv_total; pos += 3) {
        int_t kind = recv_records[(size_t) pos];
        int_t offset = recv_records[(size_t) pos + 1];
        int_t grow = recv_records[(size_t) pos + 2];
        std::vector<int_t> *map = kind == 0 ? &forward_rows
                                           : kind == 1 ? &backward_rows
                                                       : NULL;
        if (map == NULL || offset < 0 || (size_t) offset >= map->size() ||
            grow < 0 || grow >= state->n || (*map)[(size_t) offset] != -1)
            ABORT("Invalid SymLDL inbox destination metadata.");
        (*map)[(size_t) offset] = grow;
    }
    for (size_t i = 0; i < forward_rows.size(); ++i)
        if (forward_rows[i] < 0)
            ABORT("SymLDL forward inbox metadata is incomplete.");
    for (size_t i = 0; i < backward_rows.size(); ++i)
        if (backward_rows[i] < 0)
            ABORT("SymLDL backward inbox metadata is incomplete.");
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
        level_panel_ptr == NULL || level_block_ptr == NULL ||
        (panel_count > 0 && panels == NULL) ||
        (block_count > 0 && blocks == NULL) ||
        (row_count > 0 && rows == NULL))
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
    state->forward_inbox = NULL;
    state->backward_inbox = NULL;
    state->d_forward_inbox_rows = NULL;
    state->d_backward_inbox_rows = NULL;
    state->forward_inbox_count = 0;
    state->backward_inbox_count = 0;
    state->forward_inbox_capacity = 0;
    state->backward_inbox_capacity = 0;
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
            int_t diag_last_column = checked_product(
                panel.width - 1, panel.nsupr,
                "SymLDL inverse-diagonal stride overflows.");
            int_t diag_end = checked_sum(
                checked_sum(panel.diag_luptr, diag_last_column,
                            "SymLDL inverse-diagonal offset overflows."),
                panel.width,
                "SymLDL inverse-diagonal extent overflows.");
            if (panel.diag_luptr < 0 || diag_end > panel.value_count)
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

    std::vector<dSymLDL3DBlockDesc> inbox_blocks;
    if (block_count > 0)
        inbox_blocks.assign(blocks, blocks + block_count);
    std::vector<int_t> forward_inbox_rows;
    std::vector<int_t> backward_inbox_rows;
    build_inbox_layout(state, panels, inbox_blocks, rows,
                       level_block_ptr, comm, forward_inbox_rows,
                       backward_inbox_rows);

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
        gpuErrchk(cudaMemcpy(state->d_blocks, inbox_blocks.data(),
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
    if (state->forward_inbox_count > 0) {
        gpuErrchk(cudaMalloc((void **) &state->d_forward_inbox_rows,
                             checked_bytes(state->forward_inbox_count,
                                           sizeof(int_t),
                                           "SymLDL forward inbox metadata overflows.")));
        gpuErrchk(cudaMemcpy(state->d_forward_inbox_rows,
                             forward_inbox_rows.data(),
                             checked_bytes(state->forward_inbox_count,
                                           sizeof(int_t),
                                           "SymLDL forward inbox metadata overflows."),
                             cudaMemcpyHostToDevice));
    }
    if (state->backward_inbox_count > 0) {
        gpuErrchk(cudaMalloc((void **) &state->d_backward_inbox_rows,
                             checked_bytes(state->backward_inbox_count,
                                           sizeof(int_t),
                                           "SymLDL backward inbox metadata overflows.")));
        gpuErrchk(cudaMemcpy(state->d_backward_inbox_rows,
                             backward_inbox_rows.data(),
                             checked_bytes(state->backward_inbox_count,
                                           sizeof(int_t),
                                           "SymLDL backward inbox metadata overflows."),
                             cudaMemcpyHostToDevice));
    }

    state->x = (double *) nvshmem_malloc(
        checked_bytes(SUPERLU_MAX(state->x_count, (int_t) 1), sizeof(double),
                      "SymLDL symmetric solution allocation overflows."));
    if (state->x == NULL)
        ABORT("NVSHMEM allocation fails for SymLDL solution.");
    int_t forward_values = checked_product(
        state->forward_inbox_capacity, (int_t) nrhs,
        "SymLDL forward inbox dimensions overflow.");
    int_t backward_values = checked_product(
        state->backward_inbox_capacity, (int_t) nrhs,
        "SymLDL backward inbox dimensions overflow.");
    state->forward_inbox = (double *) nvshmem_malloc(
        checked_bytes(SUPERLU_MAX(forward_values, (int_t) 1),
                      sizeof(double),
                      "SymLDL forward inbox allocation overflows."));
    state->backward_inbox = (double *) nvshmem_malloc(
        checked_bytes(SUPERLU_MAX(backward_values, (int_t) 1),
                      sizeof(double),
                      "SymLDL backward inbox allocation overflows."));
    if (state->forward_inbox == NULL || state->backward_inbox == NULL)
        ABORT("NVSHMEM allocation fails for SymLDL contribution inboxes.");
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
                bb, bc, state->forward_inbox, state->pivot_work,
                state->n, state->nrhs);
        gpuErrchk(cudaGetLastError());
        quiet_and_barrier(state->stream);
        int_t inbox_begin = state->forward_level_ptr[(size_t) level];
        int_t inbox_count = state->forward_level_ptr[(size_t) level + 1] -
                            inbox_begin;
        if (inbox_count > 0) {
            int_t apply_count = checked_product(
                inbox_count, (int_t) state->nrhs,
                "SymLDL forward inbox launch dimensions overflow.");
            int_t apply_blocks = (apply_count + threads - 1) / threads;
            symldl_apply_forward_inbox<<<apply_blocks, threads, 0,
                                           state->stream>>>(
                state->x, state->forward_inbox,
                state->d_forward_inbox_rows, inbox_begin, inbox_count,
                state->n, state->nrhs);
            gpuErrchk(cudaGetLastError());
        }
        nvshmemx_barrier_all_on_stream(state->stream);
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
                state->d_panels, state->d_blocks, bb, bc,
                state->backward_inbox,
                state->row_work, state->n, state->nrhs);
            gpuErrchk(cudaGetLastError());
        }
        quiet_and_barrier(state->stream);
        int_t inbox_begin = state->backward_level_ptr[(size_t) level - 1];
        int_t inbox_count = state->backward_level_ptr[(size_t) level] -
                            inbox_begin;
        if (inbox_count > 0) {
            int_t apply_count = checked_product(
                inbox_count, (int_t) state->nrhs,
                "SymLDL backward inbox launch dimensions overflow.");
            int_t apply_blocks = (apply_count + threads - 1) / threads;
            symldl_apply_backward_inbox<<<apply_blocks, threads, 0,
                                            state->stream>>>(
                state->x, state->backward_inbox,
                state->d_backward_inbox_rows, inbox_begin, inbox_count,
                state->n, state->nrhs);
            gpuErrchk(cudaGetLastError());
        }
        nvshmemx_barrier_all_on_stream(state->stream);
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
    if (state->backward_inbox) nvshmem_free(state->backward_inbox);
    if (state->forward_inbox) nvshmem_free(state->forward_inbox);
    if (state->x) nvshmem_free(state->x);
#endif
    if (state->d_panels) gpuErrchk(cudaFree(state->d_panels));
    if (state->d_blocks) gpuErrchk(cudaFree(state->d_blocks));
    if (state->d_rows) gpuErrchk(cudaFree(state->d_rows));
    if (state->d_forward_inbox_rows)
        gpuErrchk(cudaFree(state->d_forward_inbox_rows));
    if (state->d_backward_inbox_rows)
        gpuErrchk(cudaFree(state->d_backward_inbox_rows));
    if (state->diag_work) gpuErrchk(cudaFree(state->diag_work));
    if (state->pivot_work) gpuErrchk(cudaFree(state->pivot_work));
    if (state->row_work) gpuErrchk(cudaFree(state->row_work));
    if (state->stream) gpuErrchk(cudaStreamDestroy(state->stream));
#ifdef HAVE_NVSHMEM
    nvshmem_finalize();
#endif
    delete state;
}
