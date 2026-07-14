/*! @file
 * \brief NVSHMEM L-D-L^T solve for the native 3D symmetric factor.
 */

#include "superlu_ddefs.h"
#include "dsymldl_v2_nvshmem_solve.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>
#include <vector>

#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace {

enum { SYMLDL_NVSHMEM_WAIT_THREADS = DIM_X * DIM_Y };

struct dSymLDLNVSHMEMSolveState {
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t x_count;
    int_t lsum_count;
    int rank;
    int nprocs;
    MPI_Comm comm;

    int_t panel_count;
    int_t block_count;
    int_t row_count;
    dSymLDLNVPanelDesc *d_panels;
    dSymLDLNVBlockDesc *d_blocks;
    int_t *d_rows;
    int_t *d_xsup;
    int *d_diag_owner;
    int_t *d_x_offsets;
    int_t *d_lsum_offsets;
    C_Tree *d_bcast_trees;
    C_Tree *d_reduce_trees;
    C_Tree *d_backward_bcast_trees;
    C_Tree *d_backward_reduce_trees;
    int *d_fmod;
    int *d_fmod_initial;
    int *d_bmod_initial;
    int_t *d_wait_gids;
    int_t wait_count;
    int_t *d_backward_wait_gids;
    int_t backward_wait_count;
    int *d_wait_status;
    int *d_wait_status_initial;
    int *d_backward_wait_status_initial;
    int_t *d_backward_block_order;
    int_t *d_backward_target_gids;
    int_t *d_backward_target_offsets;
    int_t backward_target_count;
    int *d_backward_forwarded;
    int *d_next_panel;
    int progress_blocks;
    int worker_blocks;
    int backward_waiter_blocks;
    int waiter_handshake_threads;
    int backward_worker_blocks;
    int *d_panel_stage;
    int *h_panel_stage;
    int diagnostics;
    int_t *h_panel_gids;

    double *d_x;
    double *d_lsum;
    double *d_bsum;
    uint64_t *flag_bc;
    uint64_t *flag_rd;
    double *ready_x;
    double *ready_lsum;
    int *h_waiter_started;
    int *d_waiter_started;
    cudaStream_t streams[2];

    double timer_h2d;
    double timer_forward;
    double timer_d2h;
    double timer_forward_phase;
    double timer_diagonal_phase;
    double timer_backward_phase;
};

static int symldl_nvshmem_initialized = 0;

static bool
symldl_cuda_ok(cudaError_t status, const char *what)
{
    if (status == cudaSuccess)
        return true;
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    fflush(stderr);
    return false;
}

static bool
symldl_count_bytes(int_t count, size_t size, size_t *bytes)
{
    if (count < 0 || (size_t) count > SIZE_MAX / size)
        return false;
    *bytes = (size_t) count * size;
    return true;
}

template <typename T>
static bool
symldl_cuda_alloc(T **ptr, int_t count, const char *what)
{
    size_t bytes;
    *ptr = NULL;
    if (count == 0)
        return true;
    if (!symldl_count_bytes(count, sizeof(T), &bytes)) {
        fprintf(stderr, "%s overflows.\n", what);
        return false;
    }
    return symldl_cuda_ok(cudaMalloc((void **) ptr, bytes), what);
}

template <typename T>
static bool
symldl_cuda_copy_to_device(T **dst, const T *src, int_t count,
                           const char *what)
{
    size_t bytes;
    if (!symldl_cuda_alloc(dst, count, what))
        return false;
    if (count == 0)
        return true;
    if (src == NULL)
        return false;
    if (!symldl_count_bytes(count, sizeof(T), &bytes))
        return false;
    return symldl_cuda_ok(cudaMemcpy(*dst, src, bytes,
                                    cudaMemcpyHostToDevice), what);
}

static void
symldl_nvshmem_state_free(dSymLDLNVSHMEMSolveState *state)
{
    if (state == NULL)
        return;

#ifdef HAVE_NVSHMEM
    if (state->flag_bc) nvshmem_free(state->flag_bc);
    if (state->flag_rd) nvshmem_free(state->flag_rd);
    if (state->ready_x) nvshmem_free(state->ready_x);
    if (state->ready_lsum) nvshmem_free(state->ready_lsum);
#endif
    if (state->h_waiter_started) cudaFreeHost(state->h_waiter_started);
    if (state->streams[0]) cudaStreamDestroy(state->streams[0]);
    if (state->streams[1]) cudaStreamDestroy(state->streams[1]);
    cudaFree(state->d_wait_status);
    cudaFree(state->d_wait_status_initial);
    cudaFree(state->d_backward_wait_status_initial);
    cudaFree(state->d_next_panel);
    cudaFree(state->d_backward_forwarded);
    cudaFree(state->d_backward_target_offsets);
    cudaFree(state->d_backward_target_gids);
    cudaFree(state->d_backward_block_order);
    if (state->h_panel_stage)
        cudaFreeHost(state->h_panel_stage);
    else
        cudaFree(state->d_panel_stage);
    cudaFree(state->d_wait_gids);
    cudaFree(state->d_backward_wait_gids);
    cudaFree(state->d_bmod_initial);
    cudaFree(state->d_fmod_initial);
    cudaFree(state->d_fmod);
    cudaFree(state->d_reduce_trees);
    cudaFree(state->d_bcast_trees);
    cudaFree(state->d_backward_reduce_trees);
    cudaFree(state->d_backward_bcast_trees);
    cudaFree(state->d_lsum_offsets);
    cudaFree(state->d_x_offsets);
    cudaFree(state->d_diag_owner);
    cudaFree(state->d_xsup);
    cudaFree(state->d_rows);
    cudaFree(state->d_blocks);
    cudaFree(state->d_panels);
    cudaFree(state->d_lsum);
    cudaFree(state->d_bsum);
    cudaFree(state->d_x);
    free(state->h_panel_gids);
    delete state;
}

#ifdef HAVE_NVSHMEM

static __device__ __forceinline__ void
symldl_nvshmem_bcast(C_Tree *tree, int_t gid, int_t count,
                     uint64_t *flag_bc, double *ready_x,
                     int_t data_offset)
{
    for (int child = 0; child < tree->destCnt_; ++child) {
        nvshmemx_double_put_signal_nbi_block(
            ready_x + data_offset, ready_x + data_offset, count,
            flag_bc + gid, 1, NVSHMEM_SIGNAL_SET, tree->myDests_[child]);
    }
}

static __device__ __forceinline__ void
symldl_nvshmem_reduce_send_thread(C_Tree *tree, int_t gid, int width,
                                  int nrhs, const int_t *xsup,
                                  const int_t *lsum_offsets,
                                  const double *lsum, uint64_t *flag_rd,
                                  double *ready_lsum)
{
    if (tree->empty_ == YES || tree->myRoot_ == tree->myRank_)
        return;

    int_t count = (int_t) width * nrhs;
    int_t source = 2 * xsup[gid] * nrhs;
    int slot = tree->myIdx & 1;
    int_t target = source + (slot ? count : 0);
    int_t local = lsum_offsets[gid];
    for (int_t i = 0; i < count; ++i)
        ready_lsum[source + i] = lsum[local + i];
    nvshmem_double_put_signal_nbi(
        ready_lsum + target, ready_lsum + source, count,
        flag_rd + 2 * gid + slot, 1, NVSHMEM_SIGNAL_SET,
        tree->myRoot_);
}

/*
 * This is the reduction half of the established dwait_bcrd protocol. Each
 * thread owns a contiguous set of target supernodes and waits on any child
 * signal in that range, so an early dependency cannot block later arrivals.
 */
static __device__ void
symldl_nvshmem_forward_wait(
    int nrhs, int_t nsupers, C_Tree *reduce_trees,
    const int_t *xsup, const int_t *lsum_offsets,
    const int_t *wait_gids, int_t wait_count, int *wait_status,
    uint64_t *flag_rd, double *ready_lsum, double *lsum, int *fmod,
    int *waiter_started, int tid, int nthreads)
{
    if (tid == 0) {
        *waiter_started = 1;
        __threadfence_system();
    }
    __syncthreads();
    int_t begin = wait_count * tid / nthreads;
    int_t end = wait_count * (tid + 1) / nthreads;
    if (begin >= end)
        return;

    int_t first_gid = wait_gids[begin];
    int_t last_gid = wait_gids[end - 1];
    size_t flag_begin = (size_t) 2 * first_gid;
    size_t flag_count = (size_t) 2 * (last_gid - first_gid + 1);
    int expected = 0;
    for (int_t p = begin; p < end; ++p)
        expected += reduce_trees[wait_gids[p]].destCnt_;
    waiter_started[1 + tid] = expected;
    waiter_started[1 + nthreads + tid] = 0;
    __threadfence_system();

    for (int received = 0; received < expected; ++received) {
        size_t rel = nvshmem_uint64_wait_until_any(
            flag_rd + flag_begin, flag_count, wait_status + flag_begin,
            NVSHMEM_CMP_EQ, 1);
        size_t flag_index = flag_begin + rel;
        wait_status[flag_index] = 1;
        waiter_started[1 + nthreads + tid] = received + 1;
        __threadfence_system();

        int_t gid = (int_t) (flag_index / 2);
        int slot = (int) (flag_index & 1);
        if (gid < 0 || gid >= nsupers)
            continue;
        int width = (int) (xsup[gid + 1] - xsup[gid]);
        int_t count = (int_t) width * nrhs;
        int_t recv_offset = 2 * xsup[gid] * nrhs +
                            (slot ? count : 0);
        int_t local = lsum_offsets[gid];
        for (int_t i = 0; i < count; ++i)
            atomicAdd(lsum + local + i, ready_lsum[recv_offset + i]);
        __threadfence();
        int old = atomicSub(fmod + gid, 1);
        if (old == 1)
            symldl_nvshmem_reduce_send_thread(
                reduce_trees + gid, gid, width, nrhs, xsup,
                lsum_offsets, lsum, flag_rd, ready_lsum);
    }
}

/*
 * Adaptation of dlsum_fmod_inv_gpu_mrhs_nvshmem. The panel and destination
 * lookup is explicit because SymLDL uses selected diagonal/panel roots rather
 * than the LU block-cyclic LBi/LBj mapping. L is unit lower, so the owner
 * publishes x_k directly after its accumulated modifications are ready.
 */
static __device__ void
symldl_nvshmem_forward_panel(
    int_t panel_id, const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *xsup, const int *diag_owner,
    const int_t *x_offsets, const int_t *lsum_offsets,
    int nrhs, int rank, double *x, double *lsum, int *fmod,
    C_Tree *bcast_trees, C_Tree *reduce_trees,
    uint64_t *flag_bc, uint64_t *flag_rd,
    double *ready_x, double *ready_lsum, int *panel_stage)
{
    dSymLDLNVPanelDesc panel = panels[panel_id];
    int_t k = panel.gid;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    int_t ksupc = panel.width;
    int_t x_ready_offset = xsup[k] * nrhs;
    int_t x_local_offset = x_offsets[k];
    int_t lsum_local_offset = lsum_offsets[k];

    if (panel_stage != NULL && tid == 0)
        panel_stage[panel_id] = 1;

    if (rank == diag_owner[k]) {
        if (tid == 0) {
            while (atomicAdd(fmod + k, 0) > 0)
                __threadfence();
        }
        __syncthreads();
        if (panel_stage != NULL && tid == 0)
            panel_stage[panel_id] = 2;
        for (int_t p = tid; p < ksupc * nrhs; p += nthreads)
            x[x_local_offset + p] += lsum[lsum_local_offset + p];
        __syncthreads();
        for (int_t p = tid; p < ksupc * nrhs; p += nthreads)
            ready_x[x_ready_offset + p] = x[x_local_offset + p];
        __syncthreads();
    } else {
        if (tid == 0) {
            while (((volatile uint64_t *) flag_bc)[k] != 1) {
                __threadfence();
                __nanosleep(64);
            }
        }
        __syncthreads();
        if (panel_stage != NULL && tid == 0)
            panel_stage[panel_id] = 2;
    }

    symldl_nvshmem_bcast(bcast_trees + k, k, ksupc * nrhs,
                         flag_bc, ready_x, x_ready_offset);
    __syncthreads();
    if (panel_stage != NULL && tid == 0)
        panel_stage[panel_id] = 3;

    for (int_t b = 0; b < panel.block_count; ++b) {
        dSymLDLNVBlockDesc block = blocks[panel.block_begin + b];
        int_t target = block.target_gid;
        int_t target_width = xsup[target + 1] - xsup[target];
        int_t target_lsum = lsum_offsets[target];
        int_t count = block.nbrow * nrhs;

        for (int_t p = tid; p < count; p += nthreads) {
            int rhs = (int) (p / block.nbrow);
            int_t r = p - (int_t) rhs * block.nbrow;
            double sum = 0.0;
            for (int_t col = 0; col < ksupc; ++col)
                sum += panel.values[block.luptr + r + col * panel.nsupr] *
                       ready_x[x_ready_offset + col +
                               (int_t) rhs * ksupc];
            int_t relative_row = rows[block.row_begin + r] - xsup[target];
            atomicAdd(lsum + target_lsum + relative_row +
                      (int_t) rhs * target_width, -sum);
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            int old = atomicSub(fmod + target, 1);
            if (old == 1)
                symldl_nvshmem_reduce_send_thread(
                    reduce_trees + target, target, (int) target_width,
                    nrhs, xsup, lsum_offsets, lsum, flag_rd, ready_lsum);
        }
        __syncthreads();
    }
    if (panel_stage != NULL && tid == 0)
        panel_stage[panel_id] = 4;
}

/*
 * Keep reduction progress and panel work resident in one kernel. Separate
 * persistent kernels can serialize on CUDA work queues, leaving diagonal
 * owners waiting for reductions while their panel grid has not started.
 */
static __global__ void
symldl_nvshmem_forward_progress_kernel(
    const dSymLDLNVPanelDesc *panels, int_t panel_count, int_t nsupers,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *xsup, const int *diag_owner,
    const int_t *x_offsets, const int_t *lsum_offsets,
    int nrhs, int rank, double *x, double *lsum, int *fmod,
    C_Tree *bcast_trees, C_Tree *reduce_trees,
    uint64_t *flag_bc, uint64_t *flag_rd,
    double *ready_x, double *ready_lsum,
    const int_t *wait_gids, int_t wait_count, int *wait_status,
    int *next_panel, int *waiter_started, int *panel_stage)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;

    if (blockIdx.x == 0) {
        symldl_nvshmem_forward_wait(
            nrhs, nsupers, reduce_trees, xsup, lsum_offsets,
            wait_gids, wait_count,
            wait_status, flag_rd, ready_lsum, lsum, fmod,
            waiter_started, tid, nthreads);
        return;
    }

    __shared__ int panel_id;
    for (;;) {
        if (tid == 0)
            panel_id = atomicAdd(next_panel, 1);
        __syncthreads();
        if ((int_t) panel_id >= panel_count)
            return;
        symldl_nvshmem_forward_panel(
            (int_t) panel_id, panels, blocks, rows, xsup, diag_owner,
            x_offsets, lsum_offsets, nrhs, rank, x, lsum, fmod,
            bcast_trees, reduce_trees, flag_bc, flag_rd, ready_x,
            ready_lsum, panel_stage);
        __syncthreads();
    }
}

static __global__ void
symldl_nvshmem_diagonal_kernel(
    const dSymLDLNVPanelDesc *panels, int_t panel_count,
    const int_t *x_offsets, const int_t *lsum_offsets,
    int nrhs, int rank, double *x, double *work)
{
    int_t panel_id = blockIdx.x;
    if (panel_id >= panel_count)
        return;
    dSymLDLNVPanelDesc panel = panels[panel_id];
    if (panel.owner != rank)
        return;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    int_t width = panel.width;
    int_t x_offset = x_offsets[panel.gid];
    int_t work_offset = lsum_offsets[panel.gid];
    for (int_t p = tid; p < width * nrhs; p += nthreads) {
        int rhs = (int) (p / width);
        int_t row = p - (int_t) rhs * width;
        double sum = 0.0;
        for (int_t col = 0; col < width; ++col)
            sum += panel.values[panel.diag_luptr + row + col * panel.nsupr] *
                   x[x_offset + col + (int_t) rhs * width];
        work[work_offset + row + (int_t) rhs * width] = sum;
    }
    __syncthreads();
    for (int_t p = tid; p < width * nrhs; p += nthreads)
        x[x_offset + p] = work[work_offset + p];
}

static __device__ __forceinline__ void
symldl_nvshmem_backward_reduce_send_thread(
    C_Tree *tree, int_t gid, int width, int nrhs, const int_t *xsup,
    const double *bsum, uint64_t *flag_rd, double *ready_lsum)
{
    if (tree->empty_ == YES || tree->myRoot_ == tree->myRank_)
        return;

    int_t count = (int_t) width * nrhs;
    int_t source = 2 * xsup[gid] * nrhs;
    int slot = tree->myIdx & 1;
    int_t target = source + (slot ? count : 0);
    int_t local = xsup[gid] * nrhs;
    for (int_t i = 0; i < count; ++i)
        ready_lsum[source + i] = bsum[local + i];
    nvshmem_double_put_signal_nbi(
        ready_lsum + target, ready_lsum + source, count,
        flag_rd + 2 * gid + slot, 1, NVSHMEM_SIGNAL_SET,
        tree->myRoot_);
}

/* Receive reverse-solve reductions using the same two-slot tree protocol. */
static __device__ void
symldl_nvshmem_backward_wait(
    int nrhs, int_t nsupers, C_Tree *reduce_trees, const int_t *xsup,
    const int_t *wait_gids, int_t wait_count, int *wait_status,
    uint64_t *flag_rd, double *ready_lsum, double *bsum, int *bmod,
    int *waiter_started, int tid, int nthreads)
{
    if (tid == 0) {
        *waiter_started = 1;
        __threadfence_system();
    }
    __syncthreads();

    int_t begin = wait_count * tid / nthreads;
    int_t end = wait_count * (tid + 1) / nthreads;
    if (begin >= end)
        return;

    int_t first_gid = wait_gids[begin];
    int_t last_gid = wait_gids[end - 1];
    size_t flag_begin = (size_t) 2 * first_gid;
    size_t flag_count = (size_t) 2 * (last_gid - first_gid + 1);
    int expected = 0;
    for (int_t p = begin; p < end; ++p)
        expected += reduce_trees[wait_gids[p]].destCnt_;
    waiter_started[1 + tid] = expected;
    waiter_started[1 + nthreads + tid] = 0;
    __threadfence_system();

    for (int received = 0; received < expected; ++received) {
        size_t rel = nvshmem_uint64_wait_until_any(
            flag_rd + flag_begin, flag_count, wait_status + flag_begin,
            NVSHMEM_CMP_EQ, 1);
        size_t flag_index = flag_begin + rel;
        wait_status[flag_index] = 1;
        waiter_started[1 + nthreads + tid] = received + 1;
        __threadfence_system();

        int_t gid = (int_t) (flag_index / 2);
        int slot = (int) (flag_index & 1);
        if (gid < 0 || gid >= nsupers)
            continue;
        int width = (int) (xsup[gid + 1] - xsup[gid]);
        int_t count = (int_t) width * nrhs;
        int_t recv_offset = 2 * xsup[gid] * nrhs +
                            (slot ? count : 0);
        int_t local = xsup[gid] * nrhs;
        for (int_t i = 0; i < count; ++i)
            atomicAdd(bsum + local + i, ready_lsum[recv_offset + i]);
        __threadfence();
        int old = atomicSub(bmod + gid, 1);
        if (old == 1)
            symldl_nvshmem_backward_reduce_send_thread(
                reduce_trees + gid, gid, width, nrhs, xsup, bsum,
                flag_rd, ready_lsum);
    }
}

static __device__ void
symldl_nvshmem_backward_target_ready(
    int_t target, const int_t *xsup, const int *diag_owner,
    int nrhs, int rank, C_Tree *bcast_trees, uint64_t *flag_bc,
    double *ready_x, int *forwarded)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int_t target_width = xsup[target + 1] - xsup[target];
    int_t target_ready = xsup[target] * nrhs;
    __shared__ int forward_here;

    if (rank != diag_owner[target]) {
        if (tid == 0) {
            while (((volatile uint64_t *) flag_bc)[target] != 1) {
                __threadfence();
                __nanosleep(64);
            }
            forward_here = atomicCAS(forwarded + target, 0, 1) == 0;
        }
        __syncthreads();
        if (forward_here) {
            symldl_nvshmem_bcast(
                bcast_trees + target, target,
                (int) (target_width * nrhs), flag_bc, ready_x,
                target_ready);
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicExch(forwarded + target, 2);
            }
        }
    }
    __syncthreads();
    if (rank == diag_owner[target] && tid == 0) {
        while (atomicAdd(forwarded + target, 0) < 1) {
            __threadfence();
            __nanosleep(64);
        }
    }
    __syncthreads();
}

/* Mirror the established U-solve role: the target CTA publishes owned x_k. */
static __device__ void
symldl_nvshmem_backward_target_publish(
    int_t target, const int_t *xsup, const int *diag_owner,
    const int_t *x_offsets, int nrhs, int rank, double *x,
    const double *bsum, int *bmod, C_Tree *bcast_trees,
    uint64_t *flag_bc, double *ready_x, int *forwarded)
{
    if (rank != diag_owner[target]) {
        symldl_nvshmem_backward_target_ready(
            target, xsup, diag_owner, nrhs, rank, bcast_trees, flag_bc,
            ready_x, forwarded);
        return;
    }

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    if (tid == 0) {
        while (atomicAdd(bmod + target, 0) > 0) {
            __threadfence();
            __nanosleep(64);
        }
    }
    __syncthreads();

    int_t width = xsup[target + 1] - xsup[target];
    int_t count = width * nrhs;
    int_t x_local = x_offsets[target];
    int_t dense = xsup[target] * nrhs;
    for (int_t p = tid; p < count; p += nthreads)
        x[x_local + p] += bsum[dense + p];
    __syncthreads();
    for (int_t p = tid; p < count; p += nthreads)
        ready_x[dense + p] = x[x_local + p];
    __syncthreads();
    symldl_nvshmem_bcast(bcast_trees + target, target, (int) count,
                         flag_bc, ready_x, dense);
    if (tid == 0) {
        __threadfence();
        atomicExch(forwarded + target, 2);
    }
    __syncthreads();
}

static __device__ void
symldl_nvshmem_backward_block_apply(
    int_t block_id, const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *xsup, int nrhs, double *bsum, int *bmod,
    C_Tree *reduce_trees, uint64_t *flag_rd, double *ready_x,
    double *ready_lsum)
{
    dSymLDLNVBlockDesc block = blocks[block_id];
    dSymLDLNVPanelDesc panel = panels[block.panel_id];
    int_t source = panel.gid;
    int_t target = block.target_gid;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    int_t source_width = panel.width;
    int_t target_width = xsup[target + 1] - xsup[target];
    int_t target_ready = xsup[target] * nrhs;

    int_t count = source_width * nrhs;
    int_t source_sum = xsup[source] * nrhs;
    for (int_t p = tid; p < count; p += nthreads) {
        int rhs = (int) (p / source_width);
        int_t col = p - (int_t) rhs * source_width;
        double sum = 0.0;
        for (int_t r = 0; r < block.nbrow; ++r) {
            int_t target_row = rows[block.row_begin + r] - xsup[target];
            sum += panel.values[block.luptr + r + col * panel.nsupr] *
                   ready_x[target_ready + target_row +
                           (int_t) rhs * target_width];
        }
        atomicAdd(bsum + source_sum + p, -sum);
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int old = atomicSub(bmod + source, 1);
        if (old == 1)
            symldl_nvshmem_backward_reduce_send_thread(
                reduce_trees + source, source, (int) source_width,
                nrhs, xsup, bsum, flag_rd, ready_lsum);
    }
    __syncthreads();
}

static __device__ void
symldl_nvshmem_backward_target(
    int_t target_pos, const int_t *target_gids,
    const int_t *target_offsets, const int_t *block_order,
    const dSymLDLNVPanelDesc *panels, const dSymLDLNVBlockDesc *blocks,
    const int_t *rows, const int_t *xsup, const int *diag_owner,
    const int_t *x_offsets, int nrhs, int rank, double *x,
    double *bsum, int *bmod,
    C_Tree *bcast_trees, C_Tree *reduce_trees,
    uint64_t *flag_bc, uint64_t *flag_rd, double *ready_x,
    double *ready_lsum, int *forwarded)
{
    int_t target = target_gids[target_pos];
    symldl_nvshmem_backward_target_publish(
        target, xsup, diag_owner, x_offsets, nrhs, rank, x, bsum,
        bmod, bcast_trees, flag_bc, ready_x, forwarded);
    for (int_t pos = target_offsets[target_pos];
         pos < target_offsets[target_pos + 1]; ++pos)
        symldl_nvshmem_backward_block_apply(
            block_order[pos], panels, blocks, rows, xsup, nrhs,
            bsum, bmod, reduce_trees, flag_rd, ready_x, ready_lsum);
}

/*
 * The resident roles prevent dependency waiters from occupying every SM:
 * the first waiter_blocks CTAs progress reductions and the remaining CTAs
 * publish solved target blocks and apply their retained-L contributions.
 */
static __global__ void
symldl_nvshmem_backward_progress_kernel(
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks,
    const int_t *block_order, const int_t *target_gids,
    const int_t *target_offsets, int_t target_count,
    const int_t *rows, int_t nsupers,
    const int_t *xsup, const int *diag_owner, const int_t *x_offsets,
    int nrhs, int rank, double *x, double *bsum, int *bmod,
    C_Tree *bcast_trees, C_Tree *reduce_trees,
    uint64_t *flag_bc, uint64_t *flag_rd,
    double *ready_x, double *ready_lsum,
    const int_t *wait_gids, int_t wait_count, int *wait_status,
    int *next_block, int *forwarded, int *waiter_started,
    int waiter_blocks)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;

    if ((int) blockIdx.x < waiter_blocks) {
        int waiter_tid = (int) blockIdx.x * nthreads + tid;
        int waiter_threads = waiter_blocks * nthreads;
        symldl_nvshmem_backward_wait(
            nrhs, nsupers, reduce_trees, xsup, wait_gids, wait_count,
            wait_status, flag_rd, ready_lsum, bsum, bmod,
            waiter_started, waiter_tid, waiter_threads);
        return;
    }
    __shared__ int block_pos;
    for (;;) {
        if (tid == 0)
            block_pos = atomicAdd(next_block, 1);
        __syncthreads();
        if ((int_t) block_pos >= target_count)
            return;
        symldl_nvshmem_backward_target(
            block_pos, target_gids, target_offsets, block_order,
            panels, blocks, rows, xsup, diag_owner, x_offsets, nrhs,
            rank, x, bsum, bmod, bcast_trees, reduce_trees,
            flag_bc, flag_rd, ready_x, ready_lsum, forwarded);
        __syncthreads();
    }
}

static int
symldl_nvshmem_wait_with_diagnostics(dSymLDLNVSHMEMSolveState *state)
{
    const double deadline = SuperLU_timer_() + 10.0;
    cudaError_t compute_status = cudaErrorNotReady;
    cudaError_t wait_status = cudaErrorNotReady;
    do {
        compute_status = cudaStreamQuery(state->streams[1]);
        wait_status = cudaStreamQuery(state->streams[0]);
        if (compute_status == cudaSuccess && wait_status == cudaSuccess)
            return 0;
        if ((compute_status != cudaSuccess &&
             compute_status != cudaErrorNotReady) ||
            (wait_status != cudaSuccess && wait_status != cudaErrorNotReady))
            return -1;
        usleep(100000);
    } while (SuperLU_timer_() < deadline);

    int expected_signals = 0;
    int received_signals = 0;
    int stalled_waiters = 0;
    for (int tid = 0; tid < SYMLDL_NVSHMEM_WAIT_THREADS; ++tid) {
        int expected = ((volatile int *) state->h_waiter_started)[1 + tid];
        int received =
            ((volatile int *) state->h_waiter_started)
                [1 + SYMLDL_NVSHMEM_WAIT_THREADS + tid];
        expected_signals += expected;
        received_signals += received;
        stalled_waiters += received < expected;
    }
    fprintf(stderr,
            "SymLDL NVSHMEM waiter rank %d: expected=%d received=%d "
            "stalled_threads=%d\n",
            state->rank, expected_signals, received_signals,
            stalled_waiters);
    int shown_waiters = 0;
    for (int tid = 0;
         tid < SYMLDL_NVSHMEM_WAIT_THREADS && shown_waiters < 16; ++tid) {
        int expected = ((volatile int *) state->h_waiter_started)[1 + tid];
        int received =
            ((volatile int *) state->h_waiter_started)
                [1 + SYMLDL_NVSHMEM_WAIT_THREADS + tid];
        if (received >= expected)
            continue;
        fprintf(stderr,
                "  waiter thread=%d expected=%d received=%d\n",
                tid, expected, received);
        ++shown_waiters;
    }
    if (state->h_panel_stage != NULL) {
        int stage_counts[5] = {0, 0, 0, 0, 0};
        for (int_t p = 0; p < state->panel_count; ++p) {
            int stage = ((volatile int *) state->h_panel_stage)[p];
            if (stage >= 0 && stage <= 4)
                ++stage_counts[stage];
        }
        fprintf(stderr,
                "  mapped stages 0=%d 1=%d 2=%d 3=%d 4=%d\n",
                stage_counts[0], stage_counts[1], stage_counts[2],
                stage_counts[3], stage_counts[4]);
        int shown = 0;
        for (int_t p = 0; p < state->panel_count && shown < 16; ++p) {
            int stage = ((volatile int *) state->h_panel_stage)[p];
            if (stage == 4)
                continue;
            fprintf(stderr,
                    "  mapped pending panel=%lld gid=%lld stage=%d\n",
                    (long long) p,
                    (long long) state->h_panel_gids[p], stage);
            ++shown;
        }
    }
    fflush(stderr);

    int *panel_stage = NULL;
    int *fmod = NULL;
    cudaStream_t stream = NULL;
    int least_priority = 0;
    int greatest_priority = 0;
    size_t panel_bytes = (size_t) state->panel_count * sizeof(int);
    size_t super_bytes = (size_t) state->nsupers * sizeof(int);
    int snapshot_ok =
        cudaMallocHost((void **) &panel_stage, panel_bytes) == cudaSuccess &&
        cudaMallocHost((void **) &fmod, super_bytes) == cudaSuccess &&
        cudaDeviceGetStreamPriorityRange(&least_priority,
                                         &greatest_priority) == cudaSuccess &&
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                     greatest_priority) == cudaSuccess;
    if (snapshot_ok) {
        snapshot_ok =
            cudaMemcpyAsync(panel_stage, state->d_panel_stage, panel_bytes,
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaMemcpyAsync(fmod, state->d_fmod, super_bytes,
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
    }

    fprintf(stderr,
            "SymLDL NVSHMEM diagnostics rank %d: compute=%d wait=%d "
            "snapshot=%d panels=%lld waits=%lld\n",
            state->rank, (int) compute_status, (int) wait_status,
            snapshot_ok, (long long) state->panel_count,
            (long long) state->wait_count);
    if (snapshot_ok) {
        int stage_counts[5] = {0, 0, 0, 0, 0};
        int positive_fmod = 0;
        for (int_t p = 0; p < state->panel_count; ++p) {
            int stage = panel_stage[p];
            if (stage >= 0 && stage <= 4)
                ++stage_counts[stage];
        }
        for (int_t k = 0; k < state->nsupers; ++k)
            positive_fmod += fmod[k] > 0;
        fprintf(stderr,
                "  stages 0=%d 1=%d 2=%d 3=%d 4=%d; fmod>0=%d\n",
                stage_counts[0], stage_counts[1], stage_counts[2],
                stage_counts[3], stage_counts[4], positive_fmod);
        int shown = 0;
        for (int_t p = 0; p < state->panel_count && shown < 16; ++p) {
            if (panel_stage[p] == 4)
                continue;
            int_t gid = state->h_panel_gids[p];
            fprintf(stderr,
                    "  pending panel=%lld gid=%lld stage=%d fmod=%d\n",
                    (long long) p, (long long) gid, panel_stage[p], fmod[gid]);
            ++shown;
        }
    }
    fflush(stderr);

    /* A live kernel cannot be safely torn down after a watchdog timeout. */
    usleep(1000000);
    MPI_Abort(state->comm, -1);
    abort();

    if (stream) cudaStreamDestroy(stream);
    if (fmod) cudaFreeHost(fmod);
    if (panel_stage) cudaFreeHost(panel_stage);
    return -1;
}

#endif /* HAVE_NVSHMEM */

} /* namespace */

extern "C" int
dSymLDLNVSHMEMSolveAvailable(void)
{
#ifdef HAVE_NVSHMEM
    return 1;
#else
    return 0;
#endif
}

extern "C" dSymLDLNVSHMEMSolveHandle
dSymLDLNVSHMEMSolveCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLNVPanelDesc *panels,
    int_t block_count, const dSymLDLNVBlockDesc *blocks,
    int_t row_count, const int_t *rows, const int_t *xsup,
    const int *diag_owner, const int_t *x_offsets,
    const int_t *lsum_offsets, int znp, MPI_Comm comm)
{
#ifndef HAVE_NVSHMEM
    (void) n; (void) nsupers; (void) nrhs; (void) x_count;
    (void) lsum_count; (void) panel_count; (void) panels;
    (void) block_count; (void) blocks; (void) row_count; (void) rows;
    (void) xsup; (void) diag_owner; (void) x_offsets;
    (void) lsum_offsets; (void) znp; (void) comm;
    return NULL;
#else
    if (n <= 0 || nsupers <= 0 || nrhs <= 0 || x_count <= 0 ||
        lsum_count <= 0 || panel_count < 0 || block_count < 0 ||
        row_count < 0 || znp <= 0 || xsup == NULL || diag_owner == NULL ||
        x_offsets == NULL || lsum_offsets == NULL || comm == MPI_COMM_NULL)
        return NULL;
    if (nsupers > INT_MAX || panel_count > INT_MAX || block_count > INT_MAX ||
        (uint64_t) n >
            (uint64_t) std::numeric_limits<int_t>::max() / (uint64_t) nrhs)
        return NULL;

    dSymLDLNVSHMEMSolveState *state = new dSymLDLNVSHMEMSolveState();
    memset(state, 0, sizeof(*state));
    state->n = n;
    state->nsupers = nsupers;
    state->nrhs = nrhs;
    state->x_count = x_count;
    state->lsum_count = lsum_count;
    state->comm = comm;
    state->panel_count = panel_count;
    state->block_count = block_count;
    state->row_count = row_count;
    const char *diagnostics = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_DIAGNOSTICS");
    state->diagnostics = diagnostics != NULL && diagnostics[0] != '\0' &&
                         strcmp(diagnostics, "0") != 0;
    MPI_Comm_rank(comm, &state->rank);
    MPI_Comm_size(comm, &state->nprocs);

    if (state->diagnostics && panel_count > 0) {
        state->h_panel_gids = (int_t *) malloc(
            (size_t) panel_count * sizeof(*state->h_panel_gids));
        if (state->h_panel_gids == NULL) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        for (int_t p = 0; p < panel_count; ++p)
            state->h_panel_gids[p] = panels[p].gid;
    }

    if (!symldl_nvshmem_initialized) {
        nv_init_wrapper(comm);
        symldl_nvshmem_initialized = 1;
    }
    if (nvshmem_my_pe() != state->rank ||
        nvshmem_n_pes() != state->nprocs) {
        if (state->rank == 0)
            fprintf(stderr, "SymLDL NVSHMEM and MPI rank spaces differ.\n");
        symldl_nvshmem_state_free(state);
        return NULL;
    }

    std::vector<unsigned char> local_panel((size_t) nsupers, 0);
    std::vector<unsigned char> local_bcast((size_t) nsupers, 0);
    std::vector<unsigned char> local_reduce((size_t) nsupers, 0);
    std::vector<unsigned char> local_backward_bcast((size_t) nsupers, 0);
    std::vector<unsigned char> local_backward_reduce((size_t) nsupers, 0);
    std::vector<int> local_contrib((size_t) nsupers, 0);
    std::vector<int> local_backward_contrib((size_t) nsupers, 0);
    for (int_t p = 0; p < panel_count; ++p) {
        int_t gid = panels[p].gid;
        if (gid < 0 || gid >= nsupers || panels[p].width <= 0 ||
            panels[p].block_begin < 0 || panels[p].block_count < 0 ||
            panels[p].block_begin + panels[p].block_count > block_count) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        local_panel[(size_t) gid] = 1;
        local_bcast[(size_t) gid] = 1;
        for (int_t b = 0; b < panels[p].block_count; ++b) {
            int_t target = blocks[panels[p].block_begin + b].target_gid;
            if (target < 0 || target >= nsupers ||
                blocks[panels[p].block_begin + b].row_begin < 0 ||
                blocks[panels[p].block_begin + b].nbrow < 0 ||
                blocks[panels[p].block_begin + b].row_begin +
                    blocks[panels[p].block_begin + b].nbrow > row_count) {
                symldl_nvshmem_state_free(state);
                return NULL;
            }
            ++local_contrib[(size_t) target];
            local_reduce[(size_t) target] = 1;
            local_backward_bcast[(size_t) target] = 1;
        }
        if (panels[p].block_count > 0) {
            if (panels[p].block_count > INT_MAX -
                    local_backward_contrib[(size_t) gid]) {
                symldl_nvshmem_state_free(state);
                return NULL;
            }
            local_backward_contrib[(size_t) gid] +=
                (int) panels[p].block_count;
            local_backward_reduce[(size_t) gid] = 1;
        }
    }
    for (int_t k = 0; k < nsupers; ++k) {
        if (diag_owner[k] == state->rank) {
            local_bcast[(size_t) k] = 1;
            local_reduce[(size_t) k] = 1;
            local_backward_bcast[(size_t) k] = 1;
            local_backward_reduce[(size_t) k] = 1;
        }
    }

    size_t presence_count = (size_t) state->nprocs * (size_t) nsupers;
    std::vector<unsigned char> all_bcast(presence_count);
    std::vector<unsigned char> all_reduce(presence_count);
    std::vector<unsigned char> all_backward_bcast(presence_count);
    std::vector<unsigned char> all_backward_reduce(presence_count);
    MPI_Allgather(local_bcast.data(), (int) nsupers, MPI_UNSIGNED_CHAR,
                  all_bcast.data(), (int) nsupers, MPI_UNSIGNED_CHAR, comm);
    MPI_Allgather(local_reduce.data(), (int) nsupers, MPI_UNSIGNED_CHAR,
                  all_reduce.data(), (int) nsupers, MPI_UNSIGNED_CHAR, comm);
    MPI_Allgather(local_backward_bcast.data(), (int) nsupers,
                  MPI_UNSIGNED_CHAR, all_backward_bcast.data(),
                  (int) nsupers, MPI_UNSIGNED_CHAR, comm);
    MPI_Allgather(local_backward_reduce.data(), (int) nsupers,
                  MPI_UNSIGNED_CHAR, all_backward_reduce.data(),
                  (int) nsupers, MPI_UNSIGNED_CHAR, comm);

    std::vector<C_Tree> bcast_trees((size_t) nsupers);
    std::vector<C_Tree> reduce_trees((size_t) nsupers);
    std::vector<C_Tree> backward_bcast_trees((size_t) nsupers);
    std::vector<C_Tree> backward_reduce_trees((size_t) nsupers);
    std::vector<int> fmod((size_t) nsupers, 0);
    std::vector<int> bmod((size_t) nsupers, 0);
    std::vector<int_t> wait_gids;
    std::vector<int_t> backward_wait_gids;
    std::vector<int> wait_status((size_t) 2 * (size_t) nsupers, 1);
    std::vector<int> backward_wait_status(
        (size_t) 2 * (size_t) nsupers, 1);
    std::vector<int> ranks((size_t) state->nprocs);

    for (int_t k = 0; k < nsupers; ++k) {
        int root = diag_owner[k];
        if (root < 0 || root >= state->nprocs) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }

        int count = 0;
        ranks[(size_t) count++] = root;
        for (int p = 0; p < state->nprocs; ++p)
            if (p != root && all_bcast[(size_t) p * nsupers + k])
                ranks[(size_t) count++] = p;
        if (local_bcast[(size_t) k]) {
            int needrecv = 0;
            C_BcTree_Create_nv(&bcast_trees[(size_t) k], comm,
                               ranks.data(), count,
                               (int) ((xsup[k + 1] - xsup[k]) * nrhs),
                               'd', &needrecv);
        } else {
            C_BcTree_Nullify(&bcast_trees[(size_t) k]);
        }

        count = 0;
        ranks[(size_t) count++] = root;
        for (int p = 0; p < state->nprocs; ++p)
            if (p != root && all_reduce[(size_t) p * nsupers + k])
                ranks[(size_t) count++] = p;
        if (local_reduce[(size_t) k]) {
            int needrecv = 0;
            int needsend = 0;
            C_RdTree_Create_nv(&reduce_trees[(size_t) k], comm,
                               ranks.data(), count,
                               (int) ((xsup[k + 1] - xsup[k]) * nrhs),
                               'd', &needrecv, &needsend);
            fmod[(size_t) k] = local_contrib[(size_t) k] + needrecv;
            if (needrecv > 0) {
                wait_gids.push_back(k);
                for (int child = 0; child < needrecv; ++child) {
                    int child_index = reduce_trees[(size_t) k].myIdx *
                                      DEG_TREE + 1 + child;
                    int slot = child_index & 1;
                    wait_status[(size_t) 2 * k + slot] = 0;
                }
            }
        } else {
            C_RdTree_Nullify(&reduce_trees[(size_t) k]);
        }

        count = 0;
        ranks[(size_t) count++] = root;
        for (int p = 0; p < state->nprocs; ++p)
            if (p != root &&
                all_backward_bcast[(size_t) p * nsupers + k])
                ranks[(size_t) count++] = p;
        if (local_backward_bcast[(size_t) k]) {
            int needrecv = 0;
            C_BcTree_Create_nv(&backward_bcast_trees[(size_t) k], comm,
                               ranks.data(), count,
                               (int) ((xsup[k + 1] - xsup[k]) * nrhs),
                               'd', &needrecv);
        } else {
            C_BcTree_Nullify(&backward_bcast_trees[(size_t) k]);
        }

        count = 0;
        ranks[(size_t) count++] = root;
        for (int p = 0; p < state->nprocs; ++p)
            if (p != root &&
                all_backward_reduce[(size_t) p * nsupers + k])
                ranks[(size_t) count++] = p;
        if (local_backward_reduce[(size_t) k]) {
            int needrecv = 0;
            int needsend = 0;
            C_RdTree_Create_nv(&backward_reduce_trees[(size_t) k], comm,
                               ranks.data(), count,
                               (int) ((xsup[k + 1] - xsup[k]) * nrhs),
                               'd', &needrecv, &needsend);
            bmod[(size_t) k] = local_backward_contrib[(size_t) k] +
                              needrecv;
            if (needrecv > 0) {
                backward_wait_gids.push_back(k);
                for (int child = 0; child < needrecv; ++child) {
                    int child_index =
                        backward_reduce_trees[(size_t) k].myIdx *
                        DEG_TREE + 1 + child;
                    int slot = child_index & 1;
                    backward_wait_status[(size_t) 2 * k + slot] = 0;
                }
            }
        } else {
            C_RdTree_Nullify(&backward_reduce_trees[(size_t) k]);
        }
    }

    std::vector<int_t> backward_block_order((size_t) block_count);
    for (int_t b = 0; b < block_count; ++b)
        backward_block_order[(size_t) b] = b;
    std::sort(backward_block_order.begin(), backward_block_order.end(),
              [blocks, panels](int_t a, int_t b) {
                  if (blocks[a].target_gid != blocks[b].target_gid)
                      return blocks[a].target_gid > blocks[b].target_gid;
                  int_t ak = panels[blocks[a].panel_id].gid;
                  int_t bk = panels[blocks[b].panel_id].gid;
                  if (ak != bk)
                      return ak > bk;
                  return a < b;
              });
    std::vector<int_t> backward_target_gids;
    std::vector<int_t> backward_target_offsets;
    backward_target_offsets.push_back(0);
    int_t pos = 0;
    for (int_t next = nsupers; next > 0; --next) {
        int_t target = next - 1;
        int_t begin = pos;
        while (pos < block_count &&
               blocks[backward_block_order[(size_t) pos]].target_gid ==
                   target)
            ++pos;
        if (pos > begin || diag_owner[target] == state->rank) {
            backward_target_gids.push_back(target);
            backward_target_offsets.push_back(pos);
        }
    }
    if (pos != block_count) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }

    for (int_t k = 0; k < nsupers; ++k) {
        if (diag_owner[k] == state->rank && !local_panel[(size_t) k]) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        if (local_reduce[(size_t) k] && lsum_offsets[k] < 0) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        if (diag_owner[k] == state->rank && x_offsets[k] < 0) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
    }

    if (!symldl_cuda_copy_to_device(&state->d_panels, panels, panel_count,
                                     "SymLDL NVSHMEM panel metadata") ||
        !symldl_cuda_copy_to_device(&state->d_blocks, blocks, block_count,
                                     "SymLDL NVSHMEM block metadata") ||
        !symldl_cuda_copy_to_device(&state->d_rows, rows, row_count,
                                     "SymLDL NVSHMEM row metadata") ||
        !symldl_cuda_copy_to_device(&state->d_xsup, xsup, nsupers + 1,
                                     "SymLDL NVSHMEM supernode metadata") ||
        !symldl_cuda_copy_to_device(&state->d_diag_owner, diag_owner,
                                     nsupers, "SymLDL NVSHMEM owners") ||
        !symldl_cuda_copy_to_device(&state->d_x_offsets, x_offsets,
                                     nsupers, "SymLDL NVSHMEM X offsets") ||
        !symldl_cuda_copy_to_device(&state->d_lsum_offsets, lsum_offsets,
                                     nsupers, "SymLDL NVSHMEM sum offsets") ||
        !symldl_cuda_copy_to_device(&state->d_bcast_trees,
                                     bcast_trees.data(), nsupers,
                                     "SymLDL NVSHMEM broadcast trees") ||
        !symldl_cuda_copy_to_device(&state->d_reduce_trees,
                                     reduce_trees.data(), nsupers,
                                     "SymLDL NVSHMEM reduction trees") ||
        !symldl_cuda_copy_to_device(&state->d_backward_bcast_trees,
                                     backward_bcast_trees.data(), nsupers,
                                     "SymLDL NVSHMEM backward broadcast trees") ||
        !symldl_cuda_copy_to_device(&state->d_backward_reduce_trees,
                                     backward_reduce_trees.data(), nsupers,
                                     "SymLDL NVSHMEM backward reduction trees") ||
        !symldl_cuda_copy_to_device(&state->d_fmod_initial, fmod.data(),
                                     nsupers, "SymLDL NVSHMEM fmod") ||
        !symldl_cuda_copy_to_device(&state->d_bmod_initial, bmod.data(),
                                     nsupers, "SymLDL NVSHMEM bmod") ||
        !symldl_cuda_copy_to_device(&state->d_wait_gids, wait_gids.data(),
                                     (int_t) wait_gids.size(),
                                     "SymLDL NVSHMEM wait list") ||
        !symldl_cuda_copy_to_device(&state->d_backward_wait_gids,
                                     backward_wait_gids.data(),
                                     (int_t) backward_wait_gids.size(),
                                     "SymLDL NVSHMEM backward wait list") ||
        !symldl_cuda_copy_to_device(&state->d_wait_status,
                                     wait_status.data(), 2 * nsupers,
                                     "SymLDL NVSHMEM wait status") ||
        !symldl_cuda_copy_to_device(&state->d_wait_status_initial,
                                     wait_status.data(), 2 * nsupers,
                                     "SymLDL NVSHMEM initial wait status") ||
        !symldl_cuda_copy_to_device(
                                     &state->d_backward_wait_status_initial,
                                     backward_wait_status.data(), 2 * nsupers,
                                     "SymLDL NVSHMEM backward wait status") ||
        !symldl_cuda_copy_to_device(&state->d_backward_block_order,
                                     backward_block_order.data(), block_count,
                                     "SymLDL NVSHMEM backward block order") ||
        !symldl_cuda_copy_to_device(&state->d_backward_target_gids,
                                     backward_target_gids.data(),
                                     (int_t) backward_target_gids.size(),
                                     "SymLDL NVSHMEM backward target list") ||
        !symldl_cuda_copy_to_device(&state->d_backward_target_offsets,
                                     backward_target_offsets.data(),
                                     (int_t) backward_target_offsets.size(),
                                     "SymLDL NVSHMEM backward target offsets") ||
        !symldl_cuda_alloc(&state->d_backward_forwarded, nsupers,
                           "SymLDL NVSHMEM backward broadcast status") ||
        !symldl_cuda_alloc(&state->d_fmod, nsupers,
                           "SymLDL NVSHMEM fmod workspace") ||
        !symldl_cuda_alloc(&state->d_next_panel, 1,
                           "SymLDL NVSHMEM panel work counter") ||
        !symldl_cuda_alloc(&state->d_x, x_count,
                           "SymLDL NVSHMEM X workspace") ||
        !symldl_cuda_alloc(&state->d_lsum, lsum_count,
                           "SymLDL NVSHMEM sum workspace") ||
        !symldl_cuda_alloc(&state->d_bsum, n * nrhs,
                           "SymLDL NVSHMEM backward sum workspace")) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    if (state->diagnostics && panel_count > 0) {
        size_t panel_stage_bytes =
            (size_t) panel_count * sizeof(*state->h_panel_stage);
        if (!symldl_cuda_ok(cudaHostAlloc(
                                (void **) &state->h_panel_stage,
                                panel_stage_bytes, cudaHostAllocMapped),
                            "SymLDL NVSHMEM panel diagnostics") ||
            !symldl_cuda_ok(cudaHostGetDevicePointer(
                                (void **) &state->d_panel_stage,
                                state->h_panel_stage, 0),
                            "SymLDL NVSHMEM panel diagnostics mapping")) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        memset(state->h_panel_stage, 0, panel_stage_bytes);
    }
    state->wait_count = (int_t) wait_gids.size();
    state->backward_wait_count = (int_t) backward_wait_gids.size();
    state->backward_target_count =
        (int_t) backward_target_gids.size();

    state->flag_bc = (uint64_t *) nvshmem_calloc((size_t) nsupers,
                                                 sizeof(uint64_t));
    state->flag_rd = (uint64_t *) nvshmem_calloc((size_t) 2 * nsupers,
                                                 sizeof(uint64_t));
    state->ready_x = (double *) nvshmem_calloc(
        (size_t) n * nrhs, sizeof(double));
    state->ready_lsum = (double *) nvshmem_calloc(
        (size_t) 2 * n * nrhs, sizeof(double));
    int device = 0;
    if (state->flag_bc == NULL || state->flag_rd == NULL ||
        state->ready_x == NULL || state->ready_lsum == NULL ||
        !symldl_cuda_ok(cudaGetDevice(&device),
                        "SymLDL NVSHMEM active device") ||
        !symldl_cuda_ok(cudaDeviceGetAttribute(
                            &state->progress_blocks,
                            cudaDevAttrMultiProcessorCount, device),
                        "SymLDL NVSHMEM multiprocessor count") ||
        !symldl_cuda_ok(cudaStreamCreateWithFlags(
                            &state->streams[0], cudaStreamNonBlocking),
                        "SymLDL NVSHMEM wait stream") ||
        !symldl_cuda_ok(cudaStreamCreateWithFlags(
                            &state->streams[1], cudaStreamNonBlocking),
                        "SymLDL NVSHMEM compute stream")) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    if (state->progress_blocks < 3) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    int max_waiter_blocks = state->progress_blocks - 2;
    state->backward_waiter_blocks =
        znp > max_waiter_blocks / 2 ? max_waiter_blocks : 2 * znp;
    const char *waiter_blocks_env = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_WAITERS");
    if (waiter_blocks_env != NULL && waiter_blocks_env[0] != '\0') {
        char *end = NULL;
        long waiter_blocks = strtol(waiter_blocks_env, &end, 10);
        if (end == waiter_blocks_env || *end != '\0' ||
            waiter_blocks < 1 ||
            waiter_blocks > max_waiter_blocks) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        state->backward_waiter_blocks = (int) waiter_blocks;
    }
    if (state->backward_waiter_blocks >
        INT_MAX / SYMLDL_NVSHMEM_WAIT_THREADS) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    state->waiter_handshake_threads =
        state->backward_waiter_blocks * SYMLDL_NVSHMEM_WAIT_THREADS;
    size_t waiter_handshake_count =
        1 + 2 * (size_t) state->waiter_handshake_threads;
    if (waiter_handshake_count > SIZE_MAX / sizeof(int) ||
        !symldl_cuda_ok(cudaHostAlloc(
                            (void **) &state->h_waiter_started,
                            waiter_handshake_count * sizeof(int),
                            cudaHostAllocMapped),
                        "SymLDL NVSHMEM waiter handshake") ||
        !symldl_cuda_ok(cudaHostGetDevicePointer(
                            (void **) &state->d_waiter_started,
                            state->h_waiter_started, 0),
                        "SymLDL NVSHMEM waiter handshake mapping")) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    state->worker_blocks = state->progress_blocks - 1;
    state->backward_worker_blocks =
        state->progress_blocks - state->backward_waiter_blocks;
    const char *worker_blocks_env = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_WORKER_BLOCKS");
    if (worker_blocks_env != NULL && worker_blocks_env[0] != '\0') {
        char *end = NULL;
        long worker_blocks = strtol(worker_blocks_env, &end, 10);
        if (end == worker_blocks_env || *end != '\0' || worker_blocks < 1 ||
            worker_blocks >= INT_MAX) {
            symldl_nvshmem_state_free(state);
            return NULL;
        }
        state->worker_blocks = std::min(
            state->worker_blocks, (int) worker_blocks);
        state->backward_worker_blocks = std::min(
            state->backward_worker_blocks, (int) worker_blocks);
    }
    memset(state->h_waiter_started, 0,
           waiter_handshake_count * sizeof(int));

    nvshmem_barrier_all();
    return (dSymLDLNVSHMEMSolveHandle) state;
#endif
}

extern "C" int
dSymLDLNVSHMEMForward(dSymLDLNVSHMEMSolveHandle handle,
                      double *x, int_t x_count)
{
#ifndef HAVE_NVSHMEM
    (void) handle; (void) x; (void) x_count;
    return -1;
#else
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;

    double start = SuperLU_timer_();
    size_t x_bytes = (size_t) state->x_count * sizeof(double);
    size_t lsum_bytes = (size_t) state->lsum_count * sizeof(double);
    size_t fmod_bytes = (size_t) state->nsupers * sizeof(int);
    size_t flag_bc_bytes = (size_t) state->nsupers * sizeof(uint64_t);
    size_t flag_rd_bytes = (size_t) 2 * state->nsupers * sizeof(uint64_t);
    size_t wait_status_bytes = (size_t) 2 * state->nsupers * sizeof(int);
    size_t ready_x_bytes = (size_t) state->n * state->nrhs * sizeof(double);
    size_t ready_lsum_bytes = (size_t) 2 * state->n * state->nrhs *
                              sizeof(double);

    if (!symldl_cuda_ok(cudaMemcpyAsync(state->d_x, x, x_bytes,
                                        cudaMemcpyHostToDevice,
                                        state->streams[1]),
                        "SymLDL NVSHMEM X upload") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->d_lsum, 0, lsum_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM sum reset") ||
        !symldl_cuda_ok(cudaMemcpyAsync(state->d_fmod,
                                        state->d_fmod_initial, fmod_bytes,
                                        cudaMemcpyDeviceToDevice,
                                        state->streams[1]),
                        "SymLDL NVSHMEM fmod reset") ||
        !symldl_cuda_ok(cudaMemcpyAsync(state->d_wait_status,
                                        state->d_wait_status_initial,
                                        wait_status_bytes,
                                        cudaMemcpyDeviceToDevice,
                                        state->streams[1]),
                        "SymLDL NVSHMEM wait status reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->d_next_panel, 0,
                                        sizeof(*state->d_next_panel),
                                        state->streams[1]),
                        "SymLDL NVSHMEM panel work counter reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->flag_bc, 0, flag_bc_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM broadcast reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->flag_rd, 0, flag_rd_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM reduction reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->ready_x, 0, ready_x_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM ready-X reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->ready_lsum, 0,
                                        ready_lsum_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM ready-sum reset") ||
        (state->diagnostics && state->panel_count > 0 &&
         !symldl_cuda_ok(cudaMemsetAsync(
                            state->d_panel_stage, 0,
                            (size_t) state->panel_count * sizeof(int),
                            state->streams[1]),
                         "SymLDL NVSHMEM panel diagnostics reset")) ||
        !symldl_cuda_ok(cudaStreamSynchronize(state->streams[1]),
                        "SymLDL NVSHMEM initialization"))
        return -2;
    state->timer_h2d += SuperLU_timer_() - start;

    nvshmem_barrier_all();
    start = SuperLU_timer_();
    memset(state->h_waiter_started, 0,
           (size_t) (1 + 2 * state->waiter_handshake_threads) *
               sizeof(int));
    __sync_synchronize();

    dim3 blocks(state->worker_blocks + 1);
    dim3 threads(DIM_X, DIM_Y);
    void *args[] = {
        &state->d_panels, &state->panel_count, &state->nsupers,
        &state->d_blocks, &state->d_rows, &state->d_xsup,
        &state->d_diag_owner, &state->d_x_offsets,
        &state->d_lsum_offsets, &state->nrhs, &state->rank,
        &state->d_x, &state->d_lsum, &state->d_fmod,
        &state->d_bcast_trees, &state->d_reduce_trees,
        &state->flag_bc, &state->flag_rd, &state->ready_x,
        &state->ready_lsum, &state->d_wait_gids,
        &state->wait_count, &state->d_wait_status,
        &state->d_next_panel, &state->d_waiter_started,
        &state->d_panel_stage
    };
    int launch_status = nvshmemx_collective_launch(
        (const void *) symldl_nvshmem_forward_progress_kernel,
        blocks, threads, args, 0, state->streams[1]);
    if (launch_status != 0 ||
        !symldl_cuda_ok(cudaGetLastError(),
                        "SymLDL NVSHMEM forward progress launch"))
        return -3;

    double waiter_deadline = SuperLU_timer_() + 2.0;
    while (*(volatile int *) state->h_waiter_started == 0 &&
           SuperLU_timer_() < waiter_deadline)
        usleep(100);
    if (*(volatile int *) state->h_waiter_started == 0) {
        fprintf(stderr,
                "SymLDL NVSHMEM progress kernel did not start on rank %d.\n",
                state->rank);
        fflush(stderr);
        return -8;
    }

    if (state->diagnostics &&
        symldl_nvshmem_wait_with_diagnostics(state) != 0)
        return -7;
    if (!symldl_cuda_ok(cudaStreamSynchronize(state->streams[1]),
                        "SymLDL NVSHMEM forward compute") ||
        !symldl_cuda_ok(cudaStreamSynchronize(state->streams[0]),
                        "SymLDL NVSHMEM forward reductions"))
        return -5;
    nvshmem_barrier_all();
    double elapsed = SuperLU_timer_() - start;
    state->timer_forward += elapsed;
    state->timer_forward_phase += elapsed;

    start = SuperLU_timer_();
    if (!symldl_cuda_ok(cudaMemcpy(x, state->d_x, x_bytes,
                                  cudaMemcpyDeviceToHost),
                        "SymLDL NVSHMEM X download"))
        return -6;
    state->timer_d2h += SuperLU_timer_() - start;
    return 0;
#endif
}

extern "C" int
dSymLDLNVSHMEMDiagonal(dSymLDLNVSHMEMSolveHandle handle,
                       double *x, int_t x_count)
{
#ifndef HAVE_NVSHMEM
    (void) handle; (void) x; (void) x_count;
    return -1;
#else
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;

    double start = SuperLU_timer_();
    if (state->panel_count > 0) {
        dim3 threads(DIM_X, DIM_Y);
        symldl_nvshmem_diagonal_kernel<<<
            state->panel_count, threads, 0, state->streams[1]>>>(
                state->d_panels, state->panel_count, state->d_x_offsets,
                state->d_lsum_offsets, state->nrhs, state->rank,
                state->d_x, state->d_lsum);
        if (!symldl_cuda_ok(cudaGetLastError(),
                            "SymLDL NVSHMEM diagonal launch") ||
            !symldl_cuda_ok(cudaStreamSynchronize(state->streams[1]),
                            "SymLDL NVSHMEM diagonal apply"))
            return -2;
    }
    double elapsed = SuperLU_timer_() - start;
    state->timer_forward += elapsed;
    state->timer_diagonal_phase += elapsed;

    start = SuperLU_timer_();
    size_t x_bytes = (size_t) state->x_count * sizeof(double);
    if (!symldl_cuda_ok(cudaMemcpy(x, state->d_x, x_bytes,
                                  cudaMemcpyDeviceToHost),
                        "SymLDL NVSHMEM diagonal download"))
        return -3;
    state->timer_d2h += SuperLU_timer_() - start;
    return 0;
#endif
}

extern "C" int
dSymLDLNVSHMEMBackward(dSymLDLNVSHMEMSolveHandle handle,
                       double *x, int_t x_count)
{
#ifndef HAVE_NVSHMEM
    (void) handle; (void) x; (void) x_count;
    return -1;
#else
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count ||
        state->worker_blocks < 1 || state->backward_worker_blocks < 1 ||
        state->backward_waiter_blocks < 1)
        return -1;

    double start = SuperLU_timer_();
    int_t dense_count = state->n * state->nrhs;
    size_t x_bytes = (size_t) state->x_count * sizeof(double);
    size_t dense_bytes = (size_t) dense_count * sizeof(double);
    size_t mod_bytes = (size_t) state->nsupers * sizeof(int);
    size_t flag_bc_bytes = (size_t) state->nsupers * sizeof(uint64_t);
    size_t flag_rd_bytes = (size_t) 2 * state->nsupers * sizeof(uint64_t);
    size_t wait_status_bytes = (size_t) 2 * state->nsupers * sizeof(int);
    size_t ready_lsum_bytes = (size_t) 2 * dense_count * sizeof(double);

    if (!symldl_cuda_ok(cudaMemcpyAsync(state->d_x, x, x_bytes,
                                        cudaMemcpyHostToDevice,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward X upload") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->d_bsum, 0, dense_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward sum reset") ||
        !symldl_cuda_ok(cudaMemcpyAsync(state->d_fmod,
                                        state->d_bmod_initial, mod_bytes,
                                        cudaMemcpyDeviceToDevice,
                                        state->streams[1]),
                        "SymLDL NVSHMEM bmod reset") ||
        !symldl_cuda_ok(cudaMemcpyAsync(
                            state->d_wait_status,
                            state->d_backward_wait_status_initial,
                            wait_status_bytes, cudaMemcpyDeviceToDevice,
                            state->streams[1]),
                        "SymLDL NVSHMEM backward wait status reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->d_next_panel, 0,
                                        sizeof(*state->d_next_panel),
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward work counter reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->d_backward_forwarded, 0,
                                        mod_bytes, state->streams[1]),
                        "SymLDL NVSHMEM backward broadcast reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->flag_bc, 0, flag_bc_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward broadcast flags reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->flag_rd, 0, flag_rd_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward reduction flags reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->ready_x, 0, dense_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward ready-X reset") ||
        !symldl_cuda_ok(cudaMemsetAsync(state->ready_lsum, 0,
                                        ready_lsum_bytes,
                                        state->streams[1]),
                        "SymLDL NVSHMEM backward ready-sum reset") ||
        !symldl_cuda_ok(cudaStreamSynchronize(state->streams[1]),
                        "SymLDL NVSHMEM backward initialization"))
        return -2;
    state->timer_h2d += SuperLU_timer_() - start;

    nvshmem_barrier_all();
    start = SuperLU_timer_();
    memset(state->h_waiter_started, 0,
           (size_t) (1 + 2 * state->waiter_handshake_threads) *
               sizeof(int));
    __sync_synchronize();

    dim3 blocks(state->backward_waiter_blocks +
                state->backward_worker_blocks);
    dim3 threads(DIM_X, DIM_Y);
    void *args[] = {
        &state->d_panels, &state->d_blocks,
        &state->d_backward_block_order, &state->d_backward_target_gids,
        &state->d_backward_target_offsets, &state->backward_target_count,
        &state->d_rows, &state->nsupers, &state->d_xsup,
        &state->d_diag_owner,
        &state->d_x_offsets, &state->nrhs, &state->rank,
        &state->d_x, &state->d_bsum, &state->d_fmod,
        &state->d_backward_bcast_trees,
        &state->d_backward_reduce_trees,
        &state->flag_bc, &state->flag_rd, &state->ready_x,
        &state->ready_lsum, &state->d_backward_wait_gids,
        &state->backward_wait_count, &state->d_wait_status,
        &state->d_next_panel, &state->d_backward_forwarded,
        &state->d_waiter_started, &state->backward_waiter_blocks
    };
    int launch_status = nvshmemx_collective_launch(
        (const void *) symldl_nvshmem_backward_progress_kernel,
        blocks, threads, args, 0, state->streams[1]);
    if (launch_status != 0 ||
        !symldl_cuda_ok(cudaGetLastError(),
                        "SymLDL NVSHMEM backward progress launch"))
        return -3;

    double waiter_deadline = SuperLU_timer_() + 2.0;
    while (*(volatile int *) state->h_waiter_started == 0 &&
           SuperLU_timer_() < waiter_deadline)
        usleep(100);
    if (*(volatile int *) state->h_waiter_started == 0) {
        fprintf(stderr,
                "SymLDL NVSHMEM backward progress kernel did not start "
                "on rank %d.\n", state->rank);
        fflush(stderr);
        return -4;
    }

    if (!symldl_cuda_ok(cudaStreamSynchronize(state->streams[1]),
                        "SymLDL NVSHMEM backward compute"))
        return -5;
    nvshmem_barrier_all();
    double elapsed = SuperLU_timer_() - start;
    state->timer_forward += elapsed;
    state->timer_backward_phase += elapsed;

    start = SuperLU_timer_();
    if (!symldl_cuda_ok(cudaMemcpy(x, state->d_x, x_bytes,
                                  cudaMemcpyDeviceToHost),
                        "SymLDL NVSHMEM backward X download"))
        return -6;
    state->timer_d2h += SuperLU_timer_() - start;
    return 0;
#endif
}

extern "C" void
dSymLDLNVSHMEMSolveTakeTimers(dSymLDLNVSHMEMSolveHandle handle,
                              double *h2d, double *forward, double *d2h)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (h2d) *h2d = state ? state->timer_h2d : 0.0;
    if (forward) *forward = state ? state->timer_forward : 0.0;
    if (d2h) *d2h = state ? state->timer_d2h : 0.0;
    if (state) {
        state->timer_h2d = 0.0;
        state->timer_forward = 0.0;
        state->timer_d2h = 0.0;
    }
}

extern "C" void
dSymLDLNVSHMEMSolveTakePhaseTimers(
    dSymLDLNVSHMEMSolveHandle handle, double *forward,
    double *diagonal, double *backward)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (forward) *forward = state ? state->timer_forward_phase : 0.0;
    if (diagonal) *diagonal = state ? state->timer_diagonal_phase : 0.0;
    if (backward) *backward = state ? state->timer_backward_phase : 0.0;
    if (state) {
        state->timer_forward_phase = 0.0;
        state->timer_diagonal_phase = 0.0;
        state->timer_backward_phase = 0.0;
    }
}

extern "C" void
dSymLDLNVSHMEMSolveDestroy(dSymLDLNVSHMEMSolveHandle handle)
{
    symldl_nvshmem_state_free((dSymLDLNVSHMEMSolveState *) handle);
}
