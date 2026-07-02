#pragma once
#include <cassert>
#include <algorithm>
#include <cmath>
#include "superlu_defs.h"
#include "superlu_dist_config.h"

// #ifdef HAVE_CUDA
#define EPSILON 1e-3
#include <cuda_runtime.h>

#include "cublas_v2.h"


#include "lupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "gpu_mpi_utils.hpp"

#include <cmath>
#include <complex>
#include <cassert>
#include <cstdio>
#include <cstdlib>

static inline bool sym_v2_trace_exchange_enabled()
{
    const char *env = std::getenv("GPU3DV2_TRACE_EXCHANGE");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

#define SYM_V2_TRACE_EXCHANGE(grid3d_, k_, fmt_, ...)                         \
    do                                                                        \
    {                                                                         \
        if (sym_v2_trace_exchange_enabled())                                  \
        {                                                                     \
            std::printf("[sym-v2-exchange] rank %d k %d: " fmt_ "\n",        \
                        (grid3d_ != NULL) ? (grid3d_)->iam : -1,              \
                        static_cast<int>(k_), ##__VA_ARGS__);                 \
            std::fflush(stdout);                                              \
        }                                                                     \
    } while (0)

static __global__ void sym_l2u_pack_kernel(const double *lpanel,
                                           double *sendbuf,
                                           const int_t *sendmap,
                                           int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    sendbuf[idx] = (src < 0) ? (double)(-src - 1) : lpanel[src];
}

static __global__ void sym_l2u_pack_segments_kernel(
    const double *lpanel,
    double *sendbuf,
    const int_t *base_sendmap,
    const SymV2RowDownSendSegmentGPU *segments,
    int nsegments,
    int_t ksupc,
    int_t dst_lda)
{
    int seg_id = blockIdx.x;
    if (seg_id >= nsegments)
        return;

    SymV2RowDownSendSegmentGPU seg = segments[seg_id];
    int_t nrows = seg.nrows;
    if (nrows <= 0 || ksupc <= 0 || dst_lda <= 0)
        return;

    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        size_t map_pos = seg.map_offset +
                         static_cast<size_t>(row) +
                         static_cast<size_t>(col) *
                             static_cast<size_t>(nrows);
        int_t src = base_sendmap[map_pos];
        sendbuf[seg.dst_row_offset + row + col * dst_lda] =
            (src < 0) ? (double)(-src - 1) : lpanel[src];
    }
}

static __global__ void sym_l2u_pack_segments_warp_kernel(
    const double *lpanel,
    double *sendbuf,
    const int_t *base_sendmap,
    const SymV2RowDownSendSegmentGPU *segments,
    int nsegments,
    int_t ksupc,
    int_t dst_lda)
{
    const int warp_size = 32;
    int warps_per_block = blockDim.x / warp_size;
    if (warps_per_block <= 0)
        return;

    int warp_in_block = threadIdx.x / warp_size;
    int lane = threadIdx.x - warp_in_block * warp_size;
    int seg_id = blockIdx.x * warps_per_block + warp_in_block;
    if (seg_id >= nsegments)
        return;

    SymV2RowDownSendSegmentGPU seg = segments[seg_id];
    int_t nrows = seg.nrows;
    if (nrows <= 0 || ksupc <= 0 || dst_lda <= 0)
        return;

    int_t count = nrows * ksupc;
    for (int_t idx = lane; idx < count; idx += warp_size)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        size_t map_pos = seg.map_offset +
                         static_cast<size_t>(row) +
                         static_cast<size_t>(col) *
                             static_cast<size_t>(nrows);
        int_t src = base_sendmap[map_pos];
        sendbuf[seg.dst_row_offset + row + col * dst_lda] =
            (src < 0) ? (double)(-src - 1) : lpanel[src];
    }
}

static __global__ void sym_l2u_pack_raw_kernel(const double *lpanel,
                                               double *sendbuf,
                                               const int_t *sendmap,
                                               int count,
                                               int_t panel_ld,
                                               const double *diag,
                                               int_t diag_ld)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    if (src < 0)
    {
        sendbuf[idx] = (double)(-src - 1);
        return;
    }

    int_t row = src % panel_ld;
    int_t col = src / panel_ld;
    double sum = 0.0;
    for (int_t p = 0; p < diag_ld; ++p)
        sum += lpanel[row + p * panel_ld] * diag[p + col * diag_ld];
    sendbuf[idx] = sum;
}

static __global__ void sym_l2u_local_gather_kernel(const double *lpanel,
                                                   double *upanel,
                                                   const int_t *local_map,
                                                   int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = local_map[idx];
    upanel[idx] = (src < 0) ? 0.0 : lpanel[src];
}

static __global__ void sym_lfrag_assemble_kernel(const double *stage,
                                                 double *frag,
                                                 const int_t *recv_map,
                                                 int pieces,
                                                 int_t ksupc,
                                                 int_t frag_lda)
{
    int piece = blockIdx.x;
    if (piece >= pieces)
        return;

    int_t dst_offset = recv_map[3 * piece];
    int_t nrows = recv_map[3 * piece + 1];
    int_t src_offset = recv_map[3 * piece + 2];
    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        frag[dst_offset + row + col * frag_lda] =
            stage[src_offset + row + col * nrows];
    }
}

template <typename T>
static __global__ void sym_lpanel_transform_inplace_kernel(T *panel,
                                                           int_t panel_ld,
                                                           const T *diag,
                                                           int_t diag_ld,
                                                           int_t len,
                                                           int_t ksupsz)
{
    extern __shared__ unsigned char raw_storage[];
    T *raw_row = reinterpret_cast<T *>(raw_storage);
    int_t row = blockIdx.x;

    if (row >= len)
        return;

    for (int_t p = threadIdx.x; p < ksupsz; p += blockDim.x)
        raw_row[p] = panel[row + p * panel_ld];
    __syncthreads();

    for (int_t j = threadIdx.x; j < ksupsz; j += blockDim.x)
    {
        T sum = T(0);
        for (int_t p = 0; p < ksupsz; ++p)
            sum += raw_row[p] * diag[p + j * diag_ld];
        panel[row + j * panel_ld] = sum;
    }
}

static inline bool superlu_sym_gpu_fuse_lpanel()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;

    const char *env = std::getenv("GPU3DFUSE_LPANEL");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }

    int enabled = superlu_env_truthy(env);
    if (enabled < 0)
        ABORT("GPU3DFUSE_LPANEL must be a boolean value.");

    cached = (enabled > 0) ? 1 : 0;
    return cached != 0;
}

template<typename T>
int checkArr(const T *A, const T *B, int n)
{
    double nrmA = 0;
    for (int i = 0; i < n; i++) {
        // For complex numbers, std::norm gives the squared magnitude.
        nrmA += sqnorm(A[i]);
    }
    nrmA = std::sqrt(nrmA);

    for (int i = 0; i < n; i++) {
        // Use std::abs for both real and complex numbers to get the magnitude.
        // assert(std::abs(A[i] - B[i]) <= EPSILON * nrmA / n);
        assert(std::sqrt(sqnorm(A[i] - B[i])) <= EPSILON * nrmA / n);
    }

    return 0;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuErrchk(cudaMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(cudaMalloc(&gpuPanel.val, valSize));

    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();
    size_t valOffset = idxSize;
    const size_t align = alignof(T);
    const size_t mask = align - 1;
    if (align > 1)
        valOffset = (valOffset + mask) & ~mask;

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr + valOffset;
    gpuPanel.val = (T *) basePtr; 

    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int_t xlpanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(cudaMemcpy(val, gpuPanel.val,  valSize, cudaMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int_t xupanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(cudaMemcpy(val, gpuPanel.val,  valSize, cudaMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int xupanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(cudaMemcpy(gpuPanel.val, val,  valSize, cudaMemcpyHostToDevice));
    return 0;
}

template <typename T>
int xlpanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(cudaMemcpy(gpuPanel.val, val,  valSize, cudaMemcpyHostToDevice));
    return 0;
}

template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuErrchk(cudaMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(cudaMalloc(&gpuPanel.val, valSize));

    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));
    return gpuPanel;
}

template <typename T>
int_t xupanel_t<T>::ensureGPUValueStorage()
{
    if (isEmpty())
        return 0;
    if (gpuPanel.val != NULL)
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    if (valSize == 0)
        return 0;
    gpuErrchk(cudaMalloc(&gpuPanel.val, valSize));
    return 0;
}

template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();
    size_t valOffset = idxSize;
    const size_t align = alignof(T);
    const size_t mask = align - 1;
    if (align > 1)
        valOffset = (valOffset + mask) & ~mask;

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr + valOffset;
    gpuPanel.val = (T *) basePtr; 

    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int xlpanel_t<T>::checkGPU()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    gpuErrchk(cudaMemcpy(tmpArr.data(), gpuPanel.val, valSize, cudaMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymStartL2UGPU(int_t k, int_t stream_offset)
{
    ABORT("LUv1 SymFact GPU L2U packing is implemented for double precision only.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    ABORT("SymFact GPU3D V2 raw L-fragment prepack is implemented for double precision only.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("GPU3DVERSION=2 raw L-fragment prepack requires GPU offload.");
    if (k < 0 || k >= nsupers || mycol != symV2PanelRoot(k))
        return 0;
    if (Pr <= 1)
    {
        int_t lk = symV2PanelIndex(k);
        if (lk < 0)
            return 0;
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (!lpanel.isEmpty() && superlu_sym_v2_wpanel_cache())
        {
            if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
                stream_offset = 0;
            if (static_cast<size_t>(stream_offset) >= symV2RawPanelNodes.size())
                ABORT("SymFact V2 W-panel ring is not initialized.");
            if (A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
                A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
                ABORT("SymFact V2 W-panel ring is not initialized.");
            cudaStream_t stream = A_gpu.cuStreams[stream_offset];
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
                static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
            gpuErrchk(cudaEventRecord(
                A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
            symV2RawPanelNodes[stream_offset] = k;
        }
        return 0;
    }
    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLPrepacked.empty())
        ABORT("SymFact V2 raw L-fragment prepack buffers are not allocated.");

    int_t lk = symV2PanelIndex(k);
    if (lk < 0 ||
        static_cast<size_t>(lk) >= symV2PartnerLPrepacked.size())
        ABORT("SymFact V2 raw L-fragment prepack has an invalid local panel.");

    /* A retained handle may execute another numerical factorization. */
    symV2PartnerLPrepacked[static_cast<size_t>(lk)] = 0;
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return 0;

    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];

    xlpanel_t<double> &lpanel = lPanelVec[lk];
    if (lpanel.isEmpty())
    {
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
        symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
            static_cast<unsigned char>(stream_offset + 1);
        return 0;
    }

    if (superlu_sym_v2_wpanel_cache())
    {
        if (static_cast<size_t>(stream_offset) >= symV2RawPanelNodes.size() ||
            A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
            A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
            ABORT("SymFact V2 W-panel ring is not initialized.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
            static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
        symV2RawPanelNodes[stream_offset] = k;
    }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double pack_issue_t = SuperLU_timer_();
#endif
    bool packed_any = false;
    for (int pc = 0; pc < Pc; ++pc)
    {
        size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc);
        if (flat >= symV2PartnerLSendSizes.size())
            ABORT("SymFact V2 raw L-fragment prepack size is missing.");
        int size = symV2PartnerLSendSizes[flat];
        if (size <= 0)
            continue;

        bool active_dest = false;
        for (int pr = 0; pr < Pr; ++pr)
        {
            size_t active_pos =
                flat * static_cast<size_t>(Pr) + static_cast<size_t>(pr);
            if (active_pos >= symV2PartnerLSendRowActive.size())
                ABORT("SymFact V2 raw L-fragment prepack row mask is missing.");
            if (symV2PartnerLSendRowActive[active_pos])
            {
                active_dest = true;
                break;
            }
        }
        if (!active_dest)
            continue;

        double *sendbuf = symV2PartnerLSendBufsGPU[flat];
        int_t *sendmap = symL2LSendMapsGPU[flat];
        if (sendbuf == NULL || sendmap == NULL)
            ABORT("SymFact V2 raw L-fragment prepack buffer is missing.");

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
            lpanel.gpuPanel.val, sendbuf, sendmap, size);
        packed_any = true;
    }
    if (packed_any)
        gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                     SYM_GPU3D_T_PARTNER_LFRAG_PACK_ISSUE,
                     SuperLU_timer_() - pack_issue_t);
#endif

    gpuErrchk(cudaEventRecord(
        A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
    symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
        static_cast<unsigned char>(stream_offset + 1);
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeGPU(
    int_t k, int_t stream_offset)
{
    ABORT("SymFact GPU3D V2 true symmetric L-fragment exchange is implemented for double precision only.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowBeginGPU(
    int_t k, int_t stream_offset)
{
    ABORT("SymFact GPU3D V2 Pc-fragment taskflow is implemented for double precision only.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
    int_t k, unsigned char kind, const Ftype *stage,
    const std::vector<int_t> &recv_map, int_t ksupc, cudaStream_t stream)
{
    (void)k;
    (void)kind;
    (void)stage;
    (void)recv_map;
    (void)ksupc;
    (void)stream;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowProgressExchangeGPU(
    int_t k, int drain)
{
    (void)k;
    (void)drain;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowProgressGPU(
    int_t k, int budget)
{
    (void)k;
    (void)budget;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowDispatchGPU(
    int streamId, int_t k, unsigned mode_mask, int_t mode_gid, int drain)
{
    (void)streamId;
    (void)k;
    (void)mode_mask;
    (void)mode_gid;
    (void)drain;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowDrainGPU(
    int_t k, unsigned mode_mask, int_t mode_gid)
{
    (void)k;
    (void)mode_mask;
    (void)mode_gid;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PcFragTaskflowReleaseGPU(int_t k)
{
    (void)k;
    return 0;
}

template <typename T>
static inline void dSymV2PcFragTaskflowRecyclePoolPush(
    std::vector<T> &pool, const T &block, const char *message)
{
    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
        pool.size() >= pool.capacity())
        ABORT(message);
    pool.push_back(block);
}

template <typename T>
static inline bool dSymV2PcFragTaskflowPoolHasBlock(
    const std::vector<T> &pool, size_t count)
{
    for (size_t i = 0; i < pool.size(); ++i)
        if (pool[i].ptr != NULL && pool[i].capacity >= count)
            return true;
    return false;
}

static inline double *dSymV2PcFragTaskflowEnsurePinnedHost(
    std::vector<xLUstruct_t<double>::SymV2PcFragHostValueBlock> &pool,
    double **buffer, size_t *capacity, size_t count,
    bool allow_late_alloc,
    long long *late_allocs = NULL)
{
    if (count == 0)
        return NULL;
    if (buffer == NULL || capacity == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW pinned host buffer handle is missing.");
    if (*capacity < count)
    {
        if (*buffer != NULL)
            dSymV2PcFragTaskflowRecyclePoolPush(
                pool,
                xLUstruct_t<double>::SymV2PcFragHostValueBlock(
                    *buffer, *capacity),
                "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host pool capacity is undersized.");
        size_t best = pool.size();
        size_t best_capacity = 0;
        for (size_t i = 0; i < pool.size(); ++i)
        {
            if (pool[i].ptr == NULL || pool[i].capacity < count)
                continue;
            if (best == pool.size() || pool[i].capacity < best_capacity)
            {
                best = i;
                best_capacity = pool[i].capacity;
            }
        }
        if (best != pool.size())
        {
            *buffer = pool[best].ptr;
            *capacity = pool[best].capacity;
            pool[best] = pool.back();
            pool.pop_back();
        }
        else
        {
            if (!allow_late_alloc)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host arena missed a required block.");
            gpuErrchk(cudaMallocHost(
                reinterpret_cast<void **>(buffer),
                sizeof(double) * count));
            if (late_allocs != NULL)
                ++(*late_allocs);
            *capacity = count;
        }
    }
    return *buffer;
}

static inline void dSymV2PcFragTaskflowReleasePinnedHost(
    std::vector<xLUstruct_t<double>::SymV2PcFragHostValueBlock> &pool,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state)
{
    if (state.producer_partner_recv_host_values != NULL)
    {
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragHostValueBlock(
                state.producer_partner_recv_host_values,
                state.producer_partner_recv_host_capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host pool capacity is undersized.");
        state.producer_partner_recv_host_values = NULL;
    }
    if (state.producer_row_recv_host_values != NULL)
    {
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragHostValueBlock(
                state.producer_row_recv_host_values,
                state.producer_row_recv_host_capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host pool capacity is undersized.");
        state.producer_row_recv_host_values = NULL;
    }
    if (state.producer_partner_send_host_values != NULL)
    {
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragHostValueBlock(
                state.producer_partner_send_host_values,
                state.producer_partner_send_host_capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host pool capacity is undersized.");
        state.producer_partner_send_host_values = NULL;
    }
    if (state.producer_row_send_host_values != NULL)
    {
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragHostValueBlock(
                state.producer_row_send_host_values,
                state.producer_row_send_host_capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE pinned host pool capacity is undersized.");
        state.producer_row_send_host_values = NULL;
    }
    state.producer_partner_recv_host_capacity = 0;
    state.producer_row_recv_host_capacity = 0;
    state.producer_partner_send_host_capacity = 0;
    state.producer_row_send_host_capacity = 0;
}

static inline void dSymV2PcFragTaskflowNoteProducerRecvPost(
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats,
    bool pinned_host)
{
    if (pinned_host)
        ++stats.producer_recv_pinned_posts;
    else
        ++stats.producer_recv_pageable_posts;
}

static inline void dSymV2PcFragTaskflowCompactProducerSends(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state)
{
    size_t write = 0;
    for (size_t i = 0; i < state.producer_send_reqs.size(); ++i)
    {
        if (state.producer_send_reqs[i] == MPI_REQUEST_NULL)
            continue;
        state.producer_send_reqs[write++] = state.producer_send_reqs[i];
    }
    if (write != state.producer_send_reqs.size())
        state.producer_send_reqs.resize(write);
}

static inline void dSymV2PcFragTaskflowWaitProducerSends(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats,
    int boundary_wait)
{
    dSymV2PcFragTaskflowCompactProducerSends(state);
    if (state.producer_send_reqs.empty())
        return;
    ++stats.producer_send_wait_calls;
    if (boundary_wait)
        ++stats.producer_send_boundary_wait_calls;
    else
        ++stats.producer_send_nonboundary_wait_calls;
    stats.producer_mpi_wait_requests +=
        static_cast<long long>(state.producer_send_reqs.size());
    MPI_Waitall(static_cast<int>(state.producer_send_reqs.size()),
                state.producer_send_reqs.data(), MPI_STATUSES_IGNORE);
    state.producer_send_reqs.clear();
}

template <typename T>
static inline void dSymV2PcFragTaskflowEnsureVectorCapacity(
    std::vector<T> &buffer, size_t count,
    long long *growth_counter = NULL)
{
    if (count == 0 || buffer.capacity() >= count)
        return;
    if (superlu_sym_v2_pcfrag_taskflow_async_core())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE vector scratch is undersized.");
    if (growth_counter != NULL)
        ++(*growth_counter);
    buffer.reserve(count);
}

static inline void dSymV2PcFragTaskflowEnsureProgressScratch(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats,
    size_t request_count)
{
    if (request_count == 0)
        return;
    dSymV2PcFragTaskflowEnsureVectorCapacity(
        state.producer_progress_indices, request_count,
        &stats.producer_progress_vector_growths);
    dSymV2PcFragTaskflowEnsureVectorCapacity(
        state.producer_progress_statuses, request_count,
        &stats.producer_progress_vector_growths);
    if (state.producer_progress_indices.size() < request_count)
        state.producer_progress_indices.resize(request_count);
    if (state.producer_progress_statuses.size() < request_count)
        state.producer_progress_statuses.resize(request_count);
}

static inline int dSymV2PcFragTaskflowProgressProducerSends(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats)
{
    dSymV2PcFragTaskflowCompactProducerSends(state);
    if (state.producer_send_reqs.empty())
        return 0;
    const int request_count =
        static_cast<int>(state.producer_send_reqs.size());
    if (state.producer_progress_indices.size() <
            static_cast<size_t>(request_count) ||
        state.producer_progress_statuses.size() <
            static_cast<size_t>(request_count))
    {
        if (superlu_sym_v2_pcfrag_taskflow_async_core())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE send progress scratch is undersized.");
        dSymV2PcFragTaskflowEnsureProgressScratch(
            state, stats, static_cast<size_t>(request_count));
    }
    int completed = 0;
    ++stats.producer_send_test_calls;
    int mpi_rc = MPI_Testsome(
        request_count, state.producer_send_reqs.data(), &completed,
        state.producer_progress_indices.data(),
        state.producer_progress_statuses.data());
    if (mpi_rc != MPI_SUCCESS)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW send progress failed.");
    if (completed == MPI_UNDEFINED)
    {
        state.producer_send_reqs.clear();
        return 0;
    }
    if (completed <= 0)
        return 0;
    stats.producer_send_test_completions +=
        static_cast<long long>(completed);
    dSymV2PcFragTaskflowCompactProducerSends(state);
    return completed;
}

static inline int dSymV2PcFragTaskflowProducerSendsComplete(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats, int drain)
{
    dSymV2PcFragTaskflowCompactProducerSends(state);
    if (state.producer_send_reqs.empty())
        return 1;
    if (drain || !superlu_sym_v2_pcfrag_taskflow_async_core())
    {
        dSymV2PcFragTaskflowWaitProducerSends(state, stats, 0);
        return 1;
    }
    dSymV2PcFragTaskflowProgressProducerSends(state, stats);
    return state.producer_send_reqs.empty() ? 1 : 0;
}

static inline bool dSymV2PcFragTaskflowUseCompactOutputLocks(
    const xLUstruct_t<double> &xlu)
{
    return superlu_sym_v2_pcfrag_taskflow_async_core() &&
           superlu_sym_v2_pcfrag_taskflow_global_output_locks() &&
           !xlu.symV2PcFragTaskflowGlobalOutputLockState.empty();
}

static inline size_t dSymV2PcFragTaskflowOutputCount(
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    if (task.output_begin < 0 || task.output_count < 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW task output range is invalid.");
    return static_cast<size_t>(task.output_count);
}

static inline const xLUstruct_t<double>::SymV2PcFragOutputKey &
dSymV2PcFragTaskflowOutputAt(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    size_t output_offset)
{
    size_t count = dSymV2PcFragTaskflowOutputCount(task);
    if (output_offset >= count)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW task output offset is invalid.");
    size_t begin = static_cast<size_t>(task.output_begin);
    if (begin > state.task_output_pool.size() ||
        count > state.task_output_pool.size() - begin)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool range is invalid.");
    return state.task_output_pool[begin + output_offset];
}

static inline int_t dSymV2PcFragTaskflowCompactOutputIdAt(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    size_t output_offset)
{
    if (output_offset == 0 && task.output_count == 1 &&
        task.output_id != GLOBAL_BLOCK_NOT_FOUND)
        return task.output_id;
    return dSymV2PcFragTaskflowOutputAt(
               state, task, output_offset).output_id;
}

static inline long long dSymV2PcFragTaskflowReleaseOutputLocks(
    xLUstruct_t<double> &xlu,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    bool strict_output_conflicts,
    bool released_by_event)
{
    if (!strict_output_conflicts)
        return 0;
    long long released = 0;
    const size_t output_count =
        dSymV2PcFragTaskflowOutputCount(task);
    for (size_t o = 0; o < output_count; ++o)
    {
        if (dSymV2PcFragTaskflowUseCompactOutputLocks(xlu))
        {
            int_t output_id =
                dSymV2PcFragTaskflowCompactOutputIdAt(state, task, o);
            if (output_id < 0 ||
                static_cast<size_t>(output_id) >=
                    xlu.symV2PcFragTaskflowGlobalOutputLockState.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock id is invalid.");
            if (!xlu.symV2PcFragTaskflowGlobalOutputLockState[
                    static_cast<size_t>(output_id)])
                ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock release found an unlocked output.");
            xlu.symV2PcFragTaskflowGlobalOutputLockState[
                static_cast<size_t>(output_id)] = 0;
            if (state.active_output_lock_count <= 0 ||
                xlu.symV2PcFragTaskflowGlobalOutputLocksLive <= 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock count underflowed.");
            --state.active_output_lock_count;
            --xlu.symV2PcFragTaskflowGlobalOutputLocksLive;
            ++released;
            continue;
        }
        const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
            dSymV2PcFragTaskflowOutputAt(state, task, o);
        size_t local_erased = state.active_output_key_set.erase(key);
        size_t global_erased = 0;
        if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
            global_erased =
                xlu.symV2PcFragTaskflowGlobalOutputLocks.erase(key);
        if (local_erased != 0 || global_erased != 0)
            ++released;
    }
    if (released_by_event)
        xlu.symV2PcFragTaskflowStats.output_locks_released_by_event +=
            released;
    else
        xlu.symV2PcFragTaskflowStats.output_locks_released_by_launch_sync +=
            released;
    if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
        xlu.symV2PcFragTaskflowStats.global_output_locks_released +=
            released;
    return released;
}

static inline void dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    if (task.mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
    {
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        if (output_count == 1 && task.lookahead_col_gid_index >= 0)
        {
            const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, 0);
            int idx = task.lookahead_col_gid_index;
            if (static_cast<size_t>(idx) >=
                    state.incomplete_lookahead_col_members_by_gid.size() ||
                state.incomplete_lookahead_col_members_by_gid.gid_at(idx) !=
                    key.gj ||
                state.incomplete_lookahead_col_members_by_gid.value_at(idx) <=
                    0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column incomplete compact counter underflowed.");
            --state.incomplete_lookahead_col_members_by_gid.value_at(idx);
        }
        else
        {
            for (size_t o = 0; o < output_count; ++o)
            {
                const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                auto it =
                    state.incomplete_lookahead_col_members_by_gid.find(
                        key.gj);
                if (it == state.incomplete_lookahead_col_members_by_gid.end() ||
                    it->second <= 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column incomplete counter underflowed.");
                --it->second;
            }
        }
    }
    if (task.mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
    {
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        if (output_count == 1 && task.lookahead_row_gid_index >= 0)
        {
            const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, 0);
            int idx = task.lookahead_row_gid_index;
            if (static_cast<size_t>(idx) >=
                    state.incomplete_lookahead_row_members_by_gid.size() ||
                state.incomplete_lookahead_row_members_by_gid.gid_at(idx) !=
                    key.gi ||
                state.incomplete_lookahead_row_members_by_gid.value_at(idx) <=
                    0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row incomplete compact counter underflowed.");
            --state.incomplete_lookahead_row_members_by_gid.value_at(idx);
        }
        else
        {
            for (size_t o = 0; o < output_count; ++o)
            {
                const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                auto it =
                    state.incomplete_lookahead_row_members_by_gid.find(
                        key.gi);
                if (it == state.incomplete_lookahead_row_members_by_gid.end() ||
                    it->second <= 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row incomplete counter underflowed.");
                --it->second;
            }
        }
    }
}

static inline void dSymV2PcFragTaskflowNoteGemmResourceComplete(
    xLUstruct_t<double> &xlu,
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task);

static inline void dSymV2PcFragTaskflowCompleteLaunchedTask(
    xLUstruct_t<double> &xlu,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    bool strict_output_conflicts)
{
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats =
        xlu.symV2PcFragTaskflowStats;
    if (task.complete)
        return;
    if (task.row_piece < 0 || task.partner_piece < 0 ||
        static_cast<size_t>(task.row_piece) >= state.row_pieces.size() ||
        static_cast<size_t>(task.partner_piece) >=
            state.partner_pieces.size())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW completed task has invalid pieces.");
    xLUstruct_t<double>::SymV2PcFragPieceDesc &row =
        state.row_pieces[static_cast<size_t>(task.row_piece)];
    xLUstruct_t<double>::SymV2PcFragPieceDesc &col =
        state.partner_pieces[static_cast<size_t>(task.partner_piece)];
    dSymV2PcFragTaskflowReleaseOutputLocks(
        xlu, state, task, strict_output_conflicts, true);
    dSymV2PcFragTaskflowNoteGemmResourceComplete(xlu, state, task);
    dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(state, task);
    task.complete = 1;
    --state.incomplete_task_count;
    --row.pending_consumers;
    --col.pending_consumers;
    if (row.pending_consumers < 0 || col.pending_consumers < 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
    ++stats.tasks_completed;
    ++stats.tasks_completed_async_core;
}

static inline int dSymV2PcFragTaskflowTaskStreamKind(
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    int kind = static_cast<int>(task.launch_stream_kind);
    if (kind <= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_NONE ||
        kind >= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_COUNT)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW async-core task has no stream order tag.");
    return kind;
}

static inline int dSymV2PcFragTaskflowGemmResourceForLaunchMode(
    unsigned launch_mode)
{
    if (launch_mode &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
        return xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_LOOKAHEAD_L;
    return xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_MAIN;
}

static inline int dSymV2PcFragTaskflowGemmResourceSlot(
    int stream_id, int resource)
{
    if (stream_id < 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW GEMM resource stream id is invalid.");
    const char *global_tail_env =
        std::getenv("GPU3DV2_PCFRAG_TASKFLOW_GLOBAL_GEMM_TAIL");
    if (global_tail_env != NULL && global_tail_env[0] != '\0' &&
        std::atoi(global_tail_env) != 0)
        return 0;
    if (resource <= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_NONE ||
        resource >= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW GEMM resource id is invalid.");
    return stream_id *
               xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT +
           resource;
}

static inline xLUstruct_t<double>::SymV2PcFragGemmResourceState &
dSymV2PcFragTaskflowGemmResource(
    xLUstruct_t<double> &xlu, int stream_id, int resource)
{
    int slot = dSymV2PcFragTaskflowGemmResourceSlot(stream_id, resource);
    if (static_cast<size_t>(slot) >=
        xlu.symV2PcFragTaskflowGemmResources.size())
    {
        std::fprintf(stderr,
                     "GPU3DV2 taskflow GEMM resource missing: stream_id=%d resource=%d slot=%d resources=%zu numCudaStreams=%d\n",
                     stream_id, resource, slot,
                     xlu.symV2PcFragTaskflowGemmResources.size(),
                     xlu.A_gpu.numCudaStreams);
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE GEMM resource state is missing.");
    }
    xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
        xlu.symV2PcFragTaskflowGemmResources[static_cast<size_t>(slot)];
    if (res.tail_event == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE GEMM resource event was not prewarmed.");
    return res;
}

static inline void dSymV2PcFragTaskflowWaitOnGemmResource(
    xLUstruct_t<double> &xlu,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    int stream_id,
    cudaStream_t stream,
    unsigned launch_mode)
{
    if (!superlu_sym_v2_pcfrag_taskflow_async_core())
        return;
    (void)state;
    int resource = dSymV2PcFragTaskflowGemmResourceForLaunchMode(
        launch_mode);
    xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
        dSymV2PcFragTaskflowGemmResource(xlu, stream_id, resource);
    task.gemm_resource_kind = static_cast<unsigned char>(resource);
    if (res.recorded)
    {
        gpuErrchk(cudaStreamWaitEvent(stream, res.tail_event, 0));
        ++xlu.symV2PcFragTaskflowStats.gemm_resource_tail_waits;
        ++res.waits;
    }
    res.active_task_id = task.task_id;
}

static inline void dSymV2PcFragTaskflowPublishGemmResourceTail(
    xLUstruct_t<double> &xlu,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    int stream_id,
    cudaStream_t stream)
{
    if (!superlu_sym_v2_pcfrag_taskflow_async_core())
        return;
    (void)state;
    int resource = static_cast<int>(task.gemm_resource_kind);
    xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
        dSymV2PcFragTaskflowGemmResource(xlu, stream_id, resource);
    gpuErrchk(cudaEventRecord(res.tail_event, stream));
    res.recorded = 1;
    res.active_task_id = task.task_id;
    ++res.updates;
    ++xlu.symV2PcFragTaskflowStats.gemm_resource_tail_updates;
}

static inline void dSymV2PcFragTaskflowNoteGemmResourceComplete(
    xLUstruct_t<double> &xlu,
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    if (!superlu_sym_v2_pcfrag_taskflow_async_core())
        return;
    int resource = static_cast<int>(task.gemm_resource_kind);
    xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
        dSymV2PcFragTaskflowGemmResource(
            xlu, state.stream_offset, resource);
    if (res.active_task_id == task.task_id)
        res.active_task_id = -1;
}

static inline bool dSymV2PcFragTaskflowSkipEventQuery(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    int kind = dSymV2PcFragTaskflowTaskStreamKind(task);
    if (state.task_event_poll_skip[kind] <= 0)
        return false;
    --state.task_event_poll_skip[kind];
    ++stats.task_completion_event_query_skips;
    return true;
}

static inline bool dSymV2PcFragTaskflowForceTaskSync()
{
    const char *env =
        std::getenv("GPU3DV2_PCFRAG_TASKFLOW_FORCE_TASK_SYNC");
    return env != NULL && env[0] != '\0' && std::atoi(env) != 0;
}

static inline void dSymV2PcFragTaskflowNoteEventNotReady(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    int kind = dSymV2PcFragTaskflowTaskStreamKind(task);
    int base = superlu_sym_v2_pcfrag_taskflow_event_poll_backoff();
    int cap = superlu_sym_v2_pcfrag_taskflow_event_poll_backoff_max();
    if (cap < base)
        cap = base;
    if (base <= 0 || cap <= 0)
    {
        state.task_event_poll_backoff[kind] = 0;
        state.task_event_poll_skip[kind] = 0;
        return;
    }
    int current = state.task_event_poll_backoff[kind];
    if (current <= 0)
        current = base;
    else
        current = SUPERLU_MIN(cap, SUPERLU_MAX(base, current * 2));
    state.task_event_poll_backoff[kind] = current;
    state.task_event_poll_skip[kind] = current;
}

static inline void dSymV2PcFragTaskflowNoteEventComplete(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    int kind = dSymV2PcFragTaskflowTaskStreamKind(task);
    state.task_event_poll_skip[kind] = 0;
    state.task_event_poll_backoff[kind] = 0;
}

static inline bool dSymV2PcFragTaskflowTaskRequiredForMode(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid)
{
    if (!drain || required_mode_mask == 0)
        return true;
    if ((task.mode_mask & required_mode_mask) == 0)
        return false;
    if (required_mode_mask & xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_FULL)
        return true;
    if (required_mode_gid == GLOBAL_BLOCK_NOT_FOUND)
        return true;
    const size_t output_count =
        dSymV2PcFragTaskflowOutputCount(task);
    for (size_t o = 0; o < output_count; ++o)
    {
        const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
            dSymV2PcFragTaskflowOutputAt(state, task, o);
        if ((required_mode_mask &
             xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL) &&
            key.gj == required_mode_gid)
            return true;
        if ((required_mode_mask &
             xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW) &&
            key.gi == required_mode_gid)
            return true;
        if ((required_mode_mask &
             xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_EXCLUDE) &&
            key.gj != required_mode_gid && key.gi != required_mode_gid)
            return true;
    }
    return false;
}

static inline int dSymV2PcFragTaskflowModeMaskIndex(unsigned mode_mask)
{
    return static_cast<int>(
        mode_mask &
        (xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
         xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW |
         xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_EXCLUDE |
         xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_FULL));
}

static inline int dSymV2PcFragTaskflowRequiredIncompleteFast(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    unsigned required_mode_mask,
    int_t required_mode_gid,
    int *known)
{
    *known = 0;
    if (required_mode_gid == GLOBAL_BLOCK_NOT_FOUND)
        return 0;
    const unsigned supported =
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
    if ((required_mode_mask & ~supported) != 0)
        return 0;
    *known = 1;
    int count = 0;
    if (required_mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
    {
        auto it =
            state.incomplete_lookahead_col_members_by_gid.find(
                required_mode_gid);
        if (it != state.incomplete_lookahead_col_members_by_gid.end())
            count += it->second;
    }
    if (required_mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
    {
        auto it =
            state.incomplete_lookahead_row_members_by_gid.find(
                required_mode_gid);
        if (it != state.incomplete_lookahead_row_members_by_gid.end())
            count += it->second;
    }
    return count;
}

static inline int dSymV2PcFragTaskflowPendingLaunchedFast(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    unsigned required_mode_mask,
    int_t required_mode_gid,
    int *known)
{
    *known = 0;
    if (required_mode_gid == GLOBAL_BLOCK_NOT_FOUND)
        return 0;
    const unsigned supported =
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
    if ((required_mode_mask & ~supported) != 0)
        return 0;
    *known = 1;
    int count = 0;
    if (required_mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
    {
        auto it =
            state.launched_lookahead_col_members_by_gid.find(
                required_mode_gid);
        if (it != state.launched_lookahead_col_members_by_gid.end())
            count += it->second;
    }
    if (required_mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
    {
        auto it =
            state.launched_lookahead_row_members_by_gid.find(
                required_mode_gid);
        if (it != state.launched_lookahead_row_members_by_gid.end())
            count += it->second;
    }
    return count;
}

static inline void dSymV2PcFragTaskflowAdjustLaunchedTaskCounts(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task,
    int delta)
{
    int kind = dSymV2PcFragTaskflowTaskStreamKind(task);
    state.launched_task_pending_by_stream[kind] += delta;
    if (state.launched_task_pending_by_stream[kind] < 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW launched stream pending count underflowed.");
    int task_mask = dSymV2PcFragTaskflowModeMaskIndex(task.mode_mask);
    for (int mask = 1; mask < 16; ++mask)
    {
        if ((task_mask & mask) == 0)
            continue;
        state.launched_task_pending_mode_by_stream[kind][mask] += delta;
        if (state.launched_task_pending_mode_by_stream[kind][mask] < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW launched mode pending count underflowed.");
    }
    if (task.mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
    {
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        if (output_count == 1 && task.lookahead_col_gid_index >= 0)
        {
            const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, 0);
            int idx = task.lookahead_col_gid_index;
            if (static_cast<size_t>(idx) >=
                    state.launched_lookahead_col_members_by_gid.size() ||
                state.launched_lookahead_col_members_by_gid.gid_at(idx) !=
                    key.gj ||
                static_cast<size_t>(idx) >=
                    state.launched_lookahead_col_members_by_gid_by_stream[
                        kind].size() ||
                state.launched_lookahead_col_members_by_gid_by_stream[
                    kind].gid_at(idx) != key.gj)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column launched compact counter is missing a gid.");
            state.launched_lookahead_col_members_by_gid.value_at(idx) +=
                delta;
            if (state.launched_lookahead_col_members_by_gid.value_at(idx) < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column launched counter underflowed.");
            state.launched_lookahead_col_members_by_gid_by_stream[
                kind].value_at(idx) += delta;
            if (state.launched_lookahead_col_members_by_gid_by_stream[
                    kind].value_at(idx) < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column stream launched counter underflowed.");
        }
        else
        {
            for (size_t o = 0; o < output_count; ++o)
            {
                const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                auto it =
                    state.launched_lookahead_col_members_by_gid.find(
                        key.gj);
                if (it == state.launched_lookahead_col_members_by_gid.end())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column launched counter is missing a gid.");
                it->second += delta;
                if (it->second < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column launched counter underflowed.");
                auto stream_it =
                    state.launched_lookahead_col_members_by_gid_by_stream[
                        kind].find(key.gj);
                if (stream_it ==
                    state.launched_lookahead_col_members_by_gid_by_stream[
                        kind].end())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column stream launched counter is missing a gid.");
                stream_it->second += delta;
                if (stream_it->second < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column stream launched counter underflowed.");
            }
        }
    }
    if (task.mode_mask &
        xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
    {
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        if (output_count == 1 && task.lookahead_row_gid_index >= 0)
        {
            const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, 0);
            int idx = task.lookahead_row_gid_index;
            if (static_cast<size_t>(idx) >=
                    state.launched_lookahead_row_members_by_gid.size() ||
                state.launched_lookahead_row_members_by_gid.gid_at(idx) !=
                    key.gi ||
                static_cast<size_t>(idx) >=
                    state.launched_lookahead_row_members_by_gid_by_stream[
                        kind].size() ||
                state.launched_lookahead_row_members_by_gid_by_stream[
                    kind].gid_at(idx) != key.gi)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row launched compact counter is missing a gid.");
            state.launched_lookahead_row_members_by_gid.value_at(idx) +=
                delta;
            if (state.launched_lookahead_row_members_by_gid.value_at(idx) < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row launched counter underflowed.");
            state.launched_lookahead_row_members_by_gid_by_stream[
                kind].value_at(idx) += delta;
            if (state.launched_lookahead_row_members_by_gid_by_stream[
                    kind].value_at(idx) < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row stream launched counter underflowed.");
        }
        else
        {
            for (size_t o = 0; o < output_count; ++o)
            {
                const xLUstruct_t<double>::SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                auto it =
                    state.launched_lookahead_row_members_by_gid.find(
                        key.gi);
                if (it == state.launched_lookahead_row_members_by_gid.end())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row launched counter is missing a gid.");
                it->second += delta;
                if (it->second < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row launched counter underflowed.");
                auto stream_it =
                    state.launched_lookahead_row_members_by_gid_by_stream[
                        kind].find(key.gi);
                if (stream_it ==
                    state.launched_lookahead_row_members_by_gid_by_stream[
                        kind].end())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row stream launched counter is missing a gid.");
                stream_it->second += delta;
                if (stream_it->second < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row stream launched counter underflowed.");
            }
        }
    }
}

static inline int dSymV2PcFragTaskflowPendingLaunchedForMode(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    int kind,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid)
{
    if (kind <= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_NONE ||
        kind >= xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_COUNT)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW pending-count stream is invalid.");
    if (!drain || required_mode_mask == 0)
        return state.launched_task_pending_by_stream[kind];
    if (required_mode_gid != GLOBAL_BLOCK_NOT_FOUND &&
        !(required_mode_mask & xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_FULL))
    {
        const unsigned lookahead_supported =
            xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
            xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
        if ((required_mode_mask & ~lookahead_supported) == 0)
        {
            int pending = 0;
            if (required_mode_mask &
                xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
            {
                auto it =
                    state.launched_lookahead_col_members_by_gid_by_stream[
                        kind].find(required_mode_gid);
                if (it !=
                    state.launched_lookahead_col_members_by_gid_by_stream[
                        kind].end())
                    pending += it->second;
            }
            if (required_mode_mask &
                xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
            {
                auto it =
                    state.launched_lookahead_row_members_by_gid_by_stream[
                        kind].find(required_mode_gid);
                if (it !=
                    state.launched_lookahead_row_members_by_gid_by_stream[
                        kind].end())
                    pending += it->second;
            }
            return pending;
        }
        int pending = 0;
        const std::vector<int> &task_ids =
            state.launched_task_ids_by_stream[kind];
        for (size_t i = 0; i < task_ids.size(); ++i)
        {
            int tid = task_ids[i];
            if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW launched task id is invalid.");
            const xLUstruct_t<double>::SymV2PcFragTaskDesc &task =
                state.tasks[static_cast<size_t>(tid)];
            if (!task.launched || task.complete)
                continue;
            if (dSymV2PcFragTaskflowTaskRequiredForMode(
                    state, task, drain, required_mode_mask,
                    required_mode_gid))
                ++pending;
        }
        return pending;
    }
    int mask = dSymV2PcFragTaskflowModeMaskIndex(required_mode_mask);
    if (mask <= 0 || mask >= 16)
        return 0;
    return state.launched_task_pending_mode_by_stream[kind][mask];
}

static inline int dSymV2PcFragTaskflowPendingLaunchedAllForMode(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid)
{
    if (drain && required_mode_mask != 0)
    {
        int known = 0;
        int fast_count =
            dSymV2PcFragTaskflowPendingLaunchedFast(
                state, required_mode_mask, required_mode_gid, &known);
        if (known)
            return fast_count;
    }
    int pending = 0;
    for (int kind = xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_MAIN;
         kind < xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++kind)
    {
        pending += dSymV2PcFragTaskflowPendingLaunchedForMode(
            state, kind, drain, required_mode_mask, required_mode_gid);
    }
    return pending;
}

static inline void dSymV2PcFragTaskflowRecycleEvent(
    std::vector<cudaEvent_t> &pool, cudaEvent_t &event);

static inline void dSymV2PcFragTaskflowRecordLaunchedTask(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    int kind = dSymV2PcFragTaskflowTaskStreamKind(task);
    std::vector<int> &launched =
        state.launched_task_ids_by_stream[kind];
    dSymV2PcFragTaskflowEnsureVectorCapacity(
        launched, launched.size() + 1);
    launched.push_back(task.task_id);
    dSymV2PcFragTaskflowAdjustLaunchedTaskCounts(state, task, 1);
}

static inline void dSymV2PcFragTaskflowUnrecordLaunchedTask(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task)
{
    dSymV2PcFragTaskflowAdjustLaunchedTaskCounts(state, task, -1);
}

static inline int dSymV2PcFragTaskflowTaskIdRequiredForMode(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    int tid,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid)
{
    if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW launched grouped task id is invalid.");
    const xLUstruct_t<double>::SymV2PcFragTaskDesc &task =
        state.tasks[static_cast<size_t>(tid)];
    if (!task.launched || task.complete)
        return 0;
    return dSymV2PcFragTaskflowTaskRequiredForMode(
        state, task, drain, required_mode_mask, required_mode_gid) ? 1 : 0;
}

static inline int dSymV2PcFragTaskflowGroupRequiredForMode(
    const xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    const xLUstruct_t<double>::SymV2PcFragLaunchedTaskGroup &group,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid)
{
    int required = 0;
    if (group.task_begin < 0 || group.task_count < 0 ||
        static_cast<size_t>(group.task_begin + group.task_count) >
            state.launched_group_task_ids.size())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW launched group task range is invalid.");
    for (int i = 0; i < group.task_count; ++i)
    {
        int tid = state.launched_group_task_ids[
            static_cast<size_t>(group.task_begin + i)];
        required += dSymV2PcFragTaskflowTaskIdRequiredForMode(
            state, tid, drain, required_mode_mask, required_mode_gid);
    }
    return required;
}

static inline void dSymV2PcFragTaskflowNoteEventNotReadyForKind(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    int kind)
{
    int base = superlu_sym_v2_pcfrag_taskflow_event_poll_backoff();
    int cap = superlu_sym_v2_pcfrag_taskflow_event_poll_backoff_max();
    if (cap < base)
        cap = base;
    if (base <= 0 || cap <= 0)
    {
        state.task_event_poll_backoff[kind] = 0;
        state.task_event_poll_skip[kind] = 0;
        return;
    }
    int current = state.task_event_poll_backoff[kind];
    if (current <= 0)
        current = base;
    else
        current = SUPERLU_MIN(cap, SUPERLU_MAX(base, current * 2));
    state.task_event_poll_backoff[kind] = current;
    state.task_event_poll_skip[kind] = current;
}

static inline void dSymV2PcFragTaskflowNoteEventCompleteForKind(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    int kind)
{
    state.task_event_poll_skip[kind] = 0;
    state.task_event_poll_backoff[kind] = 0;
}

static inline int dSymV2PcFragTaskflowProgressLaunchedTasks(
    xLUstruct_t<double> &xlu,
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state,
    xLUstruct_t<double>::SymV2PcFragTaskflowStats &stats,
    bool strict_output_conflicts,
    int drain,
    unsigned required_mode_mask,
    int_t required_mode_gid,
    int *pending_required_out = NULL)
{
    int completed = 0;
    int pending_required =
        dSymV2PcFragTaskflowPendingLaunchedAllForMode(
            state, drain, required_mode_mask, required_mode_gid);
    if (pending_required <= 0)
    {
        if (pending_required_out != NULL)
            *pending_required_out = 0;
        return 0;
    }
    pending_required = 0;
    ++stats.task_completion_poll_calls;
    if (drain)
        ++stats.task_completion_drain_poll_calls;
    for (int kind = xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_MAIN;
         kind < xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++kind)
    {
        std::vector<int> &task_ids =
            state.launched_task_ids_by_stream[kind];
        std::vector<xLUstruct_t<double>::SymV2PcFragLaunchedTaskGroup>
            &task_groups = state.launched_task_groups_by_stream[kind];
        size_t launched_write = 0;
        size_t group_write = 0;
        if (task_ids.empty() && task_groups.empty())
            continue;
        int stream_pending_required =
            dSymV2PcFragTaskflowPendingLaunchedForMode(
                state, kind, drain, required_mode_mask,
                required_mode_gid);
        if (stream_pending_required <= 0)
            continue;
        if (!drain && state.task_event_poll_skip[kind] > 0)
        {
            --state.task_event_poll_skip[kind];
            ++stats.task_completion_event_query_skips;
            pending_required += stream_pending_required;
            continue;
        }
        for (size_t g = 0; g < task_groups.size(); ++g)
        {
            xLUstruct_t<double>::SymV2PcFragLaunchedTaskGroup group =
                task_groups[g];
            if (group.done_event == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW async-core grouped task has no completion event.");
            int group_required =
                dSymV2PcFragTaskflowGroupRequiredForMode(
                    state, group, drain, required_mode_mask,
                    required_mode_gid);
            if (group_required > 0)
            {
                pending_required += group_required;
                stats.task_completion_poll_required_seen += group_required;
                if (drain)
                    stats.task_completion_drain_required_seen +=
                        group_required;
            }
            else if (drain && required_mode_mask != 0 &&
                     !(required_mode_mask &
                       xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_FULL))
            {
                task_groups[group_write++] = group;
                continue;
            }
            ++stats.task_completion_poll_task_scans;
            if (drain)
                ++stats.task_completion_drain_task_scans;
            cudaError_t event_rc = cudaEventQuery(group.done_event);
            ++stats.task_completion_event_queries;
            if (event_rc == cudaSuccess)
            {
                ++stats.task_completion_event_successes;
                dSymV2PcFragTaskflowNoteEventCompleteForKind(state, kind);
                if (group.task_begin < 0 || group.task_count < 0 ||
                    static_cast<size_t>(group.task_begin + group.task_count) >
                        state.launched_group_task_ids.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW launched group task range is invalid.");
                for (int ti = 0; ti < group.task_count; ++ti)
                {
                    int tid = state.launched_group_task_ids[
                        static_cast<size_t>(group.task_begin + ti)];
                    if (tid < 0 ||
                        static_cast<size_t>(tid) >= state.tasks.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW launched group task id is invalid.");
                    xLUstruct_t<double>::SymV2PcFragTaskDesc &task =
                        state.tasks[static_cast<size_t>(tid)];
                    if (!task.launched || task.complete)
                        continue;
                    dSymV2PcFragTaskflowUnrecordLaunchedTask(state, task);
                    dSymV2PcFragTaskflowCompleteLaunchedTask(
                        xlu, state, task, strict_output_conflicts);
                    ++completed;
                }
                dSymV2PcFragTaskflowRecycleEvent(
                    xlu.symV2PcFragTaskflowEventPool, group.done_event);
                if (group.d_group_index_pool != NULL)
                {
                    dSymV2PcFragTaskflowRecyclePoolPush(
                        xlu.symV2PcFragTaskflowGroupIndexBlockPool,
                        xLUstruct_t<double>::SymV2PcFragGpuIndexBlock(
                            group.d_group_index_pool,
                            group.group_index_pool_capacity),
                        "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE group index pool capacity is undersized.");
                    group.d_group_index_pool = NULL;
                    group.group_index_pool_capacity = 0;
                }
                if (group.d_group_value_pool != NULL)
                {
                    dSymV2PcFragTaskflowRecyclePoolPush(
                        xlu.symV2PcFragTaskflowGroupValueBlockPool,
                        xLUstruct_t<double>::SymV2PcFragGpuValueBlock(
                            group.d_group_value_pool,
                            group.group_value_pool_capacity),
                        "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE group value pool capacity is undersized.");
                    group.d_group_value_pool = NULL;
                    group.group_value_pool_capacity = 0;
                }
                if (group.d_group_index_pool == NULL &&
                    group.d_group_value_pool == NULL)
                    state.group_scratch_in_use = 0;
                if (group_required > 0)
                    pending_required -= group_required;
                continue;
            }
            if (event_rc != cudaErrorNotReady)
                gpuErrchk(event_rc);
            dSymV2PcFragTaskflowNoteEventNotReadyForKind(state, kind);
            task_groups[group_write++] = group;
        }
        if (group_write != task_groups.size())
            task_groups.resize(group_write);
        for (size_t i = 0; i < task_ids.size(); ++i)
        {
            int tid = task_ids[i];
            if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW launched task id is invalid.");
            xLUstruct_t<double>::SymV2PcFragTaskDesc &task =
                state.tasks[static_cast<size_t>(tid)];
            if (!task.launched || task.complete)
                continue;
            ++stats.task_completion_poll_task_scans;
            if (drain)
                ++stats.task_completion_drain_task_scans;
            if (task.done_event == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW async-core task has no completion event.");
            const bool required_task =
                dSymV2PcFragTaskflowTaskRequiredForMode(
                    state, task, drain, required_mode_mask,
                    required_mode_gid);
            if (required_task)
            {
                ++pending_required;
                ++stats.task_completion_poll_required_seen;
                if (drain)
                    ++stats.task_completion_drain_required_seen;
            }
            else if (drain && required_mode_mask != 0 &&
                     !(required_mode_mask &
                       xLUstruct_t<double>::SYM_V2_PCFRAG_TASK_FULL))
            {
                task_ids[launched_write++] = tid;
                continue;
            }
            if (!drain &&
                dSymV2PcFragTaskflowSkipEventQuery(state, stats, task))
            {
                task_ids[launched_write++] = tid;
                if (required_task)
                    --pending_required;
                pending_required +=
                    dSymV2PcFragTaskflowPendingLaunchedForMode(
                        state, kind, drain, required_mode_mask,
                        required_mode_gid);
                for (size_t j = i + 1; j < task_ids.size(); ++j)
                    task_ids[launched_write++] = task_ids[j];
                break;
            }
            cudaError_t event_rc = cudaEventQuery(task.done_event);
            ++stats.task_completion_event_queries;
            if (event_rc == cudaSuccess)
            {
                ++stats.task_completion_event_successes;
                dSymV2PcFragTaskflowNoteEventComplete(state, task);
                dSymV2PcFragTaskflowUnrecordLaunchedTask(state, task);
                dSymV2PcFragTaskflowCompleteLaunchedTask(
                    xlu, state, task, strict_output_conflicts);
                dSymV2PcFragTaskflowRecycleEvent(
                    xlu.symV2PcFragTaskflowEventPool, task.done_event);
                ++completed;
                if (required_task)
                    --pending_required;
                continue;
            }
            if (event_rc != cudaErrorNotReady)
                gpuErrchk(event_rc);
            dSymV2PcFragTaskflowNoteEventNotReady(state, task);
            task_ids[launched_write++] = tid;
            if (required_task)
                --pending_required;
            pending_required +=
                dSymV2PcFragTaskflowPendingLaunchedForMode(
                    state, kind, drain, required_mode_mask,
                    required_mode_gid);
            for (size_t j = i + 1; j < task_ids.size(); ++j)
                task_ids[launched_write++] = task_ids[j];
            break;
        }
        if (launched_write != task_ids.size())
            task_ids.resize(launched_write);
    }
    if (pending_required_out != NULL)
        *pending_required_out = pending_required;
    return completed;
}

static inline cudaEvent_t dSymV2PcFragTaskflowAcquireEvent(
    std::vector<cudaEvent_t> &pool,
    bool allow_late_alloc = true,
    long long *late_allocs = NULL)
{
    cudaEvent_t event = NULL;
    if (!pool.empty())
    {
        event = pool.back();
        pool.pop_back();
    }
    else
    {
        if (!allow_late_alloc)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE event pool missed a required event.");
        gpuErrchk(cudaEventCreateWithFlags(
            &event, cudaEventDisableTiming));
        if (late_allocs != NULL)
            ++(*late_allocs);
    }
    return event;
}

static inline void dSymV2PcFragTaskflowRecycleEvent(
    std::vector<cudaEvent_t> &pool, cudaEvent_t &event)
{
    if (event == NULL)
        return;
    dSymV2PcFragTaskflowRecyclePoolPush(
        pool, event,
        "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE event pool capacity is undersized.");
    event = NULL;
}

static inline int dSymV2PcFragTaskflowProducerStreamComplete(
    xLUstruct_t<double>::SymV2PcFragPanelTaskState &state, int drain)
{
    if (!state.producer_stream_pending)
        return 1;
    if (state.row_pieces_ready_count != state.row_pieces.size() ||
        state.partner_pieces_ready_count != state.partner_pieces.size())
    {
        if (drain)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW producer stream drain saw unready pieces.");
        return 0;
    }
    if (state.row_pieces.empty() && state.partner_pieces.empty())
    {
        state.producer_stream_pending = 0;
        state.producer_exchange_active = 0;
        return 1;
    }
    if (state.producer_last_ready_event == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW producer stream has no final ready event.");
    cudaError_t status = cudaSuccess;
    if (drain && !superlu_sym_v2_pcfrag_taskflow_async_core())
    {
        status = cudaEventSynchronize(state.producer_last_ready_event);
    }
    else
    {
        int idle_polls = 0;
        for (;;)
        {
            status = cudaEventQuery(state.producer_last_ready_event);
            if (status == cudaSuccess)
                break;
            if (status != cudaErrorNotReady)
                gpuErrchk(status);
            if (!drain)
                return 0;
            if (++idle_polls > 10000000)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW producer stream drain made no CUDA progress.");
        }
    }
    if (status == cudaSuccess)
    {
        state.producer_stream_pending = 0;
        state.producer_exchange_active = 0;
        return 1;
    }
    gpuErrchk(status);
    return 0;
}

static inline int_t *dSymV2PcFragTaskflowAcquireIndexBlock(
    std::vector<xLUstruct_t<double>::SymV2PcFragGpuIndexBlock> &pool,
    size_t count, size_t *capacity, bool allow_late_alloc,
    long long *late_allocs = NULL)
{
    if (capacity == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW index block capacity handle is missing.");
    *capacity = 0;
    if (count == 0)
        return NULL;
    size_t best = pool.size();
    size_t best_capacity = 0;
    for (size_t i = 0; i < pool.size(); ++i)
    {
        if (pool[i].ptr == NULL || pool[i].capacity < count)
            continue;
        if (best == pool.size() || pool[i].capacity < best_capacity)
        {
            best = i;
            best_capacity = pool[i].capacity;
        }
    }
    if (best != pool.size())
    {
        int_t *ptr = pool[best].ptr;
        *capacity = pool[best].capacity;
        pool[best] = pool.back();
        pool.pop_back();
        return ptr;
    }
    if (!allow_late_alloc)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE index arena missed a required block.");
    int_t *ptr = NULL;
    gpuErrchk(cudaMalloc(
        reinterpret_cast<void **>(&ptr), sizeof(int_t) * count));
    if (late_allocs != NULL)
        ++(*late_allocs);
    *capacity = count;
    return ptr;
}

static inline double *dSymV2PcFragTaskflowAcquireValueBlock(
    std::vector<xLUstruct_t<double>::SymV2PcFragGpuValueBlock> &pool,
    size_t count, size_t *capacity, bool allow_late_alloc,
    long long *late_allocs = NULL)
{
    if (capacity == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW value block capacity handle is missing.");
    *capacity = 0;
    if (count == 0)
        return NULL;
    size_t best = pool.size();
    size_t best_capacity = 0;
    for (size_t i = 0; i < pool.size(); ++i)
    {
        if (pool[i].ptr == NULL || pool[i].capacity < count)
            continue;
        if (best == pool.size() || pool[i].capacity < best_capacity)
        {
            best = i;
            best_capacity = pool[i].capacity;
        }
    }
    if (best != pool.size())
    {
        double *ptr = pool[best].ptr;
        *capacity = pool[best].capacity;
        pool[best] = pool.back();
        pool.pop_back();
        return ptr;
    }
    if (!allow_late_alloc)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE value arena missed a required block.");
    double *ptr = NULL;
    gpuErrchk(cudaMalloc(
        reinterpret_cast<void **>(&ptr), sizeof(double) * count));
    if (late_allocs != NULL)
        ++(*late_allocs);
    *capacity = count;
    return ptr;
}

static inline void dSymV2PcFragTaskflowRecycleIndexBlock(
    std::vector<xLUstruct_t<double>::SymV2PcFragGpuIndexBlock> &pool,
    int_t *&ptr, size_t &capacity)
{
    if (ptr != NULL)
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragGpuIndexBlock(ptr, capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE index pool capacity is undersized.");
    ptr = NULL;
    capacity = 0;
}

static inline void dSymV2PcFragTaskflowRecycleValueBlock(
    std::vector<xLUstruct_t<double>::SymV2PcFragGpuValueBlock> &pool,
    double *&ptr, size_t &capacity)
{
    if (ptr != NULL)
        dSymV2PcFragTaskflowRecyclePoolPush(
            pool,
            xLUstruct_t<double>::SymV2PcFragGpuValueBlock(ptr, capacity),
            "GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE value pool capacity is undersized.");
    ptr = NULL;
    capacity = 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowBeginGPU(
    int_t k, int_t stream_offset)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    if (superlu_cuda_aware_mpi())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW keeps CUDA-aware MPI fail-closed.");
    if (!superlu_sym_v2_pc_fragment_ldl_native())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW requires GPU3DV2_PC_FRAGMENT_LDL_NATIVE=1.");
    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
        superlu_sym_v2_pcfrag_taskflow_validate())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE is incompatible with GPU3DV2_PCFRAG_TASKFLOW_VALIDATE.");
    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
        !superlu_sym_v2_pcfrag_taskflow_async_pieces())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE requires GPU3DV2_PCFRAG_TASKFLOW_ASYNC_PIECES=1.");
    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
        superlu_sym_v2_pcfrag_taskflow_strict() &&
        !superlu_sym_v2_pcfrag_taskflow_global_output_locks())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE strict mode requires global output locks.");
    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
        superlu_sym_v2_pcfrag_taskflow_piece_max_rows() > 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_PIECE_MAX_ROWS>0 requires mode-split task planning; currently disabled.");
    if (!superlu_sym_v2_row_l_plan_v2_exchange() ||
        !superlu_sym_v2_row_l_direct_recv() ||
        !superlu_sym_v2_row_l_compressed_plan() ||
        !superlu_sym_v2_row_l_lazy_sendmap())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW requires the LDL-native row-down plan-v2 lazy path.");
    if (k < 0 || k >= nsupers)
        return 0;
    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    if (static_cast<size_t>(k) >= symV2RowFragRecvIndex.size() ||
        static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment taskflow metadata is missing.");

    if (symV2PcFragTaskStates.size() < static_cast<size_t>(nsupers))
        symV2PcFragTaskStates.resize(static_cast<size_t>(nsupers));
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    const bool async_core =
        superlu_sym_v2_pcfrag_taskflow_async_core();
    const bool allow_taskflow_late_alloc =
        !async_core;
    const bool need_piece_pair_lookup = !async_core;

    auto release_taskflow_state = [&](SymV2PcFragPanelTaskState &s) {
        for (size_t i = 0; i < s.row_pieces.size(); ++i)
        {
            if (s.row_pieces[i].pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a row piece with pending consumers.");
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                s.row_pieces[i].ready_event);
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                s.row_pieces[i].done_event);
            s.row_pieces[i].d_index = NULL;
            s.row_pieces[i].d_val = NULL;
            s.row_pieces[i].h_index.clear();
        }
        for (size_t i = 0; i < s.partner_pieces.size(); ++i)
        {
            if (s.partner_pieces[i].pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a partner piece with pending consumers.");
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                s.partner_pieces[i].ready_event);
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                s.partner_pieces[i].done_event);
            s.partner_pieces[i].d_index = NULL;
            s.partner_pieces[i].d_val = NULL;
            s.partner_pieces[i].h_index.clear();
        }
        for (size_t i = 0; i < s.tasks.size(); ++i)
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                s.tasks[i].done_event);
        dSymV2PcFragTaskflowProducerStreamComplete(s, 1);
        dSymV2PcFragTaskflowRecycleIndexBlock(
            symV2PcFragTaskflowIndexBlockPool,
            s.d_index_pool, s.index_pool_capacity);
        dSymV2PcFragTaskflowRecycleValueBlock(
            symV2PcFragTaskflowValueBlockPool,
            s.d_value_pool, s.value_pool_capacity);
        dSymV2PcFragTaskflowRecycleIndexBlock(
            symV2PcFragTaskflowIndexBlockPool,
            s.d_group_index_pool, s.group_index_pool_capacity);
        dSymV2PcFragTaskflowRecycleValueBlock(
            symV2PcFragTaskflowValueBlockPool,
            s.d_group_value_pool, s.group_value_pool_capacity);
        dSymV2PcFragTaskflowWaitProducerSends(
            s, symV2PcFragTaskflowStats, 1);
        dSymV2PcFragTaskflowReleasePinnedHost(
            symV2PcFragTaskflowPinnedBlockPool, s);
        s.reset();
    };
    release_taskflow_state(state);
    state.k = k;
    state.stream_offset = stream_offset;
    state.initialized = 1;
    state.exchange_posted = 1;
    state.producer_exchange_active = 1;
    ++symV2PcFragTaskflowStats.taskflow_entries;

    const bool compact_output_locks =
        async_core && superlu_sym_v2_pcfrag_taskflow_global_output_locks();
    if (compact_output_locks)
    {
        const int_t local_panel_count = symV2PanelCount();
        if (local_panel_count < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW local panel count is invalid.");
        if (symV2PcFragTaskflowOutputPanelOffsets.size() !=
            static_cast<size_t>(local_panel_count + 1))
        {
            if (symV2PcFragTaskflowGlobalOutputLocksLive != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW cannot rebuild output lock ids while locks are live.");
            symV2PcFragTaskflowOutputPanelOffsets.assign(
                static_cast<size_t>(local_panel_count + 1), 0);
            int_t prefix = 0;
            for (int_t lj = 0; lj < local_panel_count; ++lj)
            {
                symV2PcFragTaskflowOutputPanelOffsets[
                    static_cast<size_t>(lj)] = prefix;
                int_t nblk = lPanelVec[static_cast<size_t>(lj)].isEmpty()
                                  ? 0
                                  : lPanelVec[static_cast<size_t>(lj)].nblocks();
                if (nblk < 0 ||
                    prefix > std::numeric_limits<int_t>::max() - nblk)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW output lock id space is too large.");
                prefix += nblk;
            }
            symV2PcFragTaskflowOutputPanelOffsets[
                static_cast<size_t>(local_panel_count)] = prefix;
            symV2PcFragTaskflowGlobalOutputLockState.assign(
                static_cast<size_t>(prefix), 0);
            symV2PcFragTaskflowGlobalOutputLocksLive = 0;
        }
    }

    const std::vector<int_t> &row_frag = symV2RowFragRecvIndex[k];
    const std::vector<int_t> &partner_frag = symV2PartnerLRecvIndex[k];
    auto nblocks = [](const std::vector<int_t> &frag) -> int_t {
        return frag.empty() ? 0 : frag[0];
    };
    auto gid_at = [](const std::vector<int_t> &frag, int_t b) -> int_t {
        return frag[LPANEL_HEADER_SIZE + b];
    };
    auto strow_at = [](const std::vector<int_t> &frag, int_t b) -> int_t {
        int_t nb = frag.empty() ? 0 : frag[0];
        return frag[LPANEL_HEADER_SIZE + nb + b];
    };
    auto compact_piece_index =
        [&](const std::vector<int_t> &frag,
            int_t begin, int_t end) -> std::vector<int_t> {
        if (frag.size() < LPANEL_HEADER_SIZE)
            ABORT("SymFact V2 Pc-fragment taskflow has truncated metadata.");
        int_t nb = frag[0];
        if (begin < 0 || begin >= end || end > nb)
            ABORT("SymFact V2 Pc-fragment taskflow block index is invalid.");
        int_t row_begin = strow_at(frag, begin);
        int_t row_end = strow_at(frag, end);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        int_t nrows = row_end - row_begin;
        size_t src_rows =
            static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * nb + 1 + row_begin);
        size_t src_rows_end = src_rows + static_cast<size_t>(nrows);
        if (src_rows_end > frag.size() || src_rows_end < src_rows)
            ABORT("SymFact V2 Pc-fragment taskflow row-list range is invalid.");

        int_t piece_blocks = end - begin;
        std::vector<int_t> piece_index(
            static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * piece_blocks +
                                1 + nrows));
        piece_index[0] = piece_blocks;
        piece_index[1] = nrows;
        piece_index[2] = 0;
        piece_index[3] = frag[3];
        for (int_t local = 0; local < piece_blocks; ++local)
            piece_index[LPANEL_HEADER_SIZE + local] =
                gid_at(frag, begin + local);
        size_t px_base = static_cast<size_t>(LPANEL_HEADER_SIZE +
                                             piece_blocks);
        for (int_t local = 0; local <= piece_blocks; ++local)
            piece_index[px_base + static_cast<size_t>(local)] =
                strow_at(frag, begin + local) - row_begin;
        std::copy(frag.begin() + static_cast<std::ptrdiff_t>(src_rows),
                  frag.begin() + static_cast<std::ptrdiff_t>(src_rows_end),
                  piece_index.begin() + static_cast<std::ptrdiff_t>(
                      LPANEL_HEADER_SIZE + 2 * piece_blocks + 1));
        return piece_index;
    };
    auto piece_index_count =
        [&](const std::vector<int_t> &frag,
            int_t begin, int_t end) -> size_t {
        int_t row_begin = strow_at(frag, begin);
        int_t row_end = strow_at(frag, end);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        return static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * (end - begin) +
                                   1 +
                                   (row_end - row_begin));
    };
    auto piece_value_count =
        [&](const std::vector<int_t> &frag,
            int_t begin, int_t end) -> size_t {
        int_t row_begin = strow_at(frag, begin);
        int_t row_end = strow_at(frag, end);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        int_t ksupc = frag.size() >= 4 ? frag[3] : SuperSize(k);
        return static_cast<size_t>(row_end - row_begin) *
               static_cast<size_t>(ksupc);
    };

    struct TaskflowPieceRange
    {
        int_t begin;
        int_t end;
    };
    const int taskflow_piece_max_rows =
        superlu_sym_v2_pcfrag_taskflow_piece_max_rows();
    if (taskflow_piece_max_rows > 0 && !async_core)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_PIECE_MAX_ROWS currently requires GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE=1.");
    auto build_piece_ranges =
        [&](const std::vector<int_t> &frag, int_t nb,
            std::vector<TaskflowPieceRange> &ranges) {
        ranges.clear();
        ranges.reserve(static_cast<size_t>(nb));
        for (int_t b = 0; b < nb;)
        {
            int_t begin = b;
            int_t end = b + 1;
            if (taskflow_piece_max_rows > 0 && gid_at(frag, begin) != k)
            {
                int_t row_begin = strow_at(frag, begin);
                while (end < nb && gid_at(frag, end) != k)
                {
                    int_t candidate_rows = strow_at(frag, end + 1) -
                                           row_begin;
                    if (candidate_rows > taskflow_piece_max_rows)
                        break;
                    ++end;
                }
            }
            ranges.push_back(TaskflowPieceRange{begin, end});
            b = end;
        }
    };

    int_t nr = nblocks(row_frag);
    int_t nc = nblocks(partner_frag);
    if (nr < 0 || nc < 0)
        ABORT("SymFact V2 Pc-fragment taskflow block count is invalid.");
    std::vector<TaskflowPieceRange> row_piece_ranges;
    std::vector<TaskflowPieceRange> partner_piece_ranges;
    build_piece_ranges(row_frag, nr, row_piece_ranges);
    build_piece_ranges(partner_frag, nc, partner_piece_ranges);
    state.row_pieces.reserve(row_piece_ranges.size());
    state.partner_pieces.reserve(partner_piece_ranges.size());
    state.row_block_piece.assign(static_cast<size_t>(nr), -1);
    state.partner_block_piece.assign(static_cast<size_t>(nc), -1);
    size_t total_index_count = 0;
    size_t total_value_count = 0;
    for (size_t r = 0; r < row_piece_ranges.size(); ++r)
    {
        total_index_count += piece_index_count(
            row_frag, row_piece_ranges[r].begin, row_piece_ranges[r].end);
        total_value_count += piece_value_count(
            row_frag, row_piece_ranges[r].begin, row_piece_ranges[r].end);
    }
    for (size_t r = 0; r < partner_piece_ranges.size(); ++r)
    {
        total_index_count += piece_index_count(
            partner_frag, partner_piece_ranges[r].begin,
            partner_piece_ranges[r].end);
        total_value_count += piece_value_count(
            partner_frag, partner_piece_ranges[r].begin,
            partner_piece_ranges[r].end);
    }
    if (total_index_count > 0)
    {
        state.d_index_pool =
            dSymV2PcFragTaskflowAcquireIndexBlock(
                symV2PcFragTaskflowIndexBlockPool,
                total_index_count, &state.index_pool_capacity,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_index_late_allocs);
    }
    if (total_value_count > 0)
    {
        state.d_value_pool =
            dSymV2PcFragTaskflowAcquireValueBlock(
                symV2PcFragTaskflowValueBlockPool,
                total_value_count, &state.value_pool_capacity,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_value_late_allocs);
        gpuErrchk(cudaMemset(
            state.d_value_pool, 0, sizeof(double) * total_value_count));
    }
    if (!async_core && total_index_count > 0)
    {
        state.d_group_index_pool =
            dSymV2PcFragTaskflowAcquireIndexBlock(
                symV2PcFragTaskflowIndexBlockPool,
                total_index_count, &state.group_index_pool_capacity,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_index_late_allocs);
    }
    if (!async_core && total_value_count > 0)
    {
        state.d_group_value_pool =
            dSymV2PcFragTaskflowAcquireValueBlock(
                symV2PcFragTaskflowValueBlockPool,
                total_value_count, &state.group_value_pool_capacity,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_value_late_allocs);
    }

    auto add_piece = [&](std::vector<SymV2PcFragPieceDesc> &pieces,
                         const std::vector<int_t> &frag,
                         std::vector<int> &block_piece,
                         unsigned char kind, int_t begin, int_t end) {
        SymV2PcFragPieceDesc piece;
        piece.k = k;
        piece.kind = kind;
        piece.piece_id = static_cast<int>(pieces.size());
        piece.frag_blk_begin = begin;
        piece.frag_blk_end = end;
        piece.frag_row_offset = strow_at(frag, begin);
        piece.nblocks = end - begin;
        piece.gid_first = gid_at(frag, begin);
        piece.gid_last = gid_at(frag, end - 1);
        piece.ksupc = frag.size() >= 4 ? frag[3] : SuperSize(k);
        piece.nrows = strow_at(frag, end) - strow_at(frag, begin);
        piece.lda = piece.nrows;
        piece.h_index = compact_piece_index(frag, begin, end);
        piece.index_count = static_cast<int_t>(piece.h_index.size());
        piece.value_count = piece.nrows * piece.ksupc;
        if (piece.index_count > 0)
        {
            size_t count = static_cast<size_t>(piece.index_count);
            if (state.index_pool_used + count > state.index_pool_capacity ||
                state.d_index_pool == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW index arena exhausted.");
            piece.d_index = state.d_index_pool + state.index_pool_used;
            state.index_pool_used += count;
            gpuErrchk(cudaMemcpy(
                piece.d_index, piece.h_index.data(),
                sizeof(int_t) * count,
                cudaMemcpyHostToDevice));
        }
        if (piece.value_count > 0)
        {
            size_t count = static_cast<size_t>(piece.value_count);
            if (state.value_pool_used + count > state.value_pool_capacity ||
                state.d_value_pool == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW value arena exhausted.");
            piece.d_val = state.d_value_pool + state.value_pool_used;
            state.value_pool_used += count;
        }
        piece.ready_event =
            dSymV2PcFragTaskflowAcquireEvent(
                symV2PcFragTaskflowEventPool,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_event_late_allocs);
        for (int_t b = begin; b < end; ++b)
        {
            if (b < 0 || static_cast<size_t>(b) >= block_piece.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW block-to-piece map is invalid.");
            block_piece[static_cast<size_t>(b)] = piece.piece_id;
        }
        pieces.push_back(piece);
    };

    for (size_t r = 0; r < row_piece_ranges.size(); ++r)
        add_piece(state.row_pieces, row_frag, state.row_block_piece,
                  SYM_V2_PCFRAG_PIECE_ROW,
                  row_piece_ranges[r].begin, row_piece_ranges[r].end);
    for (size_t r = 0; r < partner_piece_ranges.size(); ++r)
        add_piece(state.partner_pieces, partner_frag,
                  state.partner_block_piece,
                  SYM_V2_PCFRAG_PIECE_PARTNER,
                  partner_piece_ranges[r].begin,
                  partner_piece_ranges[r].end);
    symV2PcFragTaskflowStats.row_pieces_created +=
        static_cast<long long>(state.row_pieces.size());
    symV2PcFragTaskflowStats.partner_pieces_created +=
        static_cast<long long>(state.partner_pieces.size());

    struct TaskflowGidPiece
    {
        int_t gid;
        int piece;
        bool operator<(const TaskflowGidPiece &other) const
        {
            return gid < other.gid;
        }
    };
    std::vector<TaskflowGidPiece> row_gid_to_piece;
    row_gid_to_piece.reserve(static_cast<size_t>(nr));
    for (int_t rb = 0; rb < nr; ++rb)
    {
        int piece = state.row_block_piece[static_cast<size_t>(rb)];
        if (piece >= 0)
            row_gid_to_piece.push_back(
                TaskflowGidPiece{gid_at(row_frag, rb), piece});
    }
    std::sort(row_gid_to_piece.begin(), row_gid_to_piece.end());
    auto row_piece_for_gid = [&](int_t gid) -> int {
        TaskflowGidPiece key{gid, -1};
        std::vector<TaskflowGidPiece>::const_iterator it =
            std::lower_bound(
                row_gid_to_piece.begin(), row_gid_to_piece.end(), key);
        if (it == row_gid_to_piece.end() || it->gid != gid)
            return -1;
        return it->piece;
    };
    std::vector<size_t> row_task_degrees(state.row_pieces.size(), 0);
    std::vector<size_t> partner_task_degrees(
        state.partner_pieces.size(), 0);
    size_t mode_counts[16] = {};
    SymV2PcFragGidCounterMap lookahead_col_degrees;
    SymV2PcFragGidCounterMap lookahead_row_degrees;
    size_t planned_task_count = 0;
    for (int_t cb = 0; cb < nc; ++cb)
    {
        int_t gj = gid_at(partner_frag, cb);
        if (gj == k)
            continue;
        int partner_piece =
            state.partner_block_piece[static_cast<size_t>(cb)];
        if (partner_piece < 0 ||
            static_cast<size_t>(partner_piece) >=
                state.partner_pieces.size())
            continue;
        int_t local_panel_j = symV2PanelIndex(gj);
        if (local_panel_j == GLOBAL_BLOCK_NOT_FOUND ||
            local_panel_j < 0 ||
            local_panel_j >= symV2PanelCount())
            continue;
        xlpanel_t<double> &dst_panel =
            lPanelVec[static_cast<size_t>(local_panel_j)];
        if (dst_panel.index == NULL)
            continue;
        for (int_t li = 0; li < dst_panel.nblocks(); ++li)
        {
            int_t gi = dst_panel.gid(li);
            if (gi == k || gi < gj)
                continue;
            int row_piece = row_piece_for_gid(gi);
            if (row_piece < 0 ||
                static_cast<size_t>(row_piece) >= state.row_pieces.size())
                continue;
            SymV2PcFragOutputKey output(gj, gi);
            output.local_panel_j = local_panel_j;
            output.local_block_i = li;
            if (compact_output_locks)
            {
                if (static_cast<size_t>(local_panel_j + 1) >=
                    symV2PcFragTaskflowOutputPanelOffsets.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW output lock panel id is invalid.");
                output.output_id =
                    symV2PcFragTaskflowOutputPanelOffsets[
                        static_cast<size_t>(local_panel_j)] + li;
                if (output.output_id < 0 ||
                    static_cast<size_t>(output.output_id) >=
                        symV2PcFragTaskflowGlobalOutputLockState.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW output lock id is invalid.");
            }
            unsigned char mode_mask =
                SYM_V2_PCFRAG_TASK_FULL |
                SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                SYM_V2_PCFRAG_TASK_EXCLUDE;
            if (gi != gj)
                mode_mask |= SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
            ++planned_task_count;
            ++row_task_degrees[static_cast<size_t>(row_piece)];
            ++partner_task_degrees[static_cast<size_t>(partner_piece)];
            for (int mask = 1; mask < 16; mask <<= 1)
                if (mode_mask & mask)
                    ++mode_counts[mask];
            ++lookahead_col_degrees[gj];
            if (mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                ++lookahead_row_degrees[gi];
        }
    }
    size_t max_planned_tasks =
        superlu_sym_v2_pcfrag_taskflow_max_planned_tasks();
    if (planned_task_count > max_planned_tasks)
    {
        std::fprintf(stderr,
                     "[rank %d] GPU3DV2 taskflow refusing k=%lld "
                     "planned_tasks=%zu max_planned_tasks=%zu "
                     "row_pieces=%zu partner_pieces=%zu\n",
                     iam, static_cast<long long>(k), planned_task_count,
                     max_planned_tasks, state.row_pieces.size(),
                     state.partner_pieces.size());
        std::fflush(stderr);
        ABORT("GPU3DV2_PCFRAG_TASKFLOW planned task count exceeds guard.");
    }
    auto byte_product_or_max = [](size_t count, size_t bytes) -> size_t {
        if (bytes != 0 &&
            count > std::numeric_limits<size_t>::max() / bytes)
            return std::numeric_limits<size_t>::max();
        return count * bytes;
    };
    size_t estimated_task_bytes =
        byte_product_or_max(planned_task_count,
                            sizeof(SymV2PcFragTaskDesc));
    size_t estimated_pair_bytes =
        need_piece_pair_lookup
            ? byte_product_or_max(planned_task_count,
                                  sizeof(SymV2PcFragPairTaskEntry))
            : 0;
    size_t estimated_ready_bytes =
        byte_product_or_max(planned_task_count, 2 * sizeof(unsigned char));
    size_t estimated_queue_bytes =
        byte_product_or_max(planned_task_count, 6 * sizeof(int));
    size_t estimated_csr_bytes =
        byte_product_or_max(state.row_pieces.size() +
                                state.partner_pieces.size() + 2,
                            sizeof(int)) +
        byte_product_or_max(2 * planned_task_count, sizeof(int));
    size_t estimated_mode_queue_bytes = 0;
    for (int mask = 1; mask < 16; mask <<= 1)
        estimated_mode_queue_bytes +=
            byte_product_or_max(mode_counts[mask], sizeof(int));
    size_t estimated_gid_queue_bytes =
        byte_product_or_max(
            mode_counts[SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL] +
                mode_counts[SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW],
            sizeof(int));
    size_t estimated_counter_map_bytes =
        byte_product_or_max(
            6 * (lookahead_col_degrees.size() +
                 lookahead_row_degrees.size()),
            sizeof(int_t) + sizeof(int));
    size_t estimated_output_lock_bytes =
        compact_output_locks
            ? symV2PcFragTaskflowGlobalOutputLockState.size() *
                  sizeof(unsigned char)
            : 0;
    auto size_sum_or_max = [](size_t a, size_t b) -> size_t {
        if (a == std::numeric_limits<size_t>::max() ||
            b > std::numeric_limits<size_t>::max() - a)
            return std::numeric_limits<size_t>::max();
        return a + b;
    };
    size_t estimated_host_graph_bytes = 0;
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_task_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_pair_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_ready_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_queue_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_csr_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_mode_queue_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_gid_queue_bytes);
    estimated_host_graph_bytes =
        size_sum_or_max(estimated_host_graph_bytes,
                        estimated_counter_map_bytes);
    size_t estimated_event_count =
        size_sum_or_max(
            state.row_pieces.size() + state.partner_pieces.size(),
            planned_task_count);
    auto add_size_to_stat = [](long long &stat, size_t value) {
        const long long max_ll = std::numeric_limits<long long>::max();
        if (stat == max_ll ||
            value > static_cast<size_t>(max_ll - stat))
        {
            stat = max_ll;
            return;
        }
        stat += static_cast<long long>(value);
    };
    auto max_size_to_stat = [](long long &stat, size_t value) {
        const long long max_ll = std::numeric_limits<long long>::max();
        long long converted =
            (value > static_cast<size_t>(max_ll))
                ? max_ll
                : static_cast<long long>(value);
        if (converted > stat)
            stat = converted;
    };
    add_size_to_stat(symV2PcFragTaskflowStats.graph_host_bytes,
                     estimated_host_graph_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_task_desc_bytes,
                     estimated_task_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_pair_bytes,
                     estimated_pair_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_ready_bytes,
                     estimated_ready_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_queue_bytes,
                     estimated_queue_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_csr_bytes,
                     estimated_csr_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_mode_queue_bytes,
                     estimated_mode_queue_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_gid_queue_bytes,
                     estimated_gid_queue_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_counter_map_bytes,
                     estimated_counter_map_bytes);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_event_count_est,
                     estimated_event_count);
    add_size_to_stat(symV2PcFragTaskflowStats.graph_output_count,
                     planned_task_count);
    max_size_to_stat(symV2PcFragTaskflowStats.graph_host_bytes_max_panel,
                     estimated_host_graph_bytes);
    max_size_to_stat(symV2PcFragTaskflowStats.graph_event_count_max_panel,
                     estimated_event_count);
    max_size_to_stat(symV2PcFragTaskflowStats.graph_output_count_max_panel,
                     planned_task_count);
    const char *taskflow_plan_diag =
        std::getenv("GPU3DV2_PCFRAG_TASKFLOW_PLAN_DIAG");
    if (taskflow_plan_diag != NULL && taskflow_plan_diag[0] != '\0')
    {
        fprintf(stderr,
                "[rank %d] taskflow plan k=%lld row_pieces=%zu "
                "partner_pieces=%zu planned_tasks=%zu "
                "lookahead_col_gids=%zu lookahead_row_gids=%zu "
                "task_desc_bytes=%zu pair_bytes=%zu ready_bytes=%zu "
                "queue_est_bytes=%zu csr_bytes=%zu "
                "mode_queue_bytes=%zu gid_queue_bytes=%zu "
                "counter_map_bytes=%zu output_lock_bytes=%zu "
                "event_count_est=%zu\n",
                iam, static_cast<long long>(k),
                state.row_pieces.size(), state.partner_pieces.size(),
                planned_task_count, lookahead_col_degrees.size(),
                lookahead_row_degrees.size(), estimated_task_bytes,
                estimated_pair_bytes, estimated_ready_bytes,
                estimated_queue_bytes, estimated_csr_bytes,
                estimated_mode_queue_bytes, estimated_gid_queue_bytes,
                estimated_counter_map_bytes, estimated_output_lock_bytes,
                state.row_pieces.size() + state.partner_pieces.size() +
                    planned_task_count);
        fflush(stderr);
        const char *taskflow_plan_diag_abort =
            std::getenv("GPU3DV2_PCFRAG_TASKFLOW_PLAN_DIAG_ABORT");
        if (taskflow_plan_diag_abort != NULL &&
            taskflow_plan_diag_abort[0] != '\0')
            ABORT("GPU3DV2_PCFRAG_TASKFLOW plan diagnostic abort.");
    }

    state.tasks.reserve(planned_task_count);
    state.task_output_pool.reserve(planned_task_count);
    if (need_piece_pair_lookup)
        state.pair_task_entries.reserve(planned_task_count);
    state.task_ready_inputs.reserve(planned_task_count);
    state.task_enqueued.reserve(planned_task_count);
    state.runnable_task_ids.reserve(planned_task_count);
    for (int mask = 1; mask < 16; mask <<= 1)
        state.runnable_task_ids_by_mode[mask].reserve(mode_counts[mask]);
    if (async_core)
    {
        size_t launched_reserve =
            static_cast<size_t>(
                superlu_sym_v2_pcfrag_taskflow_effective_progress_budget());
        if (launched_reserve > planned_task_count)
            launched_reserve = planned_task_count;
        for (int kind = SYM_V2_PCFRAG_TASK_STREAM_MAIN;
             kind < SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++kind)
            state.launched_task_ids_by_stream[kind].reserve(
                launched_reserve);
    }
    auto build_piece_task_offsets =
        [](const std::vector<size_t> &degrees,
           std::vector<int> &offsets,
           const char *abort_msg) -> size_t {
        offsets.assign(degrees.size() + 1, 0);
        size_t prefix = 0;
        for (size_t i = 0; i < degrees.size(); ++i)
        {
            if (degrees[i] >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT(abort_msg);
            if (prefix >
                static_cast<size_t>(std::numeric_limits<int>::max()) -
                    degrees[i])
                ABORT(abort_msg);
            offsets[i] = static_cast<int>(prefix);
            prefix += degrees[i];
        }
        offsets[degrees.size()] = static_cast<int>(prefix);
        return prefix;
    };
    size_t row_piece_task_count =
        build_piece_task_offsets(
            row_task_degrees, state.row_piece_task_offsets,
            "GPU3DV2_PCFRAG_TASKFLOW row-piece task adjacency is too large.");
    size_t partner_piece_task_count =
        build_piece_task_offsets(
            partner_task_degrees, state.partner_piece_task_offsets,
            "GPU3DV2_PCFRAG_TASKFLOW partner-piece task adjacency is too large.");
    state.row_piece_task_ids.assign(row_piece_task_count, -1);
    state.partner_piece_task_ids.assign(partner_piece_task_count, -1);
    std::vector<int> row_piece_task_write =
        state.row_piece_task_offsets;
    std::vector<int> partner_piece_task_write =
        state.partner_piece_task_offsets;
    auto prebuild_gid_counter_maps =
        [&](const SymV2PcFragGidCounterMap &degrees,
            SymV2PcFragGidCounterMap &incomplete,
            SymV2PcFragGidCounterMap &launched,
            SymV2PcFragGidCounterMap launched_by_stream[
                SYM_V2_PCFRAG_TASK_STREAM_COUNT]) {
        incomplete.assign_zero_from(degrees);
        launched.assign_zero_from(degrees);
        if (async_core)
        {
            for (int kind = SYM_V2_PCFRAG_TASK_STREAM_MAIN;
                 kind < SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++kind)
                launched_by_stream[kind].assign_zero_from(degrees);
        }
    };
    prebuild_gid_counter_maps(
        lookahead_col_degrees,
        state.incomplete_lookahead_col_members_by_gid,
        state.launched_lookahead_col_members_by_gid,
        state.launched_lookahead_col_members_by_gid_by_stream);
    prebuild_gid_counter_maps(
        lookahead_row_degrees,
        state.incomplete_lookahead_row_members_by_gid,
        state.launched_lookahead_row_members_by_gid,
        state.launched_lookahead_row_members_by_gid_by_stream);
    auto create_task_for_output =
        [&](int row_piece, int partner_piece,
            const SymV2PcFragOutputKey &output,
            unsigned char mode_mask) {
        size_t rp = static_cast<size_t>(row_piece);
        size_t cp = static_cast<size_t>(partner_piece);
        if (rp >= state.row_pieces.size() ||
            cp >= state.partner_pieces.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW sparse task has invalid piece id.");
        SymV2PcFragTaskDesc task;
        task.k = k;
        task.task_id = static_cast<int>(state.tasks.size());
        task.row_piece = row_piece;
        task.partner_piece = partner_piece;
        task.row_piece_blk_begin = state.row_pieces[rp].frag_blk_begin;
        task.row_piece_blk_end = state.row_pieces[rp].frag_blk_end;
        task.partner_piece_blk_begin =
            state.partner_pieces[cp].frag_blk_begin;
        task.partner_piece_blk_end =
            state.partner_pieces[cp].frag_blk_end;
        task.gemm_m = state.row_pieces[rp].nrows;
        task.gemm_n = state.partner_pieces[cp].nrows;
        task.gemm_k = SuperSize(k);
        task.mode_mask = mode_mask;
        if (state.task_output_pool.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool is too large.");
        task.output_begin =
            static_cast<int>(state.task_output_pool.size());
        task.output_count = 1;
        task.output_id = output.output_id;
        state.task_output_pool.push_back(output);
        task.lookahead_col_gid_index =
            state.incomplete_lookahead_col_members_by_gid.index_of(
                output.gj);
        if (task.lookahead_col_gid_index < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-column compact gid is missing.");
        ++state.incomplete_lookahead_col_members_by_gid.value_at(
            task.lookahead_col_gid_index);
        if (task.mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
        {
            task.lookahead_row_gid_index =
                state.incomplete_lookahead_row_members_by_gid.index_of(
                    output.gi);
            if (task.lookahead_row_gid_index < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead-row compact gid is missing.");
            ++state.incomplete_lookahead_row_members_by_gid.value_at(
                task.lookahead_row_gid_index);
        }
        int task_id = task.task_id;
        state.tasks.push_back(task);
        state.task_ready_inputs.push_back(0);
        state.task_enqueued.push_back(0);
        if (need_piece_pair_lookup)
            state.pair_task_entries.push_back(
                SymV2PcFragPairTaskEntry(row_piece, partner_piece, task_id));
        int row_pos = row_piece_task_write[rp]++;
        int row_end = state.row_piece_task_offsets[rp + 1];
        if (row_pos < state.row_piece_task_offsets[rp] ||
            row_pos >= row_end ||
            static_cast<size_t>(row_pos) >= state.row_piece_task_ids.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW row-piece task adjacency overflowed.");
        state.row_piece_task_ids[static_cast<size_t>(row_pos)] = task_id;
        int partner_pos = partner_piece_task_write[cp]++;
        int partner_end = state.partner_piece_task_offsets[cp + 1];
        if (partner_pos < state.partner_piece_task_offsets[cp] ||
            partner_pos >= partner_end ||
            static_cast<size_t>(partner_pos) >=
                state.partner_piece_task_ids.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner-piece task adjacency overflowed.");
        state.partner_piece_task_ids[static_cast<size_t>(partner_pos)] =
            task_id;
        ++state.row_pieces[rp].pending_consumers;
        ++state.partner_pieces[cp].pending_consumers;
    };
    for (int_t cb = 0; cb < nc; ++cb)
    {
        int_t gj = gid_at(partner_frag, cb);
        if (gj == k)
            continue;
        int partner_piece =
            state.partner_block_piece[static_cast<size_t>(cb)];
        if (partner_piece < 0 ||
            static_cast<size_t>(partner_piece) >=
                state.partner_pieces.size())
            continue;
        int_t local_panel_j = symV2PanelIndex(gj);
        if (local_panel_j == GLOBAL_BLOCK_NOT_FOUND ||
            local_panel_j < 0 ||
            local_panel_j >= symV2PanelCount())
            continue;
        xlpanel_t<double> &dst_panel =
            lPanelVec[static_cast<size_t>(local_panel_j)];
        if (dst_panel.index == NULL)
            continue;
        for (int_t li = 0; li < dst_panel.nblocks(); ++li)
        {
            int_t gi = dst_panel.gid(li);
            if (gi == k || gi < gj)
                continue;
            int row_piece = row_piece_for_gid(gi);
            if (row_piece < 0 ||
                static_cast<size_t>(row_piece) >= state.row_pieces.size())
                continue;
            SymV2PcFragOutputKey output(gj, gi);
            output.local_panel_j = local_panel_j;
            output.local_block_i = li;
            if (compact_output_locks)
            {
                if (static_cast<size_t>(local_panel_j + 1) >=
                    symV2PcFragTaskflowOutputPanelOffsets.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW output lock panel id is invalid.");
                output.output_id =
                    symV2PcFragTaskflowOutputPanelOffsets[
                        static_cast<size_t>(local_panel_j)] + li;
                if (output.output_id < 0 ||
                    static_cast<size_t>(output.output_id) >=
                        symV2PcFragTaskflowGlobalOutputLockState.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW output lock id is invalid.");
            }
            unsigned char mode_mask =
                SYM_V2_PCFRAG_TASK_FULL |
                SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                SYM_V2_PCFRAG_TASK_EXCLUDE;
            if (gi != gj)
                mode_mask |= SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
            create_task_for_output(row_piece, partner_piece, output,
                                   mode_mask);
        }
    }
    for (size_t rp = 0; rp < row_task_degrees.size(); ++rp)
        if (row_piece_task_write[rp] !=
            state.row_piece_task_offsets[rp + 1])
            ABORT("GPU3DV2_PCFRAG_TASKFLOW row-piece task adjacency is incomplete.");
    for (size_t cp = 0; cp < partner_task_degrees.size(); ++cp)
        if (partner_piece_task_write[cp] !=
            state.partner_piece_task_offsets[cp + 1])
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner-piece task adjacency is incomplete.");
    if (need_piece_pair_lookup)
        std::sort(state.pair_task_entries.begin(),
                  state.pair_task_entries.end());
    state.runnable_lookahead_col_by_gid.assign_from_degrees(
        lookahead_col_degrees);
    state.runnable_lookahead_row_by_gid.assign_from_degrees(
        lookahead_row_degrees);
    if (!state.tasks.empty() &&
        superlu_sym_v2_pcfrag_taskflow_force_output_locks())
        state.output_conflicts_possible = 1;
    state.incomplete_task_count = static_cast<int>(state.tasks.size());
    symV2PcFragTaskflowStats.tasks_planned +=
        static_cast<long long>(state.tasks.size());
    symV2PcFragTaskflowStats.arena_index_high_water =
        SUPERLU_MAX(symV2PcFragTaskflowStats.arena_index_high_water,
                    static_cast<long long>(
                        (state.index_pool_capacity +
                         state.group_index_pool_capacity) *
                        sizeof(int_t)));
    symV2PcFragTaskflowStats.arena_value_high_water =
        SUPERLU_MAX(symV2PcFragTaskflowStats.arena_value_high_water,
                    static_cast<long long>(
                        (state.value_pool_capacity +
                         state.group_value_pool_capacity) *
                        sizeof(double)));
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
    int_t k, unsigned char kind, const double *stage,
    const std::vector<int_t> &recv_map, int_t ksupc, cudaStream_t stream)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    if (stage == NULL)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly has no source.");
    if (recv_map.empty())
        return 0;
    if (recv_map.size() % 3 != 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW piece receive map has invalid stride.");
    if (static_cast<size_t>(k) >= symV2PcFragTaskStates.size())
        ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly has no state.");
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    if (!state.initialized)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly has uninitialized state.");
    std::vector<SymV2PcFragPieceDesc> &pieces =
        (kind == SYM_V2_PCFRAG_PIECE_ROW)
            ? state.row_pieces
            : state.partner_pieces;
    for (size_t p = 1; p < pieces.size(); ++p)
    {
        int_t prev_begin = pieces[p - 1].frag_row_offset;
        int_t prev_end = prev_begin + pieces[p - 1].nrows;
        if (pieces[p].frag_row_offset < prev_end)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece offsets are not sorted.");
    }
    auto piece_for_range =
        [&](int_t dst_offset, int_t nrows,
            size_t *piece_cursor) -> SymV2PcFragPieceDesc * {
        while (*piece_cursor < pieces.size())
        {
            int_t begin = pieces[*piece_cursor].frag_row_offset;
            int_t end = begin + pieces[*piece_cursor].nrows;
            if (dst_offset < end)
                break;
            ++(*piece_cursor);
        }
        if (*piece_cursor < pieces.size())
        {
            int_t begin = pieces[*piece_cursor].frag_row_offset;
            int_t end = begin + pieces[*piece_cursor].nrows;
            if (dst_offset >= begin && dst_offset + nrows <= end)
                return &pieces[*piece_cursor];
        }

        size_t lo = 0;
        size_t hi = pieces.size();
        while (lo < hi)
        {
            size_t mid = lo + (hi - lo) / 2;
            if (pieces[mid].frag_row_offset <= dst_offset)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == 0)
            return NULL;
        size_t pos = lo - 1;
        int_t begin = pieces[pos].frag_row_offset;
        int_t end = begin + pieces[pos].nrows;
        if (dst_offset >= begin && dst_offset + nrows <= end)
        {
            *piece_cursor = pos;
            return &pieces[pos];
        }
        return NULL;
    };
    int ready_count = 0;
    size_t map_pos = 0;
    size_t piece_cursor = 0;
    while (map_pos < recv_map.size())
    {
        int_t dst_offset = recv_map[map_pos++];
        int_t nrows = recv_map[map_pos++];
        int_t src_offset = recv_map[map_pos++];
        if (nrows <= 0)
            continue;
        SymV2PcFragPieceDesc *piece =
            piece_for_range(dst_offset, nrows, &piece_cursor);
        if (piece == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW receive map does not match a piece.");
        if (piece->d_val == NULL || piece->lda <= 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece value storage is missing.");
        if (piece->filled_rows > piece->nrows - nrows)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly overfills a piece.");
        int_t row_offset = dst_offset - piece->frag_row_offset;
        gpuErrchk(cudaMemcpy2DAsync(
            piece->d_val + row_offset,
            sizeof(double) * static_cast<size_t>(piece->lda),
            stage + src_offset,
            sizeof(double) * static_cast<size_t>(nrows),
            sizeof(double) * static_cast<size_t>(nrows),
            static_cast<size_t>(ksupc),
            cudaMemcpyDeviceToDevice, stream));
        piece->filled_rows += nrows;
        if (!piece->ready && piece->filled_rows == piece->nrows)
        {
            if (piece->ready_event != NULL)
            {
                gpuErrchk(cudaEventRecord(piece->ready_event, stream));
                state.producer_last_ready_event = piece->ready_event;
            }
            piece->ready = 1;
            if (kind == SYM_V2_PCFRAG_PIECE_ROW)
            {
                ++symV2PcFragTaskflowStats.row_pieces_ready;
                ++state.row_pieces_ready_count;
            }
            else
            {
                ++symV2PcFragTaskflowStats.partner_pieces_ready;
                ++state.partner_pieces_ready_count;
            }
            state.note_piece_ready(kind, piece->piece_id);
            ++ready_count;
        }
    }
    return ready_count;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowProgressExchangeGPU(
    int_t k, int drain)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PcFragTaskStates.size())
        return 0;
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    if (!state.initialized)
        return 0;
    int progressed = dSymV2PcFragTaskflowProgressProducerSends(
        state, symV2PcFragTaskflowStats);
    if (!state.producer_exchange_pending)
        return progressed;
    int streamId = state.stream_offset >= 0 ? state.stream_offset : 0;
    if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
        streamId = 0;
    cudaStream_t stream = superlu_sym_v2_async_factor()
                              ? A_gpu.lookAheadUStream[streamId]
                              : A_gpu.cuStreams[streamId];
    if (stream == NULL)
        stream = A_gpu.cuStreams[streamId];
    ++symV2PcFragTaskflowStats.producer_exchange_progress_calls;
    if (drain)
        ++symV2PcFragTaskflowStats.producer_exchange_drain_calls;

    auto complete_partner_recv = [&](int req_index) {
        if (req_index < 0 ||
            static_cast<size_t>(req_index) >=
                state.producer_partner_recv_reqs.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner receive completion index is invalid.");
        if (state.producer_partner_recv_done[static_cast<size_t>(req_index)])
            return;
        int pr = state.producer_partner_recv_prs[static_cast<size_t>(req_index)];
        int count = state.producer_partner_recv_sizes[static_cast<size_t>(req_index)];
        int offset = state.producer_partner_recv_offsets[static_cast<size_t>(req_index)];
        if (pr < 0 || pr >= Pr || count <= 0 || offset < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner receive metadata is invalid.");
        if (A_gpu.symPartnerLStageBufs[streamId] == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner stage buffer is missing.");
        if (static_cast<size_t>(offset) + static_cast<size_t>(count) >
            state.producer_partner_recv_host_capacity ||
            state.producer_partner_recv_host_values == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner receive host buffer is too small.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLStageBufs[streamId] + offset,
            state.producer_partner_recv_host_values + offset,
            sizeof(double) * static_cast<size_t>(count),
            cudaMemcpyHostToDevice, stream));
        size_t recv_map_pos =
            static_cast<size_t>(k) * static_cast<size_t>(Pr) +
            static_cast<size_t>(pr);
        if (recv_map_pos >= symV2PartnerLRecvMap.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW partner receive map is missing.");
        dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
            k, SYM_V2_PCFRAG_PIECE_PARTNER,
            A_gpu.symPartnerLStageBufs[streamId] + offset,
            symV2PartnerLRecvMap[recv_map_pos],
            state.producer_ksupc, stream);
        state.producer_partner_recv_done[static_cast<size_t>(req_index)] = 1;
        --state.producer_partner_recv_remaining;
        ++symV2PcFragTaskflowStats.producer_recv_test_completions;
        ++progressed;
    };

    auto complete_row_recv = [&](int req_index) {
        if (req_index < 0 ||
            static_cast<size_t>(req_index) >=
                state.producer_row_recv_reqs.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW row receive completion index is invalid.");
        if (state.producer_row_recv_done[static_cast<size_t>(req_index)])
            return;
        int pc = state.producer_row_recv_pcs[static_cast<size_t>(req_index)];
        int count = state.producer_row_recv_sizes[static_cast<size_t>(req_index)];
        int offset = state.producer_row_recv_offsets[static_cast<size_t>(req_index)];
        if (A_gpu.symV2RowFragStageBufs[streamId] == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW row stage buffer is missing.");
        if (pc == -1)
        {
            if (static_cast<size_t>(k) >= symV2RowFragRecvIndex.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row aggregate index is missing.");
            const std::vector<int_t> &row_index =
                symV2RowFragRecvIndex[static_cast<size_t>(k)];
            if (row_index.empty() || row_index[1] <= 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row aggregate index is invalid.");
            int_t row_nrows = row_index[1];
            if (state.producer_row_recv_host_capacity <
                static_cast<size_t>(row_nrows) *
                    static_cast<size_t>(state.producer_ksupc) ||
                state.producer_row_recv_host_values == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row aggregate host buffer is too small.");
            for (size_t p = 0; p < state.row_pieces.size(); ++p)
            {
                SymV2PcFragPieceDesc &piece = state.row_pieces[p];
                if (piece.ready)
                    continue;
                if (piece.d_val == NULL || piece.lda <= 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row piece storage is missing.");
                if (piece.frag_row_offset < 0 ||
                    piece.frag_row_offset + piece.nrows > row_nrows)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row aggregate piece range is invalid.");
                gpuErrchk(cudaMemcpy2DAsync(
                    piece.d_val,
                    sizeof(double) * static_cast<size_t>(piece.lda),
                    state.producer_row_recv_host_values +
                        piece.frag_row_offset,
                    sizeof(double) * static_cast<size_t>(row_nrows),
                    sizeof(double) * static_cast<size_t>(piece.nrows),
                    static_cast<size_t>(state.producer_ksupc),
                    cudaMemcpyHostToDevice, stream));
                piece.filled_rows = piece.nrows;
                if (piece.ready_event != NULL)
                {
                    gpuErrchk(cudaEventRecord(piece.ready_event, stream));
                    state.producer_last_ready_event = piece.ready_event;
                }
                piece.ready = 1;
                ++state.row_pieces_ready_count;
                ++symV2PcFragTaskflowStats.row_pieces_ready;
                state.note_piece_ready(
                    SYM_V2_PCFRAG_PIECE_ROW, piece.piece_id);
            }
        }
        else
        {
            if (pc < 0 || pc >= Pc || count <= 0 || offset < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row receive metadata is invalid.");
            if (static_cast<size_t>(offset) + static_cast<size_t>(count) >
                state.producer_row_recv_host_capacity ||
                state.producer_row_recv_host_values == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row receive host buffer is too small.");
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symV2RowFragStageBufs[streamId] + offset,
                state.producer_row_recv_host_values + offset,
                sizeof(double) * static_cast<size_t>(count),
                cudaMemcpyHostToDevice, stream));
            size_t row_pos =
                static_cast<size_t>(k) * static_cast<size_t>(Pc) +
                static_cast<size_t>(pc);
            if (row_pos >= symV2RowFragRecvMap.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW row receive map is missing.");
            dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
                k, SYM_V2_PCFRAG_PIECE_ROW,
                A_gpu.symV2RowFragStageBufs[streamId] + offset,
                symV2RowFragRecvMap[row_pos],
                state.producer_ksupc, stream);
        }
        state.producer_row_recv_done[static_cast<size_t>(req_index)] = 1;
        --state.producer_row_recv_remaining;
        ++symV2PcFragTaskflowStats.producer_recv_test_completions;
        ++progressed;
    };

    auto progress_requests =
        [&](std::vector<MPI_Request> &reqs, int remaining,
            void (*unused)(int), int kind) -> int {
        (void)unused;
        if (remaining <= 0 || reqs.empty())
            return 0;
        const int request_count = static_cast<int>(reqs.size());
        if (state.producer_progress_indices.size() <
                static_cast<size_t>(request_count) ||
            state.producer_progress_statuses.size() <
                static_cast<size_t>(request_count))
        {
            if (superlu_sym_v2_pcfrag_taskflow_async_core())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE progress scratch is undersized.");
            dSymV2PcFragTaskflowEnsureProgressScratch(
                state, symV2PcFragTaskflowStats,
                static_cast<size_t>(request_count));
        }
        int *indices = state.producer_progress_indices.data();
        MPI_Status *statuses = state.producer_progress_statuses.data();
        int local_progress = 0;
        do
        {
            int completed = 0;
            ++symV2PcFragTaskflowStats.producer_recv_test_calls;
            int mpi_rc = drain
                ? MPI_Waitsome(request_count, reqs.data(), &completed,
                               indices, statuses)
                : MPI_Testsome(request_count, reqs.data(), &completed,
                               indices, statuses);
            if (mpi_rc != MPI_SUCCESS)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW receive progress failed.");
            if (completed == MPI_UNDEFINED || completed == 0)
                break;
            for (int item = 0; item < completed; ++item)
            {
                int req_index = indices[item];
                if (kind == 0)
                    complete_partner_recv(req_index);
                else
                    complete_row_recv(req_index);
                ++local_progress;
            }
            remaining -= completed;
        } while (drain && remaining > 0);
        return local_progress;
    };

    progress_requests(state.producer_partner_recv_reqs,
                      state.producer_partner_recv_remaining,
                      NULL, 0);
    progress_requests(state.producer_row_recv_reqs,
                      state.producer_row_recv_remaining,
                      NULL, 1);
    if (state.producer_partner_recv_remaining < 0 ||
        state.producer_row_recv_remaining < 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW receive remaining count underflowed.");
    if (state.producer_partner_recv_remaining == 0 &&
        state.producer_row_recv_remaining == 0)
    {
        state.producer_exchange_pending = 0;
        state.producer_partner_recv_reqs.clear();
        state.producer_partner_recv_prs.clear();
        state.producer_partner_recv_sizes.clear();
        state.producer_partner_recv_offsets.clear();
        state.producer_partner_recv_done.clear();
        state.producer_row_recv_reqs.clear();
        state.producer_row_recv_pcs.clear();
        state.producer_row_recv_sizes.clear();
        state.producer_row_recv_offsets.clear();
        state.producer_row_recv_done.clear();
    }
    return progressed;
}

template <>
int_t xLUstruct_t<double>::dSymV2PcFragTaskflowDispatchGPU(
    int streamId, int_t k, unsigned mode_mask, int_t mode_gid, int drain);

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowProgressGPU(
    int_t k, int budget)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    if (static_cast<size_t>(k) >= symV2PcFragTaskStates.size())
        return 0;
    if (!superlu_sym_v2_pcfrag_taskflow_eager() ||
        superlu_sym_v2_pcfrag_taskflow_validate())
        return 0;
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    if (!state.initialized || state.incomplete_task_count <= 0)
        return 0;
    int streamId = state.stream_offset >= 0 ? state.stream_offset : 0;
    if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
        streamId = 0;
    cudaStream_t stream = A_gpu.cuStreams[streamId];
    cublasHandle_t handle = A_gpu.cuHandles[streamId];
    double *gemmBuff = A_gpu.gpuGemmBuffs[streamId];
    if (budget <= 0)
        budget = superlu_sym_v2_pcfrag_taskflow_effective_progress_budget();
    const bool async_core =
        superlu_sym_v2_pcfrag_taskflow_async_core();
    const int producer_task_limit =
        superlu_sym_v2_pcfrag_taskflow_producer_task_limit();
    if (producer_task_limit > 0)
    {
        const int producer_tasks_remaining =
            producer_task_limit - state.producer_tasks_launched;
        if (producer_tasks_remaining <= 0)
        {
            if (!state.producer_launch_cap_reported &&
                state.incomplete_task_count > 0)
            {
                ++symV2PcFragTaskflowStats.producer_task_launch_cap_hits;
                symV2PcFragTaskflowStats.producer_task_launch_cap_deferred +=
                    static_cast<long long>(state.incomplete_task_count);
                state.producer_launch_cap_reported = 1;
            }
            return 0;
        }
        budget = SUPERLU_MIN(budget, producer_tasks_remaining);
    }
    if (async_core &&
        producer_task_limit == 0 &&
        superlu_sym_v2_pcfrag_taskflow_async_grouped_dispatch())
    {
        int_t launched_grouped =
            dSymV2PcFragTaskflowDispatchGPU(
                streamId, k, SYM_V2_PCFRAG_TASK_FULL,
                GLOBAL_BLOCK_NOT_FOUND, 0);
        if (launched_grouped > 0)
        {
            symV2PcFragTaskflowStats.tasks_launched_progress +=
                static_cast<long long>(launched_grouped);
            return launched_grouped;
        }
        if (static_cast<size_t>(k) >= symV2PcFragTaskStates.size() ||
            !state.initialized)
            return 0;
    }
    const bool strict_output_conflicts =
        superlu_sym_v2_pcfrag_taskflow_strict() &&
        (state.output_conflicts_possible ||
         superlu_sym_v2_pcfrag_taskflow_global_output_locks());
    const bool compact_output_locks =
        dSymV2PcFragTaskflowUseCompactOutputLocks(*this);

    auto all_pieces_ready = [&]() -> bool {
        return state.row_pieces_ready_count ==
                   state.row_pieces.size() &&
               state.partner_pieces_ready_count ==
                   state.partner_pieces.size();
    };
    auto output_locked = [&](const SymV2PcFragTaskDesc &task) -> bool {
        if (!strict_output_conflicts)
            return false;
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        for (size_t o = 0; o < output_count; ++o)
        {
            if (compact_output_locks)
            {
                int_t output_id =
                    dSymV2PcFragTaskflowCompactOutputIdAt(
                        state, task, o);
                if (output_id < 0 ||
                    static_cast<size_t>(output_id) >=
                        symV2PcFragTaskflowGlobalOutputLockState.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock id is invalid.");
                if (symV2PcFragTaskflowGlobalOutputLockState[
                        static_cast<size_t>(output_id)])
                {
                    ++symV2PcFragTaskflowStats.global_output_lock_conflicts;
                    return true;
                }
                continue;
            }
            const SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, o);
            if (state.active_output_key_set.find(key) !=
                state.active_output_key_set.end())
                return true;
            if (superlu_sym_v2_pcfrag_taskflow_global_output_locks() &&
                symV2PcFragTaskflowGlobalOutputLocks.find(key) !=
                    symV2PcFragTaskflowGlobalOutputLocks.end())
            {
                ++symV2PcFragTaskflowStats.global_output_lock_conflicts;
                return true;
            }
        }
        return false;
    };
    auto lock_outputs = [&](const SymV2PcFragTaskDesc &task) {
        if (!strict_output_conflicts)
            return;
        const size_t output_count =
            dSymV2PcFragTaskflowOutputCount(task);
        for (size_t o = 0; o < output_count; ++o)
        {
            if (compact_output_locks)
            {
                int_t output_id =
                    dSymV2PcFragTaskflowCompactOutputIdAt(
                        state, task, o);
                if (output_id < 0 ||
                    static_cast<size_t>(output_id) >=
                        symV2PcFragTaskflowGlobalOutputLockState.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock id is invalid.");
                if (symV2PcFragTaskflowGlobalOutputLockState[
                        static_cast<size_t>(output_id)])
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock was already held.");
                symV2PcFragTaskflowGlobalOutputLockState[
                    static_cast<size_t>(output_id)] = 1;
                ++symV2PcFragTaskflowGlobalOutputLocksLive;
                ++state.active_output_lock_count;
                continue;
            }
            const SymV2PcFragOutputKey &key =
                dSymV2PcFragTaskflowOutputAt(state, task, o);
            state.active_output_key_set.insert(key);
            if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
                symV2PcFragTaskflowGlobalOutputLocks.insert(key);
        }
        symV2PcFragTaskflowStats.output_locks_acquired +=
            static_cast<long long>(output_count);
        if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
            symV2PcFragTaskflowStats.global_output_locks_acquired +=
                static_cast<long long>(output_count);
        long long active_output_locks =
            compact_output_locks
                ? static_cast<long long>(state.active_output_lock_count)
                : static_cast<long long>(state.active_output_key_set.size());
        if (active_output_locks >
            symV2PcFragTaskflowStats.output_lock_high_water)
            symV2PcFragTaskflowStats.output_lock_high_water =
                active_output_locks;
    };
    auto unlock_outputs = [&](const SymV2PcFragTaskDesc &task) {
        return dSymV2PcFragTaskflowReleaseOutputLocks(
            *this, state, task, strict_output_conflicts, false);
    };
    auto progress_launched_tasks = [&](int drain) -> int {
        if (!async_core)
            return 0;
        int completed = 0;
        int idle_polls = 0;
        do
        {
            int pending = 0;
            int completed_this_pass =
                dSymV2PcFragTaskflowProgressLaunchedTasks(
                    *this, state, symV2PcFragTaskflowStats,
                    strict_output_conflicts, drain, 0,
                    GLOBAL_BLOCK_NOT_FOUND, &pending);
            completed += completed_this_pass;
            if (!drain || pending == 0)
                break;
            if (completed_this_pass > 0)
            {
                idle_polls = 0;
                continue;
            }
            dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
            if (++idle_polls > 10000000)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW async-core task drain made no CUDA progress.");
        } while (drain);
        return completed;
    };
    auto release_completed_state = [&]() -> int {
        if (!dSymV2PcFragTaskflowProducerSendsComplete(
                state, symV2PcFragTaskflowStats, 0))
            return 0;
        if (!state.active_output_key_set.empty() ||
            state.active_output_lock_count != 0)
        {
            std::fprintf(stderr,
                         "GPU3DV2 taskflow release with live local output locks: k=%lld locks=%zu compact_locks=%d incomplete=%d\n",
                         static_cast<long long>(k),
                         state.active_output_key_set.size(),
                         state.active_output_lock_count,
                         state.incomplete_task_count);
            ABORT("GPU3DV2_PCFRAG_TASKFLOW release found live local output locks.");
        }
        for (size_t i = 0; i < state.tasks.size(); ++i)
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                state.tasks[i].done_event);
        dSymV2PcFragTaskflowProducerStreamComplete(state, 1);
        auto release_piece_storage = [&](SymV2PcFragPieceDesc &piece) {
            if (piece.pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                piece.ready_event);
            dSymV2PcFragTaskflowRecycleEvent(
                symV2PcFragTaskflowEventPool,
                piece.done_event);
            piece.d_index = NULL;
            piece.d_val = NULL;
            piece.h_index.clear();
        };
        for (size_t p = 0; p < state.row_pieces.size(); ++p)
            release_piece_storage(state.row_pieces[p]);
        for (size_t p = 0; p < state.partner_pieces.size(); ++p)
            release_piece_storage(state.partner_pieces[p]);
        dSymV2PcFragTaskflowRecycleIndexBlock(
            symV2PcFragTaskflowIndexBlockPool,
            state.d_index_pool, state.index_pool_capacity);
        dSymV2PcFragTaskflowRecycleValueBlock(
            symV2PcFragTaskflowValueBlockPool,
            state.d_value_pool, state.value_pool_capacity);
        dSymV2PcFragTaskflowRecycleIndexBlock(
            symV2PcFragTaskflowIndexBlockPool,
            state.d_group_index_pool, state.group_index_pool_capacity);
        dSymV2PcFragTaskflowRecycleValueBlock(
            symV2PcFragTaskflowValueBlockPool,
            state.d_group_value_pool, state.group_value_pool_capacity);
        dSymV2PcFragTaskflowReleasePinnedHost(
            symV2PcFragTaskflowPinnedBlockPool, state);
        state.reset();
        return 1;
    };

    int launched = 0;
    progress_launched_tasks(0);
    int pending_launched =
        async_core ? dSymV2PcFragTaskflowPendingLaunchedAllForMode(
                         state, 0, 0, GLOBAL_BLOCK_NOT_FOUND)
                   : 0;
    const int in_flight_task_cap =
        async_core ? superlu_sym_v2_pcfrag_taskflow_effective_progress_budget() : 0;
    size_t runnable_write = 0;
    for (size_t i = 0; i < state.runnable_task_ids.size(); ++i)
    {
        int tid = state.runnable_task_ids[i];
        if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
            ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable task id is invalid.");
        SymV2PcFragTaskDesc &task =
            state.tasks[static_cast<size_t>(tid)];
        if (task.launched || task.complete)
            continue;
        if (launched >= budget)
        {
            state.runnable_task_ids[runnable_write++] = tid;
            continue;
        }
        if (async_core && pending_launched >= in_flight_task_cap)
        {
            state.runnable_task_ids[runnable_write++] = tid;
            continue;
        }
        SymV2PcFragPieceDesc &row =
            state.row_pieces[static_cast<size_t>(task.row_piece)];
        SymV2PcFragPieceDesc &col =
            state.partner_pieces[static_cast<size_t>(task.partner_piece)];
        if (!row.ready)
            ++symV2PcFragTaskflowStats.tasks_blocked_row;
        if (!col.ready)
            ++symV2PcFragTaskflowStats.tasks_blocked_partner;
        if (!row.ready || !col.ready)
        {
            state.runnable_task_ids[runnable_write++] = tid;
            continue;
        }
        if (output_locked(task))
        {
            ++symV2PcFragTaskflowStats.tasks_blocked_output;
            ++symV2PcFragTaskflowStats.scatter_conflict_waits;
            state.runnable_task_ids[runnable_write++] = tid;
            continue;
        }
        if (row.d_index == NULL || row.d_val == NULL ||
            col.d_index == NULL || col.d_val == NULL ||
            gemmBuff == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW progress task is missing owned device storage.");
        if (!all_pieces_ready())
            ++symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready;
        lock_outputs(task);
        if (row.ready_event != NULL)
            gpuErrchk(cudaStreamWaitEvent(stream, row.ready_event, 0));
        if (col.ready_event != NULL)
            gpuErrchk(cudaStreamWaitEvent(stream, col.ready_event, 0));
        dSymV2PcFragTaskflowWaitOnGemmResource(
            *this, state, task, streamId, stream,
            SYM_V2_PCFRAG_TASK_FULL);
        dSymSchurCompUpdateTaskDualPiecesGPU(
            k,
            row.h_index, col.h_index,
            row.d_index, row.d_val,
            col.d_index, col.d_val,
            handle, stream, gemmBuff);
        task.launched = 1;
        task.launch_stream_kind = SYM_V2_PCFRAG_TASK_STREAM_MAIN;
        if (async_core)
        {
            if (task.done_event == NULL)
                task.done_event =
                        dSymV2PcFragTaskflowAcquireEvent(
                            symV2PcFragTaskflowEventPool,
                            !async_core,
                            &symV2PcFragTaskflowStats.arena_event_late_allocs);
            gpuErrchk(cudaEventRecord(task.done_event, stream));
            dSymV2PcFragTaskflowPublishGemmResourceTail(
                *this, state, task, streamId, stream);
            dSymV2PcFragTaskflowRecordLaunchedTask(state, task);
            ++pending_launched;
            if (dSymV2PcFragTaskflowForceTaskSync())
            {
                ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
                gpuErrchk(cudaStreamSynchronize(stream));
                dSymV2PcFragTaskflowUnrecordLaunchedTask(state, task);
                dSymV2PcFragTaskflowCompleteLaunchedTask(
                    *this, state, task, strict_output_conflicts);
                dSymV2PcFragTaskflowRecycleEvent(
                    symV2PcFragTaskflowEventPool, task.done_event);
                --pending_launched;
            }
        }
        else
        {
            ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
            gpuErrchk(cudaStreamSynchronize(stream));
            symV2PcFragTaskflowStats.output_locks_released_by_launch_sync +=
                unlock_outputs(task);
            dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(
                state, task);
            task.complete = 1;
            --state.incomplete_task_count;
            --row.pending_consumers;
            --col.pending_consumers;
            if (row.pending_consumers < 0 || col.pending_consumers < 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
            ++symV2PcFragTaskflowStats.tasks_completed;
        }
        ++symV2PcFragTaskflowStats.tasks_launched;
        ++symV2PcFragTaskflowStats.tasks_launched_progress;
        ++state.producer_tasks_launched;
        ++launched;
    }
    if (runnable_write != state.runnable_task_ids.size())
        state.runnable_task_ids.resize(runnable_write);
    progress_launched_tasks(0);
    if (producer_task_limit > 0 &&
        state.incomplete_task_count > 0 &&
        state.producer_tasks_launched >= producer_task_limit &&
        !state.producer_launch_cap_reported)
    {
        ++symV2PcFragTaskflowStats.producer_task_launch_cap_hits;
        symV2PcFragTaskflowStats.producer_task_launch_cap_deferred +=
            static_cast<long long>(state.incomplete_task_count);
        state.producer_launch_cap_reported = 1;
    }
    if (state.incomplete_task_count == 0 &&
        !state.producer_exchange_pending &&
        dSymV2PcFragTaskflowProducerStreamComplete(state, 0) &&
        !state.producer_exchange_active)
        release_completed_state();
    return launched;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowDispatchGPU(
    int streamId, int_t k, unsigned mode_mask, int_t mode_gid, int drain)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    if (static_cast<size_t>(k) < symV2PcFragTaskStates.size())
    {
        if (!drain)
            dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
        SymV2PcFragPanelTaskState &state =
            symV2PcFragTaskStates[static_cast<size_t>(k)];
        if (!state.initialized)
            return 0;
        if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
            streamId = state.stream_offset >= 0 ? state.stream_offset : 0;
        if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
            streamId = 0;
        cudaStream_t stream = A_gpu.cuStreams[streamId];
        cublasHandle_t handle = A_gpu.cuHandles[streamId];
        double *gemmBuff = A_gpu.gpuGemmBuffs[streamId];
        const bool strict_output_conflicts =
            superlu_sym_v2_pcfrag_taskflow_strict() &&
            (state.output_conflicts_possible ||
             superlu_sym_v2_pcfrag_taskflow_global_output_locks());
        const bool async_core =
            superlu_sym_v2_pcfrag_taskflow_async_core();
        const bool allow_taskflow_late_alloc = !async_core;
        const bool async_grouped_dispatch =
            async_core &&
            superlu_sym_v2_pcfrag_taskflow_async_grouped_dispatch();
        const bool compact_output_locks =
            dSymV2PcFragTaskflowUseCompactOutputLocks(*this);

        auto all_pieces_ready = [&]() -> bool {
            return state.row_pieces_ready_count ==
                       state.row_pieces.size() &&
                   state.partner_pieces_ready_count ==
                       state.partner_pieces.size();
        };
        auto output_locked = [&](const SymV2PcFragTaskDesc &task) -> bool {
            if (!strict_output_conflicts)
                return false;
            const size_t output_count =
                dSymV2PcFragTaskflowOutputCount(task);
            for (size_t o = 0; o < output_count; ++o)
            {
                if (compact_output_locks)
                {
                    int_t output_id =
                        dSymV2PcFragTaskflowCompactOutputIdAt(
                            state, task, o);
                    if (output_id < 0 ||
                        static_cast<size_t>(output_id) >=
                            symV2PcFragTaskflowGlobalOutputLockState.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock id is invalid.");
                    if (symV2PcFragTaskflowGlobalOutputLockState[
                            static_cast<size_t>(output_id)])
                    {
                        ++symV2PcFragTaskflowStats.global_output_lock_conflicts;
                        return true;
                    }
                    continue;
                }
                const SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                if (state.active_output_key_set.find(key) !=
                    state.active_output_key_set.end())
                    return true;
                if (superlu_sym_v2_pcfrag_taskflow_global_output_locks() &&
                    symV2PcFragTaskflowGlobalOutputLocks.find(key) !=
                        symV2PcFragTaskflowGlobalOutputLocks.end())
                {
                    ++symV2PcFragTaskflowStats.global_output_lock_conflicts;
                    return true;
                }
            }
            return false;
        };
        auto task_stream_kind_for_launch_mode =
            [](unsigned launch_mode) -> unsigned char {
            if (launch_mode & SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
                return SYM_V2_PCFRAG_TASK_STREAM_LOOKAHEAD_L;
            if (launch_mode & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                return SYM_V2_PCFRAG_TASK_STREAM_LOOKAHEAD_U;
            return SYM_V2_PCFRAG_TASK_STREAM_MAIN;
        };
        auto lock_outputs = [&](const SymV2PcFragTaskDesc &task) {
            if (!strict_output_conflicts)
                return;
            const size_t output_count =
                dSymV2PcFragTaskflowOutputCount(task);
            for (size_t o = 0; o < output_count; ++o)
            {
                if (compact_output_locks)
                {
                    int_t output_id =
                        dSymV2PcFragTaskflowCompactOutputIdAt(
                            state, task, o);
                    if (output_id < 0 ||
                        static_cast<size_t>(output_id) >=
                            symV2PcFragTaskflowGlobalOutputLockState.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock id is invalid.");
                    if (symV2PcFragTaskflowGlobalOutputLockState[
                            static_cast<size_t>(output_id)])
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW compact output lock was already held.");
                    symV2PcFragTaskflowGlobalOutputLockState[
                        static_cast<size_t>(output_id)] = 1;
                    ++symV2PcFragTaskflowGlobalOutputLocksLive;
                    ++state.active_output_lock_count;
                    continue;
                }
                const SymV2PcFragOutputKey &key =
                    dSymV2PcFragTaskflowOutputAt(state, task, o);
                state.active_output_key_set.insert(key);
                if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
                    symV2PcFragTaskflowGlobalOutputLocks.insert(
                        key);
            }
            symV2PcFragTaskflowStats.output_locks_acquired +=
                static_cast<long long>(output_count);
            if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
                symV2PcFragTaskflowStats.global_output_locks_acquired +=
                    static_cast<long long>(output_count);
            long long active_output_locks =
                compact_output_locks
                    ? static_cast<long long>(state.active_output_lock_count)
                    : static_cast<long long>(state.active_output_key_set.size());
            if (active_output_locks >
                symV2PcFragTaskflowStats.output_lock_high_water)
                symV2PcFragTaskflowStats.output_lock_high_water =
                    active_output_locks;
        };
        auto unlock_outputs = [&](const SymV2PcFragTaskDesc &task) {
            return dSymV2PcFragTaskflowReleaseOutputLocks(
                *this, state, task, strict_output_conflicts, false);
        };
        auto progress_launched_tasks =
            [&](int drain, unsigned required_mode_mask) -> int {
            if (!async_core)
                return 0;
            int completed = 0;
            int idle_polls = 0;
            do
            {
                int pending_required = 0;
                int completed_this_pass =
                    dSymV2PcFragTaskflowProgressLaunchedTasks(
                        *this, state, symV2PcFragTaskflowStats,
                        strict_output_conflicts, drain,
                        required_mode_mask, mode_gid,
                        &pending_required);
                completed += completed_this_pass;
                if (!drain || pending_required == 0)
                    break;
                if (completed_this_pass > 0)
                {
                    idle_polls = 0;
                    continue;
                }
                dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
                if (++idle_polls > 10000000)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW async-core task drain made no CUDA progress.");
            } while (drain);
            return completed;
        };
        auto count_task_launch_mode = [&](unsigned launch_mode) {
            if (launch_mode &
                (SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                 SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW))
                ++symV2PcFragTaskflowStats.tasks_launched_lookahead;
            else if (launch_mode & SYM_V2_PCFRAG_TASK_EXCLUDE)
                ++symV2PcFragTaskflowStats.tasks_launched_exclude;
            else if (launch_mode & SYM_V2_PCFRAG_TASK_FULL)
                ++symV2PcFragTaskflowStats.tasks_launched_full;
        };
        auto task_matches_launch_mode =
            [](const SymV2PcFragTaskDesc &task,
               unsigned launch_mode) -> bool {
            return (task.mode_mask & launch_mode) != 0;
        };
        auto task_launch_mode_for_request =
            [&](const SymV2PcFragTaskDesc &task,
                unsigned requested_mode_mask,
                int_t requested_gid) -> unsigned {
            if ((requested_mode_mask & SYM_V2_PCFRAG_TASK_FULL) &&
                (task.mode_mask & SYM_V2_PCFRAG_TASK_FULL))
                return SYM_V2_PCFRAG_TASK_FULL;
            if ((requested_mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL) &&
                (task.mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL))
            {
                if (requested_gid == GLOBAL_BLOCK_NOT_FOUND)
                    return SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL;
                const size_t output_count =
                    dSymV2PcFragTaskflowOutputCount(task);
                for (size_t o = 0; o < output_count; ++o)
                    if (dSymV2PcFragTaskflowOutputAt(state, task, o).gj ==
                        requested_gid)
                        return SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL;
            }
            if ((requested_mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW) &&
                (task.mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW))
            {
                if (requested_gid == GLOBAL_BLOCK_NOT_FOUND)
                    return SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
                const size_t output_count =
                    dSymV2PcFragTaskflowOutputCount(task);
                for (size_t o = 0; o < output_count; ++o)
                    if (dSymV2PcFragTaskflowOutputAt(state, task, o).gi ==
                        requested_gid)
                        return SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
            }
            if ((requested_mode_mask & SYM_V2_PCFRAG_TASK_EXCLUDE) &&
                (task.mode_mask & SYM_V2_PCFRAG_TASK_EXCLUDE))
                return SYM_V2_PCFRAG_TASK_EXCLUDE;
            return 0;
        };
        auto release_completed_state = [&]() -> int {
            if (!dSymV2PcFragTaskflowProducerSendsComplete(
                    state, symV2PcFragTaskflowStats, 0))
                return 0;
            if (!state.active_output_key_set.empty() ||
                state.active_output_lock_count != 0)
            {
                std::fprintf(stderr,
                             "GPU3DV2 taskflow release with live local output locks: k=%lld locks=%zu compact_locks=%d incomplete=%d\n",
                             static_cast<long long>(k),
                             state.active_output_key_set.size(),
                             state.active_output_lock_count,
                             state.incomplete_task_count);
                ABORT("GPU3DV2_PCFRAG_TASKFLOW release found live local output locks.");
            }
            for (size_t i = 0; i < state.tasks.size(); ++i)
                dSymV2PcFragTaskflowRecycleEvent(
                    symV2PcFragTaskflowEventPool,
                    state.tasks[i].done_event);
            auto release_piece_storage = [&](SymV2PcFragPieceDesc &piece) {
                if (piece.pending_consumers != 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
                dSymV2PcFragTaskflowRecycleEvent(
                    symV2PcFragTaskflowEventPool,
                    piece.ready_event);
                dSymV2PcFragTaskflowRecycleEvent(
                    symV2PcFragTaskflowEventPool,
                    piece.done_event);
                piece.d_index = NULL;
                piece.d_val = NULL;
                piece.h_index.clear();
            };
            for (size_t p = 0; p < state.row_pieces.size(); ++p)
                release_piece_storage(state.row_pieces[p]);
            for (size_t p = 0; p < state.partner_pieces.size(); ++p)
                release_piece_storage(state.partner_pieces[p]);
            dSymV2PcFragTaskflowRecycleIndexBlock(
                symV2PcFragTaskflowIndexBlockPool,
                state.d_index_pool, state.index_pool_capacity);
            dSymV2PcFragTaskflowRecycleValueBlock(
                symV2PcFragTaskflowValueBlockPool,
                state.d_value_pool, state.value_pool_capacity);
            dSymV2PcFragTaskflowRecycleIndexBlock(
                symV2PcFragTaskflowIndexBlockPool,
                state.d_group_index_pool, state.group_index_pool_capacity);
            dSymV2PcFragTaskflowRecycleValueBlock(
                symV2PcFragTaskflowValueBlockPool,
                state.d_group_value_pool, state.group_value_pool_capacity);
            dSymV2PcFragTaskflowReleasePinnedHost(
                symV2PcFragTaskflowPinnedBlockPool, state);
            state.reset();
            return 1;
        };

        progress_launched_tasks(drain ? 1 : 0, mode_mask);
        if ((mode_mask & SYM_V2_PCFRAG_TASK_FULL) &&
            superlu_sym_v2_pcfrag_taskflow_eager() &&
            !superlu_sym_v2_pcfrag_taskflow_validate())
        {
            int launched_this_call = 0;
            int pending_launched =
                async_core ? dSymV2PcFragTaskflowPendingLaunchedAllForMode(
                                 state, 0, 0, GLOBAL_BLOCK_NOT_FOUND)
                           : 0;
            const int in_flight_task_cap =
                async_core ? superlu_sym_v2_pcfrag_taskflow_effective_progress_budget()
                           : 0;
            size_t runnable_write = 0;
            for (size_t i = 0; i < state.runnable_task_ids.size(); ++i)
            {
                int tid = state.runnable_task_ids[i];
                if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable task id is invalid.");
                SymV2PcFragTaskDesc &task =
                    state.tasks[static_cast<size_t>(tid)];
                if (task.launched || task.complete)
                    continue;
                if (async_core && pending_launched >= in_flight_task_cap)
                {
                    state.runnable_task_ids[runnable_write++] = tid;
                    continue;
                }
                SymV2PcFragPieceDesc &row =
                    state.row_pieces[static_cast<size_t>(task.row_piece)];
                SymV2PcFragPieceDesc &col =
                    state.partner_pieces[static_cast<size_t>(task.partner_piece)];
                if (!row.ready)
                    ++symV2PcFragTaskflowStats.tasks_blocked_row;
                if (!col.ready)
                    ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                if (!row.ready || !col.ready)
                {
                    state.runnable_task_ids[runnable_write++] = tid;
                    continue;
                }
                if (output_locked(task))
                {
                    ++symV2PcFragTaskflowStats.tasks_blocked_output;
                    ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                    state.runnable_task_ids[runnable_write++] = tid;
                    continue;
                }
                if (row.d_index == NULL || row.d_val == NULL ||
                    col.d_index == NULL || col.d_val == NULL ||
                    gemmBuff == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW task is missing owned device storage.");
                if (!all_pieces_ready())
                    ++symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready;
                lock_outputs(task);
                if (row.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(stream, row.ready_event, 0));
                if (col.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(stream, col.ready_event, 0));
                dSymV2PcFragTaskflowWaitOnGemmResource(
                    *this, state, task, streamId, stream,
                    SYM_V2_PCFRAG_TASK_FULL);
                dSymSchurCompUpdateTaskDualPiecesGPU(
                    k,
                    row.h_index, col.h_index,
                    row.d_index, row.d_val,
                    col.d_index, col.d_val,
                    handle, stream, gemmBuff);
                task.launched = 1;
                task.launch_stream_kind = SYM_V2_PCFRAG_TASK_STREAM_MAIN;
                if (async_core)
                {
                    if (task.done_event == NULL)
                        task.done_event =
                            dSymV2PcFragTaskflowAcquireEvent(
                                symV2PcFragTaskflowEventPool,
                                !async_core,
                                &symV2PcFragTaskflowStats.arena_event_late_allocs);
                    gpuErrchk(cudaEventRecord(task.done_event, stream));
                    dSymV2PcFragTaskflowPublishGemmResourceTail(
                        *this, state, task, streamId, stream);
                    dSymV2PcFragTaskflowRecordLaunchedTask(state, task);
                    ++pending_launched;
                    if (dSymV2PcFragTaskflowForceTaskSync())
                    {
                        ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
                        gpuErrchk(cudaStreamSynchronize(stream));
                        dSymV2PcFragTaskflowUnrecordLaunchedTask(state, task);
                        dSymV2PcFragTaskflowCompleteLaunchedTask(
                            *this, state, task, strict_output_conflicts);
                        dSymV2PcFragTaskflowRecycleEvent(
                            symV2PcFragTaskflowEventPool, task.done_event);
                        --pending_launched;
                    }
                }
                else
                {
                    ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
                    gpuErrchk(cudaStreamSynchronize(stream));
                    symV2PcFragTaskflowStats.output_locks_released_by_launch_sync +=
                        unlock_outputs(task);
                    dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(
                        state, task);
                    task.complete = 1;
                    --state.incomplete_task_count;
                    --row.pending_consumers;
                    --col.pending_consumers;
                    if (row.pending_consumers < 0 || col.pending_consumers < 0)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
                    ++symV2PcFragTaskflowStats.tasks_completed;
                }
                ++symV2PcFragTaskflowStats.tasks_launched;
                ++symV2PcFragTaskflowStats.tasks_launched_eager_full;
                ++launched_this_call;
            }
            if (runnable_write != state.runnable_task_ids.size())
                state.runnable_task_ids.resize(runnable_write);
            progress_launched_tasks(drain ? 1 : 0, mode_mask);
            if (state.incomplete_task_count == 0 &&
                !state.producer_exchange_active &&
                !state.producer_exchange_pending)
            {
                release_completed_state();
                return launched_this_call;
            }
            if (!drain)
                return launched_this_call;
        }

        const std::vector<int_t> &row_frag = symV2RowFragRecvIndex[k];
        const std::vector<int_t> &col_frag = symV2PartnerLRecvIndex[k];
        auto frag_nblocks = [](const std::vector<int_t> &frag) -> int_t {
            return frag.empty() ? 0 : frag[0];
        };
        auto frag_gid = [](const std::vector<int_t> &frag, int_t b) -> int_t {
            return frag[LPANEL_HEADER_SIZE + b];
        };
        auto frag_strow = [](const std::vector<int_t> &frag, int_t b) -> int_t {
            int_t nb = frag.empty() ? 0 : frag[0];
            return frag[LPANEL_HEADER_SIZE + nb + b];
        };
        auto frag_nbrow = [&](const std::vector<int_t> &frag, int_t b) -> int_t {
            return frag_strow(frag, b + 1) - frag_strow(frag, b);
        };
        auto frag_find = [&](const std::vector<int_t> &frag, int_t gid) -> int_t {
            int_t nb = frag_nblocks(frag);
            for (int_t b = 0; b < nb; ++b)
                if (frag_gid(frag, b) == gid)
                    return b;
            return GLOBAL_BLOCK_NOT_FOUND;
        };
        auto frag_get_end_block =
            [&](const std::vector<int_t> &frag, int_t b, int max_rows) -> int_t {
            int_t nb = frag_nblocks(frag);
            int_t base = frag_strow(frag, b);
            int_t e = b + 1;
            while (e < nb &&
                   frag_strow(frag, e + 1) - base <= max_rows)
                ++e;
            return e;
        };
        auto ranges_excluding =
            [](int_t begin, int_t end, int_t skip0, int_t skip1,
               int_t ranges[3][2]) -> int {
            if (begin >= end)
                return 0;
            if (skip0 == skip1)
                skip1 = GLOBAL_BLOCK_NOT_FOUND;
            if (skip1 != GLOBAL_BLOCK_NOT_FOUND &&
                (skip0 == GLOBAL_BLOCK_NOT_FOUND || skip1 < skip0))
                std::swap(skip0, skip1);
            int nr = 0;
            int_t cur = begin;
            int_t skips[2] = {skip0, skip1};
            for (int s = 0; s < 2; ++s)
            {
                int_t skip = skips[s];
                if (skip == GLOBAL_BLOCK_NOT_FOUND ||
                    skip < begin || skip >= end)
                    continue;
                if (cur < skip)
                {
                    ranges[nr][0] = cur;
                    ranges[nr][1] = skip;
                    ++nr;
                }
                cur = skip + 1;
            }
            if (cur < end)
            {
                ranges[nr][0] = cur;
                ranges[nr][1] = end;
                ++nr;
            }
            return nr;
        };
        auto pair_task =
            [&](int_t rb, int_t cb) -> SymV2PcFragTaskDesc * {
            if (rb < 0 || cb < 0 ||
                static_cast<size_t>(rb) >= state.row_block_piece.size() ||
                static_cast<size_t>(cb) >= state.partner_block_piece.size() ||
                state.partner_pieces.empty())
                return NULL;
            int row_piece = state.row_block_piece[static_cast<size_t>(rb)];
            int partner_piece =
                state.partner_block_piece[static_cast<size_t>(cb)];
            if (row_piece < 0 || partner_piece < 0 ||
                static_cast<size_t>(row_piece) >= state.row_pieces.size() ||
                static_cast<size_t>(partner_piece) >= state.partner_pieces.size())
                return NULL;
            SymV2PcFragPairTaskEntry key(row_piece, partner_piece, -1);
            std::vector<SymV2PcFragPairTaskEntry>::const_iterator it =
                std::lower_bound(state.pair_task_entries.begin(),
                                 state.pair_task_entries.end(), key);
            if (it == state.pair_task_entries.end() ||
                it->row_piece != row_piece ||
                it->partner_piece != partner_piece)
                return NULL;
            int tid = it->task_id;
            if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                return NULL;
            return &state.tasks[static_cast<size_t>(tid)];
        };
        auto row_piece_for_block = [&](int_t rb) -> SymV2PcFragPieceDesc * {
            if (rb < 0 || static_cast<size_t>(rb) >=
                              state.row_block_piece.size())
                return NULL;
            int piece_id = state.row_block_piece[static_cast<size_t>(rb)];
            if (piece_id < 0 ||
                static_cast<size_t>(piece_id) >= state.row_pieces.size())
                return NULL;
            return &state.row_pieces[static_cast<size_t>(piece_id)];
        };
        auto partner_piece_for_block =
            [&](int_t cb) -> SymV2PcFragPieceDesc * {
            if (cb < 0 || static_cast<size_t>(cb) >=
                              state.partner_block_piece.size())
                return NULL;
            int piece_id = state.partner_block_piece[static_cast<size_t>(cb)];
            if (piece_id < 0 || static_cast<size_t>(piece_id) >=
                                    state.partner_pieces.size())
                return NULL;
            return &state.partner_pieces[static_cast<size_t>(piece_id)];
        };
        auto build_group_index =
            [&](const std::vector<SymV2PcFragPieceDesc> &pieces,
                int_t begin, int_t end) -> std::vector<int_t> {
            int_t nblocks = end - begin;
            int_t nrows = 0;
            for (int_t b = begin; b < end; ++b)
                nrows += pieces[static_cast<size_t>(b)].nrows;
            std::vector<int_t> idx(
                static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                                    nrows), 0);
            idx[0] = nblocks;
            idx[1] = nrows;
            idx[2] = 0;
            idx[3] = SuperSize(k);
            int_t gid_ptr = LPANEL_HEADER_SIZE;
            int_t px_ptr = LPANEL_HEADER_SIZE + nblocks;
            int_t row_ptr = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
            idx[px_ptr] = 0;
            for (int_t local = 0; local < nblocks; ++local)
            {
                const SymV2PcFragPieceDesc &piece =
                    pieces[static_cast<size_t>(begin + local)];
                idx[gid_ptr + local] = piece.gid_first;
                idx[px_ptr + local + 1] =
                    idx[px_ptr + local] + piece.nrows;
                size_t piece_row_ptr =
                    static_cast<size_t>(LPANEL_HEADER_SIZE + 3);
                for (int_t r = 0; r < piece.nrows; ++r)
                    idx[row_ptr++] =
                        piece.h_index[piece_row_ptr + static_cast<size_t>(r)];
            }
            return idx;
        };
        auto build_group_values =
            [&](const std::vector<SymV2PcFragPieceDesc> &pieces,
                int_t begin, int_t end, int_t group_lda,
                double *group_val, cudaStream_t group_stream) {
            int_t ksupc = SuperSize(k);
            size_t value_count =
                static_cast<size_t>(group_lda) * static_cast<size_t>(ksupc);
            if (value_count == 0)
                return;
            if (group_val == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped value pool is missing.");
            int_t dst = 0;
            for (int_t b = begin; b < end; ++b)
            {
                const SymV2PcFragPieceDesc &piece =
                    pieces[static_cast<size_t>(b)];
                if (!piece.ready || piece.d_val == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped dispatch found an unready piece.");
                if (piece.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(
                        group_stream, piece.ready_event, 0));
                gpuErrchk(cudaMemcpy2DAsync(
                    group_val + dst,
                    sizeof(double) * static_cast<size_t>(group_lda),
                    piece.d_val,
                    sizeof(double) * static_cast<size_t>(piece.lda),
                    sizeof(double) * static_cast<size_t>(piece.nrows),
                    static_cast<size_t>(ksupc),
                    cudaMemcpyDeviceToDevice, group_stream));
                dst += piece.nrows;
            }
        };
        auto build_group_index_from_ids =
            [&](const std::vector<SymV2PcFragPieceDesc> &pieces,
                const std::vector<int> &piece_ids) -> std::vector<int_t> {
            int_t nblocks = static_cast<int_t>(piece_ids.size());
            int_t nrows = 0;
            for (size_t p = 0; p < piece_ids.size(); ++p)
            {
                int piece_id = piece_ids[p];
                if (piece_id < 0 ||
                    static_cast<size_t>(piece_id) >= pieces.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped explicit piece id is invalid.");
                nrows += pieces[static_cast<size_t>(piece_id)].nrows;
            }
            std::vector<int_t> idx(
                static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                                    nrows), 0);
            idx[0] = nblocks;
            idx[1] = nrows;
            idx[2] = 0;
            idx[3] = SuperSize(k);
            int_t gid_ptr = LPANEL_HEADER_SIZE;
            int_t px_ptr = LPANEL_HEADER_SIZE + nblocks;
            int_t row_ptr = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
            idx[px_ptr] = 0;
            for (int_t local = 0; local < nblocks; ++local)
            {
                const SymV2PcFragPieceDesc &piece =
                    pieces[static_cast<size_t>(
                        piece_ids[static_cast<size_t>(local)])];
                idx[gid_ptr + local] = piece.gid_first;
                idx[px_ptr + local + 1] =
                    idx[px_ptr + local] + piece.nrows;
                size_t piece_row_ptr =
                    static_cast<size_t>(LPANEL_HEADER_SIZE + 3);
                for (int_t r = 0; r < piece.nrows; ++r)
                    idx[row_ptr++] =
                        piece.h_index[piece_row_ptr + static_cast<size_t>(r)];
            }
            return idx;
        };
        auto build_group_values_from_ids =
            [&](const std::vector<SymV2PcFragPieceDesc> &pieces,
                const std::vector<int> &piece_ids, int_t group_lda,
                double *group_val, cudaStream_t group_stream) {
            int_t ksupc = SuperSize(k);
            if (group_lda <= 0)
                return;
            if (group_val == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped explicit value pool is missing.");
            int_t dst = 0;
            for (size_t p = 0; p < piece_ids.size(); ++p)
            {
                const SymV2PcFragPieceDesc &piece =
                    pieces[static_cast<size_t>(piece_ids[p])];
                if (!piece.ready || piece.d_val == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped explicit dispatch found an unready piece.");
                if (piece.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(
                        group_stream, piece.ready_event, 0));
                gpuErrchk(cudaMemcpy2DAsync(
                    group_val + dst,
                    sizeof(double) * static_cast<size_t>(group_lda),
                    piece.d_val,
                    sizeof(double) * static_cast<size_t>(piece.lda),
                    sizeof(double) * static_cast<size_t>(piece.nrows),
                    static_cast<size_t>(ksupc),
                    cudaMemcpyDeviceToDevice, group_stream));
                dst += piece.nrows;
            }
        };
        auto launch_single_task =
            [&](SymV2PcFragTaskDesc &task,
                cublasHandle_t task_handle,
                cudaStream_t task_stream,
                double *task_gemm,
                unsigned launch_mode) -> int {
            if (task.complete ||
                task.launched ||
                !task_matches_launch_mode(task, launch_mode))
                return 0;
            SymV2PcFragPieceDesc &row =
                state.row_pieces[static_cast<size_t>(task.row_piece)];
            SymV2PcFragPieceDesc &col =
                state.partner_pieces[static_cast<size_t>(task.partner_piece)];
            if (!row.ready)
            {
                ++symV2PcFragTaskflowStats.tasks_blocked_row;
                return 0;
            }
            if (!col.ready)
            {
                ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                return 0;
            }
            if (output_locked(task))
            {
                ++symV2PcFragTaskflowStats.tasks_blocked_output;
                ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                return 0;
            }
            if (row.d_index == NULL || row.d_val == NULL ||
                col.d_index == NULL || col.d_val == NULL ||
                task_handle == NULL || task_stream == NULL ||
                task_gemm == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW single task is missing owned resources.");
            if (!all_pieces_ready())
                ++symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready;
            lock_outputs(task);
            if (row.ready_event != NULL)
                gpuErrchk(cudaStreamWaitEvent(
                    task_stream, row.ready_event, 0));
            if (col.ready_event != NULL)
                gpuErrchk(cudaStreamWaitEvent(
                    task_stream, col.ready_event, 0));
            dSymV2PcFragTaskflowWaitOnGemmResource(
                *this, state, task, streamId, task_stream, launch_mode);
            dSymSchurCompUpdateTaskDualPiecesGPU(
                k,
                row.h_index, col.h_index,
                row.d_index, row.d_val,
                col.d_index, col.d_val,
                task_handle, task_stream, task_gemm);
            task.launched = 1;
            task.launch_stream_kind =
                task_stream_kind_for_launch_mode(launch_mode);
            if (async_core)
            {
                if (task.done_event == NULL)
                    task.done_event =
                        dSymV2PcFragTaskflowAcquireEvent(
                            symV2PcFragTaskflowEventPool,
                            !async_core,
                            &symV2PcFragTaskflowStats.arena_event_late_allocs);
                gpuErrchk(cudaEventRecord(task.done_event, task_stream));
                dSymV2PcFragTaskflowPublishGemmResourceTail(
                    *this, state, task, streamId, task_stream);
                dSymV2PcFragTaskflowRecordLaunchedTask(state, task);
                if (dSymV2PcFragTaskflowForceTaskSync())
                {
                    ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
                    gpuErrchk(cudaStreamSynchronize(task_stream));
                    dSymV2PcFragTaskflowUnrecordLaunchedTask(state, task);
                    dSymV2PcFragTaskflowCompleteLaunchedTask(
                        *this, state, task, strict_output_conflicts);
                    dSymV2PcFragTaskflowRecycleEvent(
                        symV2PcFragTaskflowEventPool, task.done_event);
                }
            }
            else
            {
                ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
                gpuErrchk(cudaStreamSynchronize(task_stream));
                symV2PcFragTaskflowStats.output_locks_released_by_launch_sync +=
                    unlock_outputs(task);
                dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(
                    state, task);
                task.complete = 1;
                --state.incomplete_task_count;
                --row.pending_consumers;
                --col.pending_consumers;
                if (row.pending_consumers < 0 || col.pending_consumers < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
                ++symV2PcFragTaskflowStats.tasks_completed;
            }
            ++symV2PcFragTaskflowStats.tasks_launched;
            count_task_launch_mode(launch_mode);
            return 1;
        };
        auto piece_id_range_for_blocks =
            [](const std::vector<int> &block_piece,
               int_t block_start, int_t block_end,
               int *piece_start, int *piece_end) -> bool {
            if (block_start >= block_end)
                return false;
            if (block_start < 0 || block_end < 0 ||
                static_cast<size_t>(block_start) >= block_piece.size() ||
                static_cast<size_t>(block_end - 1) >= block_piece.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW block range is outside the block-to-piece map.");
            int first = block_piece[static_cast<size_t>(block_start)];
            int last = block_piece[static_cast<size_t>(block_end - 1)];
            if (first < 0 || last < first)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW block-to-piece range is invalid.");
            *piece_start = first;
            *piece_end = last + 1;
            return true;
        };
        auto task_for_piece_pair =
            [&](int row_piece, int partner_piece) -> SymV2PcFragTaskDesc * {
            if (row_piece < 0 || partner_piece < 0 ||
                static_cast<size_t>(row_piece) >= state.row_pieces.size() ||
                static_cast<size_t>(partner_piece) >=
                    state.partner_pieces.size() ||
                state.partner_pieces.empty())
                return NULL;
            SymV2PcFragPairTaskEntry key(row_piece, partner_piece, -1);
            std::vector<SymV2PcFragPairTaskEntry>::const_iterator it =
                std::lower_bound(state.pair_task_entries.begin(),
                                 state.pair_task_entries.end(), key);
            if (it == state.pair_task_entries.end() ||
                it->row_piece != row_piece ||
                it->partner_piece != partner_piece)
                return NULL;
            int tid = it->task_id;
            if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                return NULL;
            return &state.tasks[static_cast<size_t>(tid)];
        };
        auto launch_group =
            [&](int_t row_start, int_t row_end,
                int_t col_start, int_t col_end,
                cublasHandle_t group_handle,
                cudaStream_t group_stream,
                double *group_gemm,
                unsigned launch_mode) {
            if (row_start >= row_end || col_start >= col_end)
                return;
            if (async_core)
            {
                auto launch_range_as_singles = [&]() {
                    int row_piece_start = 0;
                    int row_piece_end = 0;
                    int partner_piece_start = 0;
                    int partner_piece_end = 0;
                    if (!piece_id_range_for_blocks(
                            state.row_block_piece, row_start, row_end,
                            &row_piece_start, &row_piece_end) ||
                        !piece_id_range_for_blocks(
                            state.partner_block_piece, col_start, col_end,
                            &partner_piece_start, &partner_piece_end))
                        return;
                    if (static_cast<size_t>(row_piece_end) >
                            state.row_pieces.size() ||
                        static_cast<size_t>(partner_piece_end) >
                            state.partner_pieces.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW piece range is outside the taskflow piece arrays.");
                    for (int rp = row_piece_start; rp < row_piece_end; ++rp)
                    {
                        SymV2PcFragPieceDesc &row_piece =
                            state.row_pieces[static_cast<size_t>(rp)];
                        if (!row_piece.ready)
                        {
                            ++symV2PcFragTaskflowStats.tasks_blocked_row;
                            return;
                        }
                        for (int cp = partner_piece_start;
                             cp < partner_piece_end; ++cp)
                        {
                            SymV2PcFragPieceDesc &partner_piece =
                                state.partner_pieces[static_cast<size_t>(cp)];
                            if (!partner_piece.ready)
                            {
                                ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                                return;
                            }
                            SymV2PcFragTaskDesc *task =
                                task_for_piece_pair(rp, cp);
                            if (task == NULL || task->complete ||
                                !task_matches_launch_mode(*task, launch_mode))
                                continue;
                            launch_single_task(
                                *task, group_handle, group_stream,
                                group_gemm, launch_mode);
                        }
                    }
                };
                int row_piece_start = 0;
                int row_piece_end = 0;
                int partner_piece_start = 0;
                int partner_piece_end = 0;
                if (!piece_id_range_for_blocks(
                        state.row_block_piece, row_start, row_end,
                        &row_piece_start, &row_piece_end) ||
                    !piece_id_range_for_blocks(
                        state.partner_block_piece, col_start, col_end,
                        &partner_piece_start, &partner_piece_end))
                    return;
                if (static_cast<size_t>(row_piece_end) >
                        state.row_pieces.size() ||
                    static_cast<size_t>(partner_piece_end) >
                        state.partner_pieces.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW piece range is outside the taskflow piece arrays.");
                const bool lookahead_group =
                    (launch_mode & (SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                                    SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)) != 0;
                if (!async_grouped_dispatch || !lookahead_group ||
                    (!async_core && state.group_scratch_in_use))
                {
                    if (async_grouped_dispatch && lookahead_group &&
                        state.group_scratch_in_use)
                        ++symV2PcFragTaskflowStats.grouped_scratch_busy_deferrals;
                    launch_range_as_singles();
                    return;
                }
                std::vector<SymV2PcFragTaskDesc *> group_tasks;
                int completed_pair_count = 0;
                for (int rp = row_piece_start; rp < row_piece_end; ++rp)
                {
                    SymV2PcFragPieceDesc &row_piece =
                        state.row_pieces[static_cast<size_t>(rp)];
                    if (!row_piece.ready)
                    {
                        ++symV2PcFragTaskflowStats.tasks_blocked_row;
                        return;
                    }
                    for (int cp = partner_piece_start;
                         cp < partner_piece_end; ++cp)
                    {
                        SymV2PcFragPieceDesc &partner_piece =
                            state.partner_pieces[static_cast<size_t>(cp)];
                        if (!partner_piece.ready)
                        {
                            ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                            return;
                        }
                        SymV2PcFragTaskDesc *task =
                            task_for_piece_pair(rp, cp);
                        if (task == NULL ||
                            !task_matches_launch_mode(*task, launch_mode))
                            continue;
                        ++symV2PcFragTaskflowStats.grouped_candidate_scans;
                        if (task->complete)
                        {
                            ++completed_pair_count;
                            continue;
                        }
                        if (task->launched)
                            continue;
                        if (output_locked(*task))
                        {
                            ++symV2PcFragTaskflowStats.tasks_blocked_output;
                            ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                            ++symV2PcFragTaskflowStats.grouped_output_conflict_fallbacks;
                            launch_range_as_singles();
                            return;
                        }
                        group_tasks.push_back(task);
                    }
                }
                ++symV2PcFragTaskflowStats.grouped_dispatch_attempts;
                if (group_tasks.size() <= 1 || completed_pair_count > 0)
                {
                    if (group_tasks.size() <= 1)
                        ++symV2PcFragTaskflowStats.grouped_single_fallbacks;
                    if (completed_pair_count > 0)
                        ++symV2PcFragTaskflowStats.grouped_completed_pair_fallbacks;
                    launch_range_as_singles();
                    return;
                }
                const int in_flight_task_cap =
                    superlu_sym_v2_pcfrag_taskflow_effective_progress_budget();
                if (static_cast<int>(group_tasks.size()) >
                    in_flight_task_cap)
                {
                    ++symV2PcFragTaskflowStats.grouped_capacity_fallbacks;
                    launch_range_as_singles();
                    return;
                }
                int pending_launched =
                    dSymV2PcFragTaskflowPendingLaunchedAllForMode(
                        state, 0, 0, GLOBAL_BLOCK_NOT_FOUND);
                if (pending_launched +
                        static_cast<int>(group_tasks.size()) >
                    in_flight_task_cap)
                {
                    ++symV2PcFragTaskflowStats.grouped_pending_cap_deferrals;
                    return;
                }
                if (group_handle == NULL || group_stream == NULL ||
                    group_gemm == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped async task has no stream resources.");
                int_t row_lda = 0;
                for (int rp = row_piece_start; rp < row_piece_end; ++rp)
                    row_lda += state.row_pieces[static_cast<size_t>(rp)].nrows;
                int_t col_lda = 0;
                for (int cp = partner_piece_start; cp < partner_piece_end; ++cp)
                    col_lda += state.partner_pieces[static_cast<size_t>(cp)].nrows;
                std::vector<int_t> row_group =
                    build_group_index(
                        state.row_pieces, row_piece_start, row_piece_end);
                std::vector<int_t> col_group =
                    build_group_index(
                        state.partner_pieces, partner_piece_start,
                        partner_piece_end);
                size_t row_index_count = row_group.size();
                size_t col_index_count = col_group.size();
                size_t row_value_count =
                    static_cast<size_t>(row_lda) *
                    static_cast<size_t>(SuperSize(k));
                size_t col_value_count =
                    static_cast<size_t>(col_lda) *
                    static_cast<size_t>(SuperSize(k));
                if (async_core &&
                    (!dSymV2PcFragTaskflowPoolHasBlock(
                         symV2PcFragTaskflowGroupIndexBlockPool,
                         row_index_count + col_index_count) ||
                     !dSymV2PcFragTaskflowPoolHasBlock(
                         symV2PcFragTaskflowGroupValueBlockPool,
                         row_value_count + col_value_count)))
                {
                    ++symV2PcFragTaskflowStats.grouped_scratch_busy_deferrals;
                    launch_range_as_singles();
                    return;
                }
                size_t group_index_capacity = 0;
                size_t group_value_capacity = 0;
                int_t *group_index_pool = state.d_group_index_pool;
                double *group_value_pool = state.d_group_value_pool;
                if (async_core)
                {
                    group_index_pool =
                        dSymV2PcFragTaskflowAcquireIndexBlock(
                            symV2PcFragTaskflowGroupIndexBlockPool,
                            row_index_count + col_index_count,
                            &group_index_capacity,
                            allow_taskflow_late_alloc,
                            &symV2PcFragTaskflowStats
                                 .arena_index_late_allocs);
                    group_value_pool =
                        dSymV2PcFragTaskflowAcquireValueBlock(
                            symV2PcFragTaskflowGroupValueBlockPool,
                            row_value_count + col_value_count,
                            &group_value_capacity,
                            allow_taskflow_late_alloc,
                            &symV2PcFragTaskflowStats
                                 .arena_value_late_allocs);
                }
                else
                {
                    group_index_capacity =
                        state.group_index_pool_capacity;
                    group_value_capacity =
                        state.group_value_pool_capacity;
                }
                if (group_index_pool == NULL ||
                    row_index_count + col_index_count >
                        group_index_capacity)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped async index arena exhausted.");
                if (group_value_pool == NULL ||
                    row_value_count + col_value_count >
                        group_value_capacity)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped async value arena exhausted.");
                for (size_t i = 0; i < group_tasks.size(); ++i)
                    lock_outputs(*group_tasks[i]);
                int_t *row_group_gpu = group_index_pool;
                int_t *col_group_gpu =
                    group_index_pool + row_index_count;
                double *row_group_val = group_value_pool;
                double *col_group_val =
                    group_value_pool + row_value_count;
                gpuErrchk(cudaMemcpyAsync(
                    row_group_gpu, row_group.data(),
                    sizeof(int_t) * row_index_count,
                    cudaMemcpyHostToDevice, group_stream));
                gpuErrchk(cudaMemcpyAsync(
                    col_group_gpu, col_group.data(),
                    sizeof(int_t) * col_index_count,
                    cudaMemcpyHostToDevice, group_stream));
                build_group_values(
                    state.row_pieces, row_piece_start, row_piece_end,
                    row_lda, row_group_val, group_stream);
                build_group_values(
                    state.partner_pieces, partner_piece_start,
                    partner_piece_end, col_lda, col_group_val,
                    group_stream);
                if (!all_pieces_ready())
                    symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready +=
                        static_cast<long long>(group_tasks.size());
                int resource =
                    dSymV2PcFragTaskflowGemmResourceForLaunchMode(
                        launch_mode);
                xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
                    dSymV2PcFragTaskflowGemmResource(
                        *this, streamId, resource);
                if (res.recorded)
                {
                    gpuErrchk(cudaStreamWaitEvent(
                        group_stream, res.tail_event, 0));
                    ++symV2PcFragTaskflowStats.gemm_resource_tail_waits;
                    ++res.waits;
                }
                dSymSchurCompUpdateTaskDualPieceGroupGPU(
                    0, row_piece_end - row_piece_start,
                    0, partner_piece_end - partner_piece_start, k,
                    row_group, col_group,
                    row_group_gpu, row_group_val,
                    col_group_gpu, col_group_val,
                    group_handle, group_stream, group_gemm);
                xLUstruct_t<double>::SymV2PcFragLaunchedTaskGroup group;
                if (async_core)
                {
                    group.d_group_index_pool = group_index_pool;
                    group.group_index_pool_capacity =
                        group_index_capacity;
                    group.d_group_value_pool = group_value_pool;
                    group.group_value_pool_capacity =
                        group_value_capacity;
                }
                group.group_id = static_cast<int>(
                    state.launched_group_task_ids.size());
                group.task_begin = static_cast<int>(
                    state.launched_group_task_ids.size());
                group.task_count = static_cast<int>(group_tasks.size());
                group.launch_stream_kind =
                    task_stream_kind_for_launch_mode(launch_mode);
                group.gemm_resource_kind =
                    static_cast<unsigned char>(resource);
                group.done_event =
                    dSymV2PcFragTaskflowAcquireEvent(
                        symV2PcFragTaskflowEventPool, !async_core,
                        &symV2PcFragTaskflowStats.arena_event_late_allocs);
                gpuErrchk(cudaEventRecord(group.done_event, group_stream));
                gpuErrchk(cudaEventRecord(res.tail_event, group_stream));
                res.recorded = 1;
                res.active_task_id = group_tasks[0]->task_id;
                ++res.updates;
                ++symV2PcFragTaskflowStats.gemm_resource_tail_updates;
                ++symV2PcFragTaskflowStats.grouped_launches;
                symV2PcFragTaskflowStats.grouped_task_members +=
                    static_cast<long long>(group_tasks.size());
                for (size_t i = 0; i < group_tasks.size(); ++i)
                {
                    SymV2PcFragTaskDesc &task = *group_tasks[i];
                    task.launched = 1;
                    task.launch_stream_kind = group.launch_stream_kind;
                    task.gemm_resource_kind = group.gemm_resource_kind;
                    state.launched_group_task_ids.push_back(task.task_id);
                    dSymV2PcFragTaskflowAdjustLaunchedTaskCounts(
                        state, task, 1);
                    ++symV2PcFragTaskflowStats.tasks_launched;
                    count_task_launch_mode(launch_mode);
                }
                state.launched_task_groups_by_stream[
                    group.launch_stream_kind].push_back(group);
                if (!async_core)
                    state.group_scratch_in_use = 1;
                return;
            }
            int pair_count = 0;
            int completed_pair_count = 0;
            for (int_t rb = row_start; rb < row_end; ++rb)
            {
                SymV2PcFragPieceDesc *row_piece =
                    row_piece_for_block(rb);
                if (row_piece == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row block has no owning piece.");
                if (!row_piece->ready)
                {
                    ++symV2PcFragTaskflowStats.tasks_blocked_row;
                    return;
                }
                for (int_t cb = col_start; cb < col_end; ++cb)
                {
                    SymV2PcFragPieceDesc *partner_piece =
                        partner_piece_for_block(cb);
                    if (partner_piece == NULL)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW partner block has no owning piece.");
                    if (!partner_piece->ready)
                    {
                        ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                        return;
                    }
                    SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                    if (task != NULL &&
                        task_matches_launch_mode(*task, launch_mode))
                    {
                        if (task->complete)
                            ++completed_pair_count;
                        else
                            ++pair_count;
                    }
                }
            }
            if (pair_count == 0)
                return;
            if (completed_pair_count > 0)
            {
                for (int_t rb = row_start; rb < row_end; ++rb)
                {
                    for (int_t cb = col_start; cb < col_end; ++cb)
                    {
                        SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                        if (task == NULL || task->complete ||
                            !task_matches_launch_mode(*task, launch_mode))
                            continue;
                        launch_single_task(
                            *task, group_handle, group_stream, group_gemm,
                            launch_mode);
                    }
                }
                return;
            }
            std::vector<SymV2PcFragTaskDesc *> locked_tasks;
            if (strict_output_conflicts)
            {
                std::set<SymV2PcFragOutputKey> pending_keys =
                    state.active_output_key_set;
                if (superlu_sym_v2_pcfrag_taskflow_global_output_locks())
                {
                    const std::set<SymV2PcFragOutputKey> &global_locks =
                        symV2PcFragTaskflowGlobalOutputLocks;
                    pending_keys.insert(global_locks.begin(),
                                        global_locks.end());
                }
                for (int_t rb = row_start; rb < row_end; ++rb)
                {
                    for (int_t cb = col_start; cb < col_end; ++cb)
                    {
                        SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                        if (task == NULL || task->complete ||
                            !task_matches_launch_mode(*task, launch_mode))
                            continue;
                        const size_t output_count =
                            dSymV2PcFragTaskflowOutputCount(*task);
                        for (size_t o = 0; o < output_count; ++o)
                        {
                            const SymV2PcFragOutputKey &key =
                                dSymV2PcFragTaskflowOutputAt(
                                    state, *task, o);
                            if (pending_keys.find(key) != pending_keys.end())
                            {
                                ++symV2PcFragTaskflowStats.tasks_blocked_output;
                                ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped dispatch found an output conflict.");
                            }
                            pending_keys.insert(key);
                        }
                        locked_tasks.push_back(task);
                    }
                }
                for (size_t i = 0; i < locked_tasks.size(); ++i)
                    lock_outputs(*locked_tasks[i]);
            }
            int_t row_lda = 0;
            for (int_t rb = row_start; rb < row_end; ++rb)
                row_lda += state.row_pieces[static_cast<size_t>(rb)].nrows;
            int_t col_lda = 0;
            for (int_t cb = col_start; cb < col_end; ++cb)
                col_lda += state.partner_pieces[static_cast<size_t>(cb)].nrows;
            if (group_handle == NULL || group_stream == NULL ||
                group_gemm == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped task has no stream resources.");
            std::vector<int_t> row_group =
                build_group_index(state.row_pieces, row_start, row_end);
            std::vector<int_t> col_group =
                build_group_index(state.partner_pieces, col_start, col_end);
            size_t row_index_count = row_group.size();
            size_t col_index_count = col_group.size();
            size_t row_value_count =
                static_cast<size_t>(row_lda) *
                static_cast<size_t>(SuperSize(k));
            size_t col_value_count =
                static_cast<size_t>(col_lda) *
                static_cast<size_t>(SuperSize(k));
            if (state.d_group_index_pool == NULL ||
                row_index_count + col_index_count >
                    state.group_index_pool_capacity)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped index arena exhausted.");
            if (state.d_group_value_pool == NULL ||
                row_value_count + col_value_count >
                    state.group_value_pool_capacity)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped value arena exhausted.");
            int_t *row_group_gpu = state.d_group_index_pool;
            int_t *col_group_gpu =
                state.d_group_index_pool + row_index_count;
            double *row_group_val = state.d_group_value_pool;
            double *col_group_val =
                state.d_group_value_pool + row_value_count;
            gpuErrchk(cudaMemcpyAsync(
                row_group_gpu, row_group.data(),
                sizeof(int_t) * row_index_count,
                cudaMemcpyHostToDevice, group_stream));
            gpuErrchk(cudaMemcpyAsync(
                col_group_gpu, col_group.data(),
                sizeof(int_t) * col_index_count,
                cudaMemcpyHostToDevice, group_stream));
            build_group_values(
                state.row_pieces, row_start, row_end, row_lda,
                row_group_val, group_stream);
            build_group_values(
                state.partner_pieces, col_start, col_end, col_lda,
                col_group_val, group_stream);
            if (!all_pieces_ready())
                symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready +=
                    pair_count;
            dSymSchurCompUpdateTaskDualPieceGroupGPU(
                0, row_end - row_start, 0, col_end - col_start, k,
                row_group, col_group,
                row_group_gpu, row_group_val,
                col_group_gpu, col_group_val,
                group_handle, group_stream, group_gemm);
            ++symV2PcFragTaskflowStats.task_launch_stream_syncs;
            gpuErrchk(cudaStreamSynchronize(group_stream));
            for (size_t i = 0; i < locked_tasks.size(); ++i)
                symV2PcFragTaskflowStats.output_locks_released_by_launch_sync +=
                    unlock_outputs(*locked_tasks[i]);
            for (int_t rb = row_start; rb < row_end; ++rb)
            {
                for (int_t cb = col_start; cb < col_end; ++cb)
                {
                    SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                    if (task == NULL || task->complete ||
                        !task_matches_launch_mode(*task, launch_mode))
                        continue;
                    task->launched = 1;
                    dSymV2PcFragTaskflowNoteTaskCompleteForModeCounters(
                        state, *task);
                    task->complete = 1;
                    --state.incomplete_task_count;
                    --state.row_pieces[static_cast<size_t>(rb)].pending_consumers;
                    --state.partner_pieces[static_cast<size_t>(cb)].pending_consumers;
                    if (state.row_pieces[static_cast<size_t>(rb)].pending_consumers < 0 ||
                        state.partner_pieces[static_cast<size_t>(cb)].pending_consumers < 0)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
                    ++symV2PcFragTaskflowStats.tasks_launched;
                    ++symV2PcFragTaskflowStats.tasks_completed;
                    count_task_launch_mode(launch_mode);
                }
            }
        };
        if (async_core &&
            !superlu_sym_v2_pcfrag_taskflow_validate())
        {
            int launched_this_call = 0;
            int pending_launched =
                dSymV2PcFragTaskflowPendingLaunchedAllForMode(
                    state, 0, 0, GLOBAL_BLOCK_NOT_FOUND);
            const int in_flight_task_cap =
                superlu_sym_v2_pcfrag_taskflow_effective_progress_budget();
            for (unsigned single_mode = 1; single_mode < 16;
                 single_mode <<= 1)
            {
                if ((mode_mask & single_mode) == 0)
                    continue;
                std::vector<int> *queue_ptr =
                    &state.runnable_task_ids_by_mode[single_mode];
                if (mode_gid != GLOBAL_BLOCK_NOT_FOUND &&
                    single_mode == SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
                {
                    auto it =
                        state.runnable_lookahead_col_by_gid.find(mode_gid);
                    if (it == state.runnable_lookahead_col_by_gid.end())
                        continue;
                    queue_ptr = &it->second;
                }
                else if (mode_gid != GLOBAL_BLOCK_NOT_FOUND &&
                         single_mode == SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                {
                    auto it =
                        state.runnable_lookahead_row_by_gid.find(mode_gid);
                    if (it == state.runnable_lookahead_row_by_gid.end())
                        continue;
                    queue_ptr = &it->second;
                }
                std::vector<int> &queue = *queue_ptr;
                size_t runnable_write = 0;
                for (size_t i = 0; i < queue.size(); ++i)
                {
                    int tid = queue[i];
                    if (tid < 0 ||
                        static_cast<size_t>(tid) >= state.tasks.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable task id is invalid.");
                    SymV2PcFragTaskDesc &task =
                        state.tasks[static_cast<size_t>(tid)];
                    if (task.launched || task.complete)
                        continue;
                    if (pending_launched >= in_flight_task_cap)
                    {
                        queue[runnable_write++] = tid;
                        continue;
                    }
                    unsigned launch_mode =
                        task_launch_mode_for_request(
                            task, single_mode, mode_gid);
                    if (launch_mode == 0 ||
                        !dSymV2PcFragTaskflowTaskRequiredForMode(
                            state, task, 1, single_mode, mode_gid))
                    {
                        queue[runnable_write++] = tid;
                        continue;
                    }
                    cublasHandle_t task_handle = handle;
                    cudaStream_t task_stream = stream;
                    double *task_gemm = gemmBuff;
                    if (launch_mode & SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
                    {
                        task_handle = A_gpu.lookAheadLHandle[streamId];
                        task_stream = A_gpu.lookAheadLStream[streamId];
                        task_gemm = A_gpu.lookAheadLGemmBuffer[streamId];
                    }
                    else if (launch_mode & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                    {
                        task_handle = A_gpu.lookAheadUHandle[streamId];
                        task_stream = A_gpu.lookAheadUStream[streamId];
                        task_gemm = A_gpu.gpuGemmBuffs[streamId];
                    }
                    if (async_grouped_dispatch &&
                        !async_core &&
                        state.group_scratch_in_use &&
                        superlu_sym_v2_pcfrag_taskflow_group_defer_busy())
                    {
                        progress_launched_tasks(0, 0);
                    }
                    if (async_grouped_dispatch &&
                        !async_core &&
                        state.group_scratch_in_use &&
                        superlu_sym_v2_pcfrag_taskflow_group_defer_busy())
                    {
                        ++symV2PcFragTaskflowStats.grouped_scratch_busy_deferrals;
                        queue[runnable_write++] = tid;
                        continue;
                    }
                    if (async_grouped_dispatch &&
                        (async_core || !state.group_scratch_in_use))
                    {
                        int available_group_slots =
                            superlu_sym_v2_pcfrag_taskflow_group_budget();
                        if (available_group_slots <= 0)
                            available_group_slots =
                                in_flight_task_cap - pending_launched;
                        if (available_group_slots > 1)
                        {
                            auto task_ready_for_group =
                                [&](SymV2PcFragTaskDesc &candidate) {
                                    if (candidate.launched ||
                                        candidate.complete)
                                        return false;
                                    if (task_launch_mode_for_request(
                                            candidate, single_mode,
                                            mode_gid) != launch_mode ||
                                        !dSymV2PcFragTaskflowTaskRequiredForMode(
                                            state, candidate, 1, single_mode,
                                            mode_gid))
                                        return false;
                                    if (candidate.task_id < 0 ||
                                        static_cast<size_t>(
                                            candidate.task_id) >=
                                            state.task_ready_inputs.size() ||
                                        static_cast<size_t>(
                                            candidate.task_id) >=
                                            state.task_enqueued.size())
                                        ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped task id is invalid.");
                                    if (state.task_ready_inputs
                                            [static_cast<size_t>(
                                                candidate.task_id)] < 2 ||
                                        !state.task_enqueued
                                            [static_cast<size_t>(
                                                candidate.task_id)])
                                        return false;
                                    return true;
                                };
                            auto collect_candidate_tids =
                                [&](bool group_by_partner) {
                                    std::vector<int> candidates;
                                    candidates.reserve(static_cast<size_t>(
                                        available_group_slots));
                                    candidates.push_back(tid);
                                    for (size_t j = i + 1;
                                         j < queue.size() &&
                                         static_cast<int>(candidates.size()) <
                                             available_group_slots;
                                         ++j)
                                    {
                                        ++symV2PcFragTaskflowStats.grouped_candidate_scans;
                                        int cand_tid = queue[j];
                                        if (cand_tid < 0 ||
                                            static_cast<size_t>(cand_tid) >=
                                                state.tasks.size())
                                            ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable task id is invalid.");
                                        SymV2PcFragTaskDesc &candidate =
                                            state.tasks[static_cast<size_t>(
                                                cand_tid)];
                                        if (!task_ready_for_group(candidate))
                                            continue;
                                        if (group_by_partner)
                                        {
                                            if (candidate.partner_piece !=
                                                task.partner_piece)
                                                continue;
                                            bool seen_row_piece = false;
                                            for (size_t ci = 0;
                                                 ci < candidates.size();
                                                 ++ci)
                                            {
                                                const SymV2PcFragTaskDesc
                                                    &seen = state.tasks
                                                        [static_cast<size_t>(
                                                            candidates[ci])];
                                                if (seen.row_piece ==
                                                    candidate.row_piece)
                                                {
                                                    seen_row_piece = true;
                                                    break;
                                                }
                                            }
                                            if (seen_row_piece)
                                                continue;
                                        }
                                        else
                                        {
                                            if (candidate.row_piece !=
                                                task.row_piece)
                                                continue;
                                            bool seen_partner_piece = false;
                                            for (size_t ci = 0;
                                                 ci < candidates.size();
                                                 ++ci)
                                            {
                                                const SymV2PcFragTaskDesc
                                                    &seen = state.tasks
                                                        [static_cast<size_t>(
                                                            candidates[ci])];
                                                if (seen.partner_piece ==
                                                    candidate.partner_piece)
                                                {
                                                    seen_partner_piece = true;
                                                    break;
                                                }
                                            }
                                            if (seen_partner_piece)
                                                continue;
                                        }
                                        candidates.push_back(cand_tid);
                                    }
                                    return candidates;
                                };
                            bool group_by_partner =
                                single_mode !=
                                SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
                            std::vector<int> candidate_tids;
                            if (single_mode ==
                                SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
                            {
                                group_by_partner = true;
                                candidate_tids =
                                    collect_candidate_tids(true);
                            }
                            else if (single_mode ==
                                     SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                            {
                                group_by_partner = false;
                                candidate_tids =
                                    collect_candidate_tids(false);
                            }
                            else
                            {
                                std::vector<int> by_partner =
                                    collect_candidate_tids(true);
                                std::vector<int> by_row =
                                    collect_candidate_tids(false);
                                if (by_row.size() > by_partner.size())
                                {
                                    group_by_partner = false;
                                    candidate_tids.swap(by_row);
                                }
                                else
                                {
                                    group_by_partner = true;
                                    candidate_tids.swap(by_partner);
                                }
                            }
                            if (candidate_tids.size() > 1)
                            {
                                ++symV2PcFragTaskflowStats.grouped_dispatch_attempts;
                                bool can_group = true;
                                if (output_locked(task))
                                {
                                    ++symV2PcFragTaskflowStats.tasks_blocked_output;
                                    ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                                    ++symV2PcFragTaskflowStats.grouped_output_conflict_fallbacks;
                                    can_group = false;
                                }
                                else
                                {
                                    std::vector<int> unlocked_candidate_tids;
                                    unlocked_candidate_tids.reserve(
                                        candidate_tids.size());
                                    unlocked_candidate_tids.push_back(tid);
                                    for (size_t ci = 1;
                                         ci < candidate_tids.size(); ++ci)
                                    {
                                        SymV2PcFragTaskDesc &candidate =
                                            state.tasks[static_cast<size_t>(
                                                candidate_tids[ci])];
                                        if (output_locked(candidate))
                                        {
                                            ++symV2PcFragTaskflowStats.tasks_blocked_output;
                                            ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                                            ++symV2PcFragTaskflowStats.grouped_output_conflict_fallbacks;
                                            continue;
                                        }
                                        unlocked_candidate_tids.push_back(
                                            candidate_tids[ci]);
                                    }
                                    candidate_tids.swap(unlocked_candidate_tids);
                                    if (candidate_tids.size() <= 1)
                                    {
                                        ++symV2PcFragTaskflowStats.grouped_single_fallbacks;
                                        can_group = false;
                                    }
                                }
                                std::vector<int> row_piece_ids;
                                std::vector<int> partner_piece_ids;
                                if (group_by_partner)
                                {
                                    partner_piece_ids.push_back(
                                        task.partner_piece);
                                    for (size_t ci = 0;
                                         ci < candidate_tids.size(); ++ci)
                                    {
                                        SymV2PcFragTaskDesc &candidate =
                                            state.tasks[static_cast<size_t>(
                                                candidate_tids[ci])];
                                        row_piece_ids.push_back(
                                            candidate.row_piece);
                                    }
                                }
                                else
                                {
                                    row_piece_ids.push_back(task.row_piece);
                                    for (size_t ci = 0;
                                         ci < candidate_tids.size(); ++ci)
                                    {
                                        SymV2PcFragTaskDesc &candidate =
                                            state.tasks[static_cast<size_t>(
                                                candidate_tids[ci])];
                                        partner_piece_ids.push_back(
                                            candidate.partner_piece);
                                    }
                                }
                                if (can_group)
                                {
                                    int_t row_lda = 0;
                                    for (size_t ri = 0;
                                         ri < row_piece_ids.size(); ++ri)
                                        row_lda += state.row_pieces[
                                            static_cast<size_t>(
                                                row_piece_ids[ri])].nrows;
                                    int_t col_lda = 0;
                                    for (size_t ci = 0;
                                         ci < partner_piece_ids.size(); ++ci)
                                        col_lda += state.partner_pieces[
                                            static_cast<size_t>(
                                                partner_piece_ids[ci])].nrows;
                                    std::vector<int_t> row_group =
                                        build_group_index_from_ids(
                                            state.row_pieces, row_piece_ids);
                                    std::vector<int_t> col_group =
                                        build_group_index_from_ids(
                                            state.partner_pieces,
                                            partner_piece_ids);
                                    size_t row_index_count =
                                        row_group.size();
                                    size_t col_index_count =
                                        col_group.size();
                                    size_t row_value_count =
                                        static_cast<size_t>(row_lda) *
                                        static_cast<size_t>(SuperSize(k));
                                    size_t col_value_count =
                                        static_cast<size_t>(col_lda) *
                                        static_cast<size_t>(SuperSize(k));
                                    if (async_core &&
                                        (!dSymV2PcFragTaskflowPoolHasBlock(
                                             symV2PcFragTaskflowGroupIndexBlockPool,
                                             row_index_count +
                                                 col_index_count) ||
                                         !dSymV2PcFragTaskflowPoolHasBlock(
                                             symV2PcFragTaskflowGroupValueBlockPool,
                                             row_value_count +
                                                 col_value_count)))
                                    {
                                        ++symV2PcFragTaskflowStats.grouped_scratch_busy_deferrals;
                                    }
                                    else
                                    {
                                    size_t group_index_capacity = 0;
                                    size_t group_value_capacity = 0;
                                    int_t *group_index_pool =
                                        state.d_group_index_pool;
                                    double *group_value_pool =
                                        state.d_group_value_pool;
                                    if (async_core)
                                    {
                                        group_index_pool =
                                            dSymV2PcFragTaskflowAcquireIndexBlock(
                                                symV2PcFragTaskflowGroupIndexBlockPool,
                                                row_index_count +
                                                    col_index_count,
                                                &group_index_capacity,
                                                allow_taskflow_late_alloc,
                                                &symV2PcFragTaskflowStats
                                                     .arena_index_late_allocs);
                                        group_value_pool =
                                            dSymV2PcFragTaskflowAcquireValueBlock(
                                                symV2PcFragTaskflowGroupValueBlockPool,
                                                row_value_count +
                                                    col_value_count,
                                                &group_value_capacity,
                                                allow_taskflow_late_alloc,
                                                &symV2PcFragTaskflowStats
                                                     .arena_value_late_allocs);
                                    }
                                    else
                                    {
                                        group_index_capacity =
                                            state.group_index_pool_capacity;
                                        group_value_capacity =
                                            state.group_value_pool_capacity;
                                    }
                                    if (group_index_pool == NULL ||
                                        row_index_count + col_index_count >
                                            group_index_capacity)
                                        ABORT("GPU3DV2_PCFRAG_TASKFLOW explicit grouped async index arena exhausted.");
                                    if (group_value_pool == NULL ||
                                        row_value_count + col_value_count >
                                            group_value_capacity)
                                        ABORT("GPU3DV2_PCFRAG_TASKFLOW explicit grouped async value arena exhausted.");
                                    int_t *row_group_gpu = group_index_pool;
                                    int_t *col_group_gpu =
                                        group_index_pool +
                                        row_index_count;
                                    double *row_group_val =
                                        group_value_pool;
                                    double *col_group_val =
                                        group_value_pool + row_value_count;
                                    for (size_t ci = 0;
                                         ci < candidate_tids.size(); ++ci)
                                        lock_outputs(
                                            state.tasks[static_cast<size_t>(
                                                candidate_tids[ci])]);
                                    gpuErrchk(cudaMemcpyAsync(
                                        row_group_gpu, row_group.data(),
                                        sizeof(int_t) * row_index_count,
                                        cudaMemcpyHostToDevice, task_stream));
                                    gpuErrchk(cudaMemcpyAsync(
                                        col_group_gpu, col_group.data(),
                                        sizeof(int_t) * col_index_count,
                                        cudaMemcpyHostToDevice, task_stream));
                                    build_group_values_from_ids(
                                        state.row_pieces, row_piece_ids,
                                        row_lda, row_group_val, task_stream);
                                    build_group_values_from_ids(
                                        state.partner_pieces,
                                        partner_piece_ids, col_lda,
                                        col_group_val, task_stream);
                                    if (!all_pieces_ready())
                                        symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready +=
                                            static_cast<long long>(
                                                candidate_tids.size());
                                    int resource =
                                        dSymV2PcFragTaskflowGemmResourceForLaunchMode(
                                            launch_mode);
                                    xLUstruct_t<double>::SymV2PcFragGemmResourceState &res =
                                        dSymV2PcFragTaskflowGemmResource(
                                            *this, streamId, resource);
                                    if (res.recorded)
                                    {
                                        gpuErrchk(cudaStreamWaitEvent(
                                            task_stream, res.tail_event, 0));
                                        ++symV2PcFragTaskflowStats.gemm_resource_tail_waits;
                                        ++res.waits;
                                    }
                                    dSymSchurCompUpdateTaskDualPieceGroupGPU(
                                        0,
                                        static_cast<int_t>(
                                            row_piece_ids.size()),
                                        0,
                                        static_cast<int_t>(
                                            partner_piece_ids.size()),
                                        k, row_group, col_group,
                                        row_group_gpu, row_group_val,
                                        col_group_gpu, col_group_val,
                                        task_handle, task_stream, task_gemm);
                                    xLUstruct_t<double>::SymV2PcFragLaunchedTaskGroup group;
                                    if (async_core)
                                    {
                                        group.d_group_index_pool =
                                            group_index_pool;
                                        group.group_index_pool_capacity =
                                            group_index_capacity;
                                        group.d_group_value_pool =
                                            group_value_pool;
                                        group.group_value_pool_capacity =
                                            group_value_capacity;
                                    }
                                    group.group_id = static_cast<int>(
                                        state.launched_group_task_ids.size());
                                    group.task_begin = static_cast<int>(
                                        state.launched_group_task_ids.size());
                                    group.task_count = static_cast<int>(
                                        candidate_tids.size());
                                    group.launch_stream_kind =
                                        task_stream_kind_for_launch_mode(
                                            launch_mode);
                                    group.gemm_resource_kind =
                                        static_cast<unsigned char>(resource);
                                    group.done_event =
                                        dSymV2PcFragTaskflowAcquireEvent(
                                            symV2PcFragTaskflowEventPool,
                                            !async_core,
                                            &symV2PcFragTaskflowStats
                                                 .arena_event_late_allocs);
                                    gpuErrchk(cudaEventRecord(
                                        group.done_event, task_stream));
                                    gpuErrchk(cudaEventRecord(
                                        res.tail_event, task_stream));
                                    res.recorded = 1;
                                    res.active_task_id = task.task_id;
                                    ++res.updates;
                                    ++symV2PcFragTaskflowStats.gemm_resource_tail_updates;
                                    ++symV2PcFragTaskflowStats.grouped_launches;
                                    symV2PcFragTaskflowStats.grouped_task_members +=
                                        static_cast<long long>(
                                            candidate_tids.size());
                                    for (size_t ci = 0;
                                         ci < candidate_tids.size(); ++ci)
                                    {
                                        SymV2PcFragTaskDesc &candidate =
                                            state.tasks[static_cast<size_t>(
                                                candidate_tids[ci])];
                                        candidate.launched = 1;
                                        candidate.launch_stream_kind =
                                            group.launch_stream_kind;
                                        candidate.gemm_resource_kind =
                                            group.gemm_resource_kind;
                                        state.launched_group_task_ids
                                            .push_back(candidate.task_id);
                                        dSymV2PcFragTaskflowAdjustLaunchedTaskCounts(
                                            state, candidate, 1);
                                        ++symV2PcFragTaskflowStats.tasks_launched;
                                        count_task_launch_mode(launch_mode);
                                    }
                                    state.launched_task_groups_by_stream[
                                        group.launch_stream_kind]
                                        .push_back(group);
                                    if (!async_core)
                                        state.group_scratch_in_use = 1;
                                    launched_this_call +=
                                        static_cast<int>(
                                            candidate_tids.size());
                                    pending_launched +=
                                        static_cast<int>(
                                            candidate_tids.size());
                                    continue;
                                    }
                                }
                            }
                        }
                    }
                    if (!launch_single_task(
                            task, task_handle, task_stream, task_gemm,
                            launch_mode))
                        queue[runnable_write++] = tid;
                    else
                    {
                        ++launched_this_call;
                        ++pending_launched;
                    }
                }
                if (runnable_write != queue.size())
                    queue.resize(runnable_write);
            }
            progress_launched_tasks(drain ? 1 : 0, mode_mask);
            if (state.incomplete_task_count == 0 &&
                !state.producer_exchange_active &&
                !state.producer_exchange_pending)
                release_completed_state();
            return launched_this_call;
        }
        auto dispatch_limited =
            [&](int_t row_start, int_t row_end,
                int_t col_start, int_t col_end,
                cublasHandle_t group_handle,
                cudaStream_t group_stream,
                double *group_gemm,
                unsigned launch_mode) {
            int_t nrow = frag_nblocks(row_frag);
            int_t ncol = frag_nblocks(col_frag);
            row_start = SUPERLU_MAX((int_t)0, row_start);
            col_start = SUPERLU_MAX((int_t)0, col_start);
            row_end = SUPERLU_MIN(row_end, nrow);
            col_end = SUPERLU_MIN(col_end, ncol);
            if (row_start >= row_end || col_start >= col_end)
                return;
            int max_block_rows = 1;
            for (int_t rb = row_start; rb < row_end; ++rb)
                max_block_rows = SUPERLU_MAX(
                    max_block_rows, static_cast<int>(frag_nbrow(row_frag, rb)));
            bool row_sorted = true;
            bool col_sorted = true;
            if (superlu_sym_v2_lower_envelope_enabled())
            {
                for (int_t rb = row_start + 1; rb < row_end; ++rb)
                    if (frag_gid(row_frag, rb) < frag_gid(row_frag, rb - 1))
                        row_sorted = false;
                for (int_t cb = col_start + 1; cb < col_end; ++cb)
                    if (frag_gid(col_frag, cb) < frag_gid(col_frag, cb - 1))
                        col_sorted = false;
            }
            const bool lower_envelope =
                superlu_sym_v2_lower_envelope_enabled() &&
                row_sorted && col_sorted;
            int64_t taskflow_gemm_capacity =
                static_cast<int64_t>(A_gpu.gemmBufferSize);
            const long long taskflow_gemm_cap =
                superlu_sym_v2_pcfrag_taskflow_gemm_cap();
            if (taskflow_gemm_cap > 0)
                taskflow_gemm_capacity =
                    SUPERLU_MIN(taskflow_gemm_capacity,
                                static_cast<int64_t>(taskflow_gemm_cap));
            const int64_t gemm_capacity = SUPERLU_MAX(
                static_cast<int64_t>(1), taskflow_gemm_capacity);
            int_t envelope_row_start = row_start;
            int_t jSt = col_start;
            while (jSt < col_end)
            {
                if (lower_envelope)
                {
                    int_t min_col_gid = frag_gid(col_frag, jSt);
                    while (envelope_row_start < row_end &&
                           frag_gid(row_frag, envelope_row_start) < min_col_gid)
                        ++envelope_row_start;
                    if (envelope_row_start >= row_end)
                        break;
                }
                else
                {
                    envelope_row_start = row_start;
                }
                int group_nrows =
                    frag_strow(row_frag, row_end) -
                    frag_strow(row_frag, envelope_row_start);
                if (group_nrows <= 0)
                    break;
                int col_limit = superlu_sym_v2_batch_schur_col_limit(
                    group_nrows, gemm_capacity);
                int64_t hard_col_limit =
                    gemm_capacity / static_cast<int64_t>(max_block_rows);
                if (hard_col_limit < 1)
                    hard_col_limit = 1;
                col_limit = SUPERLU_MAX(
                    1, SUPERLU_MIN(
                           col_limit,
                           static_cast<int>(SUPERLU_MIN(
                               hard_col_limit,
                               static_cast<int64_t>(2147483647L)))));
                int_t jNext = jSt + 1;
                while (jNext < col_end)
                {
                    int candidate_cols =
                        frag_strow(col_frag, jNext + 1) -
                        frag_strow(col_frag, jSt);
                    if (candidate_cols > col_limit)
                        break;
                    ++jNext;
                }
                int group_cols =
                    frag_strow(col_frag, jNext) -
                    frag_strow(col_frag, jSt);
                int max_gemm_rows = group_nrows;
                if (static_cast<int64_t>(group_nrows) * group_cols >
                    gemm_capacity)
                    max_gemm_rows = static_cast<int>(
                        gemm_capacity / static_cast<int64_t>(group_cols));
                max_gemm_rows = SUPERLU_MAX(max_gemm_rows, 1);
                int_t iEnd = envelope_row_start;
                while (iEnd < row_end)
                {
                    int_t iSt = iEnd;
                    iEnd = frag_get_end_block(row_frag, iSt, max_gemm_rows);
                    if (iEnd > row_end)
                        iEnd = row_end;
                    if (iEnd <= iSt)
                        iEnd = iSt + 1;
                    launch_group(iSt, iEnd, jSt, jNext,
                                 group_handle, group_stream, group_gemm,
                                 launch_mode);
                }
                jSt = jNext;
            }
        };

        int_t nrow = frag_nblocks(row_frag);
        int_t ncol = frag_nblocks(col_frag);
        int_t diag_row = frag_find(row_frag, k);
        int_t diag_col = frag_find(col_frag, k);
        if (mode_mask & (SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                         SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW))
        {
            ++symV2PcFragTaskflowStats.dispatch_calls_lookahead;
            int_t row_loc = frag_find(row_frag, mode_gid);
            int_t col_loc = frag_find(col_frag, mode_gid);
            int_t row_ranges[3][2];
            int n_row_ranges = ranges_excluding(
                0, nrow, diag_row, GLOBAL_BLOCK_NOT_FOUND, row_ranges);
            if (col_loc != GLOBAL_BLOCK_NOT_FOUND && col_loc != diag_col)
                for (int r = 0; r < n_row_ranges; ++r)
                    dispatch_limited(row_ranges[r][0], row_ranges[r][1],
                                     col_loc, col_loc + 1,
                                     A_gpu.lookAheadLHandle[streamId],
                                     A_gpu.lookAheadLStream[streamId],
                                     A_gpu.lookAheadLGemmBuffer[streamId],
                                     SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL);
            if (row_loc != GLOBAL_BLOCK_NOT_FOUND && row_loc != diag_row)
            {
                int_t col_ranges[3][2];
                int n_col_ranges = ranges_excluding(
                    0, ncol, col_loc, diag_col, col_ranges);
                for (int c = 0; c < n_col_ranges; ++c)
                    dispatch_limited(row_loc, row_loc + 1,
                                     col_ranges[c][0], col_ranges[c][1],
                                     A_gpu.lookAheadUHandle[streamId],
                                     A_gpu.lookAheadUStream[streamId],
                                     A_gpu.gpuGemmBuffs[streamId],
                                     SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW);
            }
        }
        if (mode_mask & SYM_V2_PCFRAG_TASK_EXCLUDE)
        {
            ++symV2PcFragTaskflowStats.dispatch_calls_exclude;
            int_t row_loc = frag_find(row_frag, mode_gid);
            int_t col_loc = frag_find(col_frag, mode_gid);
            int_t row_ranges[3][2];
            int_t col_ranges[3][2];
            int n_row_ranges = ranges_excluding(
                0, nrow, row_loc, diag_row, row_ranges);
            int n_col_ranges = ranges_excluding(
                0, ncol, col_loc, diag_col, col_ranges);
            for (int r = 0; r < n_row_ranges; ++r)
                for (int c = 0; c < n_col_ranges; ++c)
                    dispatch_limited(row_ranges[r][0], row_ranges[r][1],
                                     col_ranges[c][0], col_ranges[c][1],
                                     handle, stream, gemmBuff,
                                     SYM_V2_PCFRAG_TASK_EXCLUDE);
        }
        if (mode_mask & SYM_V2_PCFRAG_TASK_FULL)
        {
            ++symV2PcFragTaskflowStats.dispatch_calls_full;
            int_t row_ranges[3][2];
            int_t col_ranges[3][2];
            int n_row_ranges = ranges_excluding(
                0, nrow, diag_row, GLOBAL_BLOCK_NOT_FOUND, row_ranges);
            int n_col_ranges = ranges_excluding(
                0, ncol, diag_col, GLOBAL_BLOCK_NOT_FOUND, col_ranges);
            for (int r = 0; r < n_row_ranges; ++r)
                for (int c = 0; c < n_col_ranges; ++c)
                    dispatch_limited(row_ranges[r][0], row_ranges[r][1],
                                     col_ranges[c][0], col_ranges[c][1],
                                     handle, stream, gemmBuff,
                                     SYM_V2_PCFRAG_TASK_FULL);
        }
        progress_launched_tasks(drain ? 1 : 0, mode_mask);
        if (state.incomplete_task_count == 0 &&
            !state.producer_exchange_pending &&
            dSymV2PcFragTaskflowProducerStreamComplete(state, 0) &&
            !state.producer_exchange_active)
        {
            release_completed_state();
            return 0;
        }
        if (!drain)
            return 0;
        if (!(mode_mask & SYM_V2_PCFRAG_TASK_FULL))
            return 0;
    }
    ABORT("GPU3DV2_PCFRAG_TASKFLOW reached Schur dispatch before piece-owned exchange/task execution is implemented.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowDrainGPU(
    int_t k, unsigned mode_mask, int_t mode_gid)
{
    if (!symV2UsePcFragmentTaskflowPanel(k))
        return 0;
    int streamId = 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PcFragTaskStates.size())
        return 0;
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    if (!state.initialized)
        return 0;
    if (state.stream_offset >= 0 &&
        state.stream_offset < A_gpu.numCudaStreams)
        streamId = state.stream_offset;
    if (state.incomplete_task_count > 0)
        symV2PcFragTaskflowStats.drain_incomplete_tasks +=
            state.incomplete_task_count;
    if (mode_mask & (SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                     SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW))
        ++symV2PcFragTaskflowStats.drain_calls_lookahead;
    if (mode_mask & SYM_V2_PCFRAG_TASK_EXCLUDE)
        ++symV2PcFragTaskflowStats.drain_calls_exclude;
    if (mode_mask & SYM_V2_PCFRAG_TASK_FULL)
        ++symV2PcFragTaskflowStats.drain_calls_full;
    if (superlu_sym_v2_pcfrag_taskflow_async_core())
    {
        dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
        dSymV2PcFragTaskflowProgressGPU(k, 0);
        if (!state.initialized)
            return 0;
        dSymV2PcFragTaskflowDispatchGPU(
            streamId, k, mode_mask, mode_gid, 0);
        if (!state.initialized)
            return 0;
        if (!(mode_mask & SYM_V2_PCFRAG_TASK_FULL))
        {
            const bool strict_output_conflicts =
                superlu_sym_v2_pcfrag_taskflow_strict() &&
                (state.output_conflicts_possible ||
                 superlu_sym_v2_pcfrag_taskflow_global_output_locks());
            auto required_incomplete = [&]() -> int {
                int known = 0;
                int fast_count =
                    dSymV2PcFragTaskflowRequiredIncompleteFast(
                        state, mode_mask, mode_gid, &known);
                if (known)
                    return fast_count;
                int count = 0;
                for (size_t i = 0; i < state.tasks.size(); ++i)
                {
                    SymV2PcFragTaskDesc &task = state.tasks[i];
                    if (task.complete)
                        continue;
                    if (dSymV2PcFragTaskflowTaskRequiredForMode(
                            state, task, 1, mode_mask, mode_gid))
                        ++count;
                }
                return count;
            };
            auto progress_cross_panel_locks = [&]() -> int {
                int completed = 0;
                for (size_t sidx = 0;
                     sidx < symV2PcFragTaskStates.size(); ++sidx)
                {
                    SymV2PcFragPanelTaskState &other =
                        symV2PcFragTaskStates[sidx];
                    if (!other.initialized)
                        continue;
                    dSymV2PcFragTaskflowProgressExchangeGPU(
                        static_cast<int_t>(sidx), 0);
                    const bool other_strict_output_conflicts =
                        superlu_sym_v2_pcfrag_taskflow_strict() &&
                        (other.output_conflicts_possible ||
                         superlu_sym_v2_pcfrag_taskflow_global_output_locks());
                    int pending_other = 0;
                    completed += dSymV2PcFragTaskflowProgressLaunchedTasks(
                        *this, other, symV2PcFragTaskflowStats,
                        other_strict_output_conflicts, 1, 0,
                        GLOBAL_BLOCK_NOT_FOUND, &pending_other);
                }
                return completed;
            };
            int idle_polls = 0;
            for (;;)
            {
                if (required_incomplete() == 0)
                    return 0;
                int launched_this_pass =
                    dSymV2PcFragTaskflowDispatchGPU(
                        streamId, k, mode_mask, mode_gid, 1);
                int pending_required = 0;
                int completed_this_pass =
                    dSymV2PcFragTaskflowProgressLaunchedTasks(
                        *this, state, symV2PcFragTaskflowStats,
                        strict_output_conflicts, 1, mode_mask, mode_gid,
                        &pending_required);
                int pending_any = 0;
                int completed_any =
                    dSymV2PcFragTaskflowProgressLaunchedTasks(
                        *this, state, symV2PcFragTaskflowStats,
                        strict_output_conflicts, 1, 0,
                        GLOBAL_BLOCK_NOT_FOUND, &pending_any);
                if (required_incomplete() == 0)
                    return 0;
                if (launched_this_pass > 0 ||
                    completed_this_pass > 0 || completed_any > 0)
                {
                    idle_polls = 0;
                    continue;
                }
                int completed_cross_panel = 0;
                if (!symV2PcFragTaskflowGlobalOutputLocks.empty() ||
                    symV2PcFragTaskflowGlobalOutputLocksLive != 0)
                    completed_cross_panel = progress_cross_panel_locks();
                if (completed_cross_panel > 0)
                {
                    idle_polls = 0;
                    continue;
                }
                int exchange_progress =
                    dSymV2PcFragTaskflowProgressExchangeGPU(
                        k, (pending_any == 0 && pending_required == 0) ? 1 : 0);
                if (exchange_progress > 0)
                {
                    idle_polls = 0;
                    continue;
                }
                if (++idle_polls > 10000000)
                {
                    std::fprintf(stderr,
                                 "GPU3DV2 taskflow mode drain stalled: "
                                 "k=%lld mode=%u gid=%lld required=%d "
                                 "incomplete=%d launched_pending=%d "
                                 "pending_required=%d pending_any=%d "
                                 "row_ready=%zu/%zu partner_ready=%zu/%zu "
                                 "active_locks=%zu compact_locks=%d "
                                 "global_locks=%zu compact_global_locks=%lld "
                                 "producer_active=%d producer_pending=%d "
                                 "producer_stream_pending=%d "
                                 "partner_recv_remaining=%d "
                                 "row_recv_remaining=%d "
                                 "partner_recv_reqs=%zu row_recv_reqs=%zu "
                                 "send_reqs=%zu\n",
                                 static_cast<long long>(k), mode_mask,
                                 static_cast<long long>(mode_gid),
                                 required_incomplete(),
                                 state.incomplete_task_count,
                                 dSymV2PcFragTaskflowPendingLaunchedAllForMode(
                                     state, 0, 0, GLOBAL_BLOCK_NOT_FOUND),
                                 pending_required, pending_any,
                                 static_cast<size_t>(state.row_pieces_ready_count),
                                 state.row_pieces.size(),
                                 static_cast<size_t>(state.partner_pieces_ready_count),
                                 state.partner_pieces.size(),
                                 state.active_output_key_set.size(),
                                 state.active_output_lock_count,
                                 symV2PcFragTaskflowGlobalOutputLocks.size(),
                                 symV2PcFragTaskflowGlobalOutputLocksLive,
                                 static_cast<int>(state.producer_exchange_active),
                                 static_cast<int>(state.producer_exchange_pending),
                                 static_cast<int>(state.producer_stream_pending),
                                 state.producer_partner_recv_remaining,
                                 state.producer_row_recv_remaining,
                                 state.producer_partner_recv_reqs.size(),
                                 state.producer_row_recv_reqs.size(),
                                 state.producer_send_reqs.size());
                    std::fflush(stderr);
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW mode drain made no CUDA progress.");
                }
            }
        }
    }
    dSymV2PcFragTaskflowProgressExchangeGPU(k, 1);
    return dSymV2PcFragTaskflowDispatchGPU(
        streamId, k, mode_mask, mode_gid, 1);
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PcFragTaskflowReleaseGPU(int_t k)
{
    if (k < 0 || static_cast<size_t>(k) >= symV2PcFragTaskStates.size())
        return 0;
    SymV2PcFragPanelTaskState &state =
        symV2PcFragTaskStates[static_cast<size_t>(k)];
    dSymV2PcFragTaskflowProgressExchangeGPU(k, 1);
    dSymV2PcFragTaskflowProducerStreamComplete(state, 1);
    if (state.producer_exchange_active)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW release found active producer exchange.");
    if (state.producer_exchange_pending)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW release found pending producer exchange.");
    if (superlu_sym_v2_pcfrag_taskflow_async_core())
    {
        int idle_polls = 0;
        for (;;)
        {
            int pending = 0;
            int completed_this_pass =
                dSymV2PcFragTaskflowProgressLaunchedTasks(
                    *this, state, symV2PcFragTaskflowStats,
                    superlu_sym_v2_pcfrag_taskflow_strict(),
                    1, 0, GLOBAL_BLOCK_NOT_FOUND, &pending);
            if (pending == 0)
                break;
            if (completed_this_pass > 0)
            {
                idle_polls = 0;
                continue;
            }
            dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
            if (++idle_polls > 10000000)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW release made no CUDA progress.");
        }
    }
    if (state.incomplete_task_count != 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW release found incomplete tasks.");
    if (!state.active_output_key_set.empty() ||
        state.active_output_lock_count != 0)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW release found active output locks.");
    auto release_piece_storage = [&](SymV2PcFragPieceDesc &piece) {
        if (piece.pending_consumers != 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
        dSymV2PcFragTaskflowRecycleEvent(
            symV2PcFragTaskflowEventPool,
            piece.ready_event);
        dSymV2PcFragTaskflowRecycleEvent(
            symV2PcFragTaskflowEventPool,
            piece.done_event);
        piece.d_index = NULL;
        piece.d_val = NULL;
        piece.h_index.clear();
    };
    for (size_t i = 0; i < state.row_pieces.size(); ++i)
        release_piece_storage(state.row_pieces[i]);
    for (size_t i = 0; i < state.partner_pieces.size(); ++i)
        release_piece_storage(state.partner_pieces[i]);
    for (size_t i = 0; i < state.tasks.size(); ++i)
        dSymV2PcFragTaskflowRecycleEvent(
            symV2PcFragTaskflowEventPool,
            state.tasks[i].done_event);
    dSymV2PcFragTaskflowProducerStreamComplete(state, 1);
    dSymV2PcFragTaskflowRecycleIndexBlock(
        symV2PcFragTaskflowIndexBlockPool,
        state.d_index_pool, state.index_pool_capacity);
    dSymV2PcFragTaskflowRecycleValueBlock(
        symV2PcFragTaskflowValueBlockPool,
        state.d_value_pool, state.value_pool_capacity);
    dSymV2PcFragTaskflowRecycleIndexBlock(
        symV2PcFragTaskflowIndexBlockPool,
        state.d_group_index_pool, state.group_index_pool_capacity);
    dSymV2PcFragTaskflowRecycleValueBlock(
        symV2PcFragTaskflowValueBlockPool,
        state.d_group_value_pool, state.group_value_pool_capacity);
    dSymV2PcFragTaskflowWaitProducerSends(
        state, symV2PcFragTaskflowStats, 1);
    dSymV2PcFragTaskflowReleasePinnedHost(
        symV2PcFragTaskflowPinnedBlockPool, state);
    state.reset();
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2LFragmentExchangeGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("GPU3DVERSION=2 true symmetric mode requires GPU offload.");
    if (Pr <= 1)
        return 0;
    if (k < 0 || k >= nsupers)
        return 0;
    const bool pcfrag_taskflow = symV2UsePcFragmentTaskflowPanel(k);
    if (pcfrag_taskflow)
        dSymV2PcFragTaskflowBeginGPU(k, stream_offset);
    SymV2PcFragPanelTaskState *taskflow_state =
        (pcfrag_taskflow &&
         static_cast<size_t>(k) < symV2PcFragTaskStates.size())
            ? &symV2PcFragTaskStates[static_cast<size_t>(k)]
            : NULL;
    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLHostSendBufs.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLRecvSizes.empty() ||
        symV2PartnerLRecvIndex.empty() ||
        symV2PartnerLRecvMap.empty() ||
        symV2PartnerLRecvMapsGPU.empty())
        ABORT("SymFact GPU L-fragment buffers are not allocated.");

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double lfrag_total_t = SuperLU_timer_();
#endif
    long long sym_v2_partner_payload_bytes = 0;
    long long sym_v2_rowfrag_payload_bytes = 0;

    SYM_V2_TRACE_EXCHANGE(grid3d, k,
                          "enter L-fragment exchange myrow=%d mycol=%d krow=%d kcol=%d Lidx=%d",
                          static_cast<int>(myrow), static_cast<int>(mycol),
                          static_cast<int>(symV2DiagRoot(k)),
                          static_cast<int>(symV2PanelRoot(k)),
                          static_cast<int>(LidxSendCounts[k]));

    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    bool async_factor = superlu_sym_v2_async_factor();
    cudaStream_t stream = async_factor
                              ? A_gpu.lookAheadUStream[stream_offset]
                              : A_gpu.cuStreams[stream_offset];
    int_t kcol_ = symV2PanelRoot(k);
    int_t ksupc = SuperSize(k);
    int tag_ub = symFactTagUb;
    bool cuda_aware = superlu_cuda_aware_mpi();
    const bool pc_fragment_schur =
        symV2UsePcFragmentSchurPanel(k);
    const bool exact_fragment_demand =
        pc_fragment_schur && superlu_sym_v2_exact_fragment_demand();
    const bool exact_partner_fragment_demand =
        exact_fragment_demand &&
        superlu_sym_v2_exact_partner_fragment_demand();
    const bool row_down_plan_exchange =
        pc_fragment_schur && superlu_sym_v2_row_l_plan_v2_exchange();
    const bool exact_row_fragment_demand =
        exact_fragment_demand &&
        superlu_sym_v2_exact_row_fragment_demand() &&
        !superlu_sym_v2_row_l_postsolve_send() &&
// SYM_V2_PC2_PHASE4_GPU_EXACT_ROW_GUARD_BEGIN
        !row_down_plan_exchange;
// SYM_V2_PC2_PHASE4_GPU_EXACT_ROW_GUARD_END
    const bool row_l_direct_recv =
        pc_fragment_schur && superlu_sym_v2_row_l_direct_recv() &&
        !superlu_sym_v2_row_l_postsolve_send();
    const bool ldl_native_direct_row =
        row_l_direct_recv && superlu_sym_v2_pc_fragment_ldl_native();
    const bool row_down_lazy_sendmap =
        row_down_plan_exchange &&
        ldl_native_direct_row &&
        superlu_sym_v2_row_l_compressed_plan() &&
        superlu_sym_v2_row_l_lazy_sendmap();
// SYM_V2_PC2_ASYNC_EXCHANGE_FLAG_BEGIN
    const bool pcfrag_async_pipeline =
        pc_fragment_schur &&
        async_factor &&
        row_down_lazy_sendmap &&
        superlu_sym_v2_pc_fragment_ldl_native() &&
        superlu_sym_v2_pcfrag_async_pipeline();
    const bool pcfrag_async_exchange =
        pc_fragment_schur &&
        async_factor &&
        row_down_lazy_sendmap &&
        superlu_sym_v2_pc_fragment_ldl_native() &&
        (superlu_sym_v2_pcfrag_async_exchange() ||
         pcfrag_async_pipeline);
// SYM_V2_PC2_ASYNC_EXCHANGE_FLAG_END
    const bool row_down_lazy_warp_pack =
        row_down_lazy_sendmap &&
        superlu_sym_v2_row_l_lazy_warp_pack();
    const bool pcfrag_taskflow_eager =
        pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_eager();
    const bool pcfrag_taskflow_validate =
        pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_validate();
    const bool pcfrag_taskflow_async_pieces =
        pcfrag_taskflow && !pcfrag_taskflow_validate &&
        superlu_sym_v2_pcfrag_taskflow_async_pieces();
    const bool allow_taskflow_late_alloc =
        !(pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_async_core());
    auto taskflow_update_pinned_high_water = [&]() {
        if (taskflow_state == NULL)
            return;
        size_t pinned_total =
            taskflow_state->producer_partner_recv_host_capacity +
            taskflow_state->producer_row_recv_host_capacity +
            taskflow_state->producer_partner_send_host_capacity +
            taskflow_state->producer_row_send_host_capacity;
        symV2PcFragTaskflowStats.arena_pinned_high_water =
            SUPERLU_MAX(
                symV2PcFragTaskflowStats.arena_pinned_high_water,
                static_cast<long long>(pinned_total * sizeof(double)));
    };
    auto taskflow_assemble_owned_pieces =
        [&](unsigned char kind, const double *stage,
            const std::vector<int_t> &recv_map) {
        if (pcfrag_taskflow)
            dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
                k, kind, stage, recv_map, ksupc, stream);
    };
    auto taskflow_copy_owned_pieces_from_full =
        [&](unsigned char kind, const double *full, int_t full_lda,
            cudaMemcpyKind copy_kind) {
        if (!pcfrag_taskflow)
            return;
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW full copy has no state.");
        if (full == NULL || full_lda <= 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW full copy has no source.");
        std::vector<SymV2PcFragPieceDesc> &pieces =
            (kind == SYM_V2_PCFRAG_PIECE_ROW)
                ? taskflow_state->row_pieces
                : taskflow_state->partner_pieces;
        for (size_t p = 0; p < pieces.size(); ++p)
        {
            SymV2PcFragPieceDesc &piece = pieces[p];
            if (piece.ready)
                continue;
            if (piece.d_val == NULL || piece.lda <= 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW piece value storage is missing.");
            if (piece.frag_row_offset < 0 ||
                piece.frag_row_offset + piece.nrows > full_lda)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW full copy piece range is invalid.");
            gpuErrchk(cudaMemcpy2DAsync(
                piece.d_val,
                sizeof(double) * static_cast<size_t>(piece.lda),
                full + piece.frag_row_offset,
                sizeof(double) * static_cast<size_t>(full_lda),
                sizeof(double) * static_cast<size_t>(piece.nrows),
                static_cast<size_t>(ksupc),
                copy_kind, stream));
            piece.filled_rows = piece.nrows;
            if (piece.ready_event != NULL)
            {
                gpuErrchk(cudaEventRecord(piece.ready_event, stream));
                taskflow_state->producer_last_ready_event = piece.ready_event;
            }
            piece.ready = 1;
            if (kind == SYM_V2_PCFRAG_PIECE_ROW)
            {
                ++symV2PcFragTaskflowStats.row_pieces_ready;
                ++taskflow_state->row_pieces_ready_count;
            }
            else
            {
                ++symV2PcFragTaskflowStats.partner_pieces_ready;
                ++taskflow_state->partner_pieces_ready_count;
            }
            taskflow_state->note_piece_ready(kind, piece.piece_id);
            if (pcfrag_taskflow_eager)
                dSymV2PcFragTaskflowProgressGPU(
                    k, superlu_sym_v2_pcfrag_taskflow_effective_progress_budget());
        }
    };
    auto taskflow_validate_owned_pieces =
        [&](unsigned char kind, const double *full, int_t full_lda) {
        if (!pcfrag_taskflow_validate)
            return;
        if (full == NULL || full_lda < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW validation has no full fragment.");
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW validation has no state.");
        gpuErrchk(cudaStreamSynchronize(stream));
        std::vector<double> full_host(
            static_cast<size_t>(full_lda) * static_cast<size_t>(ksupc));
        if (!full_host.empty())
            gpuErrchk(cudaMemcpy(
                full_host.data(), full,
                sizeof(double) * full_host.size(), cudaMemcpyDeviceToHost));
        const std::vector<SymV2PcFragPieceDesc> &pieces =
            (kind == SYM_V2_PCFRAG_PIECE_ROW)
                ? taskflow_state->row_pieces
                : taskflow_state->partner_pieces;
        for (size_t p = 0; p < pieces.size(); ++p)
        {
            const SymV2PcFragPieceDesc &piece = pieces[p];
            if (!piece.ready || piece.d_val == NULL)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW validation saw an unready piece.");
            std::vector<double> piece_host(
                static_cast<size_t>(piece.value_count));
            if (!piece_host.empty())
                gpuErrchk(cudaMemcpy(
                    piece_host.data(), piece.d_val,
                    sizeof(double) * piece_host.size(),
                    cudaMemcpyDeviceToHost));
            for (int_t col = 0; col < ksupc; ++col)
            {
                for (int_t row = 0; row < piece.nrows; ++row)
                {
                    size_t piece_pos =
                        static_cast<size_t>(row) +
                        static_cast<size_t>(col) *
                            static_cast<size_t>(piece.lda);
                    size_t full_pos =
                        static_cast<size_t>(piece.frag_row_offset + row) +
                        static_cast<size_t>(col) *
                            static_cast<size_t>(full_lda);
                    if (full_pos >= full_host.size() ||
                        piece_pos >= piece_host.size())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW validation index is out of range.");
                    if (std::memcmp(&piece_host[piece_pos],
                                    &full_host[full_pos],
                                    sizeof(double)) != 0)
                    {
                        std::printf(
                            "GPU3DV2_PCFRAG_TASKFLOW validation mismatch: "
                            "k=%lld kind=%d piece=%zu gid=%lld row=%lld col=%lld "
                            "piece=%+.17e full=%+.17e\n",
                            static_cast<long long>(k), static_cast<int>(kind),
                            p, static_cast<long long>(piece.gid_first),
                            static_cast<long long>(row),
                            static_cast<long long>(col),
                            piece_host[piece_pos], full_host[full_pos]);
                        std::fflush(stdout);
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW owned piece differs from full fragment.");
                    }
                }
            }
        }
    };
// SYM_V2_PC2_PHASE2_COMBO_GUARD_BEGIN
    if (pc_fragment_schur && superlu_sym_v2_row_l_one_sync() &&
        !superlu_sym_v2_row_l_pack_all_dest())
        ABORT("GPU3DV2_ROW_L_ONE_SYNC requires GPU3DV2_ROW_L_PACK_ALL_DEST=1.");
    if (pc_fragment_schur && superlu_sym_v2_row_l_pack_all_dest() &&
        !superlu_sym_v2_row_l_separate_send_staging())
        ABORT("GPU3DV2_ROW_L_PACK_ALL_DEST requires GPU3DV2_ROW_L_SEPARATE_SEND_STAGING=1.");
// SYM_V2_PC2_PHASE2_COMBO_GUARD_END
// SYM_V2_PC2_PHASE4_PLAN_EXCHANGE_GUARD_BEGIN
    if (row_down_plan_exchange)
    {
        if (!superlu_sym_v2_row_l_plan_v2())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE requires GPU3DV2_ROW_L_PLAN_V2=1.");
        if (superlu_sym_v2_row_l_plan_v2_dryrun())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE requires GPU3DV2_ROW_L_PLAN_V2_DRYRUN=0.");
        if (!superlu_sym_v2_row_l_pack_all_dest())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE requires GPU3DV2_ROW_L_PACK_ALL_DEST=1.");
        if (!superlu_sym_v2_row_l_plan_v2_aggregate_dest())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE currently requires GPU3DV2_ROW_L_PLAN_V2_AGGREGATE_DEST=1.");
        if (!superlu_sym_v2_row_l_separate_send_staging())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE requires GPU3DV2_ROW_L_SEPARATE_SEND_STAGING=1.");
    }
    if (pc_fragment_schur && superlu_sym_v2_row_l_plan_v2_compact())
    {
        if (!superlu_sym_v2_row_l_plan_v2())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PLAN_V2=1.");
        if (superlu_sym_v2_row_l_plan_v2_dryrun())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PLAN_V2_DRYRUN=0.");
        if (!superlu_sym_v2_row_l_plan_v2_exchange())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PLAN_V2_EXCHANGE=1.");
        if (!superlu_sym_v2_row_l_pack_all_dest())
            ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PACK_ALL_DEST=1.");
    }
// SYM_V2_PC2_PHASE4_PLAN_EXCHANGE_GUARD_END


    if (pc_fragment_schur)
    {
// SYM_V2_PC2_PHASE6_ASYNC_CUDA_AWARE_GUARD_BEGIN
// SYM_V2_PC2_ASYNC_EXCHANGE_GUARD_BEGIN
        if (cuda_aware)
            ABORT("Pc-fragment CUDA-aware MPI is still fail-closed.");
        if (async_factor && !pcfrag_async_exchange)
            ABORT("Pc-fragment async factor is fail-closed unless GPU3DV2_PCFRAG_ASYNC_EXCHANGE=1 with lazy LDL row-down.");
        if (superlu_sym_v2_pcfrag_cuda_aware_experiment())
            ABORT("Pc-fragment CUDA-aware experiment is fail-closed.");
        if (superlu_sym_v2_pcfrag_async_experiment() &&
            !pcfrag_async_exchange)
            ABORT("Pc-fragment async experiment is fail-closed; use GPU3DV2_PCFRAG_ASYNC_EXCHANGE=1 for the guarded slice.");
// SYM_V2_PC2_ASYNC_EXCHANGE_GUARD_END
// SYM_V2_PC2_PHASE6_ASYNC_CUDA_AWARE_GUARD_END
        if (exact_partner_fragment_demand &&
            superlu_sym_v2_pc_fragment_ldl_native())
            ABORT("GPU3DV2_PC_FRAGMENT_LDL_NATIVE does not support exact partner-fragment demand with stream-staged partner sends.");
        if (exact_partner_fragment_demand &&
            (symV2PartnerLExactSendSizes.empty() ||
             symV2PartnerLExactSendBufsGPU.empty() ||
             symV2PartnerLExactSendMapsGPU.empty()))
            ABORT("GPU3DV2_EXACT_FRAGMENT_DEMAND partner buffers are not allocated.");
        if (exact_row_fragment_demand &&
            !superlu_sym_v2_rowfrag_destination_path())
            ABORT("GPU3DV2_EXACT_ROW_FRAGMENT_DEMAND requires GPU3DV2_ROW_L_SOURCE_PACK=1, GPU3DV2_ROW_L_DIRECT_RECV=1, or GPU3DV2_ROWFRAG_DEST_PACK=1.");
        if (exact_row_fragment_demand &&
            !ldl_native_direct_row &&
            (symV2RowFragExactSendSizes.empty() ||
             symV2RowFragExactSendMapOffsets.empty() ||
             symV2RowFragExactSendMapsHost.empty()))
            ABORT("GPU3DV2_EXACT_FRAGMENT_DEMAND row maps are not allocated.");
    }
    const bool pooled_staging =
        !cuda_aware && superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool();
    if (pc_fragment_schur && !pooled_staging)
        ABORT("GPU3DV2_PC_FRAGMENT_SCHUR requires pooled pinned staging.");
    std::vector<int> local_send_sizes;
    std::vector<int> &send_sizes = pooled_staging
        ? symV2ExchangeSendSizesScratch
        : local_send_sizes;
    if (pooled_staging)
    {
        if (send_sizes.size() != static_cast<size_t>(Pc))
            ABORT("SymFact V2 pooled send-size scratch is invalid.");
        std::fill(send_sizes.begin(), send_sizes.end(), 0);
    }
    else
    {
        send_sizes.assign(static_cast<size_t>(Pc), 0);
    }
    std::vector<size_t> taskflow_partner_send_offsets;
    double *taskflow_partner_send_host_base = NULL;
    size_t taskflow_partner_send_total = 0;

    auto partner_send_buffer = [&](size_t flat, int size) -> double *
    {
        if (flat >= symV2PartnerLSendBufsGPU.size())
            ABORT("SymFact V2 partner-L send buffer slot is invalid.");
        double *sendbuf = symV2PartnerLSendBufsGPU[flat];
        if (superlu_sym_v2_pc_fragment_ldl_native())
        {
            if (A_gpu.symPartnerLSendStageBufs[stream_offset] == NULL)
                ABORT("SymFact V2 partner-L stream send stage is missing.");
            if (flat >= symV2PartnerLHostSendScratchOffsets.size())
                ABORT("SymFact V2 partner-L send staging offset is missing.");
            size_t send_offset =
                symV2PartnerLHostSendScratchOffsets[flat];
            size_t count = size > 0 ? static_cast<size_t>(size) : 0;
            if (send_offset + count >
                    static_cast<size_t>(maxSymPartnerLSendStageCount) ||
                send_offset + count < send_offset)
                ABORT("SymFact V2 partner-L stream send stage is too small.");
            sendbuf = A_gpu.symPartnerLSendStageBufs[stream_offset] +
                      send_offset;
        }
        return sendbuf;
    };
    auto setup_taskflow_partner_send_host = [&]() -> double *
    {
        if (!pcfrag_taskflow_async_pieces || cuda_aware)
            return NULL;
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW async partner send has no state.");
        if (exact_partner_fragment_demand)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW async partner send does not support exact partner demand.");
        int_t lk = symV2PanelIndex(k);
        taskflow_partner_send_offsets.assign(static_cast<size_t>(Pc),
                                             static_cast<size_t>(-1));
        taskflow_partner_send_total = 0;
        for (int pc = 0; pc < Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(Pc) +
                          static_cast<size_t>(pc);
            if (flat >= symV2PartnerLSendSizes.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send size is missing.");
            int size = send_sizes[static_cast<size_t>(pc)];
            if (size <= 0)
                continue;
            bool active_remote_dest = false;
            for (int pr = 0; pr < Pr; ++pr)
            {
                size_t active_pos =
                    flat * static_cast<size_t>(Pr) +
                    static_cast<size_t>(pr);
                if (active_pos >= symV2PartnerLSendRowActive.size())
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send row mask is missing.");
                if (symV2PartnerLSendRowActive[active_pos] &&
                    PNUM(pr, pc, grid) != iam)
                {
                    active_remote_dest = true;
                    break;
                }
            }
            if (!active_remote_dest)
                continue;
            taskflow_partner_send_offsets[static_cast<size_t>(pc)] =
                taskflow_partner_send_total;
            if (taskflow_partner_send_total >
                std::numeric_limits<size_t>::max() -
                    static_cast<size_t>(size))
                ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send host total overflows.");
            taskflow_partner_send_total += static_cast<size_t>(size);
        }
        if (taskflow_partner_send_total == 0)
            return NULL;
        taskflow_partner_send_host_base =
            dSymV2PcFragTaskflowEnsurePinnedHost(
                symV2PcFragTaskflowPinnedBlockPool,
                &taskflow_state->producer_partner_send_host_values,
                &taskflow_state->producer_partner_send_host_capacity,
                taskflow_partner_send_total,
                allow_taskflow_late_alloc,
                &symV2PcFragTaskflowStats.arena_pinned_late_allocs);
        taskflow_update_pinned_high_water();
        return taskflow_partner_send_host_base;
    };

// SYM_V2_PC2_ASYNC_PIPELINE_EARLY_RECV_BEGIN
    size_t recv_count_base =
        static_cast<size_t>(k) * static_cast<size_t>(Pr);
    std::vector<int> local_recv_sizes;
    std::vector<int> local_recv_offsets;
    std::vector<int> &recv_sizes = pooled_staging
        ? symV2ExchangeRecvSizesScratch
        : local_recv_sizes;
    std::vector<int> &recv_offsets = pooled_staging
        ? symV2ExchangeRecvOffsetsScratch
        : local_recv_offsets;
    std::vector<MPI_Request> local_recv_reqs;
    std::vector<MPI_Request> local_send_reqs;
    std::vector<int> local_recv_request_peers;
    std::vector<int> local_wait_indices;
    std::vector<MPI_Status> local_wait_statuses;
    std::vector<MPI_Request> &recv_reqs = pooled_staging
        ? symV2ExchangeRecvReqsScratch
        : local_recv_reqs;
    std::vector<MPI_Request> &send_reqs = pooled_staging
        ? symV2ExchangeSendReqsScratch
        : local_send_reqs;
    std::vector<int> &recv_request_peers = pooled_staging
        ? symV2ExchangeRecvPeersScratch
        : local_recv_request_peers;
    std::vector<int> &wait_indices = pooled_staging
        ? symV2ExchangeWaitIndicesScratch
        : local_wait_indices;
    std::vector<MPI_Status> &wait_statuses = pooled_staging
        ? symV2ExchangeWaitStatusesScratch
        : local_wait_statuses;
    int recv_total = 0;
    double *recv_host_base = NULL;
    bool partner_recvs_posted = false;
    size_t deferred_partner_send_req_count = 0;

    auto setup_partner_recv_metadata = [&]()
    {
        if (recv_count_base + static_cast<size_t>(Pr) >
                symV2PartnerLRecvSizes.size() ||
            recv_count_base + static_cast<size_t>(Pr) >
                symV2PartnerLRecvMap.size())
            ABORT("SymFact V2 true symmetric L-fragment receive sizes are missing.");

        if (pooled_staging)
        {
            if (recv_sizes.size() != static_cast<size_t>(Pr) ||
                recv_offsets.size() != static_cast<size_t>(Pr))
                ABORT("SymFact V2 pooled receive scratch is invalid.");
            std::fill(recv_sizes.begin(), recv_sizes.end(), 0);
            std::fill(recv_offsets.begin(), recv_offsets.end(), -1);
        }
        else
        {
            recv_sizes.assign(static_cast<size_t>(Pr), 0);
            recv_offsets.assign(static_cast<size_t>(Pr), -1);
        }
        recv_total = 0;
        for (int pr = 0; pr < Pr; ++pr)
        {
            size_t pos = recv_count_base + static_cast<size_t>(pr);
            recv_sizes[pr] = symV2PartnerLRecvSizes[pos];
            int src = PNUM(pr, kcol_, grid);
            if (recv_sizes[pr] > 0 && src != iam)
            {
                recv_offsets[pr] = recv_total;
                recv_total += recv_sizes[pr];
            }
        }
        if (recv_total > maxSymPartnerLvalCount)
            ABORT("SymFact V2 true symmetric L-fragment receive exceeds staging buffer.");
        if (recv_total > 0 && A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 true symmetric L-fragment staging buffer is missing.");

        recv_reqs.clear();
        send_reqs.clear();
        recv_request_peers.clear();
        recv_host_base = NULL;
        if (!cuda_aware && recv_total > 0)
        {
            if (pcfrag_taskflow_async_pieces)
            {
                if (taskflow_state == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW async partner receive has no state.");
                recv_host_base = dSymV2PcFragTaskflowEnsurePinnedHost(
                    symV2PcFragTaskflowPinnedBlockPool,
                    &taskflow_state->producer_partner_recv_host_values,
                    &taskflow_state->producer_partner_recv_host_capacity,
                    static_cast<size_t>(recv_total),
                    allow_taskflow_late_alloc,
                    &symV2PcFragTaskflowStats.arena_pinned_late_allocs);
                taskflow_update_pinned_high_water();
            }
            else
            {
                if (static_cast<size_t>(stream_offset) >=
                        symPartnerLvalRecvBufs.size() ||
                    symPartnerLvalRecvBufs[stream_offset] == NULL)
                    ABORT("SymFact V2 host receive staging buffer is missing.");
                recv_host_base = symPartnerLvalRecvBufs[stream_offset];
            }
        }
    };

    auto post_partner_recvs = [&]()
    {
        if (partner_recvs_posted)
            return;
        recv_reqs.reserve(Pr);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double recv_post_t = SuperLU_timer_();
#endif
        for (int pr = 0; pr < Pr; ++pr)
        {
            int size = recv_sizes[pr];
            if (size <= 0)
                continue;
            int src = PNUM(pr, kcol_, grid);
            if (src == iam)
                continue;
            MPI_Request req;
            double *recv_ptr = NULL;
            long long recv_bytes =
                static_cast<long long>(size) *
                static_cast<long long>(sizeof(double));
            sym_v2_partner_payload_bytes += recv_bytes;
            symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_MPI_RECV,
                                   recv_bytes);
            if (cuda_aware)
            {
                recv_ptr = A_gpu.symPartnerLStageBufs[stream_offset] +
                           recv_offsets[pr];
            }
            else
            {
                recv_ptr = recv_host_base + recv_offsets[pr];
            }
            if (pcfrag_taskflow_async_pieces)
            {
                const bool pinned_taskflow_recv =
                    !cuda_aware && taskflow_state != NULL &&
                    recv_host_base != NULL &&
                    recv_host_base ==
                        taskflow_state->producer_partner_recv_host_values;
                dSymV2PcFragTaskflowNoteProducerRecvPost(
                    symV2PcFragTaskflowStats, pinned_taskflow_recv);
            }
            MPI_Irecv(recv_ptr, size, MPI_DOUBLE, src,
                      SLU_MPI_TAG(5, k), grid->comm, &req);
            recv_reqs.push_back(req);
            recv_request_peers.push_back(pr);
        }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAddBoth(SYM_GPU3D_T_LFRAG_RECV_POST,
                         SYM_GPU3D_T_PARTNER_LFRAG_RECV_POST,
                         SuperLU_timer_() - recv_post_t);
        symStatAdd(SYM_GPU3D_S_L2U_RECV_REQUESTS,
                   static_cast<long long>(recv_reqs.size()));
#endif
        partner_recvs_posted = true;
    };

    if (pcfrag_async_pipeline)
    {
        setup_partner_recv_metadata();
        post_partner_recvs();
    }
// SYM_V2_PC2_ASYNC_PIPELINE_EARLY_RECV_END

    if (mycol == kcol_)
    {
        int_t lk = symV2PanelIndex(k);
        unsigned char prepacked_slot =
            (lk >= 0 &&
             static_cast<size_t>(lk) < symV2PartnerLPrepacked.size())
                ? symV2PartnerLPrepacked[static_cast<size_t>(lk)]
                : 0;
        bool prepacked = prepacked_slot != 0;
        if (async_factor && prepacked)
        {
            int pack_event_id = static_cast<int>(prepacked_slot) - 1;
            if (pack_event_id < 0 ||
                pack_event_id >= A_gpu.numCudaStreams)
                ABORT("SymFact V2 raw L-fragment pack event is invalid.");
            gpuErrchk(cudaStreamWaitEvent(
                stream,
                A_gpu.symV2PartnerLPackReadyEvents[pack_event_id], 0));
        }
        if (!prepacked &&
            (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
             symV2DiagBlocksGPU[k] == NULL))
            ABORT("SymFact V2 true symmetric device diagonal block is missing.");
        if (async_factor && !prepacked &&
            k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
            symPanelReadyEventIds[k] >= 0)
        {
            int panel_event_id = symPanelReadyEventIds[k];
            if (panel_event_id >= A_gpu.numCudaStreams)
                ABORT("SymFact V2 transformed-panel event is invalid.");
            gpuErrchk(cudaStreamWaitEvent(
                stream, A_gpu.panelReadyEvents[panel_event_id], 0));
        }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double pack_issue_t = SuperLU_timer_();
#endif
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        bool packed_any = false;
        bool issued_pack = false;
        if (exact_partner_fragment_demand)
        {
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc);
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t active_pos =
                        flat * static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLSendRowActive.size() ||
                        active_pos >= symV2PartnerLExactSendSizes.size() ||
                        active_pos >= symV2PartnerLExactSendBufsGPU.size() ||
                        active_pos >= symV2PartnerLExactSendMapsGPU.size())
                        ABORT("SymFact V2 exact partner-L send slot is invalid.");
                    if (!symV2PartnerLSendRowActive[active_pos])
                        continue;
                    int size = symV2PartnerLExactSendSizes[active_pos];
                    if (size <= 0)
                        ABORT("SymFact V2 exact partner-L active send has no data.");
                    if (lpanel.isEmpty())
                        ABORT("SymFact V2 true symmetric source L panel is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symStatAdd(SYM_GPU3D_S_L2U_SEND_BYTES,
                               static_cast<long long>(size) *
                                   static_cast<long long>(sizeof(double)));
#endif
                    double *sendbuf =
                        symV2PartnerLExactSendBufsGPU[active_pos];
                    int_t *sendmap =
                        symV2PartnerLExactSendMapsGPU[active_pos];
                    if (sendbuf == NULL || sendmap == NULL)
                        ABORT("SymFact V2 exact partner-L send buffer is missing.");
                    int threads = 256;
                    int blocks = (size + threads - 1) / threads;
                    if (prepacked)
                    {
                        sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                            lpanel.gpuPanel.val, sendbuf, sendmap, size);
                    }
                    else
                    {
                        sym_l2u_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                            lpanel.gpuPanel.val, sendbuf, sendmap, size,
                            lpanel.LDA(), symV2DiagBlocksGPU[k], ksupc);
                        issued_pack = true;
                    }
                    packed_any = true;
                }
            }
        }
        else
        {
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);
                if (flat >= symV2PartnerLSendSizes.size())
                    ABORT("SymFact V2 true symmetric L-fragment send size is missing.");
                int size = symV2PartnerLSendSizes[flat];
                send_sizes[pc] = size;
                if (size <= 0)
                    continue;
                bool active_dest = false;
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t active_pos =
                        flat * static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLSendRowActive.size())
                        ABORT("SymFact V2 true symmetric L-fragment send row mask is missing.");
                    if (symV2PartnerLSendRowActive[active_pos])
                    {
                        active_dest = true;
                        break;
                    }
                }
                if (!active_dest)
                    continue;
                if (lpanel.isEmpty())
                    ABORT("SymFact V2 true symmetric source L panel is missing.");

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symStatAdd(SYM_GPU3D_S_L2U_SEND_BYTES,
                           static_cast<long long>(size) *
                               static_cast<long long>(sizeof(double)));
#endif
                double *sendbuf = partner_send_buffer(flat, size);
                int_t *sendmap = symL2LSendMapsGPU[flat];
                if (sendbuf == NULL || sendmap == NULL)
                    ABORT("SymFact V2 true symmetric L-fragment buffer is missing.");

                if (!prepacked)
                {
                    int threads = 256;
                    int blocks = (size + threads - 1) / threads;
                    sym_l2u_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                        lpanel.gpuPanel.val, sendbuf, sendmap, size,
                        lpanel.LDA(), symV2DiagBlocksGPU[k], ksupc);
                    issued_pack = true;
                }
                packed_any = true;
            }
        }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        if (!prepacked)
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                             SYM_GPU3D_T_PARTNER_LFRAG_PACK_ISSUE,
                             SuperLU_timer_() - pack_issue_t);
#endif

        if (packed_any)
        {
            if (issued_pack)
                gpuErrchk(cudaGetLastError());
            if (!cuda_aware)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double d2h_issue_t = SuperLU_timer_();
#endif
                double *owned_partner_send_base =
                    setup_taskflow_partner_send_host();
                if (exact_partner_fragment_demand)
                {
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            size_t active_pos =
                                flat * static_cast<size_t>(Pr) +
                                static_cast<size_t>(pr);
                            if (active_pos >= symV2PartnerLSendRowActive.size() ||
                                active_pos >= symV2PartnerLExactSendSizes.size())
                                ABORT("SymFact V2 exact partner-L send slot is invalid.");
                            if (!symV2PartnerLSendRowActive[active_pos] ||
                                PNUM(pr, pc, grid) == iam)
                                continue;
                            int size =
                                symV2PartnerLExactSendSizes[active_pos];
                            double *host_stage =
                                (active_pos <
                                     symV2PartnerLExactHostSendBufsPinned.size() &&
                                 symV2PartnerLExactHostSendBufsPinned[active_pos] !=
                                     NULL)
                                    ? symV2PartnerLExactHostSendBufsPinned[active_pos]
                                    : (symV2PartnerLExactHostSendBufs[active_pos].empty()
                                           ? NULL
                                           : symV2PartnerLExactHostSendBufs[active_pos].data());
                            if (size <= 0 || host_stage == NULL ||
                                symV2PartnerLExactSendBufsGPU[active_pos] == NULL)
                                ABORT("SymFact V2 exact partner-L host send buffer is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                            symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                                       static_cast<long long>(size) *
                                           static_cast<long long>(sizeof(double)));
#endif
                            gpuErrchk(cudaMemcpyAsync(
                                host_stage,
                                symV2PartnerLExactSendBufsGPU[active_pos],
                                sizeof(double) * static_cast<size_t>(size),
                                cudaMemcpyDeviceToHost, stream));
                        }
                    }
                }
                else
                {
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        size_t flat = static_cast<size_t>(lk) *
                                          static_cast<size_t>(Pc) +
                                      static_cast<size_t>(pc);
                        int size = symV2PartnerLSendSizes[flat];
                        if (size <= 0)
                            continue;
                        bool active_remote_dest = false;
                        for (int pr = 0; pr < Pr; ++pr)
                        {
                            size_t active_pos =
                                flat * static_cast<size_t>(Pr) +
                                static_cast<size_t>(pr);
                            if (active_pos >= symV2PartnerLSendRowActive.size())
                                ABORT("SymFact V2 true symmetric L-fragment send row mask is missing.");
                            if (symV2PartnerLSendRowActive[active_pos] &&
                                PNUM(pr, pc, grid) != iam)
                            {
                                active_remote_dest = true;
                                break;
                            }
                        }
                        if (!active_remote_dest)
                            continue;
                        double *host_stage = NULL;
                        if (owned_partner_send_base != NULL)
                        {
                            size_t off =
                                taskflow_partner_send_offsets[
                                    static_cast<size_t>(pc)];
                            if (off == static_cast<size_t>(-1) ||
                                off + static_cast<size_t>(size) >
                                    taskflow_partner_send_total)
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send host offset is invalid.");
                            host_stage = owned_partner_send_base + off;
                        }
                        else
                        {
                            host_stage =
                                (flat < symV2PartnerLHostSendBufsPinned.size() &&
                                 symV2PartnerLHostSendBufsPinned[flat] != NULL)
                                    ? symV2PartnerLHostSendBufsPinned[flat]
                                    : (symV2PartnerLHostSendBufs[flat].empty()
                                           ? NULL
                                           : symV2PartnerLHostSendBufs[flat].data());
                        }
                        if (host_stage == NULL)
                            ABORT("SymFact V2 true symmetric L-fragment host send buffer is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                        symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                                   static_cast<long long>(size) *
                                       static_cast<long long>(sizeof(double)));
#endif
                        double *d2h_src =
                            partner_send_buffer(flat, size);
                        if (d2h_src == NULL)
                            ABORT("SymFact V2 true symmetric L-fragment send buffer is missing.");
                        gpuErrchk(cudaMemcpyAsync(
                            host_stage, d2h_src,
                            sizeof(double) * static_cast<size_t>(size),
                            cudaMemcpyDeviceToHost, stream));
                    }
                }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                                 SYM_GPU3D_T_PARTNER_LFRAG_D2H_STAGE_ISSUE,
                                 SuperLU_timer_() - d2h_issue_t);
#endif
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double pack_stage_sync_t = SuperLU_timer_();
#endif
            if (pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_async_core())
                ++symV2PcFragTaskflowStats.producer_exchange_stream_syncs;
            gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                             SYM_GPU3D_T_PARTNER_LFRAG_PACK_STAGE_SYNC,
                             SuperLU_timer_() - pack_stage_sync_t);
#endif
        }
    }

// SYM_V2_PC2_ASYNC_PIPELINE_LATE_RECV_BEGIN
    if (!pcfrag_async_pipeline)
    {
        setup_partner_recv_metadata();
        post_partner_recvs();
    }
// SYM_V2_PC2_ASYNC_PIPELINE_LATE_RECV_END


#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double send_post_t = SuperLU_timer_();
#endif
    if (mycol == kcol_)
    {
        int_t send_lk = symV2PanelIndex(k);
        send_reqs.reserve(static_cast<size_t>(Pr) * static_cast<size_t>(Pc));
        if (exact_partner_fragment_demand)
        {
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(send_lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc);
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t active_pos =
                        flat * static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLSendRowActive.size() ||
                        active_pos >= symV2PartnerLExactSendSizes.size())
                        ABORT("SymFact V2 exact partner-L send slot is invalid.");
                    if (!symV2PartnerLSendRowActive[active_pos])
                        continue;
                    int dest = PNUM(pr, pc, grid);
                    if (dest == iam)
                        continue;
                    int size = symV2PartnerLExactSendSizes[active_pos];
                    if (size <= 0)
                        ABORT("SymFact V2 exact partner-L active send has no data.");
                    double *sendbuf =
                        symV2PartnerLExactSendBufsGPU[active_pos];
                    double *hostbuf = NULL;
                    if (!cuda_aware)
                    {
                        hostbuf =
                            (active_pos <
                                 symV2PartnerLExactHostSendBufsPinned.size() &&
                             symV2PartnerLExactHostSendBufsPinned[active_pos] !=
                                 NULL)
                                ? symV2PartnerLExactHostSendBufsPinned[active_pos]
                                : (symV2PartnerLExactHostSendBufs[active_pos].empty()
                                       ? NULL
                                       : symV2PartnerLExactHostSendBufs[active_pos].data());
                        if (hostbuf == NULL)
                            ABORT("SymFact V2 exact partner-L host send staging is missing.");
                    }
                    if (sendbuf == NULL)
                        ABORT("SymFact V2 exact partner-L send buffer is missing.");
                    MPI_Request req;
                    long long send_bytes =
                        static_cast<long long>(size) *
                        static_cast<long long>(sizeof(double));
                    sym_v2_partner_payload_bytes += send_bytes;
                    symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_MPI_SEND,
                                           send_bytes);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    if (cuda_aware)
                        symStatAdd(SYM_GPU3D_S_L2U_CUDA_AWARE_SEND_BYTES,
                                   static_cast<long long>(size) *
                                       static_cast<long long>(sizeof(double)));
#endif
                    MPI_Isend(cuda_aware ? sendbuf : hostbuf,
                              size, MPI_DOUBLE, dest,
                              SLU_MPI_TAG(5, k), grid->comm, &req);
                    send_reqs.push_back(req);
                }
            }
        }
        else
        {
            for (int pc = 0; pc < Pc; ++pc)
            {
                int size = send_sizes[pc];
                size_t flat = static_cast<size_t>(send_lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);
                if (size <= 0)
                    continue;
                double *sendbuf = partner_send_buffer(flat, size);
                double *hostbuf = NULL;
                if (!cuda_aware)
                {
                    if (taskflow_partner_send_host_base != NULL)
                    {
                        size_t off =
                            taskflow_partner_send_offsets[
                                static_cast<size_t>(pc)];
                        if (off != static_cast<size_t>(-1))
                        {
                            if (off + static_cast<size_t>(size) >
                                taskflow_partner_send_total)
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send host offset is invalid.");
                            hostbuf = taskflow_partner_send_host_base + off;
                        }
                    }
                    else
                    {
                        hostbuf =
                            (flat < symV2PartnerLHostSendBufsPinned.size() &&
                             symV2PartnerLHostSendBufsPinned[flat] != NULL)
                                ? symV2PartnerLHostSendBufsPinned[flat]
                                : (symV2PartnerLHostSendBufs[flat].empty()
                                       ? NULL
                                       : symV2PartnerLHostSendBufs[flat].data());
                    }
                    if (hostbuf == NULL &&
                        taskflow_partner_send_host_base == NULL)
                        ABORT("SymFact V2 host send staging is missing.");
                }
                if (sendbuf == NULL)
                    ABORT("SymFact V2 true symmetric L-fragment send buffer is missing.");
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t active_pos =
                        flat * static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLSendRowActive.size())
                        ABORT("SymFact V2 true symmetric L-fragment send row mask is missing.");
                    if (!symV2PartnerLSendRowActive[active_pos])
                        continue;
                    int dest = PNUM(pr, pc, grid);
                    if (dest == iam)
                        continue;
                    if (!cuda_aware && hostbuf == NULL)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW partner send host staging is missing for an active destination.");
                    MPI_Request req;
                    long long send_bytes =
                        static_cast<long long>(size) *
                        static_cast<long long>(sizeof(double));
                    sym_v2_partner_payload_bytes += send_bytes;
                    symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_MPI_SEND,
                                           send_bytes);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    if (cuda_aware)
                        symStatAdd(SYM_GPU3D_S_L2U_CUDA_AWARE_SEND_BYTES,
                                   static_cast<long long>(size) *
                                       static_cast<long long>(sizeof(double)));
#endif
                    MPI_Isend(cuda_aware ? sendbuf : hostbuf,
                              size, MPI_DOUBLE, dest,
                              SLU_MPI_TAG(5, k), grid->comm, &req);
                    send_reqs.push_back(req);
                }
            }
        }
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_POST,
                     SYM_GPU3D_T_PARTNER_LFRAG_SEND_POST,
                     SuperLU_timer_() - send_post_t);
    symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS,
               static_cast<long long>(send_reqs.size()));
#endif
    if (pcfrag_async_pipeline || pcfrag_taskflow_async_pieces)
        deferred_partner_send_req_count = send_reqs.size();

    bool recv_h2d_issued = false;
    bool pipeline_large_receives = false;
    std::vector<unsigned char> *taskflow_partner_progressive_assembled = NULL;
    if (pcfrag_taskflow)
    {
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW progressive partner scratch has no state.");
        taskflow_partner_progressive_assembled =
            &taskflow_state->producer_partner_progressive_assembled;
        if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
            taskflow_partner_progressive_assembled->capacity() <
                static_cast<size_t>(Pr))
            ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE progressive partner scratch is undersized.");
        taskflow_partner_progressive_assembled->assign(
            static_cast<size_t>(Pr), 0);
    }
    if (superlu_sym_v2_large_recv_pipeline())
    {
        if (cuda_aware)
            ABORT("GPU3DV2_LARGE_RECV_PIPELINE currently requires non-CUDA-aware MPI.");
        if (!pooled_staging)
            ABORT("GPU3DV2_LARGE_RECV_PIPELINE requires pooled pinned staging.");
        const size_t threshold = superlu_sym_v2_large_recv_bytes();
        for (int pr = 0; pr < Pr; ++pr)
        {
            int src = PNUM(pr, kcol_, grid);
            if (src == iam || recv_sizes[pr] <= 0)
                continue;
            size_t bytes = static_cast<size_t>(recv_sizes[pr]) *
                           sizeof(double);
            if (bytes >= threshold)
            {
                pipeline_large_receives = true;
                break;
            }
        }
    }

    if (pcfrag_taskflow_async_pieces && !recv_reqs.empty())
    {
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW async partner receive has no state.");
        taskflow_state->producer_exchange_pending = 1;
        taskflow_state->producer_ksupc = static_cast<int>(ksupc);
        dSymV2PcFragTaskflowEnsureVectorCapacity(
            taskflow_state->producer_partner_recv_reqs,
            recv_reqs.size());
        dSymV2PcFragTaskflowEnsureVectorCapacity(
            taskflow_state->producer_partner_recv_prs,
            recv_request_peers.size());
        dSymV2PcFragTaskflowEnsureVectorCapacity(
            taskflow_state->producer_partner_recv_sizes,
            recv_request_peers.size());
        dSymV2PcFragTaskflowEnsureVectorCapacity(
            taskflow_state->producer_partner_recv_offsets,
            recv_request_peers.size());
        dSymV2PcFragTaskflowEnsureVectorCapacity(
            taskflow_state->producer_partner_recv_done,
            recv_reqs.size());
        taskflow_state->producer_partner_recv_reqs = recv_reqs;
        taskflow_state->producer_partner_recv_prs = recv_request_peers;
        taskflow_state->producer_partner_recv_sizes.clear();
        taskflow_state->producer_partner_recv_offsets.clear();
        taskflow_state->producer_partner_recv_sizes.reserve(
            recv_request_peers.size());
        taskflow_state->producer_partner_recv_offsets.reserve(
            recv_request_peers.size());
        for (size_t i = 0; i < recv_request_peers.size(); ++i)
        {
            int pr = recv_request_peers[i];
            if (pr < 0 || pr >= Pr)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW async partner peer is invalid.");
            taskflow_state->producer_partner_recv_sizes.push_back(
                recv_sizes[static_cast<size_t>(pr)]);
            taskflow_state->producer_partner_recv_offsets.push_back(
                recv_offsets[static_cast<size_t>(pr)]);
        }
        taskflow_state->producer_partner_recv_done.assign(
            recv_reqs.size(), 0);
        taskflow_state->producer_partner_recv_remaining =
            static_cast<int>(recv_reqs.size());
        dSymV2PcFragTaskflowEnsureProgressScratch(
            *taskflow_state, symV2PcFragTaskflowStats, recv_reqs.size());
        recv_reqs.clear();
        recv_request_peers.clear();
        recv_h2d_issued = true;
        dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
    }

    if (!recv_reqs.empty())
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double recv_wait_t = SuperLU_timer_();
        double progressive_h2d_issue = 0.0;
#endif
        const bool taskflow_progressive_receives =
            pcfrag_taskflow && !recv_reqs.empty();
        if (!pipeline_large_receives && !taskflow_progressive_receives)
        {
            if (pcfrag_taskflow)
            {
                ++symV2PcFragTaskflowStats.producer_recv_wait_calls;
                symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                    static_cast<long long>(recv_reqs.size());
            }
            MPI_Waitall(static_cast<int>(recv_reqs.size()),
                        recv_reqs.data(),
                        MPI_STATUSES_IGNORE);
        }
        else
        {
            /* SymFact V2 progressive large receive staging: completed peers
               enter H2D DMA while MPI is still receiving the other peers. */
            const int request_count = static_cast<int>(recv_reqs.size());
            if (recv_request_peers.size() != recv_reqs.size())
                ABORT("SymFact V2 large receive peer map is inconsistent.");
            if (wait_indices.size() < recv_reqs.size())
                wait_indices.resize(recv_reqs.size());
            if (wait_statuses.size() < recv_reqs.size())
                wait_statuses.resize(recv_reqs.size());

            int remaining = request_count;
            while (remaining > 0)
            {
                int completed = 0;
                if (pcfrag_taskflow)
                    ++symV2PcFragTaskflowStats.producer_recv_wait_calls;
                int mpi_rc = MPI_Waitsome(
                    request_count, recv_reqs.data(), &completed,
                    wait_indices.data(), wait_statuses.data());
                if (mpi_rc != MPI_SUCCESS || completed == MPI_UNDEFINED ||
                    completed <= 0 || completed > remaining)
                    ABORT("SymFact V2 progressive receive wait failed.");
                if (pcfrag_taskflow)
                    symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                        static_cast<long long>(completed);

                for (int item = 0; item < completed; ++item)
                {
                    int req_index = wait_indices[static_cast<size_t>(item)];
                    if (req_index < 0 || req_index >= request_count)
                        ABORT("SymFact V2 progressive receive index is invalid.");
                    int pr = recv_request_peers[
                        static_cast<size_t>(req_index)];
                    if (pr < 0 || pr >= Pr || recv_sizes[pr] <= 0 ||
                        recv_offsets[pr] < 0)
                        ABORT("SymFact V2 progressive receive metadata is invalid.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double h2d_issue_t = SuperLU_timer_();
#endif
                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symPartnerLStageBufs[stream_offset] +
                            recv_offsets[pr],
                        recv_host_base + recv_offsets[pr],
                        sizeof(double) *
                            static_cast<size_t>(recv_sizes[pr]),
                        cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    progressive_h2d_issue +=
                        SuperLU_timer_() - h2d_issue_t;
#endif
                    if (pcfrag_taskflow)
                    {
                        size_t recv_map_pos =
                            recv_count_base + static_cast<size_t>(pr);
                        if (recv_map_pos >= symV2PartnerLRecvMap.size())
                            ABORT("GPU3DV2_PCFRAG_TASKFLOW progressive partner receive map is missing.");
                        taskflow_assemble_owned_pieces(
                            SYM_V2_PCFRAG_PIECE_PARTNER,
                            A_gpu.symPartnerLStageBufs[stream_offset] +
                                recv_offsets[pr],
                            symV2PartnerLRecvMap[recv_map_pos]);
                        (*taskflow_partner_progressive_assembled)[
                            static_cast<size_t>(pr)] = 1;
                    }
                }
                remaining -= completed;
            }
            recv_h2d_issued = true;
        }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAddBoth(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                         SYM_GPU3D_T_PARTNER_LFRAG_MPI_RECV_WAIT,
                         SuperLU_timer_() - recv_wait_t);
        if (progressive_h2d_issue > 0.0)
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                             SYM_GPU3D_T_PARTNER_LFRAG_H2D_STAGE_ISSUE,
                             progressive_h2d_issue);
#endif
    }

    if (static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 true symmetric L-fragment cached index is missing.");
    const std::vector<int_t> &cached_index = symV2PartnerLRecvIndex[k];
    if (recv_total > 0 && cached_index.empty())
        ABORT("SymFact V2 true symmetric received L fragments without cached metadata.");

    int_t frag_nblocks = cached_index.empty() ? 0 : cached_index[0];
    int_t frag_nrows = cached_index.empty() ? 0 : cached_index[1];
    int_t frag_index_size = static_cast<int_t>(cached_index.size());
    if (frag_index_size > maxSymPartnerLidxCount)
        ABORT("SymFact V2 true symmetric cached L-fragment index exceeds buffer.");
    if (static_cast<int64_t>(frag_nrows) * static_cast<int64_t>(ksupc) >
        static_cast<int64_t>(maxSymPartnerLvalCount))
        ABORT("SymFact V2 true symmetric L-fragment values exceed receive buffer.");

    int_t empty_header[LPANEL_HEADER_SIZE] = {0, 0, 0, ksupc};
    if (A_gpu.symPartnerLidxRecvBufs[stream_offset] == NULL)
        ABORT("SymFact V2 true symmetric L-fragment device index buffer is missing.");
    if (!pcfrag_taskflow || pcfrag_taskflow_validate)
    {
        if (cached_index.empty())
        {
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symPartnerLidxRecvBufs[stream_offset], empty_header,
                sizeof(int_t) * LPANEL_HEADER_SIZE,
                cudaMemcpyHostToDevice, stream));
        }
        else
        {
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symPartnerLidxRecvBufs[stream_offset],
                cached_index.data(),
                sizeof(int_t) *
                    static_cast<size_t>(frag_index_size),
                cudaMemcpyHostToDevice, stream));
        }
    }

    if (!cuda_aware && !recv_reqs.empty() && !recv_h2d_issued)
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double h2d_issue_t = SuperLU_timer_();
#endif
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLStageBufs[stream_offset], recv_host_base,
            sizeof(double) * static_cast<size_t>(recv_total),
            cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAddBoth(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                         SYM_GPU3D_T_PARTNER_LFRAG_H2D_STAGE_ISSUE,
                         SuperLU_timer_() - h2d_issue_t);
#endif
    }

    if (frag_nblocks > 0 && frag_nrows > 0)
    {
        if (A_gpu.symPartnerLStageBufs[stream_offset] == NULL ||
            ((!pcfrag_taskflow || pcfrag_taskflow_validate) &&
             A_gpu.symPartnerLvalRecvBufs[stream_offset] == NULL))
            ABORT("SymFact V2 true symmetric L-fragment device buffers are missing.");
        if (!pcfrag_taskflow || pcfrag_taskflow_validate)
            gpuErrchk(cudaMemsetAsync(A_gpu.symPartnerLvalRecvBufs[stream_offset], 0,
                                      sizeof(double) *
                                          static_cast<size_t>(frag_nrows) *
                                          static_cast<size_t>(ksupc),
                                      stream));

        for (int pr = 0; pr < Pr; ++pr)
        {
            int count = recv_sizes[pr];
            if (count <= 0)
                continue;
            const std::vector<int_t> &recv_map =
                symV2PartnerLRecvMap[recv_count_base +
                                      static_cast<size_t>(pr)];
            int src = PNUM(pr, kcol_, grid);
            double *stage = NULL;
            if (src == iam)
            {
                if (mycol != kcol_)
                    ABORT("SymFact V2 self fragment has an invalid source column.");
                long long self_bytes =
                    static_cast<long long>(count) *
                    static_cast<long long>(sizeof(double));
                sym_v2_partner_payload_bytes += self_bytes;
                symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_SELF,
                                       self_bytes);
                int_t send_lk = symV2PanelIndex(k);
                size_t self_flat =
                    static_cast<size_t>(send_lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(mycol);
                if (exact_partner_fragment_demand)
                {
                    size_t active_pos =
                        self_flat * static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLExactSendBufsGPU.size() ||
                        active_pos >= symV2PartnerLExactSendSizes.size() ||
                        symV2PartnerLExactSendBufsGPU[active_pos] == NULL ||
                        symV2PartnerLExactSendSizes[active_pos] != count)
                        ABORT("SymFact V2 exact self fragment buffer is invalid.");
                    stage = symV2PartnerLExactSendBufsGPU[active_pos];
                }
                else
                {
                    if (self_flat >= symV2PartnerLSendSizes.size() ||
                        symV2PartnerLSendSizes[self_flat] != count)
                        ABORT("SymFact V2 self fragment buffer is invalid.");
                    stage = partner_send_buffer(self_flat, count);
                    if (stage == NULL)
                        ABORT("SymFact V2 self fragment buffer is invalid.");
                }
            }
            else
            {
                if (pcfrag_taskflow_async_pieces)
                    continue;
                stage = A_gpu.symPartnerLStageBufs[stream_offset] +
                        recv_offsets[pr];
            }

            size_t pos = 0;
            size_t end = static_cast<size_t>(count);
            if (recv_map.size() % 3 != 0)
                ABORT("SymFact V2 true symmetric L-fragment receive map has invalid stride.");
            int pieces = static_cast<int>(recv_map.size() / 3);
            int_t *recv_map_gpu =
                symV2PartnerLRecvMapsGPU[recv_count_base +
                                          static_cast<size_t>(pr)];
            if (pieces > 0 && recv_map_gpu == NULL)
                ABORT("SymFact V2 true symmetric L-fragment device receive map is missing.");
            size_t map_pos = 0;
            while (map_pos < recv_map.size())
            {
                if (map_pos + 3 > recv_map.size())
                    ABORT("SymFact V2 true symmetric L-fragment receive map is truncated.");
                int_t dst_offset = recv_map[map_pos++];
                int_t nrows = recv_map[map_pos++];
                int_t src_offset = recv_map[map_pos++];
                if (dst_offset < 0 || nrows < 0 ||
                    dst_offset + nrows > frag_nrows)
                    ABORT("SymFact V2 true symmetric L-fragment receive map is invalid.");
                size_t need = static_cast<size_t>(nrows) *
                              static_cast<size_t>(ksupc);
                if (src_offset < 0 ||
                    static_cast<size_t>(src_offset) + need > end)
                    ABORT("SymFact V2 true symmetric L-fragment buffer is truncated.");
                pos += need;
            }
            if (pieces > 0)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double assemble_issue_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow)
                {
                    if (taskflow_partner_progressive_assembled == NULL ||
                        static_cast<size_t>(pr) >=
                            taskflow_partner_progressive_assembled->size() ||
                        !(*taskflow_partner_progressive_assembled)[
                            static_cast<size_t>(pr)])
                    {
                        taskflow_assemble_owned_pieces(
                            SYM_V2_PCFRAG_PIECE_PARTNER, stage, recv_map);
                    }
                }
                if (!pcfrag_taskflow || pcfrag_taskflow_validate)
                {
                    sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                        stage, A_gpu.symPartnerLvalRecvBufs[stream_offset],
                        recv_map_gpu, pieces, ksupc, frag_nrows);
                    gpuErrchk(cudaGetLastError());
                }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
                                 SYM_GPU3D_T_PARTNER_LFRAG_ASSEMBLE_ISSUE,
                                 SuperLU_timer_() - assemble_issue_t);
#endif
            }
            if (pos != end)
                ABORT("SymFact V2 true symmetric L-fragment buffer has extra data.");
        }
    }

    if (pc_fragment_schur)
    {
        if (!pcfrag_async_pipeline && !pcfrag_taskflow_async_pieces &&
            !send_reqs.empty())
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double send_wait_t = SuperLU_timer_();
#endif
            if (pcfrag_taskflow)
            {
                ++symV2PcFragTaskflowStats.producer_send_wait_calls;
                symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                    static_cast<long long>(send_reqs.size());
            }
            MPI_Waitall(static_cast<int>(send_reqs.size()),
                        send_reqs.data(), MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                             SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
                             SuperLU_timer_() - send_wait_t);
#endif
            send_reqs.clear();
        } // SYM_V2_PC2_ASYNC_PIPELINE_DEFER_PARTNER_SEND_WAIT

        if (static_cast<size_t>(k) >= symV2RowFragRecvIndex.size())
            ABORT("SymFact V2 row-fragment cached index is missing.");
        const std::vector<int_t> &row_index = symV2RowFragRecvIndex[k];
        int_t row_index_size = row_index.empty()
                                   ? LPANEL_HEADER_SIZE
                                   : static_cast<int_t>(row_index.size());
        if (row_index_size > maxSymV2RowFragIdxRecvCount)
            ABORT("SymFact V2 row-fragment index exceeds device buffer.");
        if (A_gpu.symV2RowFragIdxRecvBufs[stream_offset] == NULL ||
            A_gpu.symV2RowFragValRecvBufs[stream_offset] == NULL ||
            A_gpu.symV2RowFragStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment GPU buffers are missing.");

        if (!superlu_sym_v2_rowfrag_destination_path())
        {
            int_t row_send_lk = symV2PanelIndex(k);
            if (mycol == kcol_)
            {
                if (row_send_lk < 0)
                    ABORT("SymFact V2 row-fragment source panel is invalid.");
                xlpanel_t<double> &row_lpanel = lPanelVec[row_send_lk];

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_pack_issue_t = SuperLU_timer_();
#endif
	                bool row_packed_any = false;
                    if (exact_row_fragment_demand)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size() ||
                                    active_pos >= symV2RowFragExactSendSizes.size() ||
                                    active_pos >= symV2RowFragExactSendBufsGPU.size() ||
                                    active_pos >= symV2RowFragExactSendMapsGPU.size())
                                    ABORT("SymFact V2 exact row-fragment send slot is invalid.");
                                if (!symV2RowFragSendActive[active_pos])
                                    continue;
                                int size =
                                    symV2RowFragExactSendSizes[active_pos];
                                if (size <= 0)
                                    ABORT("SymFact V2 exact row-fragment active send has no data.");
                                if (row_lpanel.isEmpty())
                                    ABORT("SymFact V2 row-fragment source L panel is missing.");
                                double *sendbuf =
                                    symV2RowFragExactSendBufsGPU[active_pos];
                                int_t *sendmap =
                                    symV2RowFragExactSendMapsGPU[active_pos];
                                if (sendbuf == NULL || sendmap == NULL)
                                    ABORT("SymFact V2 exact row-fragment send buffer is missing.");

                                int threads = 256;
                                int blocks = (size + threads - 1) / threads;
                                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                                    row_lpanel.gpuPanel.val, sendbuf, sendmap, size);
                                row_packed_any = true;
                            }
                        }
                    }
                    else
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            if (flat >= symV2PartnerLSendSizes.size() ||
                                flat >= symV2PartnerLSendBufsGPU.size() ||
                                flat >= symL2LSendMapsGPU.size())
                                ABORT("SymFact V2 row-fragment send metadata is missing.");
                            int size = symV2PartnerLSendSizes[flat];
                            bool active_dest = false;
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size())
                                    ABORT("SymFact V2 row-fragment send mask is missing.");
                                if (symV2RowFragSendActive[active_pos])
                                {
                                    active_dest = true;
                                    break;
                                }
                            }
                            if (!active_dest)
                                continue;
                            if (size <= 0)
                                ABORT("SymFact V2 row-fragment active send has no data.");
                            if (row_lpanel.isEmpty())
                                ABORT("SymFact V2 row-fragment source L panel is missing.");

                            double *sendbuf = symV2PartnerLSendBufsGPU[flat];
                            int_t *sendmap = symL2LSendMapsGPU[flat];
                            if (sendbuf == NULL || sendmap == NULL)
                                ABORT("SymFact V2 row-fragment send buffer is missing.");

                            int threads = 256;
                            int blocks = (size + threads - 1) / threads;
                            sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                                row_lpanel.gpuPanel.val, sendbuf, sendmap, size);
                            row_packed_any = true;
                        }
                    }
                if (row_packed_any)
                    gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
                                 SuperLU_timer_() - row_pack_issue_t);
#endif

                bool row_d2h_any = false;
                if (row_packed_any)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_d2h_issue_t = SuperLU_timer_();
#endif
                    if (exact_row_fragment_demand)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size() ||
                                    active_pos >= symV2RowFragExactSendSizes.size())
                                    ABORT("SymFact V2 exact row-fragment send slot is invalid.");
                                if (!symV2RowFragSendActive[active_pos] ||
                                    pc_dest == mycol)
                                    continue;
                                int size =
                                    symV2RowFragExactSendSizes[active_pos];
                                double *host_stage =
                                    (active_pos <
                                         symV2RowFragExactHostSendBufsPinned.size() &&
                                     symV2RowFragExactHostSendBufsPinned[active_pos] !=
                                         NULL)
                                        ? symV2RowFragExactHostSendBufsPinned[active_pos]
                                        : (symV2RowFragExactHostSendBufs[active_pos].empty()
                                               ? NULL
                                               : symV2RowFragExactHostSendBufs[active_pos].data());
                                if (size <= 0 || host_stage == NULL ||
                                    symV2RowFragExactSendBufsGPU[active_pos] == NULL)
                                    ABORT("SymFact V2 exact row-fragment host send buffer is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                                symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                                           static_cast<long long>(size) *
                                               static_cast<long long>(sizeof(double)));
#endif
                                symV2PayloadProfileAdd(
                                    SYM_V2_PAYLOAD_ROWFRAG_HOST_STAGING,
                                    static_cast<long long>(size) *
                                        static_cast<long long>(sizeof(double)));
                                gpuErrchk(cudaMemcpyAsync(
                                    host_stage,
                                    symV2RowFragExactSendBufsGPU[active_pos],
                                    sizeof(double) * static_cast<size_t>(size),
                                    cudaMemcpyDeviceToHost, stream));
                                row_d2h_any = true;
                            }
                        }
                    }
                    else
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            int size = symV2PartnerLSendSizes[flat];
                            if (size <= 0)
                                continue;
                            bool active_remote_dest = false;
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size())
                                    ABORT("SymFact V2 row-fragment send mask is missing.");
                                if (symV2RowFragSendActive[active_pos] &&
                                    pc_dest != mycol)
                                {
                                    active_remote_dest = true;
                                    break;
                                }
                            }
                            if (!active_remote_dest)
                                continue;
                            double *host_stage =
                                (flat < symV2PartnerLHostSendBufsPinned.size() &&
                                 symV2PartnerLHostSendBufsPinned[flat] != NULL)
                                    ? symV2PartnerLHostSendBufsPinned[flat]
                                    : (symV2PartnerLHostSendBufs[flat].empty()
                                           ? NULL
                                           : symV2PartnerLHostSendBufs[flat].data());
                            if (host_stage == NULL)
                                ABORT("SymFact V2 row-fragment host send buffer is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                            symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                                       static_cast<long long>(size) *
                                           static_cast<long long>(sizeof(double)));
#endif
                            symV2PayloadProfileAdd(
                                SYM_V2_PAYLOAD_ROWFRAG_HOST_STAGING,
                                static_cast<long long>(size) *
                                    static_cast<long long>(sizeof(double)));
                            gpuErrchk(cudaMemcpyAsync(
                                host_stage, symV2PartnerLSendBufsGPU[flat],
                                sizeof(double) * static_cast<size_t>(size),
                                cudaMemcpyDeviceToHost, stream));
                            row_d2h_any = true;
                        }
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                                     SYM_GPU3D_T_ROW_LFRAG_D2H_STAGE_ISSUE,
                                     SuperLU_timer_() - row_d2h_issue_t);
#endif
                }
                if (row_packed_any || row_d2h_any)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_pack_stage_sync_t = SuperLU_timer_();
#endif
                    if (pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_async_core())
                        ++symV2PcFragTaskflowStats.producer_exchange_stream_syncs;
                    gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                                     SYM_GPU3D_T_ROW_LFRAG_PACK_STAGE_SYNC,
                                     SuperLU_timer_() - row_pack_stage_sync_t);
#endif
                }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_send_post_t = SuperLU_timer_();
#endif
                    if (exact_row_fragment_demand)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size() ||
                                    active_pos >= symV2RowFragExactSendSizes.size())
                                    ABORT("SymFact V2 exact row-fragment send slot is invalid.");
                                if (!symV2RowFragSendActive[active_pos] ||
                                    pc_dest == mycol)
                                    continue;
                                int size =
                                    symV2RowFragExactSendSizes[active_pos];
                                double *hostbuf =
                                    (active_pos <
                                         symV2RowFragExactHostSendBufsPinned.size() &&
                                     symV2RowFragExactHostSendBufsPinned[active_pos] !=
                                         NULL)
                                        ? symV2RowFragExactHostSendBufsPinned[active_pos]
                                        : (symV2RowFragExactHostSendBufs[active_pos].empty()
                                               ? NULL
                                               : symV2RowFragExactHostSendBufs[active_pos].data());
                                if (size <= 0 || hostbuf == NULL)
                                    ABORT("SymFact V2 exact row-fragment host send buffer is missing.");
                                MPI_Request req;
                                long long send_bytes =
                                    static_cast<long long>(size) *
                                    static_cast<long long>(sizeof(double));
                                sym_v2_rowfrag_payload_bytes += send_bytes;
                                symV2PayloadProfileAdd(
                                    SYM_V2_PAYLOAD_ROWFRAG_MPI_SEND,
                                    send_bytes);
                                MPI_Isend(hostbuf, size, MPI_DOUBLE, pc_dest,
                                          SLU_MPI_TAG(5, k),
                                          grid3d->rscp.comm, &req);
                                send_reqs.push_back(req);
                            }
                        }
                    }
                    else
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(row_send_lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            int size = symV2PartnerLSendSizes[flat];
                            if (size <= 0)
                                continue;
                            double *hostbuf =
                                (flat < symV2PartnerLHostSendBufsPinned.size() &&
                                 symV2PartnerLHostSendBufsPinned[flat] != NULL)
                                    ? symV2PartnerLHostSendBufsPinned[flat]
                                    : (symV2PartnerLHostSendBufs[flat].empty()
                                           ? NULL
                                           : symV2PartnerLHostSendBufs[flat].data());
                            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (active_pos >= symV2RowFragSendActive.size())
                                    ABORT("SymFact V2 row-fragment send mask is missing.");
                                if (!symV2RowFragSendActive[active_pos] ||
                                    pc_dest == mycol)
                                    continue;
                                if (hostbuf == NULL)
                                    ABORT("SymFact V2 row-fragment host send buffer is missing.");
                                MPI_Request req;
                                long long send_bytes =
                                    static_cast<long long>(size) *
                                    static_cast<long long>(sizeof(double));
                                sym_v2_rowfrag_payload_bytes += send_bytes;
                                symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_MPI_SEND,
                                                       send_bytes);
                                MPI_Isend(hostbuf, size, MPI_DOUBLE, pc_dest,
                                          SLU_MPI_TAG(5, k), grid3d->rscp.comm, &req);
                                send_reqs.push_back(req);
                            }
                        }
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_POST,
                                 SYM_GPU3D_T_ROW_LFRAG_SEND_POST,
                                 SuperLU_timer_() - row_send_post_t);
                if (send_reqs.size() < deferred_partner_send_req_count)
                    ABORT("SymFact V2 row send request accounting is invalid.");
                symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS,
                           static_cast<long long>(
                               send_reqs.size() -
                               deferred_partner_send_req_count));
#endif
            }

            if (!pcfrag_taskflow || pcfrag_taskflow_validate)
            {
                if (row_index.empty())
                {
                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
                        empty_header, sizeof(int_t) * LPANEL_HEADER_SIZE,
                        cudaMemcpyHostToDevice, stream));
                }
                else
                {
                    if (row_index[3] != ksupc)
                        ABORT("SymFact V2 row-fragment index has wrong panel width.");
                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
                        row_index.data(),
                        sizeof(int_t) * static_cast<size_t>(row_index_size),
                        cudaMemcpyHostToDevice, stream));
                }
            }

            if (!row_index.empty())
            {
                int_t row_nrows = row_index[1];
                if (row_nrows <= 0 ||
                    static_cast<int64_t>(row_nrows) *
                            static_cast<int64_t>(ksupc) >
                        static_cast<int64_t>(maxSymV2RowFragValRecvCount))
                    ABORT("SymFact V2 row-fragment value buffer is too small.");

                std::vector<int> row_recv_offsets(static_cast<size_t>(Pc), -1);
                int row_recv_total = 0;
                int row_src_pc = static_cast<int>(kcol_);
                size_t row_recv_base =
                    static_cast<size_t>(k) * static_cast<size_t>(Pc);
                if (row_recv_base + static_cast<size_t>(Pc) >
                        symV2RowFragRecvSizes.size() ||
                    row_recv_base + static_cast<size_t>(Pc) >
                        symV2RowFragRecvMap.size())
                    ABORT("SymFact V2 row-fragment receive metadata is missing.");
                for (int pc = 0; pc < Pc; ++pc)
                {
                    int count = symV2RowFragRecvSizes[
                        row_recv_base + static_cast<size_t>(pc)];
                    if (count <= 0)
                        continue;
                    if (row_src_pc != mycol)
                    {
                        row_recv_offsets[static_cast<size_t>(pc)] =
                            row_recv_total;
                        row_recv_total += count;
                    }
                }
                if (row_recv_total > maxSymV2RowFragStageCount)
                    ABORT("SymFact V2 row-fragment receive exceeds staging buffer.");
                double *row_recv_host_base = NULL;
                std::vector<MPI_Request> row_recv_reqs;
                std::vector<int> row_recv_request_pcs;
                std::vector<unsigned char> row_recv_progressive_assembled(
                    static_cast<size_t>(Pc), 0);
                bool row_recv_h2d_issued = false;
                if (row_recv_total > 0)
                {
                    if (static_cast<size_t>(stream_offset) >=
                            symV2RowFragHostRecvBufs.size() ||
                        symV2RowFragHostRecvBufs[stream_offset] == NULL)
                        ABORT("SymFact V2 row-fragment host receive staging is missing.");
                    row_recv_host_base = symV2RowFragHostRecvBufs[stream_offset];
                    row_recv_reqs.reserve(static_cast<size_t>(Pc));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_recv_post_t = SuperLU_timer_();
#endif
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        int count = symV2RowFragRecvSizes[
                            row_recv_base + static_cast<size_t>(pc)];
                        if (count <= 0)
                            continue;
                        int offset = row_recv_offsets[static_cast<size_t>(pc)];
                        if (offset < 0)
                            continue;
                        MPI_Request req;
                        long long recv_bytes =
                            static_cast<long long>(count) *
                            static_cast<long long>(sizeof(double));
                        sym_v2_rowfrag_payload_bytes += recv_bytes;
                        symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_MPI_RECV,
                                               recv_bytes);
                        MPI_Irecv(row_recv_host_base + offset, count,
                                  MPI_DOUBLE, row_src_pc, SLU_MPI_TAG(5, k),
                                  grid3d->rscp.comm, &req);
                        row_recv_reqs.push_back(req);
                        row_recv_request_pcs.push_back(pc);
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_RECV_POST,
                                     SYM_GPU3D_T_ROW_LFRAG_RECV_POST,
                                     SuperLU_timer_() - row_recv_post_t);
#endif
                }
                if (!row_recv_reqs.empty())
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_recv_wait_t = SuperLU_timer_();
                    double row_progressive_h2d_issue = 0.0;
#endif
                    if (!pcfrag_taskflow)
                    {
                        MPI_Waitall(static_cast<int>(row_recv_reqs.size()),
                                    row_recv_reqs.data(),
                                    MPI_STATUSES_IGNORE);
                    }
                    else
                    {
                        const int request_count =
                            static_cast<int>(row_recv_reqs.size());
                        if (row_recv_request_pcs.size() !=
                            row_recv_reqs.size())
                            ABORT("GPU3DV2_PCFRAG_TASKFLOW row receive request map is inconsistent.");
                        if (wait_indices.size() < row_recv_reqs.size())
                            wait_indices.resize(row_recv_reqs.size());
                        if (wait_statuses.size() < row_recv_reqs.size())
                            wait_statuses.resize(row_recv_reqs.size());

                        int remaining = request_count;
                        while (remaining > 0)
                        {
                            int completed = 0;
                            ++symV2PcFragTaskflowStats.producer_recv_wait_calls;
                            int mpi_rc = MPI_Waitsome(
                                request_count, row_recv_reqs.data(),
                                &completed, wait_indices.data(),
                                wait_statuses.data());
                            if (mpi_rc != MPI_SUCCESS ||
                                completed == MPI_UNDEFINED ||
                                completed <= 0 || completed > remaining)
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW row progressive receive wait failed.");
                            symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                                static_cast<long long>(completed);

                            for (int item = 0; item < completed; ++item)
                            {
                                int req_index =
                                    wait_indices[static_cast<size_t>(item)];
                                if (req_index < 0 ||
                                    req_index >= request_count)
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row progressive receive index is invalid.");
                                int pc = row_recv_request_pcs[
                                    static_cast<size_t>(req_index)];
                                if (pc < 0 || pc >= Pc)
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row progressive receive pc is invalid.");
                                size_t row_pos =
                                    row_recv_base + static_cast<size_t>(pc);
                                int count = symV2RowFragRecvSizes[row_pos];
                                int offset = row_recv_offsets[
                                    static_cast<size_t>(pc)];
                                if (count <= 0 || offset < 0)
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row progressive receive metadata is invalid.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                                double h2d_issue_t = SuperLU_timer_();
#endif
                                gpuErrchk(cudaMemcpyAsync(
                                    A_gpu.symV2RowFragStageBufs[stream_offset] +
                                        offset,
                                    row_recv_host_base + offset,
                                    sizeof(double) *
                                        static_cast<size_t>(count),
                                    cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                                row_progressive_h2d_issue +=
                                    SuperLU_timer_() - h2d_issue_t;
#endif
                                taskflow_assemble_owned_pieces(
                                    SYM_V2_PCFRAG_PIECE_ROW,
                                    A_gpu.symV2RowFragStageBufs[stream_offset] +
                                        offset,
                                    symV2RowFragRecvMap[row_pos]);
                                row_recv_progressive_assembled[
                                    static_cast<size_t>(pc)] = 1;
                            }
                            remaining -= completed;
                        }
                        row_recv_h2d_issued = true;
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                                     SYM_GPU3D_T_ROW_LFRAG_MPI_RECV_WAIT,
                                     SuperLU_timer_() - row_recv_wait_t);
                    if (row_progressive_h2d_issue > 0.0)
                        symTimingAddBoth(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                                         SYM_GPU3D_T_ROW_LFRAG_H2D_STAGE_ISSUE,
                                         row_progressive_h2d_issue);
#endif
                }
                if (row_recv_total > 0 && !row_recv_h2d_issued)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_h2d_issue_t = SuperLU_timer_();
#endif
                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symV2RowFragStageBufs[stream_offset],
                        row_recv_host_base,
                        sizeof(double) * static_cast<size_t>(row_recv_total),
                        cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                                     SYM_GPU3D_T_ROW_LFRAG_H2D_STAGE_ISSUE,
                                     SuperLU_timer_() - row_h2d_issue_t);
#endif
                }

                if (!pcfrag_taskflow || pcfrag_taskflow_validate)
                    gpuErrchk(cudaMemsetAsync(
                        A_gpu.symV2RowFragValRecvBufs[stream_offset], 0,
                        sizeof(double) * static_cast<size_t>(row_nrows) *
                            static_cast<size_t>(ksupc),
                        stream));

                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t row_pos = row_recv_base + static_cast<size_t>(pc);
                    int count = symV2RowFragRecvSizes[row_pos];
                    if (count <= 0)
                        continue;
                    const std::vector<int_t> &row_map =
                        symV2RowFragRecvMap[row_pos];
                    if (row_map.empty() || row_map.size() % 3 != 0)
                        ABORT("SymFact V2 row-fragment receive map is invalid.");
                    int pieces = static_cast<int>(row_map.size() / 3);
                    int_t *row_map_gpu = symV2RowFragRecvMapsGPU[row_pos];
                    if (row_map_gpu == NULL)
                        ABORT("SymFact V2 row-fragment device receive map is missing.");

                    double *stage = NULL;
                    if (row_src_pc == mycol)
                    {
                        if (row_send_lk < 0)
                            ABORT("SymFact V2 row-fragment self source panel is invalid.");
	                        size_t self_flat =
	                            static_cast<size_t>(row_send_lk) *
	                                static_cast<size_t>(Pc) +
	                            static_cast<size_t>(pc);
                        if (exact_row_fragment_demand)
                        {
                            size_t active_pos =
                                self_flat * static_cast<size_t>(Pc) +
                                static_cast<size_t>(mycol);
                            if (active_pos >= symV2RowFragExactSendBufsGPU.size() ||
                                active_pos >= symV2RowFragExactSendSizes.size() ||
                                symV2RowFragExactSendBufsGPU[active_pos] == NULL ||
                                symV2RowFragExactSendSizes[active_pos] != count)
                                ABORT("SymFact V2 exact row-fragment self L buffer is invalid.");
                            stage = symV2RowFragExactSendBufsGPU[active_pos];
                        }
                        else
                        {
                            if (self_flat >= symV2PartnerLSendBufsGPU.size() ||
                                self_flat >= symV2PartnerLSendSizes.size() ||
                                symV2PartnerLSendBufsGPU[self_flat] == NULL ||
                                symV2PartnerLSendSizes[self_flat] != count)
                                ABORT("SymFact V2 row-fragment self L buffer is invalid.");
                            stage = symV2PartnerLSendBufsGPU[self_flat];
                        }
                        long long self_bytes =
                            static_cast<long long>(count) *
                            static_cast<long long>(sizeof(double));
                        sym_v2_rowfrag_payload_bytes += self_bytes;
                        symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_SELF,
                                               self_bytes);
                    }
                    else
                    {
                        int offset = row_recv_offsets[static_cast<size_t>(pc)];
                        if (offset < 0)
                            ABORT("SymFact V2 row-fragment remote offset is invalid.");
                        stage = A_gpu.symV2RowFragStageBufs[stream_offset] +
                                offset;
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_assemble_issue_t = SuperLU_timer_();
#endif
                    if (pcfrag_taskflow)
                    {
                        if (static_cast<size_t>(pc) >=
                                row_recv_progressive_assembled.size() ||
                            !row_recv_progressive_assembled[
                                static_cast<size_t>(pc)])
                        {
                            taskflow_assemble_owned_pieces(
                                SYM_V2_PCFRAG_PIECE_ROW, stage, row_map);
                        }
                    }
                    if (!pcfrag_taskflow || pcfrag_taskflow_validate)
                    {
                        sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                            stage, A_gpu.symV2RowFragValRecvBufs[stream_offset],
                            row_map_gpu, pieces, ksupc, row_nrows);
                        gpuErrchk(cudaGetLastError());
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
                                     SYM_GPU3D_T_ROW_LFRAG_ASSEMBLE_ISSUE,
                                     SuperLU_timer_() - row_assemble_issue_t);
#endif
                }
            }
        }
        else
        {
        int_t row_send_lk = symV2PanelIndex(k);
        xlpanel_t<double> *row_lpanel_ptr = NULL;
        if (mycol == kcol_)
        {
            if (row_send_lk < 0)
                ABORT("SymFact V2 row-fragment source panel is invalid.");
            row_lpanel_ptr = &lPanelVec[row_send_lk];
        }

        std::vector<int> row_dest_offsets(static_cast<size_t>(Pc), -1);
// SYM_V2_PC2_PHASE2_ROW_COUNT_DEST_BEGIN
        auto row_count_destination = [&](int pc_dest) -> int
        {
// SYM_V2_PC2_PHASE4_ROW_COUNT_PLAN_BRANCH_BEGIN
            if (superlu_sym_v2_row_l_plan_v2_exchange())
            {
                size_t slot =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (slot >= symV2RowDownSendSizes.size())
                    ABORT("SymFact V2 row-down plan send size is missing.");
                return symV2RowDownSendSizes[slot];
            }
// SYM_V2_PC2_PHASE4_ROW_COUNT_PLAN_BRANCH_END

            if (mycol != kcol_)
                return 0;
            if (row_l_direct_recv)
            {
                size_t slot =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (slot >= symV2RowDirectSendSizes.size())
                    ABORT("SymFact V2 direct row-L count metadata is missing.");
                return symV2RowDirectSendSizes[slot];
            }

            int total = 0;
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc);
                size_t active_pos =
                    flat * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (active_pos >= symV2RowFragSendActive.size())
                    ABORT("SymFact V2 row-fragment send mask is missing.");
                if (!symV2RowFragSendActive[active_pos])
                    continue;
                int size = 0;
                if (exact_row_fragment_demand)
                {
                    if (active_pos >= symV2RowFragExactSendSizes.size())
                        ABORT("SymFact V2 exact row-fragment size is missing.");
                    size = symV2RowFragExactSendSizes[active_pos];
                }
                else
                {
                    if (flat >= symV2PartnerLSendSizes.size())
                        ABORT("SymFact V2 row-fragment size is missing.");
                    size = symV2PartnerLSendSizes[flat];
                }
                if (size < 0 || total > std::numeric_limits<int>::max() - size)
                    ABORT("SymFact V2 row-fragment destination count overflows.");
                total += size;
            }
            return total;
        };
// SYM_V2_PC2_PHASE2_ROW_COUNT_DEST_END
        auto row_pack_destination = [&](int pc_dest,
                                        double *dst_buf) -> int
        {
            if (mycol != kcol_)
                ABORT("SymFact V2 row-fragment pack called on a non-source rank.");
            if (dst_buf == NULL)
                ABORT("SymFact V2 row-fragment destination buffer is missing.");
            if (A_gpu.symV2RowFragStageBufs[stream_offset] == NULL)
                ABORT("SymFact V2 row-fragment device staging buffer is missing.");
            if ((exact_row_fragment_demand || row_l_direct_recv) &&
                A_gpu.symV2RowFragSendMapStageBufs[stream_offset] == NULL)
                ABORT("SymFact V2 row-fragment map staging buffer is missing.");
            const bool exact_map_index =
                exact_row_fragment_demand && superlu_sym_v2_exact_map_index();
// SYM_V2_PC2_PHASE4_ROW_PACK_PLAN_BRANCH_BEGIN
            if (superlu_sym_v2_row_l_plan_v2_exchange())
            {
                size_t slot =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (slot >= symV2RowDownSendSizes.size())
                    ABORT("SymFact V2 row-down plan send slot is invalid.");
                int total = symV2RowDownSendSizes[slot];
                if (total <= 0)
                    return 0;
                if (row_lpanel_ptr == NULL || row_lpanel_ptr->isEmpty())
                    ABORT("SymFact V2 row-down source L panel is missing.");
                if (total > maxSymV2RowFragValSendCount)
                    ABORT("SymFact V2 row-down packed destination exceeds send staging capacity.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_pack_issue_t = SuperLU_timer_();
#endif
                int threads = 256;
// SYM_V2_PC2_LAZY_SENDMAP_GPU_PACK_BEGIN
                if (row_down_lazy_sendmap)
                {
                    if (ksupc <= 0 || total % ksupc != 0)
                        ABORT("SymFact V2 row-down lazy-sendmap total has invalid width.");
                    if (slot >= symV2RowDownSendSegsGPU.size() ||
                        slot >= symV2RowDownSendSegCounts.size())
                        ABORT("SymFact V2 row-down lazy-sendmap slot is invalid.");
                    int nsegments = symV2RowDownSendSegCounts[slot];
                    SymV2RowDownSendSegmentGPU *segments =
                        symV2RowDownSendSegsGPU[slot];
                    if (nsegments <= 0 || segments == NULL ||
                        symL2LSendMapPoolGPU == NULL)
                        ABORT("SymFact V2 row-down lazy-sendmap descriptors are missing.");
                    int_t dst_lda = static_cast<int_t>(total / ksupc);
// SYM_V2_PC2_LAZY_WARP_PACK_BRANCH_BEGIN
                    if (row_down_lazy_warp_pack)
                    {
                        const int warp_size = 32;
                        if (threads % warp_size != 0)
                            ABORT("SymFact V2 row-down lazy warp-pack thread count is invalid.");
                        int warps_per_block = threads / warp_size;
                        int blocks =
                            (nsegments + warps_per_block - 1) /
                            warps_per_block;
                        sym_l2u_pack_segments_warp_kernel<<<blocks, threads, 0, stream>>>(
                            row_lpanel_ptr->gpuPanel.val, dst_buf,
                            symL2LSendMapPoolGPU, segments, nsegments,
                            ksupc, dst_lda);
                    }
                    else
                    {
                        sym_l2u_pack_segments_kernel<<<nsegments, threads, 0, stream>>>(
                            row_lpanel_ptr->gpuPanel.val, dst_buf,
                            symL2LSendMapPoolGPU, segments, nsegments,
                            ksupc, dst_lda);
                    }
// SYM_V2_PC2_LAZY_WARP_PACK_BRANCH_END
                }
                else
                {
                    if (slot >= symV2RowDownSendMapsGPU.size())
                        ABORT("SymFact V2 row-down plan send-map slot is invalid.");
                    int_t *sendmap = symV2RowDownSendMapsGPU[slot];
                    if (sendmap == NULL)
                        ABORT("SymFact V2 row-down send map is missing.");
                    int blocks = (total + threads - 1) / threads;
                    sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                        row_lpanel_ptr->gpuPanel.val, dst_buf, sendmap, total);
                }
// SYM_V2_PC2_LAZY_SENDMAP_GPU_PACK_END
                gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
                                 SuperLU_timer_() - row_pack_issue_t);
#endif
                return total;
            }
// SYM_V2_PC2_PHASE4_ROW_PACK_PLAN_BRANCH_END


            if (row_l_direct_recv)
            {
                size_t slot =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (slot >= symV2RowDirectSendSizes.size() ||
                    slot >= symV2RowDirectSendMapOffsets.size())
                    ABORT("SymFact V2 direct row-L send metadata is missing.");
                int total = symV2RowDirectSendSizes[slot];
                if (total <= 0)
                    return 0;
                if (row_lpanel_ptr == NULL || row_lpanel_ptr->isEmpty())
                    ABORT("SymFact V2 direct row-L source panel is missing.");
                if (total > maxSymV2RowFragStageCount)
                    ABORT("SymFact V2 direct row-L packed destination exceeds staging buffer.");
                if (ldl_native_direct_row)
                {
                    if (slot >= symV2RowDirectSendBlocksHost.size())
                        ABORT("SymFact V2 direct row-L block descriptors are missing.");
                    const std::vector<int_t> &block_desc =
                        symV2RowDirectSendBlocksHost[slot];
                    if (block_desc.empty() || block_desc.size() % 2 != 0)
                        ABORT("SymFact V2 direct row-L block descriptors are invalid.");
                    if (ksupc <= 0)
                        ABORT("SymFact V2 direct row-L panel width is invalid.");
                    int stream_count = A_gpu.numCudaStreams > 0
                                           ? A_gpu.numCudaStreams
                                           : 1;
                    size_t scratch_slots =
                        static_cast<size_t>(stream_count) *
                        static_cast<size_t>(Pc);
                    if (Pc <= 0 ||
                        scratch_slots / static_cast<size_t>(Pc) !=
                            static_cast<size_t>(stream_count))
                        ABORT("SymFact V2 direct row-L scratch slot count overflows.");
                    if (symV2RowDirectSendMapScratchHost.size() <
                        scratch_slots)
                        symV2RowDirectSendMapScratchHost.resize(
                            scratch_slots);
                    size_t scratch_slot =
                        static_cast<size_t>(stream_offset) *
                            static_cast<size_t>(Pc) +
                        static_cast<size_t>(pc_dest);
                    if (scratch_slot >=
                        symV2RowDirectSendMapScratchHost.size())
                        ABORT("SymFact V2 direct row-L scratch slot is invalid.");
                    std::vector<int_t> &direct_map =
                        symV2RowDirectSendMapScratchHost[scratch_slot];
                    direct_map.clear();
                    direct_map.reserve(static_cast<size_t>(total));

                    auto append_direct_source_map =
                        [&](size_t flat, int_t gid,
                            std::vector<int_t> &out)
                    {
                        if (flat >= symL2LSendMeta.size() ||
                            flat >= symV2PartnerLSendSizes.size() ||
                            flat >= symV2PartnerLMapOffsets.size())
                            ABORT("SymFact V2 direct row-L source map is invalid.");
                        if (symV2PartnerLSendSizes[flat] < 0)
                            ABORT("SymFact V2 direct row-L source map size is invalid.");
                        const std::vector<int_t> &meta =
                            symL2LSendMeta[flat];
                        size_t map_pos = symV2PartnerLMapOffsets[flat];
                        size_t map_end =
                            map_pos +
                            static_cast<size_t>(
                                symV2PartnerLSendSizes[flat]);
                        if (map_end > symV2PartnerLPackedMaps.size() ||
                            map_end < map_pos)
                            ABORT("SymFact V2 direct row-L source map bounds are invalid.");
                        bool found = false;
                        size_t meta_pos = 0;
                        while (meta_pos < meta.size())
                        {
                            if (meta_pos + 2 > meta.size())
                                ABORT("SymFact V2 direct row-L metadata is truncated.");
                            int_t block_gid = meta[meta_pos++];
                            int_t len = meta[meta_pos++];
                            if (len < 0 ||
                                meta_pos + static_cast<size_t>(len) >
                                    meta.size())
                                ABORT("SymFact V2 direct row-L metadata block is invalid.");
                            if (ksupc <= 0 ||
                                static_cast<size_t>(len) >
                                    std::numeric_limits<size_t>::max() /
                                        static_cast<size_t>(ksupc))
                                ABORT("SymFact V2 direct row-L source map segment overflows.");
                            size_t value_count =
                                static_cast<size_t>(len) *
                                static_cast<size_t>(ksupc);
                            if (map_pos + value_count > map_end ||
                                map_pos + value_count < map_pos)
                                ABORT("SymFact V2 direct row-L source map segment is invalid.");
                            if (block_gid == gid)
                            {
                                out.insert(out.end(),
                                           symV2PartnerLPackedMaps.begin() +
                                               map_pos,
                                           symV2PartnerLPackedMaps.begin() +
                                               map_pos + value_count);
                                found = true;
                            }
                            map_pos += value_count;
                            meta_pos += static_cast<size_t>(len);
                        }
                        if (map_pos != map_end)
                            ABORT("SymFact V2 direct row-L source map size mismatch.");
                        if (!found)
                            ABORT("SymFact V2 direct row-L source map cannot find a requested block.");
                    };

                    size_t nblocks = block_desc.size() / 2;
                    std::vector<std::vector<int_t> > block_maps(nblocks);
                    std::vector<int_t> block_nrows(nblocks, 0);
                    for (size_t bi = 0; bi < nblocks; ++bi)
                    {
                        int_t gid = block_desc[2 * bi];
                        int_t chunk_pc = block_desc[2 * bi + 1];
                        if (chunk_pc < 0 || chunk_pc >= Pc)
                            ABORT("SymFact V2 direct row-L block source column is invalid.");
                        size_t flat =
                            static_cast<size_t>(row_send_lk) *
                                static_cast<size_t>(Pc) +
                            static_cast<size_t>(chunk_pc);
                        append_direct_source_map(flat, gid, block_maps[bi]);
                        if (block_maps[bi].empty() ||
                            block_maps[bi].size() %
                                    static_cast<size_t>(ksupc) !=
                                0)
                            ABORT("SymFact V2 direct row-L block map has invalid width.");
                        size_t nrows =
                            block_maps[bi].size() /
                            static_cast<size_t>(ksupc);
                        if (nrows >
                            static_cast<size_t>(
                                std::numeric_limits<int_t>::max()))
                            ABORT("SymFact V2 direct row-L block is too large.");
                        block_nrows[bi] = static_cast<int_t>(nrows);
                    }
                    for (int_t col = 0; col < ksupc; ++col)
                    {
                        for (size_t bi = 0; bi < nblocks; ++bi)
                        {
                            size_t src =
                                static_cast<size_t>(col) *
                                static_cast<size_t>(block_nrows[bi]);
                            direct_map.insert(
                                direct_map.end(),
                                block_maps[bi].begin() + src,
                                block_maps[bi].begin() + src +
                                    block_nrows[bi]);
                        }
                    }
                    if (direct_map.size() != static_cast<size_t>(total))
                        ABORT("SymFact V2 direct row-L compact map size mismatch.");
                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symV2RowFragSendMapStageBufs[stream_offset],
                        direct_map.data(),
                        sizeof(int_t) * static_cast<size_t>(total),
                        cudaMemcpyHostToDevice, stream));
                }
                else
                {
                    size_t map_offset = symV2RowDirectSendMapOffsets[slot];
                    if (map_offset + static_cast<size_t>(total) >
                            symV2RowDirectSendMapsHost.size() ||
                        map_offset + static_cast<size_t>(total) < map_offset)
                        ABORT("SymFact V2 direct row-L host map is invalid.");

                    gpuErrchk(cudaMemcpyAsync(
                        A_gpu.symV2RowFragSendMapStageBufs[stream_offset],
                        symV2RowDirectSendMapsHost.data() + map_offset,
                        sizeof(int_t) * static_cast<size_t>(total),
                        cudaMemcpyHostToDevice, stream));
                }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_pack_issue_t = SuperLU_timer_();
#endif
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                    row_lpanel_ptr->gpuPanel.val, dst_buf,
                    A_gpu.symV2RowFragSendMapStageBufs[stream_offset],
                    total);
                gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
                                 SuperLU_timer_() - row_pack_issue_t);
#endif
                return total;
            }

            std::fill(row_dest_offsets.begin(), row_dest_offsets.end(), -1);
            int total = 0;
            size_t exact_map_host_offset = 0;
            size_t exact_map_host_next = 0;
            bool exact_map_host_valid = false;
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(row_send_lk) *
                        static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc);
                if (flat >= symV2PartnerLSendSizes.size() ||
                    flat >= symL2LSendMapsGPU.size())
                    ABORT("SymFact V2 row-fragment send metadata is missing.");
                size_t active_pos =
                    flat * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (active_pos >= symV2RowFragSendActive.size())
                    ABORT("SymFact V2 row-fragment send mask is missing.");
                if (!symV2RowFragSendActive[active_pos])
                    continue;
                if (exact_row_fragment_demand &&
                    (active_pos >= symV2RowFragExactSendSizes.size() ||
                     active_pos >= symV2RowFragExactSendMapOffsets.size()))
                    ABORT("SymFact V2 exact row-fragment destination slot is invalid.");
                int size = exact_row_fragment_demand
                    ? symV2RowFragExactSendSizes[active_pos]
                    : symV2PartnerLSendSizes[flat];
                if (size <= 0)
                    ABORT("SymFact V2 row-fragment active send has no data.");
                if (exact_map_index)
                {
                    size_t map_offset =
                        symV2RowFragExactSendMapOffsets[active_pos];
                    if (map_offset + static_cast<size_t>(size) >
                            symV2RowFragExactSendMapsHost.size() ||
                        map_offset + static_cast<size_t>(size) < map_offset)
                        ABORT("SymFact V2 exact row-fragment host map is invalid.");
                    if (!exact_map_host_valid)
                    {
                        exact_map_host_offset = map_offset;
                        exact_map_host_next = map_offset;
                        exact_map_host_valid = true;
                    }
                    if (map_offset != exact_map_host_next)
                        ABORT("SymFact V2 exact row-fragment destination maps are not contiguous.");
                    exact_map_host_next += static_cast<size_t>(size);
                }
                row_dest_offsets[static_cast<size_t>(pc)] = total;
                total += size;
            }
            if (total <= 0)
                return 0;
            if (row_lpanel_ptr == NULL || row_lpanel_ptr->isEmpty())
                ABORT("SymFact V2 row-fragment source L panel is missing.");
            if (total > maxSymV2RowFragStageCount)
                ABORT("SymFact V2 row-fragment packed destination exceeds staging buffer.");
            if (exact_map_index)
            {
                if (!exact_map_host_valid)
                    ABORT("SymFact V2 exact row-fragment destination has no maps.");
                gpuErrchk(cudaMemcpyAsync(
                    A_gpu.symV2RowFragSendMapStageBufs[stream_offset],
                    symV2RowFragExactSendMapsHost.data() +
                        exact_map_host_offset,
                    sizeof(int_t) * static_cast<size_t>(total),
                    cudaMemcpyHostToDevice, stream));
            }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double row_pack_issue_t = SuperLU_timer_();
#endif
            bool packed_any = false;
            for (int pc = 0; pc < Pc; ++pc)
            {
                int dst_offset = row_dest_offsets[static_cast<size_t>(pc)];
                if (dst_offset < 0)
                    continue;
	                size_t flat =
	                    static_cast<size_t>(row_send_lk) *
	                        static_cast<size_t>(Pc) +
	                    static_cast<size_t>(pc);
                size_t active_pos =
                    flat * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc_dest);
                if (exact_row_fragment_demand &&
                    (active_pos >= symV2RowFragExactSendSizes.size() ||
                     active_pos >= symV2RowFragExactSendMapOffsets.size()))
                    ABORT("SymFact V2 exact row-fragment destination slot is invalid.");
                int size = exact_row_fragment_demand
                    ? symV2RowFragExactSendSizes[active_pos]
                    : symV2PartnerLSendSizes[flat];
                int_t *sendmap = NULL;
                if (exact_row_fragment_demand)
                {
                    sendmap =
                        A_gpu.symV2RowFragSendMapStageBufs[stream_offset] +
                        dst_offset;
                    if (!exact_map_index)
                    {
                        if (active_pos >= symV2RowFragExactSendMapOffsets.size())
                            ABORT("SymFact V2 exact row-fragment destination slot is invalid.");
                        size_t map_offset =
                            symV2RowFragExactSendMapOffsets[active_pos];
                        if (map_offset + static_cast<size_t>(size) >
                                symV2RowFragExactSendMapsHost.size() ||
                            map_offset + static_cast<size_t>(size) < map_offset)
                            ABORT("SymFact V2 exact row-fragment host map is invalid.");
                        gpuErrchk(cudaMemcpyAsync(
                            sendmap,
                            symV2RowFragExactSendMapsHost.data() + map_offset,
                            sizeof(int_t) * static_cast<size_t>(size),
                            cudaMemcpyHostToDevice, stream));
                    }
                }
                else
                {
                    sendmap = symL2LSendMapsGPU[flat];
                }
                if (sendmap == NULL)
                    ABORT("SymFact V2 row-fragment send map is missing.");

                int threads = 256;
                int blocks = (size + threads - 1) / threads;
                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                    row_lpanel_ptr->gpuPanel.val,
                    dst_buf + dst_offset,
                    sendmap, size);
                packed_any = true;
            }
            if (packed_any)
                gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                             SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
                             SuperLU_timer_() - row_pack_issue_t);
#endif
            return total;
        };

        if (mycol == kcol_ && superlu_sym_v2_row_l_pack_all_dest())
        {
// SYM_V2_PC2_PHASE2_PACK_ALL_BRANCH_BEGIN
            /* Phase 2: destination-packed aggregate send.  This mostly helps
               Pc>2 by removing per-destination synchronization and immediate
               MPI_Wait.  Pc=2 has only one remote destination column and may
               therefore be neutral. */
            if (static_cast<size_t>(stream_offset) >= symV2RowFragHostSendBufs.size() ||
                symV2RowFragHostSendBufs[stream_offset] == NULL)
                ABORT("SymFact V2 row-fragment send staging buffer is missing.");
            if (A_gpu.symV2RowFragStageBufs[stream_offset] == NULL)
                ABORT("SymFact V2 row-fragment device staging buffer is missing.");

            if (symV2RowFragSendCountsScratch.size() != static_cast<size_t>(Pc))
                symV2RowFragSendCountsScratch.assign(static_cast<size_t>(Pc), 0);
            if (symV2RowFragSendOffsetsScratch.size() != static_cast<size_t>(Pc))
                symV2RowFragSendOffsetsScratch.assign(static_cast<size_t>(Pc), -1);
            std::fill(symV2RowFragSendCountsScratch.begin(),
                      symV2RowFragSendCountsScratch.end(), 0);
            std::fill(symV2RowFragSendOffsetsScratch.begin(),
                      symV2RowFragSendOffsetsScratch.end(), -1);

            int row_send_total = 0;
            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
            {
                if (pc_dest == mycol)
                    continue;
                int count = row_count_destination(pc_dest);
                if (count <= 0)
                    continue;
                if (count > maxSymV2RowFragValSendCount ||
                    row_send_total > maxSymV2RowFragValSendCount - count)
                    ABORT("SymFact V2 aggregate row-fragment send exceeds send staging capacity.");
                symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)] =
                    row_send_total;
                symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)] =
                    count;
                row_send_total += count;
            }

// SYM_V2_PC2_COMPRESSED_PLAN_AGG_PACK_BEGIN
            auto row_pack_plan_destination_range =
                [&](int pc_begin, int pc_end)
            {
                int run_first = -1;
                int run_last = -1;
                int run_total = 0;
                int expected_off = -1;
                for (int pc_dest = pc_begin; pc_dest < pc_end; ++pc_dest)
                {
                    int count = symV2RowFragSendCountsScratch[
                        static_cast<size_t>(pc_dest)];
                    if (count <= 0)
                        continue;
                    int off = symV2RowFragSendOffsetsScratch[
                        static_cast<size_t>(pc_dest)];
                    if (off < 0)
                        ABORT("SymFact V2 aggregate row-fragment offset is invalid.");
                    if (run_first < 0)
                    {
                        run_first = pc_dest;
                        expected_off = off;
                    }
                    else if (off != expected_off)
                    {
                        ABORT("SymFact V2 aggregate row-fragment staging is not contiguous.");
                    }
                    run_last = pc_dest;
                    if (run_total > maxSymV2RowFragValSendCount - count)
                        ABORT("SymFact V2 aggregate row-fragment range pack exceeds staging capacity.");
                    run_total += count;
                    expected_off += count;
                }
                if (run_total <= 0)
                    return;
                if (row_lpanel_ptr == NULL || row_lpanel_ptr->isEmpty())
                    ABORT("SymFact V2 row-down source L panel is missing.");
                size_t first_slot =
                    static_cast<size_t>(row_send_lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(run_first);
                size_t last_slot =
                    static_cast<size_t>(row_send_lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(run_last);
                if (first_slot >= symV2RowDownSendSizes.size() ||
                    last_slot >= symV2RowDownSendSizes.size() ||
                    first_slot >= symV2RowDownSendMapOffsets.size() ||
                    last_slot >= symV2RowDownSendMapOffsets.size() ||
                    first_slot >= symV2RowDownSendMapsGPU.size())
                    ABORT("SymFact V2 row-down range send slot is invalid.");
                size_t map_expected = symV2RowDownSendMapOffsets[first_slot];
                size_t map_begin = map_expected;
                for (int pc_dest = run_first; pc_dest <= run_last; ++pc_dest)
                {
                    size_t slot =
                        static_cast<size_t>(row_send_lk) *
                            static_cast<size_t>(Pc) +
                        static_cast<size_t>(pc_dest);
                    if (slot >= symV2RowDownSendSizes.size() ||
                        slot >= symV2RowDownSendMapOffsets.size())
                        ABORT("SymFact V2 row-down range send slot is invalid.");
                    int expected_count = symV2RowFragSendCountsScratch[
                        static_cast<size_t>(pc_dest)];
                    int map_count = symV2RowDownSendSizes[slot];
                    if (expected_count < 0 || map_count < 0 ||
                        expected_count != map_count)
                        ABORT("SymFact V2 row-down range send size mismatch.");
                    if (symV2RowDownSendMapOffsets[slot] != map_expected)
                        ABORT("SymFact V2 row-down range send maps are not contiguous.");
                    map_expected += static_cast<size_t>(map_count);
                }
                if (map_expected < map_begin ||
                    map_expected - map_begin != static_cast<size_t>(run_total))
                    ABORT("SymFact V2 row-down range send map size mismatch.");
                int_t *sendmap = symV2RowDownSendMapsGPU[first_slot];
                if (sendmap == NULL)
                    ABORT("SymFact V2 row-down range send map is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_pack_issue_t = SuperLU_timer_();
#endif
                int threads = 256;
                int blocks = (run_total + threads - 1) / threads;
                int first_off = symV2RowFragSendOffsetsScratch[
                    static_cast<size_t>(run_first)];
                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                    row_lpanel_ptr->gpuPanel.val,
                    A_gpu.symV2RowFragStageBufs[stream_offset] + first_off,
                    sendmap, run_total);
                gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
                                 SuperLU_timer_() - row_pack_issue_t);
#endif
            };

            if (superlu_sym_v2_row_l_plan_v2_exchange() &&
                superlu_sym_v2_row_l_compressed_plan() &&
                !row_down_lazy_sendmap) // SYM_V2_PC2_LAZY_SENDMAP_AGG_GUARD
            {
                row_pack_plan_destination_range(0, mycol);
                row_pack_plan_destination_range(mycol + 1, Pc);
            }
            else
            {
                for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                {
                    int count = symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)];
                    if (count <= 0)
                        continue;
                    int off = symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)];
                    int packed = row_pack_destination(
                        pc_dest,
                        A_gpu.symV2RowFragStageBufs[stream_offset] + off);
                    if (packed != count)
                        ABORT("SymFact V2 aggregate row-fragment pack size mismatch.");
                }
            }
// SYM_V2_PC2_COMPRESSED_PLAN_AGG_PACK_END

            if (row_send_total > 0)
            {
                double *row_send_host_base =
                    symV2RowFragHostSendBufs[stream_offset];
                if (pcfrag_taskflow_async_pieces)
                {
                    if (taskflow_state == NULL || !taskflow_state->initialized)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW async row send has no state.");
                    row_send_host_base =
                        dSymV2PcFragTaskflowEnsurePinnedHost(
                            symV2PcFragTaskflowPinnedBlockPool,
                            &taskflow_state->producer_row_send_host_values,
                            &taskflow_state->producer_row_send_host_capacity,
                            static_cast<size_t>(row_send_total),
                            allow_taskflow_late_alloc,
                            &symV2PcFragTaskflowStats.arena_pinned_late_allocs);
                    taskflow_update_pinned_high_water();
                }
                if (row_send_host_base == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW row send host buffer is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_d2h_issue_t = SuperLU_timer_();
                symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                           static_cast<long long>(row_send_total) *
                               static_cast<long long>(sizeof(double)));
#endif
                symV2PayloadProfileAdd(
                    SYM_V2_PAYLOAD_ROWFRAG_HOST_STAGING,
                    static_cast<long long>(row_send_total) *
                        static_cast<long long>(sizeof(double)));
                gpuErrchk(cudaMemcpyAsync(
                    row_send_host_base,
                    A_gpu.symV2RowFragStageBufs[stream_offset],
                    sizeof(double) * static_cast<size_t>(row_send_total),
                    cudaMemcpyDeviceToHost, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_D2H_STAGE_ISSUE,
                                 SuperLU_timer_() - row_d2h_issue_t);
                double row_pack_stage_sync_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_async_core())
                    ++symV2PcFragTaskflowStats.producer_exchange_stream_syncs;
                gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_STAGE_SYNC,
                                 SuperLU_timer_() - row_pack_stage_sync_t);
                double row_send_post_t = SuperLU_timer_();
#endif
                for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                {
                    int count = symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)];
                    if (count <= 0)
                        continue;
                    int off = symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)];
                    MPI_Request req;
                    long long send_bytes =
                        static_cast<long long>(count) *
                        static_cast<long long>(sizeof(double));
                    sym_v2_rowfrag_payload_bytes += send_bytes;
                    symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_MPI_SEND,
                                           send_bytes);
                    MPI_Isend(row_send_host_base + off,
                              count, MPI_DOUBLE, pc_dest,
                              SLU_MPI_TAG(5, k), grid3d->rscp.comm, &req);
                    send_reqs.push_back(req);
                }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_POST,
                                 SYM_GPU3D_T_ROW_LFRAG_SEND_POST,
                                 SuperLU_timer_() - row_send_post_t);
                symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS,
                           static_cast<long long>(send_reqs.size()));
#endif
            }
// SYM_V2_PC2_PHASE2_PACK_ALL_BRANCH_END
        }
        else if (mycol == kcol_)
        {
            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
            {
                if (pc_dest == mycol)
                    continue;
                int send_total = row_pack_destination(
                    pc_dest, A_gpu.symV2RowFragStageBufs[stream_offset]);
                if (send_total <= 0)
                    continue;
                if (static_cast<size_t>(stream_offset) >=
                        symV2RowFragHostRecvBufs.size() ||
                    symV2RowFragHostRecvBufs[stream_offset] == NULL)
                    ABORT("SymFact V2 row-fragment host send staging is missing.");

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_d2h_issue_t = SuperLU_timer_();
                symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                           static_cast<long long>(send_total) *
                               static_cast<long long>(sizeof(double)));
#endif
                symV2PayloadProfileAdd(
                    SYM_V2_PAYLOAD_ROWFRAG_HOST_STAGING,
                    static_cast<long long>(send_total) *
                        static_cast<long long>(sizeof(double)));
                gpuErrchk(cudaMemcpyAsync(
                    symV2RowFragHostRecvBufs[stream_offset],
                    A_gpu.symV2RowFragStageBufs[stream_offset],
                    sizeof(double) * static_cast<size_t>(send_total),
                    cudaMemcpyDeviceToHost, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_D2H_STAGE_ISSUE,
                                 SuperLU_timer_() - row_d2h_issue_t);
                double row_pack_stage_sync_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow && superlu_sym_v2_pcfrag_taskflow_async_core())
                    ++symV2PcFragTaskflowStats.producer_exchange_stream_syncs;
                gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                                 SYM_GPU3D_T_ROW_LFRAG_PACK_STAGE_SYNC,
                                 SuperLU_timer_() - row_pack_stage_sync_t);
                double row_send_post_t = SuperLU_timer_();
#endif
                MPI_Request req;
                long long send_bytes =
                    static_cast<long long>(send_total) *
                    static_cast<long long>(sizeof(double));
                sym_v2_rowfrag_payload_bytes += send_bytes;
                symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_MPI_SEND,
                                       send_bytes);
                MPI_Isend(symV2RowFragHostRecvBufs[stream_offset],
                          send_total, MPI_DOUBLE, pc_dest,
                          SLU_MPI_TAG(5, k), grid3d->rscp.comm, &req);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_POST,
                                 SYM_GPU3D_T_ROW_LFRAG_SEND_POST,
                                 SuperLU_timer_() - row_send_post_t);
                symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS, 1);
                double row_send_wait_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow)
                {
                    ++symV2PcFragTaskflowStats.producer_send_wait_calls;
                    ++symV2PcFragTaskflowStats.producer_mpi_wait_requests;
                }
                MPI_Wait(&req, MPI_STATUS_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                                 SYM_GPU3D_T_ROW_LFRAG_SEND_WAIT,
                                 SuperLU_timer_() - row_send_wait_t);
#endif
            }
        }

        if (!pcfrag_taskflow || pcfrag_taskflow_validate)
        {
            if (row_index.empty())
            {
                gpuErrchk(cudaMemcpyAsync(
                    A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
                    empty_header, sizeof(int_t) * LPANEL_HEADER_SIZE,
                    cudaMemcpyHostToDevice, stream));
            }
            else
            {
                if (row_index[3] != ksupc)
                    ABORT("SymFact V2 row-fragment index has wrong panel width.");
                gpuErrchk(cudaMemcpyAsync(
                    A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
                    row_index.data(),
                    sizeof(int_t) * static_cast<size_t>(row_index_size),
                    cudaMemcpyHostToDevice, stream));
            }
        }

        if (!row_index.empty())
        {
            int_t row_nrows = row_index[1];
            if (row_nrows <= 0 ||
                static_cast<int64_t>(row_nrows) *
                        static_cast<int64_t>(ksupc) >
                    static_cast<int64_t>(maxSymV2RowFragValRecvCount))
                ABORT("SymFact V2 row-fragment value buffer is too small.");

            std::vector<int> row_recv_offsets(static_cast<size_t>(Pc), -1);
            int row_recv_total = 0;
            int row_src_pc = static_cast<int>(kcol_);
            size_t row_recv_base =
                static_cast<size_t>(k) * static_cast<size_t>(Pc);
            if (row_recv_base + static_cast<size_t>(Pc) >
                    symV2RowFragRecvSizes.size() ||
                row_recv_base + static_cast<size_t>(Pc) >
                    symV2RowFragRecvMap.size())
                ABORT("SymFact V2 row-fragment receive metadata is missing.");
            for (int pc = 0; pc < Pc; ++pc)
            {
                int count = symV2RowFragRecvSizes[row_recv_base +
                                                  static_cast<size_t>(pc)];
                if (count <= 0)
                    continue;
                row_recv_offsets[static_cast<size_t>(pc)] = row_recv_total;
                row_recv_total += count;
            }
            if (row_recv_total > maxSymV2RowFragStageCount)
                ABORT("SymFact V2 row-fragment receive exceeds staging buffer.");
            double *row_recv_host_base = NULL;
            std::vector<MPI_Request> row_recv_reqs;
            if (row_recv_total > 0 && row_src_pc != mycol)
            {
                if (pcfrag_taskflow_async_pieces)
                {
                    if (taskflow_state == NULL)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW async row receive has no state.");
                    row_recv_host_base =
                        dSymV2PcFragTaskflowEnsurePinnedHost(
                            symV2PcFragTaskflowPinnedBlockPool,
                            &taskflow_state->producer_row_recv_host_values,
                            &taskflow_state->producer_row_recv_host_capacity,
                        static_cast<size_t>(row_recv_total),
                        allow_taskflow_late_alloc,
                        &symV2PcFragTaskflowStats.arena_pinned_late_allocs);
                    taskflow_update_pinned_high_water();
                }
                else
                {
                    if (static_cast<size_t>(stream_offset) >=
                            symV2RowFragHostRecvBufs.size() ||
                        symV2RowFragHostRecvBufs[stream_offset] == NULL)
                        ABORT("SymFact V2 row-fragment host receive staging is missing.");
                    row_recv_host_base = symV2RowFragHostRecvBufs[stream_offset];
                }
                MPI_Request req;
                long long recv_bytes =
                    static_cast<long long>(row_recv_total) *
                    static_cast<long long>(sizeof(double));
                sym_v2_rowfrag_payload_bytes += recv_bytes;
                symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_MPI_RECV,
                                       recv_bytes);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_recv_post_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow_async_pieces)
                {
                    const bool pinned_taskflow_recv =
                        taskflow_state != NULL &&
                        row_recv_host_base != NULL &&
                        row_recv_host_base ==
                            taskflow_state->producer_row_recv_host_values;
                    dSymV2PcFragTaskflowNoteProducerRecvPost(
                        symV2PcFragTaskflowStats, pinned_taskflow_recv);
                }
                MPI_Irecv(row_recv_host_base, row_recv_total, MPI_DOUBLE,
                          row_src_pc, SLU_MPI_TAG(5, k),
                          grid3d->rscp.comm, &req);
                row_recv_reqs.push_back(req);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_RECV_POST,
                                 SYM_GPU3D_T_ROW_LFRAG_RECV_POST,
                                 SuperLU_timer_() - row_recv_post_t);
#endif
            }
            if (pcfrag_taskflow_async_pieces && !row_recv_reqs.empty())
            {
                if (taskflow_state == NULL || !taskflow_state->initialized)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW async row receive has no state.");
                taskflow_state->producer_exchange_pending = 1;
                taskflow_state->producer_ksupc = static_cast<int>(ksupc);
                dSymV2PcFragTaskflowEnsureVectorCapacity(
                    taskflow_state->producer_row_recv_reqs,
                    row_recv_reqs.size());
                dSymV2PcFragTaskflowEnsureVectorCapacity(
                    taskflow_state->producer_row_recv_pcs,
                    row_recv_reqs.size());
                dSymV2PcFragTaskflowEnsureVectorCapacity(
                    taskflow_state->producer_row_recv_sizes,
                    static_cast<size_t>(Pc));
                dSymV2PcFragTaskflowEnsureVectorCapacity(
                    taskflow_state->producer_row_recv_offsets,
                    row_recv_offsets.size());
                dSymV2PcFragTaskflowEnsureVectorCapacity(
                    taskflow_state->producer_row_recv_done,
                    row_recv_reqs.size());
                taskflow_state->producer_row_recv_reqs = row_recv_reqs;
                taskflow_state->producer_row_recv_pcs.assign(
                    row_recv_reqs.size(), -1);
                taskflow_state->producer_row_recv_sizes.assign(
                    static_cast<size_t>(Pc), 0);
                taskflow_state->producer_row_recv_offsets =
                    row_recv_offsets;
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t row_pos =
                        row_recv_base + static_cast<size_t>(pc);
                    taskflow_state->producer_row_recv_sizes[
                        static_cast<size_t>(pc)] =
                        symV2RowFragRecvSizes[row_pos];
                }
                taskflow_state->producer_row_recv_done.assign(
                    row_recv_reqs.size(), 0);
                taskflow_state->producer_row_recv_remaining =
                    static_cast<int>(row_recv_reqs.size());
                dSymV2PcFragTaskflowEnsureProgressScratch(
                    *taskflow_state, symV2PcFragTaskflowStats,
                    row_recv_reqs.size());
                row_recv_reqs.clear();
                dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
            }
            if (!row_recv_reqs.empty())
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_recv_wait_t = SuperLU_timer_();
#endif
                if (pcfrag_taskflow)
                {
                    ++symV2PcFragTaskflowStats.producer_recv_wait_calls;
                    symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                        static_cast<long long>(row_recv_reqs.size());
                }
                MPI_Waitall(static_cast<int>(row_recv_reqs.size()),
                            row_recv_reqs.data(), MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                                 SYM_GPU3D_T_ROW_LFRAG_MPI_RECV_WAIT,
                                 SuperLU_timer_() - row_recv_wait_t);
#endif
            }
            const bool taskflow_direct_row_host_to_piece =
                pcfrag_taskflow && row_l_direct_recv &&
                !pcfrag_taskflow_validate && row_src_pc != mycol;
            const bool taskflow_async_row_pending =
                pcfrag_taskflow_async_pieces && row_src_pc != mycol &&
                taskflow_state != NULL &&
                taskflow_state->producer_row_recv_remaining > 0;
            if (row_recv_total > 0 && row_src_pc != mycol &&
                !taskflow_direct_row_host_to_piece)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_h2d_issue_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpyAsync(
                    (row_l_direct_recv && !pcfrag_taskflow)
                        ? A_gpu.symV2RowFragValRecvBufs[stream_offset]
                        : A_gpu.symV2RowFragStageBufs[stream_offset],
                    row_recv_host_base,
                    sizeof(double) * static_cast<size_t>(row_recv_total),
                    cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_H2D_STAGE_ISSUE,
                                 SuperLU_timer_() - row_h2d_issue_t);
#endif
            }
            else if (row_recv_total > 0 && row_src_pc == mycol)
            {
                int self_total = row_pack_destination(
                    mycol,
                    (row_l_direct_recv && !pcfrag_taskflow)
                        ? A_gpu.symV2RowFragValRecvBufs[stream_offset]
                        : A_gpu.symV2RowFragStageBufs[stream_offset]);
                if (self_total != row_recv_total)
                    ABORT("SymFact V2 row-fragment self pack size mismatch.");
                long long self_bytes =
                    static_cast<long long>(row_recv_total) *
                    static_cast<long long>(sizeof(double));
                sym_v2_rowfrag_payload_bytes += self_bytes;
                symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_SELF,
                                       self_bytes);
            }

            if (row_l_direct_recv)
            {
                int64_t row_value_count =
                    static_cast<int64_t>(row_nrows) *
                    static_cast<int64_t>(ksupc);
                if (row_recv_total != row_value_count)
                    ABORT("SymFact V2 direct row-L receive size does not match the row-fragment layout.");
                if (pcfrag_taskflow)
                {
                    if (!taskflow_async_row_pending &&
                        pcfrag_taskflow_validate &&
                        A_gpu.symV2RowFragValRecvBufs[stream_offset] != NULL)
                    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                        double row_assemble_issue_t = SuperLU_timer_();
#endif
                        gpuErrchk(cudaMemcpyAsync(
                            A_gpu.symV2RowFragValRecvBufs[stream_offset],
                            A_gpu.symV2RowFragStageBufs[stream_offset],
                            sizeof(double) *
                                static_cast<size_t>(row_recv_total),
                            cudaMemcpyDeviceToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                        symTimingAddBoth(SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
                                         SYM_GPU3D_T_ROW_LFRAG_ASSEMBLE_ISSUE,
                                         SuperLU_timer_() - row_assemble_issue_t);
#endif
                    }
                    if (!taskflow_async_row_pending)
                    {
                        taskflow_copy_owned_pieces_from_full(
                            SYM_V2_PCFRAG_PIECE_ROW,
                            taskflow_direct_row_host_to_piece
                                ? row_recv_host_base
                                : A_gpu.symV2RowFragStageBufs[stream_offset],
                            row_nrows,
                            taskflow_direct_row_host_to_piece
                                ? cudaMemcpyHostToDevice
                                : cudaMemcpyDeviceToDevice);
                    }
                }
            }
            else
            {
                if (!pcfrag_taskflow || pcfrag_taskflow_validate)
                    gpuErrchk(cudaMemsetAsync(
                        A_gpu.symV2RowFragValRecvBufs[stream_offset], 0,
                        sizeof(double) * static_cast<size_t>(row_nrows) *
                            static_cast<size_t>(ksupc),
                        stream));

                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t row_pos = row_recv_base + static_cast<size_t>(pc);
                    int count = symV2RowFragRecvSizes[row_pos];
                    if (count <= 0)
                        continue;
                    const std::vector<int_t> &row_map =
                        symV2RowFragRecvMap[row_pos];
                    if (row_map.empty() || row_map.size() % 3 != 0)
                        ABORT("SymFact V2 row-fragment receive map is invalid.");
                    int pieces = static_cast<int>(row_map.size() / 3);
                    int_t *row_map_gpu = symV2RowFragRecvMapsGPU[row_pos];
                    if (row_map_gpu == NULL)
                        ABORT("SymFact V2 row-fragment device receive map is missing.");

                    int offset = row_recv_offsets[static_cast<size_t>(pc)];
                    if (offset < 0)
                        ABORT("SymFact V2 row-fragment offset is invalid.");
                    double *stage =
                        A_gpu.symV2RowFragStageBufs[stream_offset] + offset;
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double row_assemble_issue_t = SuperLU_timer_();
#endif
                    if (pcfrag_taskflow)
                    {
                        taskflow_assemble_owned_pieces(
                            SYM_V2_PCFRAG_PIECE_ROW, stage, row_map);
                    }
                    if (!pcfrag_taskflow || pcfrag_taskflow_validate)
                    {
                        sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                            stage, A_gpu.symV2RowFragValRecvBufs[stream_offset],
                            row_map_gpu, pieces, ksupc, row_nrows);
                        gpuErrchk(cudaGetLastError());
                    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
                                     SYM_GPU3D_T_ROW_LFRAG_ASSEMBLE_ISSUE,
                                     SuperLU_timer_() - row_assemble_issue_t);
#endif
                }
            }
        }
        }
    }

    if (!send_reqs.empty())
    {
        if (deferred_partner_send_req_count > send_reqs.size())
            ABORT("SymFact V2 deferred partner send count is invalid.");
        if (pcfrag_taskflow_async_pieces)
        {
            if (taskflow_state == NULL || !taskflow_state->initialized)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW deferred send has no state.");
            if (!taskflow_state->producer_send_reqs.empty())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW found pending producer sends before exchange return.");
            dSymV2PcFragTaskflowEnsureVectorCapacity(
                taskflow_state->producer_send_reqs, send_reqs.size());
            taskflow_state->producer_send_reqs = send_reqs;
            dSymV2PcFragTaskflowEnsureProgressScratch(
                *taskflow_state, symV2PcFragTaskflowStats,
                taskflow_state->producer_send_reqs.size());
            send_reqs.clear();
        }
        else
        {
            auto wait_send_requests = [&](MPI_Request *reqs, size_t count) {
                if (count == 0)
                    return;
                if (!pcfrag_taskflow)
                {
                    MPI_Waitall(static_cast<int>(count), reqs,
                                MPI_STATUSES_IGNORE);
                    return;
                }
                if (wait_indices.size() < count)
                    wait_indices.resize(count);
                if (wait_statuses.size() < count)
                    wait_statuses.resize(count);

                const int request_count = static_cast<int>(count);
                int remaining = request_count;
                int idle_polls = 0;
                while (remaining > 0)
                {
                    int completed = 0;
                    ++symV2PcFragTaskflowStats.producer_send_wait_calls;
                    int mpi_rc = pcfrag_taskflow_async_pieces
                        ? MPI_Testsome(request_count, reqs, &completed,
                                       wait_indices.data(), wait_statuses.data())
                        : MPI_Waitsome(request_count, reqs, &completed,
                                       wait_indices.data(), wait_statuses.data());
                    if (mpi_rc != MPI_SUCCESS || completed == MPI_UNDEFINED ||
                        completed < 0 || completed > remaining)
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW send progressive wait failed.");
                    if (pcfrag_taskflow_async_pieces)
                        dSymV2PcFragTaskflowProgressExchangeGPU(k, 0);
                    if (completed > 0)
                    {
                        symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                            static_cast<long long>(completed);
                        dSymV2PcFragTaskflowProgressGPU(
                            k, superlu_sym_v2_pcfrag_taskflow_effective_progress_budget());
                        remaining -= completed;
                        idle_polls = 0;
                    }
                    else if (pcfrag_taskflow_async_pieces)
                    {
                        if (++idle_polls > 10000000)
                            ABORT("GPU3DV2_PCFRAG_TASKFLOW send wait made no MPI progress.");
                    }
                    else
                    {
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW send wait returned no completions.");
                    }
                }
            };
        if (deferred_partner_send_req_count > 0)
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double partner_send_wait_t = SuperLU_timer_();
#endif
            wait_send_requests(send_reqs.data(), deferred_partner_send_req_count);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                             SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
                             SuperLU_timer_() - partner_send_wait_t);
#endif
        }
        size_t row_send_req_count =
            send_reqs.size() - deferred_partner_send_req_count;
        if (row_send_req_count > 0)
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double send_wait_t = SuperLU_timer_();
#endif
            wait_send_requests(send_reqs.data() + deferred_partner_send_req_count,
                               row_send_req_count);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                             pc_fragment_schur
                                 ? SYM_GPU3D_T_ROW_LFRAG_SEND_WAIT
                                 : SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
                             SuperLU_timer_() - send_wait_t);
#endif
        }
        send_reqs.clear();
        }
    }

    const char *pcfrag_taskflow_force_return_sync_env =
        std::getenv("GPU3DV2_PCFRAG_TASKFLOW_FORCE_RETURN_SYNC");
    const bool pcfrag_taskflow_force_return_sync =
        pcfrag_taskflow_force_return_sync_env != NULL &&
        pcfrag_taskflow_force_return_sync_env[0] != '\0' &&
        std::atoi(pcfrag_taskflow_force_return_sync_env) != 0;
    const bool pcfrag_taskflow_skip_return_sync =
        pcfrag_taskflow_async_pieces &&
        superlu_sym_v2_pcfrag_taskflow_async_core() &&
        !pcfrag_taskflow_force_return_sync;
    if (pcfrag_taskflow && taskflow_state != NULL &&
        taskflow_state->initialized)
    {
        if (pcfrag_taskflow_skip_return_sync)
            taskflow_state->producer_stream_pending = 1;
        else
            taskflow_state->producer_exchange_active = 0;
    }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double stream_sync_t = SuperLU_timer_();
#endif
    if (!pcfrag_taskflow_skip_return_sync)
        gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    if (!pcfrag_taskflow_skip_return_sync)
        symTimingAdd(SYM_GPU3D_T_LFRAG_STREAM_SYNC,
                     SuperLU_timer_() - stream_sync_t);
    symTimingAdd(SYM_GPU3D_T_LFRAG_EXCHANGE_TOTAL,
                 SuperLU_timer_() - lfrag_total_t);
#endif
    if (pcfrag_taskflow)
    {
        ++symV2PcFragTaskflowStats.producer_returns;
        long long unready_pieces = 0;
        int incomplete_tasks = 0;
        if (taskflow_state != NULL && taskflow_state->initialized)
        {
            if (taskflow_state->row_pieces_ready_count >
                    taskflow_state->row_pieces.size() ||
                taskflow_state->partner_pieces_ready_count >
                    taskflow_state->partner_pieces.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW ready piece count is inconsistent.");
            unready_pieces =
                static_cast<long long>(taskflow_state->row_pieces.size()) -
                static_cast<long long>(taskflow_state->row_pieces_ready_count) +
                static_cast<long long>(taskflow_state->partner_pieces.size()) -
                static_cast<long long>(taskflow_state->partner_pieces_ready_count);
            incomplete_tasks = taskflow_state->incomplete_task_count;
            if (taskflow_state->producer_exchange_pending)
                ++symV2PcFragTaskflowStats.producer_returns_with_pending_recvs;
        }
        if (unready_pieces == 0)
            ++symV2PcFragTaskflowStats.producer_returns_all_pieces_ready;
        else
        {
            ++symV2PcFragTaskflowStats.producer_returns_incomplete_pieces;
            symV2PcFragTaskflowStats.producer_return_unready_pieces +=
                unready_pieces;
        }
        if (incomplete_tasks == 0)
            ++symV2PcFragTaskflowStats.producer_returns_all_tasks_complete;
        else
        {
            ++symV2PcFragTaskflowStats.producer_returns_incomplete_tasks;
            symV2PcFragTaskflowStats.producer_return_incomplete_task_sum +=
                static_cast<long long>(incomplete_tasks);
        }
    }
    if (pcfrag_taskflow_validate)
    {
        if (frag_nblocks > 0 && frag_nrows > 0)
            taskflow_validate_owned_pieces(
                SYM_V2_PCFRAG_PIECE_PARTNER,
                A_gpu.symPartnerLvalRecvBufs[stream_offset], frag_nrows);
        if (static_cast<size_t>(k) < symV2RowFragRecvIndex.size() &&
            !symV2RowFragRecvIndex[k].empty() &&
            symV2RowFragRecvIndex[k][1] > 0)
        {
            taskflow_validate_owned_pieces(
                SYM_V2_PCFRAG_PIECE_ROW,
                A_gpu.symV2RowFragValRecvBufs[stream_offset],
                symV2RowFragRecvIndex[k][1]);
        }
    }
    symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_CALL,
                           sym_v2_partner_payload_bytes);
    if (pc_fragment_schur)
        symV2PayloadProfileAdd(SYM_V2_PAYLOAD_ROWFRAG_CALL,
                               sym_v2_rowfrag_payload_bytes);
    SYM_V2_TRACE_EXCHANGE(grid3d, k, "leave L-fragment exchange");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymStartL2UGPU(int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || options->CommL != YES)
    {
        if (options->SymFact == YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        return 0;
    }
    if (symL2USendBufsGPU.empty() || symL2USendMapsGPU.empty())
        ABORT("SymFact GPU L2U buffers are not allocated.");

    dLocalLU_t *Llu = LUstructPtr->Llu;
    int_t krow_ = PROW(k, grid);
    int_t kcol_ = PCOL(k, grid);
    int tag_ub = symFactTagUb;
    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];

    if (Pr == 1 && Pc == 1)
    {
        int_t lk = g2lRow(k);
        xupanel_t<double> &upanel = uPanelVec[lk];
        if (upanel.isEmpty())
            return 0;

        if (lk >= (int_t)symL2ULocalMapsGPU.size() ||
            symL2ULocalMapsGPU[lk] == NULL)
            ABORT("SymFact local GPU L2U map is missing.");

        int count = upanel.nzvalSize();
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symStatAdd(SYM_GPU3D_S_L2U_LOCAL_BYTES,
                   static_cast<long long>(count) *
                   static_cast<long long>(sizeof(double)));
#endif
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        sym_l2u_local_gather_kernel<<<blocks, threads, 0, stream>>>(
            lPanelVec[g2lCol(k)].gpuPanel.val, upanel.gpuPanel.val,
            symL2ULocalMapsGPU[lk], count);
        gpuErrchk(cudaGetLastError());
        return 0;
    }

    if (mycol == kcol_)
    {
        int_t lk = LBj(k, grid);
        if (Llu->Send_CommL[lk].ComQuant != NULL)
        {
            xlpanel_t<double> &lpanel = lPanelVec[g2lCol(k)];
            bool cuda_aware = superlu_cuda_aware_mpi();
            bool packed_any = false;

            for (int pc = 0; pc < grid->npcol; ++pc)
            {
                int size = Llu->Send_CommL[lk].ComQuant[pc].size;
                if (size <= 0)
                    continue;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symStatAdd(SYM_GPU3D_S_L2U_SEND_BYTES,
                           static_cast<long long>(size) *
                           static_cast<long long>(sizeof(double)));
#endif
                size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);
                double *sendbuf = symL2USendBufsGPU[flat];
                int_t *sendmap = symL2USendMapsGPU[flat];
                if (sendbuf == NULL || sendmap == NULL)
                    ABORT("SymFact GPU L2U buffer is missing.");

                int threads = 256;
                int blocks = (size + threads - 1) / threads;
                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                    lpanel.gpuPanel.val, sendbuf, sendmap, size);
                packed_any = true;
            }

            if (packed_any)
            {
                gpuErrchk(cudaGetLastError());
                if (!cuda_aware)
                {
                    for (int pc = 0; pc < grid->npcol; ++pc)
                    {
                        int size = Llu->Send_CommL[lk].ComQuant[pc].size;
                        if (size <= 0)
                            continue;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                        symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                                   static_cast<long long>(size) *
                                   static_cast<long long>(sizeof(double)));
#endif
                        size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                                      static_cast<size_t>(pc);
                        gpuErrchk(cudaMemcpyAsync(Llu->Send_CommL[lk].ComQuant[pc].dat,
                                                  symL2USendBufsGPU[flat],
                                                  sizeof(double) * static_cast<size_t>(size),
                                                  cudaMemcpyDeviceToHost, stream));
                    }
                }
                gpuErrchk(cudaStreamSynchronize(stream));
            }

            for (int pc = 0; pc < grid->npcol; ++pc)
            {
                int size = Llu->Send_CommL[lk].ComQuant[pc].size;
                if (size <= 0)
                    continue;

                size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);
                double *sendbuf = symL2USendBufsGPU[flat];
                double *hostbuf = Llu->Send_CommL[lk].ComQuant[pc].dat;
                int dest = PNUM(krow_, pc, grid);
                if (cuda_aware)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symStatAdd(SYM_GPU3D_S_L2U_CUDA_AWARE_SEND_BYTES,
                               static_cast<long long>(size) *
                               static_cast<long long>(sizeof(double)));
#endif
                    MPI_Isend(sendbuf, size, MPI_DOUBLE, dest,
                              SLU_MPI_TAG(5, k), grid->comm,
                              &Llu->Send_CommL[lk].req[pc]);
                }
                else
                {
                    MPI_Isend(hostbuf, size, MPI_DOUBLE, dest,
                              SLU_MPI_TAG(5, k), grid->comm,
                              &Llu->Send_CommL[lk].req[pc]);
                }
            }
        }
    }

    if (myrow == krow_)
    {
        int_t lk = LBi(k, grid);
        if (Llu->Recv_CommL[lk].ComQuant != NULL)
        {
            for (int p = 0; p < grid->nprow; ++p)
            {
                int size = Llu->Recv_CommL[lk].ComQuant[p].size;
                if (size > 0)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symStatAdd(SYM_GPU3D_S_L2U_RECV_BYTES,
                               static_cast<long long>(size) *
                               static_cast<long long>(sizeof(double)));
#endif
                    int src = PNUM(p, kcol_, grid);
                    MPI_Irecv(Llu->Recv_CommL[lk].ComQuant[p].dat, size,
                              MPI_DOUBLE, src, SLU_MPI_TAG(5, k), grid->comm,
                              &Llu->Recv_CommL[lk].req[p]);
                }
            }
        }
    }

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz,
                              T *DiagBlk, // device pointer
                              int_t LDD)
{
    if (isEmpty())
        return 0;
    T *lPanelStPtr = blkPtrGPU(0); // &val[blkPtrOffset(0)];
    int_t len = nzrows();
    if (haveDiag())
    {
        lPanelStPtr = blkPtrGPU(1); // &val[blkPtrOffset(1)];
        len -= nbrow(0);
    }

    T alpha = one<T>();

    cublasSetStream(handle, cuStream);
    cublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, &alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA());

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::panelSolveSymmetricGPU(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz,
                              T *DiagBlk, // device pointer
                              int_t LDD,
                              T *Work, // device pointer
                              int_t LDWork)
{
    if (isEmpty())
        return 0;

    T *lPanelStPtr = blkPtrGPU(0);
    int_t len = nzrows();
    if (haveDiag())
    {
        lPanelStPtr = blkPtrGPU(1);
        len -= nbrow(0);
    }

    if (len <= 0)
        return 0;
    if (LDWork < len)
        ABORT("Symmetric GPU L-panel workspace has an invalid leading dimension.");

    T alpha = one<T>();
    T beta = zeroT<T>();

    if (superlu_sym_gpu_fuse_lpanel())
    {
        size_t shared_bytes = static_cast<size_t>(ksupsz) * sizeof(T);
        if (shared_bytes <= 49152)
        {
            int threads = (ksupsz <= 128) ? 128 : 256;
            sym_lpanel_transform_inplace_kernel<T><<<static_cast<unsigned int>(len),
                                                     threads,
                                                     shared_bytes,
                                                     cuStream>>>(
                lPanelStPtr, LDA(), DiagBlk, LDD, len, ksupsz);
            gpuErrchk(cudaGetLastError());
            return 0;
        }
    }

    cublasSetStream(handle, cuStream);
    myCublasGemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    len, ksupsz, ksupsz, &alpha,
                    lPanelStPtr, LDA(),
                    DiagBlk, LDD, &beta,
                    Work, LDWork);

    gpuErrchk(cudaMemcpy2DAsync(lPanelStPtr, LDA() * sizeof(T),
                                Work, LDWork * sizeof(T),
                                len * sizeof(T), ksupsz,
                                cudaMemcpyDeviceToDevice, cuStream));
    return 0;
}

template <typename T>
int_t xlpanel_t<T>::diagFactorPackDiagBlockGPU(int_t k,
                                           T *UBlk, int_t LDU,     // CPU pointers
                                           T *DiagLBlk, int_t LDD, // CPU pointers
                                           T thresh, int_t *xsup,
                                           superlu_dist_options_t *options,
                                           SuperLUStat_t *stat, int *info)
{
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    gpuErrchk(cudaMemcpy2D(DiagLBlk, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToHost));

    // call dgetrf2
    dgstrf2(k, DiagLBlk, LDD, UBlk, LDU,
            thresh, xsup, options, stat, info);

    //copy back to device
    gpuErrchk(cudaMemcpy2D(val, spitch, DiagLBlk, dpitch,
                 width, height, cudaMemcpyHostToDevice));

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::diagFactorCuSolver(int_t k,
                                     cusolverDnHandle_t cusolverH, cudaStream_t cuStream,
                                    T *dWork, int* dInfo,  // GPU pointers 
                                    T *dDiagBuf, int_t LDD, // GPU pointers
                                    threshPivValType<T> thresh, int_t *xsup,
                                    superlu_dist_options_t *options,
                                    SuperLUStat_t *stat, int *info)
{
    // cudaStream_t stream = NULL;
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);
    
    gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
    gpuCusolverErrchk(myCusolverGetrf<T>(cusolverH, kSupSize, kSupSize, val, LDA(), dWork, NULL, dInfo));

    gpuErrchk(cudaMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToDevice, cuStream));
    gpuErrchk(cudaStreamSynchronize(cuStream));
    return 0;
}

template <typename T>
int_t xupanel_t<T>::panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz, T *DiagBlk, int_t LDD)
{
    if (isEmpty())
        return 0;

    T alpha = one<T>();
    
    cublasSetStream(handle, cuStream);
    cublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                    ksupsz, nzcols(), &alpha, DiagBlk, LDD,
                    blkPtrGPU(0), LDA());
    
    return 0; 
}

template <typename T>
int xupanel_t<T>::checkGPU()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    gpuErrchk(cudaMemcpy(tmpArr.data(), gpuPanel.val, valSize, cudaMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}
