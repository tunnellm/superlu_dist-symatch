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

    auto release_taskflow_state = [](SymV2PcFragPanelTaskState &s) {
        for (size_t i = 0; i < s.row_pieces.size(); ++i)
        {
            if (s.row_pieces[i].pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a row piece with pending consumers.");
            if (s.row_pieces[i].ready_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(s.row_pieces[i].ready_event));
                s.row_pieces[i].ready_event = NULL;
            }
            if (s.row_pieces[i].done_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(s.row_pieces[i].done_event));
                s.row_pieces[i].done_event = NULL;
            }
            if (s.row_pieces[i].d_stage != NULL)
                gpuErrchk(cudaFree(s.row_pieces[i].d_stage));
            s.row_pieces[i].d_index = NULL;
            s.row_pieces[i].d_val = NULL;
            s.row_pieces[i].d_stage = NULL;
            s.row_pieces[i].h_index.clear();
        }
        for (size_t i = 0; i < s.partner_pieces.size(); ++i)
        {
            if (s.partner_pieces[i].pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a partner piece with pending consumers.");
            if (s.partner_pieces[i].ready_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(s.partner_pieces[i].ready_event));
                s.partner_pieces[i].ready_event = NULL;
            }
            if (s.partner_pieces[i].done_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(s.partner_pieces[i].done_event));
                s.partner_pieces[i].done_event = NULL;
            }
            if (s.partner_pieces[i].d_stage != NULL)
                gpuErrchk(cudaFree(s.partner_pieces[i].d_stage));
            s.partner_pieces[i].d_index = NULL;
            s.partner_pieces[i].d_val = NULL;
            s.partner_pieces[i].d_stage = NULL;
            s.partner_pieces[i].h_index.clear();
        }
        if (s.d_index_pool != NULL)
            gpuErrchk(cudaFree(s.d_index_pool));
        if (s.d_value_pool != NULL)
            gpuErrchk(cudaFree(s.d_value_pool));
        if (s.d_group_index_pool != NULL)
            gpuErrchk(cudaFree(s.d_group_index_pool));
        if (s.d_group_value_pool != NULL)
            gpuErrchk(cudaFree(s.d_group_value_pool));
        s.reset();
    };
    release_taskflow_state(state);
    state.k = k;
    state.stream_offset = stream_offset;
    state.initialized = 1;
    state.exchange_posted = 1;
    ++symV2PcFragTaskflowStats.taskflow_entries;

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
        [&](const std::vector<int_t> &frag, int_t b) -> std::vector<int_t> {
        if (frag.size() < LPANEL_HEADER_SIZE)
            ABORT("SymFact V2 Pc-fragment taskflow has truncated metadata.");
        int_t nb = frag[0];
        if (b < 0 || b >= nb)
            ABORT("SymFact V2 Pc-fragment taskflow block index is invalid.");
        int_t row_begin = strow_at(frag, b);
        int_t row_end = strow_at(frag, b + 1);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        int_t nrows = row_end - row_begin;
        size_t src_rows =
            static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * nb + 1 + row_begin);
        size_t src_rows_end = src_rows + static_cast<size_t>(nrows);
        if (src_rows_end > frag.size() || src_rows_end < src_rows)
            ABORT("SymFact V2 Pc-fragment taskflow row-list range is invalid.");

        std::vector<int_t> piece_index(
            static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * 1 + 1 + nrows));
        piece_index[0] = 1;
        piece_index[1] = nrows;
        piece_index[2] = 0;
        piece_index[3] = frag[3];
        piece_index[LPANEL_HEADER_SIZE] = gid_at(frag, b);
        piece_index[LPANEL_HEADER_SIZE + 1] = 0;
        piece_index[LPANEL_HEADER_SIZE + 2] = nrows;
        std::copy(frag.begin() + static_cast<std::ptrdiff_t>(src_rows),
                  frag.begin() + static_cast<std::ptrdiff_t>(src_rows_end),
                  piece_index.begin() + LPANEL_HEADER_SIZE + 3);
        return piece_index;
    };
    auto piece_index_count =
        [&](const std::vector<int_t> &frag, int_t b) -> size_t {
        int_t row_begin = strow_at(frag, b);
        int_t row_end = strow_at(frag, b + 1);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        return static_cast<size_t>(LPANEL_HEADER_SIZE + 2 * 1 + 1 +
                                   (row_end - row_begin));
    };
    auto piece_value_count =
        [&](const std::vector<int_t> &frag, int_t b) -> size_t {
        int_t row_begin = strow_at(frag, b);
        int_t row_end = strow_at(frag, b + 1);
        if (row_begin < 0 || row_end < row_begin)
            ABORT("SymFact V2 Pc-fragment taskflow block row range is invalid.");
        int_t ksupc = frag.size() >= 4 ? frag[3] : SuperSize(k);
        return static_cast<size_t>(row_end - row_begin) *
               static_cast<size_t>(ksupc);
    };

    int_t nr = nblocks(row_frag);
    int_t nc = nblocks(partner_frag);
    size_t total_index_count = 0;
    size_t total_value_count = 0;
    for (int_t b = 0; b < nr; ++b)
    {
        total_index_count += piece_index_count(row_frag, b);
        total_value_count += piece_value_count(row_frag, b);
    }
    for (int_t b = 0; b < nc; ++b)
    {
        total_index_count += piece_index_count(partner_frag, b);
        total_value_count += piece_value_count(partner_frag, b);
    }
    if (total_index_count > 0)
    {
        gpuErrchk(cudaMalloc(
            reinterpret_cast<void **>(&state.d_index_pool),
            sizeof(int_t) * total_index_count));
        state.index_pool_capacity = total_index_count;
    }
    if (total_value_count > 0)
    {
        gpuErrchk(cudaMalloc(
            reinterpret_cast<void **>(&state.d_value_pool),
            sizeof(double) * total_value_count));
        gpuErrchk(cudaMemset(
            state.d_value_pool, 0, sizeof(double) * total_value_count));
        state.value_pool_capacity = total_value_count;
    }
    if (total_index_count > 0)
    {
        gpuErrchk(cudaMalloc(
            reinterpret_cast<void **>(&state.d_group_index_pool),
            sizeof(int_t) * total_index_count));
        state.group_index_pool_capacity = total_index_count;
    }
    if (total_value_count > 0)
    {
        gpuErrchk(cudaMalloc(
            reinterpret_cast<void **>(&state.d_group_value_pool),
            sizeof(double) * total_value_count));
        state.group_value_pool_capacity = total_value_count;
    }

    auto add_piece = [&](std::vector<SymV2PcFragPieceDesc> &pieces,
                         const std::vector<int_t> &frag,
                         unsigned char kind, int_t b) {
        SymV2PcFragPieceDesc piece;
        piece.k = k;
        piece.kind = kind;
        piece.piece_id = static_cast<int>(pieces.size());
        piece.frag_blk_begin = b;
        piece.frag_blk_end = b + 1;
        piece.frag_row_offset = strow_at(frag, b);
        piece.nblocks = 1;
        piece.gid_first = gid_at(frag, b);
        piece.gid_last = piece.gid_first;
        piece.ksupc = frag.size() >= 4 ? frag[3] : SuperSize(k);
        piece.nrows = strow_at(frag, b + 1) - strow_at(frag, b);
        piece.lda = piece.nrows;
        piece.h_index = compact_piece_index(frag, b);
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
        gpuErrchk(cudaEventCreateWithFlags(
            &piece.ready_event, cudaEventDisableTiming));
        pieces.push_back(piece);
    };

    for (int_t b = 0; b < nr; ++b)
        add_piece(state.row_pieces, row_frag,
                  SYM_V2_PCFRAG_PIECE_ROW, b);
    for (int_t b = 0; b < nc; ++b)
        add_piece(state.partner_pieces, partner_frag,
                  SYM_V2_PCFRAG_PIECE_PARTNER, b);
    symV2PcFragTaskflowStats.row_pieces_created +=
        static_cast<long long>(state.row_pieces.size());
    symV2PcFragTaskflowStats.partner_pieces_created +=
        static_cast<long long>(state.partner_pieces.size());

    state.pair_task_index.assign(
        state.row_pieces.size() * state.partner_pieces.size(), -1);
    for (size_t rp = 0; rp < state.row_pieces.size(); ++rp)
    {
        int_t gi = state.row_pieces[rp].gid_first;
        for (size_t cp = 0; cp < state.partner_pieces.size(); ++cp)
        {
            int_t gj = state.partner_pieces[cp].gid_first;
            if (gi < gj)
                continue;
            if (gi == k || gj == k)
                continue;
            SymV2PcFragTaskDesc task;
            task.k = k;
            task.task_id = static_cast<int>(state.tasks.size());
            task.row_piece = static_cast<int>(rp);
            task.partner_piece = static_cast<int>(cp);
            task.row_piece_blk_begin = state.row_pieces[rp].frag_blk_begin;
            task.row_piece_blk_end = state.row_pieces[rp].frag_blk_end;
            task.partner_piece_blk_begin =
                state.partner_pieces[cp].frag_blk_begin;
            task.partner_piece_blk_end =
                state.partner_pieces[cp].frag_blk_end;
            task.gemm_m = state.row_pieces[rp].nrows;
            task.gemm_n = state.partner_pieces[cp].nrows;
            task.gemm_k = SuperSize(k);
            task.mode_mask = SYM_V2_PCFRAG_TASK_FULL |
                             SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL |
                             SYM_V2_PCFRAG_TASK_EXCLUDE;
            if (gi != gj)
                task.mode_mask |= SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW;
            task.outputs.push_back(SymV2PcFragOutputKey(gj, gi));
            state.tasks.push_back(task);
            state.pair_task_index[
                rp * state.partner_pieces.size() + cp] = task.task_id;
            ++state.row_pieces[rp].pending_consumers;
            ++state.partner_pieces[cp].pending_consumers;
        }
    }
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
        budget = superlu_sym_v2_pcfrag_taskflow_progress_budget();
    const bool strict_output_conflicts =
        superlu_sym_v2_pcfrag_taskflow_strict();

    auto all_pieces_ready = [&]() -> bool {
        for (size_t p = 0; p < state.row_pieces.size(); ++p)
            if (!state.row_pieces[p].ready)
                return false;
        for (size_t p = 0; p < state.partner_pieces.size(); ++p)
            if (!state.partner_pieces[p].ready)
                return false;
        return true;
    };
    auto output_locked = [&](const SymV2PcFragTaskDesc &task) -> bool {
        if (!strict_output_conflicts)
            return false;
        for (size_t o = 0; o < task.outputs.size(); ++o)
        {
            unsigned long long key = task.outputs[o].packed();
            if (std::find(state.active_output_keys.begin(),
                          state.active_output_keys.end(),
                          key) != state.active_output_keys.end())
                return true;
        }
        return false;
    };
    auto lock_outputs = [&](const SymV2PcFragTaskDesc &task) {
        if (!strict_output_conflicts)
            return;
        for (size_t o = 0; o < task.outputs.size(); ++o)
            state.active_output_keys.push_back(task.outputs[o].packed());
        symV2PcFragTaskflowStats.output_locks_acquired +=
            static_cast<long long>(task.outputs.size());
        if (static_cast<long long>(state.active_output_keys.size()) >
            symV2PcFragTaskflowStats.output_lock_high_water)
            symV2PcFragTaskflowStats.output_lock_high_water =
                static_cast<long long>(state.active_output_keys.size());
    };
    auto unlock_outputs = [&](const SymV2PcFragTaskDesc &task) {
        if (!strict_output_conflicts)
            return;
        for (size_t o = 0; o < task.outputs.size(); ++o)
        {
            unsigned long long key = task.outputs[o].packed();
            std::vector<unsigned long long>::iterator it =
                std::find(state.active_output_keys.begin(),
                          state.active_output_keys.end(), key);
            if (it != state.active_output_keys.end())
                state.active_output_keys.erase(it);
        }
    };
    auto release_completed_state = [&]() {
        auto release_piece_storage = [](SymV2PcFragPieceDesc &piece) {
            if (piece.pending_consumers != 0)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
            if (piece.ready_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(piece.ready_event));
                piece.ready_event = NULL;
            }
            if (piece.done_event != NULL)
            {
                gpuErrchk(cudaEventDestroy(piece.done_event));
                piece.done_event = NULL;
            }
            if (piece.d_stage != NULL)
            {
                gpuErrchk(cudaFree(piece.d_stage));
                piece.d_stage = NULL;
            }
            piece.d_index = NULL;
            piece.d_val = NULL;
            piece.h_index.clear();
        };
        for (size_t p = 0; p < state.row_pieces.size(); ++p)
            release_piece_storage(state.row_pieces[p]);
        for (size_t p = 0; p < state.partner_pieces.size(); ++p)
            release_piece_storage(state.partner_pieces[p]);
        if (state.d_index_pool != NULL)
            gpuErrchk(cudaFree(state.d_index_pool));
        if (state.d_value_pool != NULL)
            gpuErrchk(cudaFree(state.d_value_pool));
        if (state.d_group_index_pool != NULL)
            gpuErrchk(cudaFree(state.d_group_index_pool));
        if (state.d_group_value_pool != NULL)
            gpuErrchk(cudaFree(state.d_group_value_pool));
        state.reset();
    };

    int launched = 0;
    for (size_t i = 0;
         i < state.tasks.size() && launched < budget; ++i)
    {
        SymV2PcFragTaskDesc &task = state.tasks[i];
        if (task.complete)
            continue;
        SymV2PcFragPieceDesc &row =
            state.row_pieces[static_cast<size_t>(task.row_piece)];
        SymV2PcFragPieceDesc &col =
            state.partner_pieces[static_cast<size_t>(task.partner_piece)];
        if (!row.ready)
            ++symV2PcFragTaskflowStats.tasks_blocked_row;
        if (!col.ready)
            ++symV2PcFragTaskflowStats.tasks_blocked_partner;
        if (!row.ready || !col.ready)
            continue;
        if (output_locked(task))
        {
            ++symV2PcFragTaskflowStats.tasks_blocked_output;
            ++symV2PcFragTaskflowStats.scatter_conflict_waits;
            continue;
        }
        if (row.d_index == NULL || row.d_val == NULL ||
            col.d_index == NULL || col.d_val == NULL ||
            gemmBuff == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW progress task is missing owned device storage.");
        if (static_cast<int64_t>(row.nrows) *
                static_cast<int64_t>(col.nrows) >
            static_cast<int64_t>(A_gpu.gemmBufferSize))
            ABORT("GPU3DV2_PCFRAG_TASKFLOW progress task tiling is not implemented.");
        if (!all_pieces_ready())
            ++symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready;
        lock_outputs(task);
        if (row.ready_event != NULL)
            gpuErrchk(cudaStreamWaitEvent(stream, row.ready_event, 0));
        if (col.ready_event != NULL)
            gpuErrchk(cudaStreamWaitEvent(stream, col.ready_event, 0));
        dSymSchurCompUpdateTaskDualPiecesGPU(
            k,
            row.h_index, col.h_index,
            row.d_index, row.d_val,
            col.d_index, col.d_val,
            handle, stream, gemmBuff);
        gpuErrchk(cudaStreamSynchronize(stream));
        unlock_outputs(task);
        task.launched = 1;
        task.complete = 1;
        --state.incomplete_task_count;
        --row.pending_consumers;
        --col.pending_consumers;
        if (row.pending_consumers < 0 || col.pending_consumers < 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
        ++symV2PcFragTaskflowStats.tasks_launched;
        ++symV2PcFragTaskflowStats.tasks_completed;
        ++symV2PcFragTaskflowStats.tasks_launched_progress;
        ++launched;
    }
    if (state.incomplete_task_count == 0)
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
        SymV2PcFragPanelTaskState &state =
            symV2PcFragTaskStates[static_cast<size_t>(k)];
        if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
            streamId = state.stream_offset >= 0 ? state.stream_offset : 0;
        if (streamId < 0 || streamId >= A_gpu.numCudaStreams)
            streamId = 0;
        cudaStream_t stream = A_gpu.cuStreams[streamId];
        cublasHandle_t handle = A_gpu.cuHandles[streamId];
        double *gemmBuff = A_gpu.gpuGemmBuffs[streamId];
        const bool strict_output_conflicts =
            superlu_sym_v2_pcfrag_taskflow_strict();

        auto all_pieces_ready = [&]() -> bool {
            for (size_t p = 0; p < state.row_pieces.size(); ++p)
                if (!state.row_pieces[p].ready)
                    return false;
            for (size_t p = 0; p < state.partner_pieces.size(); ++p)
                if (!state.partner_pieces[p].ready)
                    return false;
            return true;
        };
        auto output_locked = [&](const SymV2PcFragTaskDesc &task) -> bool {
            if (!strict_output_conflicts)
                return false;
            for (size_t o = 0; o < task.outputs.size(); ++o)
            {
                unsigned long long key = task.outputs[o].packed();
                if (std::find(state.active_output_keys.begin(),
                              state.active_output_keys.end(),
                              key) != state.active_output_keys.end())
                    return true;
            }
            return false;
        };
        auto lock_outputs = [&](const SymV2PcFragTaskDesc &task) {
            if (!strict_output_conflicts)
                return;
            for (size_t o = 0; o < task.outputs.size(); ++o)
                state.active_output_keys.push_back(task.outputs[o].packed());
            symV2PcFragTaskflowStats.output_locks_acquired +=
                static_cast<long long>(task.outputs.size());
            if (static_cast<long long>(state.active_output_keys.size()) >
                symV2PcFragTaskflowStats.output_lock_high_water)
                symV2PcFragTaskflowStats.output_lock_high_water =
                    static_cast<long long>(state.active_output_keys.size());
        };
        auto unlock_outputs = [&](const SymV2PcFragTaskDesc &task) {
            if (!strict_output_conflicts)
                return;
            for (size_t o = 0; o < task.outputs.size(); ++o)
            {
                unsigned long long key = task.outputs[o].packed();
                std::vector<unsigned long long>::iterator it =
                    std::find(state.active_output_keys.begin(),
                              state.active_output_keys.end(), key);
                if (it != state.active_output_keys.end())
                    state.active_output_keys.erase(it);
            }
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
        auto release_completed_state = [&]() {
            auto release_piece_storage = [](SymV2PcFragPieceDesc &piece) {
                if (piece.pending_consumers != 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
                if (piece.ready_event != NULL)
                {
                    gpuErrchk(cudaEventDestroy(piece.ready_event));
                    piece.ready_event = NULL;
                }
                if (piece.done_event != NULL)
                {
                    gpuErrchk(cudaEventDestroy(piece.done_event));
                    piece.done_event = NULL;
                }
                if (piece.d_stage != NULL)
                {
                    gpuErrchk(cudaFree(piece.d_stage));
                    piece.d_stage = NULL;
                }
                piece.d_index = NULL;
                piece.d_val = NULL;
                piece.h_index.clear();
            };
            for (size_t p = 0; p < state.row_pieces.size(); ++p)
                release_piece_storage(state.row_pieces[p]);
            for (size_t p = 0; p < state.partner_pieces.size(); ++p)
                release_piece_storage(state.partner_pieces[p]);
            if (state.d_index_pool != NULL)
                gpuErrchk(cudaFree(state.d_index_pool));
            if (state.d_value_pool != NULL)
                gpuErrchk(cudaFree(state.d_value_pool));
            if (state.d_group_index_pool != NULL)
                gpuErrchk(cudaFree(state.d_group_index_pool));
            if (state.d_group_value_pool != NULL)
                gpuErrchk(cudaFree(state.d_group_value_pool));
            state.reset();
        };

        if ((mode_mask & SYM_V2_PCFRAG_TASK_FULL) &&
            superlu_sym_v2_pcfrag_taskflow_eager() &&
            !superlu_sym_v2_pcfrag_taskflow_validate())
        {
            for (size_t i = 0; i < state.tasks.size(); ++i)
            {
                SymV2PcFragTaskDesc &task = state.tasks[i];
                if (task.complete)
                    continue;
                SymV2PcFragPieceDesc &row =
                    state.row_pieces[static_cast<size_t>(task.row_piece)];
                SymV2PcFragPieceDesc &col =
                    state.partner_pieces[static_cast<size_t>(task.partner_piece)];
                if (!row.ready)
                    ++symV2PcFragTaskflowStats.tasks_blocked_row;
                if (!col.ready)
                    ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                if (!row.ready || !col.ready)
                    continue;
                if (output_locked(task))
                {
                    ++symV2PcFragTaskflowStats.tasks_blocked_output;
                    ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                    continue;
                }
                if (row.d_index == NULL || row.d_val == NULL ||
                    col.d_index == NULL || col.d_val == NULL ||
                    gemmBuff == NULL)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW task is missing owned device storage.");
                if (static_cast<int64_t>(row.nrows) *
                        static_cast<int64_t>(col.nrows) >
                    static_cast<int64_t>(A_gpu.gemmBufferSize))
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW eager task tiling is not implemented.");
                if (!all_pieces_ready())
                    ++symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready;
                lock_outputs(task);
                if (row.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(stream, row.ready_event, 0));
                if (col.ready_event != NULL)
                    gpuErrchk(cudaStreamWaitEvent(stream, col.ready_event, 0));
                dSymSchurCompUpdateTaskDualPiecesGPU(
                    k,
                    row.h_index, col.h_index,
                    row.d_index, row.d_val,
                    col.d_index, col.d_val,
                    handle, stream, gemmBuff);
                gpuErrchk(cudaStreamSynchronize(stream));
                unlock_outputs(task);
                task.launched = 1;
                task.complete = 1;
                --state.incomplete_task_count;
                --row.pending_consumers;
                --col.pending_consumers;
                if (row.pending_consumers < 0 || col.pending_consumers < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW pending consumer count underflowed.");
                ++symV2PcFragTaskflowStats.tasks_launched;
                ++symV2PcFragTaskflowStats.tasks_completed;
                ++symV2PcFragTaskflowStats.tasks_launched_eager_full;
            }
            if (state.incomplete_task_count == 0)
            {
                release_completed_state();
                return 0;
            }
            if (!drain)
                return 0;
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
                static_cast<size_t>(rb) >= state.row_pieces.size() ||
                static_cast<size_t>(cb) >= state.partner_pieces.size() ||
                state.partner_pieces.empty())
                return NULL;
            size_t pos = static_cast<size_t>(rb) *
                             state.partner_pieces.size() +
                         static_cast<size_t>(cb);
            if (pos >= state.pair_task_index.size())
                return NULL;
            int tid = state.pair_task_index[pos];
            if (tid < 0 || static_cast<size_t>(tid) >= state.tasks.size())
                return NULL;
            return &state.tasks[static_cast<size_t>(tid)];
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
        auto launch_group =
            [&](int_t row_start, int_t row_end,
                int_t col_start, int_t col_end,
                cublasHandle_t group_handle,
                cudaStream_t group_stream,
                double *group_gemm,
                unsigned launch_mode) {
            if (row_start >= row_end || col_start >= col_end)
                return;
            int pair_count = 0;
            for (int_t rb = row_start; rb < row_end; ++rb)
            {
                if (!state.row_pieces[static_cast<size_t>(rb)].ready)
                {
                    ++symV2PcFragTaskflowStats.tasks_blocked_row;
                    return;
                }
                for (int_t cb = col_start; cb < col_end; ++cb)
                {
                    if (!state.partner_pieces[static_cast<size_t>(cb)].ready)
                    {
                        ++symV2PcFragTaskflowStats.tasks_blocked_partner;
                        return;
                    }
                    SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                    if (task != NULL && !task->complete &&
                        task_matches_launch_mode(*task, launch_mode))
                        ++pair_count;
                }
            }
            if (pair_count == 0)
                return;
            std::vector<SymV2PcFragTaskDesc *> locked_tasks;
            if (strict_output_conflicts)
            {
                std::vector<unsigned long long> pending_keys =
                    state.active_output_keys;
                for (int_t rb = row_start; rb < row_end; ++rb)
                {
                    for (int_t cb = col_start; cb < col_end; ++cb)
                    {
                        SymV2PcFragTaskDesc *task = pair_task(rb, cb);
                        if (task == NULL || task->complete ||
                            !task_matches_launch_mode(*task, launch_mode))
                            continue;
                        for (size_t o = 0; o < task->outputs.size(); ++o)
                        {
                            unsigned long long key =
                                task->outputs[o].packed();
                            if (std::find(pending_keys.begin(),
                                          pending_keys.end(), key) !=
                                pending_keys.end())
                            {
                                ++symV2PcFragTaskflowStats.tasks_blocked_output;
                                ++symV2PcFragTaskflowStats.scatter_conflict_waits;
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped dispatch found an output conflict.");
                            }
                            pending_keys.push_back(key);
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
            if (static_cast<int64_t>(row_lda) *
                    static_cast<int64_t>(col_lda) >
                static_cast<int64_t>(A_gpu.gemmBufferSize))
                ABORT("GPU3DV2_PCFRAG_TASKFLOW grouped task tiling is not implemented.");
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
            gpuErrchk(cudaStreamSynchronize(group_stream));
            for (size_t i = 0; i < locked_tasks.size(); ++i)
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
            const int64_t gemm_capacity = SUPERLU_MAX(
                static_cast<int64_t>(1),
                static_cast<int64_t>(A_gpu.gemmBufferSize));
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
        if (state.incomplete_task_count == 0)
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
    auto release_piece_storage = [](SymV2PcFragPieceDesc &piece) {
        if (piece.pending_consumers != 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW attempted to release a piece with pending consumers.");
        if (piece.ready_event != NULL)
        {
            gpuErrchk(cudaEventDestroy(piece.ready_event));
            piece.ready_event = NULL;
        }
        if (piece.done_event != NULL)
        {
            gpuErrchk(cudaEventDestroy(piece.done_event));
            piece.done_event = NULL;
        }
        if (piece.d_stage != NULL)
        {
            gpuErrchk(cudaFree(piece.d_stage));
            piece.d_stage = NULL;
        }
        piece.d_index = NULL;
        piece.d_val = NULL;
        piece.h_index.clear();
    };
    for (size_t i = 0; i < state.row_pieces.size(); ++i)
        release_piece_storage(state.row_pieces[i]);
    for (size_t i = 0; i < state.partner_pieces.size(); ++i)
        release_piece_storage(state.partner_pieces[i]);
    if (state.d_index_pool != NULL)
        gpuErrchk(cudaFree(state.d_index_pool));
    if (state.d_value_pool != NULL)
        gpuErrchk(cudaFree(state.d_value_pool));
    if (state.d_group_index_pool != NULL)
        gpuErrchk(cudaFree(state.d_group_index_pool));
    if (state.d_group_value_pool != NULL)
        gpuErrchk(cudaFree(state.d_group_value_pool));
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
    auto taskflow_assemble_owned_pieces =
        [&](unsigned char kind, const double *stage,
            const std::vector<int_t> &recv_map) {
        if (!pcfrag_taskflow)
            return;
        if (taskflow_state == NULL || !taskflow_state->initialized)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly has no state.");
        if (stage == NULL)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece assembly has no source.");
        if (recv_map.empty())
            return;
        if (recv_map.size() % 3 != 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW piece receive map has invalid stride.");
        std::vector<SymV2PcFragPieceDesc> &pieces =
            (kind == SYM_V2_PCFRAG_PIECE_ROW)
                ? taskflow_state->row_pieces
                : taskflow_state->partner_pieces;
        size_t map_pos = 0;
        while (map_pos < recv_map.size())
        {
            int_t dst_offset = recv_map[map_pos++];
            int_t nrows = recv_map[map_pos++];
            int_t src_offset = recv_map[map_pos++];
            if (nrows <= 0)
                continue;
            SymV2PcFragPieceDesc *piece = NULL;
            for (size_t p = 0; p < pieces.size(); ++p)
            {
                int_t begin = pieces[p].frag_row_offset;
                int_t end = begin + pieces[p].nrows;
                if (dst_offset >= begin && dst_offset + nrows <= end)
                {
                    piece = &pieces[p];
                    break;
                }
            }
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
                    gpuErrchk(cudaEventRecord(piece->ready_event, stream));
                piece->ready = 1;
                if (kind == SYM_V2_PCFRAG_PIECE_ROW)
                    ++symV2PcFragTaskflowStats.row_pieces_ready;
                else
                    ++symV2PcFragTaskflowStats.partner_pieces_ready;
                if (pcfrag_taskflow_eager)
                    dSymV2PcFragTaskflowProgressGPU(
                        k,
                        superlu_sym_v2_pcfrag_taskflow_progress_budget());
            }
        }
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
                gpuErrchk(cudaEventRecord(piece.ready_event, stream));
            piece.ready = 1;
            if (kind == SYM_V2_PCFRAG_PIECE_ROW)
                ++symV2PcFragTaskflowStats.row_pieces_ready;
            else
                ++symV2PcFragTaskflowStats.partner_pieces_ready;
            if (pcfrag_taskflow_eager)
                dSymV2PcFragTaskflowProgressGPU(
                    k, superlu_sym_v2_pcfrag_taskflow_progress_budget());
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
            if (static_cast<size_t>(stream_offset) >=
                    symPartnerLvalRecvBufs.size() ||
                symPartnerLvalRecvBufs[stream_offset] == NULL)
                ABORT("SymFact V2 host receive staging buffer is missing.");
            recv_host_base = symPartnerLvalRecvBufs[stream_offset];
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
                        double *host_stage =
                            (flat < symV2PartnerLHostSendBufsPinned.size() &&
                             symV2PartnerLHostSendBufsPinned[flat] != NULL)
                                ? symV2PartnerLHostSendBufsPinned[flat]
                                : (symV2PartnerLHostSendBufs[flat].empty()
                                       ? NULL
                                       : symV2PartnerLHostSendBufs[flat].data());
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
                    hostbuf =
                        (flat < symV2PartnerLHostSendBufsPinned.size() &&
                         symV2PartnerLHostSendBufsPinned[flat] != NULL)
                            ? symV2PartnerLHostSendBufsPinned[flat]
                            : (symV2PartnerLHostSendBufs[flat].empty()
                                   ? NULL
                                   : symV2PartnerLHostSendBufs[flat].data());
                    if (hostbuf == NULL)
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
    if (pcfrag_async_pipeline)
        deferred_partner_send_req_count = send_reqs.size();

    bool recv_h2d_issued = false;
    bool pipeline_large_receives = false;
    std::vector<unsigned char> taskflow_partner_progressive_assembled;
    if (pcfrag_taskflow)
        taskflow_partner_progressive_assembled.assign(
            static_cast<size_t>(Pr), 0);
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
                        taskflow_partner_progressive_assembled[
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
                    if (static_cast<size_t>(pr) >=
                            taskflow_partner_progressive_assembled.size() ||
                        !taskflow_partner_progressive_assembled[
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
        if (!pcfrag_async_pipeline && !send_reqs.empty())
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
                    symV2RowFragHostSendBufs[stream_offset],
                    A_gpu.symV2RowFragStageBufs[stream_offset],
                    sizeof(double) * static_cast<size_t>(row_send_total),
                    cudaMemcpyDeviceToHost, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                                 SYM_GPU3D_T_ROW_LFRAG_D2H_STAGE_ISSUE,
                                 SuperLU_timer_() - row_d2h_issue_t);
                double row_pack_stage_sync_t = SuperLU_timer_();
#endif
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
                    MPI_Isend(symV2RowFragHostSendBufs[stream_offset] + off,
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
                if (static_cast<size_t>(stream_offset) >=
                        symV2RowFragHostRecvBufs.size() ||
                    symV2RowFragHostRecvBufs[stream_offset] == NULL)
                    ABORT("SymFact V2 row-fragment host receive staging is missing.");
                row_recv_host_base = symV2RowFragHostRecvBufs[stream_offset];
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
                    if (pcfrag_taskflow_validate &&
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
            while (remaining > 0)
            {
                int completed = 0;
                ++symV2PcFragTaskflowStats.producer_send_wait_calls;
                int mpi_rc = MPI_Waitsome(
                    request_count, reqs, &completed, wait_indices.data(),
                    wait_statuses.data());
                if (mpi_rc != MPI_SUCCESS ||
                    completed == MPI_UNDEFINED || completed <= 0 ||
                    completed > remaining)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW send progressive wait failed.");
                symV2PcFragTaskflowStats.producer_mpi_wait_requests +=
                    static_cast<long long>(completed);
                dSymV2PcFragTaskflowProgressGPU(
                    k, superlu_sym_v2_pcfrag_taskflow_progress_budget());
                remaining -= completed;
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

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double stream_sync_t = SuperLU_timer_();
#endif
    gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_LFRAG_STREAM_SYNC,
                 SuperLU_timer_() - stream_sync_t);
    symTimingAdd(SYM_GPU3D_T_LFRAG_EXCHANGE_TOTAL,
                 SuperLU_timer_() - lfrag_total_t);
#endif
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
