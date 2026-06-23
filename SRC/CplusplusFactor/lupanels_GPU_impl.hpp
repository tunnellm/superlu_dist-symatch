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
    symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
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
    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLHostSendBufs.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLRecvSizes.empty() ||
        symV2PartnerLRecvIndex.empty() ||
        symV2PartnerLRecvMap.empty() ||
        symV2PartnerLRecvMapsGPU.empty())
        ABORT("SymFact GPU L-fragment buffers are not allocated.");
    if (k < 0 || k >= nsupers)
        return 0;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double lfrag_total_t = SuperLU_timer_();
#endif
    long long sym_v2_partner_payload_bytes = 0;

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
    const bool pooled_staging =
        !cuda_aware && superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool();
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
            double *sendbuf = symV2PartnerLSendBufsGPU[flat];
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
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        if (!prepacked)
            symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
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
                    gpuErrchk(cudaMemcpyAsync(
                        host_stage,
                        symV2PartnerLSendBufsGPU[flat],
                        sizeof(double) * static_cast<size_t>(size),
                        cudaMemcpyDeviceToHost, stream));
                }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
                             SuperLU_timer_() - d2h_issue_t);
#endif
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double pack_stage_sync_t = SuperLU_timer_();
#endif
            gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                         SuperLU_timer_() - pack_stage_sync_t);
#endif
        }
    }

    size_t recv_count_base =
        static_cast<size_t>(k) * static_cast<size_t>(Pr);
    if (recv_count_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvSizes.size() ||
        recv_count_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvMap.size())
        ABORT("SymFact V2 true symmetric L-fragment receive sizes are missing.");

    std::vector<int> local_recv_sizes;
    std::vector<int> local_recv_offsets;
    std::vector<int> &recv_sizes = pooled_staging
        ? symV2ExchangeRecvSizesScratch
        : local_recv_sizes;
    std::vector<int> &recv_offsets = pooled_staging
        ? symV2ExchangeRecvOffsetsScratch
        : local_recv_offsets;
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
    int recv_total = 0;
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

    std::vector<MPI_Request> local_recv_reqs;
    std::vector<MPI_Request> local_send_reqs;
    std::vector<MPI_Request> &recv_reqs = pooled_staging
        ? symV2ExchangeRecvReqsScratch
        : local_recv_reqs;
    std::vector<MPI_Request> &send_reqs = pooled_staging
        ? symV2ExchangeSendReqsScratch
        : local_send_reqs;
    recv_reqs.clear();
    send_reqs.clear();
    double *recv_host_base = NULL;
    if (!cuda_aware && recv_total > 0)
    {
        if (static_cast<size_t>(stream_offset) >=
                symPartnerLvalRecvBufs.size() ||
            symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 host receive staging buffer is missing.");
        recv_host_base = symPartnerLvalRecvBufs[stream_offset];
    }
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
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_LFRAG_RECV_POST,
                 SuperLU_timer_() - recv_post_t);
    symStatAdd(SYM_GPU3D_S_L2U_RECV_REQUESTS,
               static_cast<long long>(recv_reqs.size()));
#endif

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double send_post_t = SuperLU_timer_();
#endif
    if (mycol == kcol_)
    {
        int_t send_lk = symV2PanelIndex(k);
        send_reqs.reserve(static_cast<size_t>(Pr) * static_cast<size_t>(Pc));
        for (int pc = 0; pc < Pc; ++pc)
        {
            int size = send_sizes[pc];
            size_t flat = static_cast<size_t>(send_lk) * static_cast<size_t>(Pc) +
                          static_cast<size_t>(pc);
            if (size <= 0)
                continue;
            double *sendbuf = symV2PartnerLSendBufsGPU[flat];
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
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_LFRAG_SEND_POST,
                 SuperLU_timer_() - send_post_t);
    symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS,
               static_cast<long long>(send_reqs.size()));
#endif

    if (!recv_reqs.empty())
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double recv_wait_t = SuperLU_timer_();
#endif
        MPI_Waitall(static_cast<int>(recv_reqs.size()),
                    recv_reqs.data(),
                    MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                     SuperLU_timer_() - recv_wait_t);
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
    if (cached_index.empty())
    {
        gpuErrchk(cudaMemcpyAsync(A_gpu.symPartnerLidxRecvBufs[stream_offset],
                                  empty_header,
                                  sizeof(int_t) * LPANEL_HEADER_SIZE,
                                  cudaMemcpyHostToDevice, stream));
    }
    else
    {
        gpuErrchk(cudaMemcpyAsync(A_gpu.symPartnerLidxRecvBufs[stream_offset],
                                  cached_index.data(),
                                  sizeof(int_t) *
                                      static_cast<size_t>(frag_index_size),
                                  cudaMemcpyHostToDevice, stream));
    }

    if (!cuda_aware && !recv_reqs.empty())
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double h2d_issue_t = SuperLU_timer_();
#endif
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLStageBufs[stream_offset], recv_host_base,
            sizeof(double) * static_cast<size_t>(recv_total),
            cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                     SuperLU_timer_() - h2d_issue_t);
#endif
    }

    if (frag_nblocks > 0 && frag_nrows > 0)
    {
        if (A_gpu.symPartnerLvalRecvBufs[stream_offset] == NULL ||
            A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 true symmetric L-fragment device buffers are missing.");
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
                if (self_flat >= symV2PartnerLSendBufsGPU.size() ||
                    self_flat >= symV2PartnerLSendSizes.size() ||
                    symV2PartnerLSendBufsGPU[self_flat] == NULL ||
                    symV2PartnerLSendSizes[self_flat] != count)
                    ABORT("SymFact V2 self fragment buffer is invalid.");
                stage = symV2PartnerLSendBufsGPU[self_flat];
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
                sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                    stage, A_gpu.symPartnerLvalRecvBufs[stream_offset],
                    recv_map_gpu, pieces, ksupc, frag_nrows);
                gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
                             SuperLU_timer_() - assemble_issue_t);
#endif
            }
            if (pos != end)
                ABORT("SymFact V2 true symmetric L-fragment buffer has extra data.");
        }
    }

    if (!send_reqs.empty())
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double send_wait_t = SuperLU_timer_();
#endif
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                     SuperLU_timer_() - send_wait_t);
#endif
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
    symV2PayloadProfileAdd(SYM_V2_PAYLOAD_PARTNER_CALL,
                           sym_v2_partner_payload_bytes);
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
