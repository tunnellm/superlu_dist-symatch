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
        if (cuda_aware)
            ABORT("Pc-fragment CUDA-aware MPI is still fail-closed.");
        if (async_factor)
            ABORT("Pc-fragment async factor is still fail-closed.");
        if (superlu_sym_v2_pcfrag_cuda_aware_experiment() ||
            superlu_sym_v2_pcfrag_async_experiment())
            ABORT("Pc-fragment async/CUDA-aware experiment is fail-closed.");
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
    recv_reqs.clear();
    send_reqs.clear();
    recv_request_peers.clear();
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
        recv_request_peers.push_back(pr);
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAddBoth(SYM_GPU3D_T_LFRAG_RECV_POST,
                     SYM_GPU3D_T_PARTNER_LFRAG_RECV_POST,
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

    bool recv_h2d_issued = false;
    bool pipeline_large_receives = false;
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
        if (!pipeline_large_receives)
        {
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
                int mpi_rc = MPI_Waitsome(
                    request_count, recv_reqs.data(), &completed,
                    wait_indices.data(), wait_statuses.data());
                if (mpi_rc != MPI_SUCCESS || completed == MPI_UNDEFINED ||
                    completed <= 0 || completed > remaining)
                    ABORT("SymFact V2 progressive receive wait failed.");

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
                sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                    stage, A_gpu.symPartnerLvalRecvBufs[stream_offset],
                    recv_map_gpu, pieces, ksupc, frag_nrows);
                gpuErrchk(cudaGetLastError());
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
        if (!send_reqs.empty())
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double send_wait_t = SuperLU_timer_();
#endif
            MPI_Waitall(static_cast<int>(send_reqs.size()),
                        send_reqs.data(), MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                             SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
                             SuperLU_timer_() - send_wait_t);
#endif
            send_reqs.clear();
        }

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
                symStatAdd(SYM_GPU3D_S_L2U_SEND_REQUESTS,
                           static_cast<long long>(send_reqs.size()));
#endif
            }

            if (row_index.empty())
            {
                gpuErrchk(cudaMemcpyAsync(
                    A_gpu.symV2RowFragIdxRecvBufs[stream_offset], empty_header,
                    sizeof(int_t) * LPANEL_HEADER_SIZE,
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
#endif
                    MPI_Waitall(static_cast<int>(row_recv_reqs.size()),
                                row_recv_reqs.data(), MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAddBoth(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                                     SYM_GPU3D_T_ROW_LFRAG_MPI_RECV_WAIT,
                                     SuperLU_timer_() - row_recv_wait_t);
#endif
                }
                if (row_recv_total > 0)
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
                    sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                        stage, A_gpu.symV2RowFragValRecvBufs[stream_offset],
                        row_map_gpu, pieces, ksupc, row_nrows);
                    gpuErrchk(cudaGetLastError());
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
                if (slot >= symV2RowDownSendSizes.size() ||
                    slot >= symV2RowDownSendMapsGPU.size())
                    ABORT("SymFact V2 row-down plan send slot is invalid.");
                int total = symV2RowDownSendSizes[slot];
                if (total <= 0)
                    return 0;
                if (row_lpanel_ptr == NULL || row_lpanel_ptr->isEmpty())
                    ABORT("SymFact V2 row-down source L panel is missing.");
                if (total > maxSymV2RowFragValSendCount)
                    ABORT("SymFact V2 row-down packed destination exceeds send staging capacity.");
                int_t *sendmap = symV2RowDownSendMapsGPU[slot];
                if (sendmap == NULL)
                    ABORT("SymFact V2 row-down send map is missing.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_pack_issue_t = SuperLU_timer_();
#endif
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
                    row_lpanel_ptr->gpuPanel.val, dst_buf, sendmap, total);
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
                MPI_Wait(&req, MPI_STATUS_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                                 SYM_GPU3D_T_ROW_LFRAG_SEND_WAIT,
                                 SuperLU_timer_() - row_send_wait_t);
#endif
            }
        }

        if (row_index.empty())
        {
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symV2RowFragIdxRecvBufs[stream_offset], empty_header,
                sizeof(int_t) * LPANEL_HEADER_SIZE,
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
                MPI_Waitall(static_cast<int>(row_recv_reqs.size()),
                            row_recv_reqs.data(), MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAddBoth(SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
                                 SYM_GPU3D_T_ROW_LFRAG_MPI_RECV_WAIT,
                                 SuperLU_timer_() - row_recv_wait_t);
#endif
            }
            if (row_recv_total > 0 && row_src_pc != mycol)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double row_h2d_issue_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpyAsync(
                    row_l_direct_recv
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
            else if (row_recv_total > 0)
            {
                int self_total = row_pack_destination(
                    mycol,
                    row_l_direct_recv
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
            }
            else
            {
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
                    sym_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                        stage, A_gpu.symV2RowFragValRecvBufs[stream_offset],
                        row_map_gpu, pieces, ksupc, row_nrows);
                    gpuErrchk(cudaGetLastError());
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
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double send_wait_t = SuperLU_timer_();
#endif
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAddBoth(SYM_GPU3D_T_LFRAG_SEND_WAIT,
                         pc_fragment_schur
                             ? SYM_GPU3D_T_ROW_LFRAG_SEND_WAIT
                             : SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
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
