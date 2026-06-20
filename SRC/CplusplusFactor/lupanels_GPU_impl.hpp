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
        cached = 1;
        return true;
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

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize; 
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

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize; 
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
    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLHostSendBufs.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLRecvSizes.empty() ||
        symV2PartnerLRecvIndex.empty() ||
        symV2PartnerLRecvMap.empty())
        ABORT("SymFact GPU L-fragment buffers are not allocated.");
    if (k < 0 || k >= nsupers)
        return 0;

    SYM_V2_TRACE_EXCHANGE(grid3d, k,
                          "enter L-fragment exchange myrow=%d mycol=%d krow=%d kcol=%d Lidx=%d",
                          static_cast<int>(myrow), static_cast<int>(mycol),
                          static_cast<int>(symV2DiagRoot(k)),
                          static_cast<int>(symV2PanelRoot(k)),
                          static_cast<int>(LidxSendCounts[k]));

    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];
    int_t kcol_ = symV2PanelRoot(k);
    int_t ksupc = SuperSize(k);
    int tag_ub = symFactTagUb;
    bool cuda_aware = superlu_cuda_aware_mpi();
    std::vector<int> send_sizes(Pc, 0);

    if (mycol == kcol_)
    {
        if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
            symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 true symmetric device diagonal block is missing.");

        int_t lk = symV2PanelIndex(k);
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        bool packed_any = false;
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

            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            sym_l2u_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                lpanel.gpuPanel.val, sendbuf, sendmap, size,
                lpanel.LDA(), symV2DiagBlocksGPU[k], ksupc);
            packed_any = true;
        }

        if (packed_any)
        {
            gpuErrchk(cudaGetLastError());
            if (!cuda_aware)
            {
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t flat = static_cast<size_t>(lk) *
                                      static_cast<size_t>(Pc) +
                                  static_cast<size_t>(pc);
                    int size = symV2PartnerLSendSizes[flat];
                    if (size <= 0)
                        continue;
                    if (symV2PartnerLHostSendBufs[flat].size() <
                        static_cast<size_t>(size))
                        ABORT("SymFact V2 true symmetric L-fragment host send buffer is too small.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symStatAdd(SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
                               static_cast<long long>(size) *
                                   static_cast<long long>(sizeof(double)));
#endif
                    gpuErrchk(cudaMemcpyAsync(
                        symV2PartnerLHostSendBufs[flat].data(),
                        symV2PartnerLSendBufsGPU[flat],
                        sizeof(double) * static_cast<size_t>(size),
                        cudaMemcpyDeviceToHost, stream));
                }
            }
            gpuErrchk(cudaStreamSynchronize(stream));
        }
    }

    size_t recv_count_base =
        static_cast<size_t>(k) * static_cast<size_t>(Pr);
    if (recv_count_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvSizes.size() ||
        recv_count_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvMap.size())
        ABORT("SymFact V2 true symmetric L-fragment receive sizes are missing.");

    std::vector<int> recv_sizes(Pr, 0);
    std::vector<int> recv_offsets(Pr, 0);
    int recv_total = 0;
    for (int pr = 0; pr < Pr; ++pr)
    {
        size_t pos = recv_count_base + static_cast<size_t>(pr);
        recv_sizes[pr] = symV2PartnerLRecvSizes[pos];
        if (recv_sizes[pr] > 0)
        {
            recv_offsets[pr] = recv_total;
            recv_total += recv_sizes[pr];
        }
    }
    if (recv_total > maxSymPartnerLvalCount)
        ABORT("SymFact V2 true symmetric L-fragment receive exceeds staging buffer.");
    if (recv_total > 0 && A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
        ABORT("SymFact V2 true symmetric L-fragment staging buffer is missing.");

    std::vector<MPI_Request> recv_reqs;
    std::vector<MPI_Request> send_reqs;
    std::vector<std::vector<double> > recv_buffers(cuda_aware ? 0 : Pr);
    recv_reqs.reserve(Pr);
    for (int pr = 0; pr < Pr; ++pr)
    {
        int size = recv_sizes[pr];
        if (size <= 0)
            continue;
        int src = PNUM(pr, kcol_, grid);
        MPI_Request req;
        double *recv_ptr = NULL;
        if (cuda_aware)
        {
            recv_ptr = A_gpu.symPartnerLStageBufs[stream_offset] +
                       recv_offsets[pr];
        }
        else
        {
            recv_buffers[pr].resize(size);
            recv_ptr = recv_buffers[pr].data();
        }
        MPI_Irecv(recv_ptr, size, MPI_DOUBLE, src,
                  SLU_MPI_TAG(5, k), grid->comm, &req);
        recv_reqs.push_back(req);
    }

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
            double *hostbuf = cuda_aware ? NULL
                                         : symV2PartnerLHostSendBufs[flat].data();
            if (sendbuf == NULL)
                ABORT("SymFact V2 true symmetric L-fragment send buffer is missing.");
            for (int pr = 0; pr < Pr; ++pr)
            {
                int dest = PNUM(pr, pc, grid);
                MPI_Request req;
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

    if (!recv_reqs.empty())
        MPI_Waitall(static_cast<int>(recv_reqs.size()), recv_reqs.data(),
                    MPI_STATUSES_IGNORE);

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
            double *stage = A_gpu.symPartnerLStageBufs[stream_offset];
            if (cuda_aware)
            {
                stage += recv_offsets[pr];
            }
            else
            {
                double *recv_data = recv_buffers[pr].data();
                gpuErrchk(cudaMemcpyAsync(
                    stage, recv_data,
                    sizeof(double) * static_cast<size_t>(count),
                    cudaMemcpyHostToDevice, stream));
            }

            size_t pos = 0;
            size_t end = static_cast<size_t>(count);
            size_t map_pos = 0;
            while (map_pos < recv_map.size())
            {
                if (map_pos + 2 > recv_map.size())
                    ABORT("SymFact V2 true symmetric L-fragment receive map is truncated.");
                int_t dst_offset = recv_map[map_pos++];
                int_t nrows = recv_map[map_pos++];
                if (dst_offset < 0 || nrows < 0 ||
                    dst_offset + nrows > frag_nrows)
                    ABORT("SymFact V2 true symmetric L-fragment receive map is invalid.");
                double *dst = A_gpu.symPartnerLvalRecvBufs[stream_offset] +
                              dst_offset;
                size_t need = static_cast<size_t>(nrows) *
                              static_cast<size_t>(ksupc);
                if (pos + need > end)
                    ABORT("SymFact V2 true symmetric L-fragment buffer is truncated.");
                gpuErrchk(cudaMemcpy2DAsync(
                    dst, sizeof(double) * static_cast<size_t>(frag_nrows),
                    stage + pos, sizeof(double) * static_cast<size_t>(nrows),
                    sizeof(double) * static_cast<size_t>(nrows),
                    static_cast<size_t>(ksupc),
                    cudaMemcpyDeviceToDevice, stream));
                pos += need;
            }
            if (pos != end)
                ABORT("SymFact V2 true symmetric L-fragment buffer has extra data.");
        }
    }

    if (!send_reqs.empty())
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);

    gpuErrchk(cudaStreamSynchronize(stream));
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
