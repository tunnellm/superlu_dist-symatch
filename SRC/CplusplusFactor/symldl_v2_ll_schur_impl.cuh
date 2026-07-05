#pragma once

#include <algorithm>
#include <cstdint>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_zero();

template <>
__device__ inline double symldl_v2_ll_zero<double>()
{
    return 0.0;
}

template <>
__device__ inline float symldl_v2_ll_zero<float>()
{
    return 0.0f;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_zero<doublecomplex>()
{
    doublecomplex z = {0.0, 0.0};
    return z;
}

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_mul(Ftype a, Ftype b)
{
    return a * b;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_mul<doublecomplex>(
    doublecomplex a, doublecomplex b)
{
    doublecomplex z = {
        a.r * b.r - a.i * b.i,
        a.r * b.i + a.i * b.r
    };
    return z;
}

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_add(Ftype a, Ftype b)
{
    return a + b;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_add<doublecomplex>(
    doublecomplex a, doublecomplex b)
{
    doublecomplex z = {a.r + b.r, a.i + b.i};
    return z;
}

template <typename Ftype>
__global__ void symldl_v2_build_raw_l_range_kernel(
    Ftype *rawBlock, int_t ldraw,
    xlpanelGPU_t<Ftype> lpanel,
    int_t jSt, int_t jEnd,
    const Ftype *diag, int_t diag_ld)
{
    int_t ncols = lpanel.stRow(jEnd) - lpanel.stRow(jSt);
    int_t ksupc = diag_ld;
    int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int_t total = ksupc * ncols;
    if (idx >= total)
        return;

    int_t lrow_local = idx % ncols;
    int_t diag_col = idx / ncols;
    int_t lrow = lpanel.stRow(jSt) + lrow_local;
    Ftype sum = symldl_v2_ll_zero<Ftype>();
    for (int_t p = 0; p < ksupc; ++p)
        sum = symldl_v2_ll_add(
            sum,
            symldl_v2_ll_mul(lpanel.val[lrow + p * lpanel.LDA()],
                             diag[p + diag_col * diag_ld]));
    rawBlock[lrow_local + diag_col * ldraw] = sum;
}

static __device__ void symldl_v2_ll_compute_indirect_map(
    int *src_to_dst, int_t src_len, int_t *src_vec,
    int_t dst_len, int_t *dst_vec, int *dst_idx)
{
    int thread_id = threadIdx.x;
    if (dst_vec == NULL)
    {
        if (thread_id < src_len)
            src_to_dst[thread_id] = static_cast<int>(src_vec[thread_id]);
        __syncthreads();
        return;
    }

    if (thread_id < src_len)
        dst_idx[src_vec[thread_id]] = GLOBAL_BLOCK_NOT_FOUND;
    __syncthreads();

    if (thread_id < dst_len)
        dst_idx[dst_vec[thread_id]] = thread_id;
    __syncthreads();

    if (thread_id < src_len)
        src_to_dst[thread_id] = dst_idx[src_vec[thread_id]];
    __syncthreads();
}

template <typename Ftype>
__device__ void symldl_v2_scatter_ll_range_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = lpanel.gid(ii);
    int_t gj = lpanel.gid(jj);
    if (gi < gj)
        return;

    int_t lj = dA->lPanelIndex(gj);
    if (lj < 0)
        return;
    int_t li = dA->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return;

    Ftype *dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dst_row_len = dA->lPanelVec[lj].nbrow(li);
    int_t *dst_row_list = dA->lPanelVec[lj].rowList(li);
    int_t dst_col_len = dA->supersize(gj);

    int nrows = static_cast<int>(lpanel.nbrow(ii));
    int ncols = static_cast<int>(lpanel.nbrow(jj));
    if (nrows <= 0 || ncols <= 0)
        return;

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    symldl_v2_ll_compute_indirect_map(
        row_src_to_dst, nrows, lpanel.rowList(ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_ll_compute_indirect_map(
        col_src_to_dst, ncols, lpanel.rowList(jj),
        dst_col_len, NULL, dst_idx);

    int cols_per_thread_block = blockDim.x / nrows;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(lpanel.stRow(ii) -
                                   lpanel.stRow(iSt));
    int col_off = static_cast<int>(lpanel.stRow(jj) -
                                   lpanel.stRow(jSt));
    Ftype *src = &gemmBuff[row_off + col_off * LDgemmBuff];

    if (thread_id < nrows * cols_per_thread_block)
    {
        int i = thread_id % nrows;
        int j = thread_id / nrows;
        while (j < ncols)
        {
            int di = row_src_to_dst[i];
            int dj = col_src_to_dst[j];
            if (di >= 0 && di < dst_row_len &&
                dj >= 0 && dj < dst_col_len)
                atomicAddT<Ftype>(&dst[di + lddst * dj],
                                  -src[i + LDgemmBuff * j]);
            j += cols_per_thread_block;
        }
    }
}

template <typename Ftype>
__global__ void symldl_v2_scatter_ll_range_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_ll_range_dev(
        iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_ll_range(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_ll_range_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
    gpuErrchk(cudaGetLastError());
}

template <typename Ftype>
static int_t symldl_v2_ll_part_update(
    xLUstruct_t<Ftype> *lu,
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *rawBlock, Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty())
        return 0;
    if (lu->symV2DiagBlocksGPU.size() !=
            static_cast<size_t>(lu->nsupers) ||
        lu->symV2DiagBlocksGPU[static_cast<size_t>(k)] == NULL)
        ABORT("SymFact V2 LL update diagonal block is missing.");

    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = lpanel.stRow(jEnd) - lpanel.stRow(jSt);
    int gemm_k = lu->supersize(k);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    Ftype *raw_rhs = NULL;
    int raw_ld = gemm_n;
    if (symldl_v2_use_wpanel_cache(lu->grid3d))
    {
        for (size_t slot = 0; slot < lu->symV2RawPanelNodes.size(); ++slot)
        {
            if (lu->symV2RawPanelNodes[slot] == k)
            {
                if (lu->A_gpu.symV2RawPanelBufs[slot] == NULL ||
                    lu->A_gpu.symV2RawPanelReadyEvents[slot] == NULL)
                    ABORT("SymFact V2 cached W panel is missing.");
                gpuErrchk(cudaStreamWaitEvent(
                    cuStream, lu->A_gpu.symV2RawPanelReadyEvents[slot], 0));
                raw_rhs = lu->A_gpu.symV2RawPanelBufs[slot] +
                          lpanel.stRow(jSt);
                raw_ld = lpanel.LDA();
                break;
            }
        }
    }
    if (raw_rhs == NULL)
    {
        int threads = 256;
        int blocks = (gemm_k * gemm_n + threads - 1) / threads;
        symldl_v2_build_raw_l_range_kernel<Ftype>
            <<<blocks, threads, 0, cuStream>>>(
                rawBlock, gemm_n, lpanel.gpuPanel, jSt, jEnd,
                lu->symV2DiagBlocksGPU[static_cast<size_t>(k)], gemm_k);
        gpuErrchk(cudaGetLastError());
        raw_rhs = rawBlock;
        raw_ld = gemm_n;
    }

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    cublasSetStream(handle, cuStream);
    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        raw_rhs, raw_ld, &beta,
                        gemmBuff, gemm_m);

    symldl_v2_scatter_ll_range<Ftype>(
        iSt, iEnd, jSt, jEnd, gemmBuff, gemm_m,
        lu->A_gpu.maxSuperSize, lu->ldt,
        lpanel.gpuPanel, lu->dA_gpu, cuStream);
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_ll_limited_update(
    xLUstruct_t<Ftype> *lu,
    int_t lStart, int_t lEnd,
    int_t jStart, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *rawBlock, Ftype *gemmBuff)
{
    if (lStart >= lEnd || jStart >= jEnd || lpanel.isEmpty())
        return 0;

    int_t nlb = lpanel.nblocks();
    lStart = SUPERLU_MAX((int_t) 0, lStart);
    jStart = SUPERLU_MAX((int_t) 0, jStart);
    lEnd = SUPERLU_MIN(lEnd, nlb);
    jEnd = SUPERLU_MIN(jEnd, nlb);
    if (lStart >= lEnd || jStart >= jEnd)
        return 0;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(lu->A_gpu.gemmBufferSize));
    int max_block_rows = 1;
    for (int_t ii = lStart; ii < lEnd; ++ii)
        max_block_rows = SUPERLU_MAX(max_block_rows,
                                     static_cast<int>(lpanel.nbrow(ii)));

    bool gid_sorted = true;
    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    if (envelope_requested)
    {
        const int_t order_begin = SUPERLU_MIN(lStart, jStart);
        const int_t order_end = SUPERLU_MAX(lEnd, jEnd);
        for (int_t ii = order_begin + 1; ii < order_end; ++ii)
        {
            if (lpanel.gid(ii) < lpanel.gid(ii - 1))
            {
                gid_sorted = false;
                break;
            }
        }
    }
    const bool lower_envelope = envelope_requested && gid_sorted;

    int_t envelope_l_start = lStart;
    int_t jSt = jStart;
    while (jSt < jEnd)
    {
        if (lower_envelope)
        {
            int_t min_col_gid = lpanel.gid(jSt);
            while (envelope_l_start < lEnd &&
                   lpanel.gid(envelope_l_start) < min_col_gid)
                ++envelope_l_start;
            if (envelope_l_start >= lEnd)
                break;
        }
        else
        {
            envelope_l_start = lStart;
        }

        int group_nrows =
            lpanel.stRow(lEnd) - lpanel.stRow(envelope_l_start);
        if (group_nrows <= 0)
            break;

        int col_limit = superlu_sym_v2_batch_schur_col_limit(
            group_nrows, gemm_capacity);
        int64_t hard_col_limit =
            gemm_capacity / static_cast<int64_t>(max_block_rows);
        if (hard_col_limit < 1)
            hard_col_limit = 1;
        col_limit = SUPERLU_MIN(
            col_limit,
            static_cast<int>(SUPERLU_MIN(
                hard_col_limit, static_cast<int64_t>(2147483647L))));
        int remaining_cols = static_cast<int>(lpanel.stRow(jEnd) -
                                              lpanel.stRow(jSt));
        col_limit = SUPERLU_MAX(1, SUPERLU_MIN(col_limit,
                                               remaining_cols));

        int_t jNext = jSt + 1;
        while (jNext < jEnd)
        {
            int candidate_cols = lpanel.stRow(jNext + 1) -
                                 lpanel.stRow(jSt);
            if (candidate_cols > col_limit)
                break;
            ++jNext;
        }

        int group_cols = lpanel.stRow(jNext) - lpanel.stRow(jSt);
        int maxGemmRows = group_nrows;
        if (static_cast<int64_t>(group_nrows) * group_cols > gemm_capacity)
            maxGemmRows = static_cast<int>(
                gemm_capacity / static_cast<int64_t>(group_cols));
        maxGemmRows = SUPERLU_MAX(maxGemmRows, 1);

        int_t iEnd = envelope_l_start;
        while (iEnd < lEnd)
        {
            int_t iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
            if (iEnd > lEnd)
                iEnd = lEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;
            symldl_v2_ll_part_update(
                lu, iSt, iEnd, jSt, jNext, k, lpanel,
                handle, cuStream, rawBlock, gemmBuff);
        }
        jSt = jNext;
    }

    return 0;
}

template <typename Ftype>
static int_t symldl_v2_ll_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    int_t laLoc = lpanel.find(laIdx);
    if (laLoc == GLOBAL_BLOCK_NOT_FOUND)
        return 0;

    if (k >= 0 &&
        static_cast<size_t>(k) < lu->symPanelReadyEventIds.size() &&
        lu->symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = lu->symPanelReadyEventIds[static_cast<size_t>(k)];
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.lookAheadLStream[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.lookAheadUStream[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
    }

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colRawBlock = lu->A_gpu.symPartnerLStageBufs[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowRawBlock = lu->A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    if (colRawBlock == NULL || colGemmBuff == NULL ||
        rowRawBlock == NULL || rowGemmBuff == NULL)
        ABORT("SymFact V2 LL lookahead buffers are missing.");

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t la_gid = lpanel.gid(laLoc);
    for (int_t ii = st_lb; ii < nlb; ++ii)
    {
        if (ii == laLoc || lpanel.gid(ii) >= la_gid)
        {
            symldl_v2_ll_part_update(
                lu, ii, ii + 1, laLoc, laLoc + 1, k, lpanel,
                colHandle, colStream, colRawBlock, colGemmBuff);
        }
        else
        {
            symldl_v2_ll_part_update(
                lu, laLoc, laLoc + 1, ii, ii + 1, k, lpanel,
                rowHandle, rowStream, rowRawBlock, rowGemmBuff);
        }
    }

    return 0;
}

template <typename Ftype>
static int_t symldl_v2_ll_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    if (k >= 0 &&
        static_cast<size_t>(k) < lu->symPanelReadyEventIds.size() &&
        lu->symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = lu->symPanelReadyEventIds[static_cast<size_t>(k)];
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.cuStreams[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
    }

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t exLoc = lpanel.find(ex);
    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *rawBlock = lu->A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];
    if (rawBlock == NULL || gemmBuff == NULL)
        ABORT("SymFact V2 LL Schur buffers are missing.");

    if (exLoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        symldl_v2_ll_limited_update(
            lu, st_lb, nlb, st_lb, nlb, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
    }
    else
    {
        symldl_v2_ll_limited_update(
            lu, st_lb, exLoc, st_lb, exLoc, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
        symldl_v2_ll_limited_update(
            lu, exLoc + 1, nlb, st_lb, exLoc, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
        symldl_v2_ll_limited_update(
            lu, exLoc + 1, nlb, exLoc + 1, nlb, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
    }

    return 0;
}

#endif
