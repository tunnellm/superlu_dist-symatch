#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static __device__ int_t symldl_v2_lfrag_nblocks(const int_t *frag_index)
{
    return frag_index[0];
}

static __device__ int_t symldl_v2_lfrag_gid(const int_t *frag_index,
                                            int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static __device__ int_t symldl_v2_lfrag_st_row(const int_t *frag_index,
                                               int_t k)
{
    int_t nblocks = symldl_v2_lfrag_nblocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static __device__ int_t symldl_v2_lfrag_nbrow(const int_t *frag_index,
                                              int_t k)
{
    return symldl_v2_lfrag_st_row(frag_index, k + 1) -
           symldl_v2_lfrag_st_row(frag_index, k);
}

static __device__ int_t *symldl_v2_lfrag_row_list(int_t *frag_index,
                                                  int_t k)
{
    int_t nblocks = symldl_v2_lfrag_nblocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       frag_index[LPANEL_HEADER_SIZE + nblocks + k]];
}

static __device__ void symldl_v2_lfrag_compute_indirect_map(
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
__device__ void symldl_v2_scatter_lpanel_fragment_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = rowPanel.gid(ii);
    int_t gj = symldl_v2_lfrag_gid(fragIndex, jj);
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

    int nrows = static_cast<int>(rowPanel.nbrow(ii));
    int ncols = static_cast<int>(symldl_v2_lfrag_nbrow(fragIndex, jj));
    if (nrows <= 0 || ncols <= 0)
        return;

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    symldl_v2_lfrag_compute_indirect_map(
        row_src_to_dst, nrows, rowPanel.rowList(ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_lfrag_compute_indirect_map(
        col_src_to_dst, ncols, symldl_v2_lfrag_row_list(fragIndex, jj),
        dst_col_len, NULL, dst_idx);

    int cols_per_thread_block = blockDim.x / nrows;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(rowPanel.stRow(ii) -
                                   rowPanel.stRow(iSt));
    int col_off = static_cast<int>(
        symldl_v2_lfrag_st_row(fragIndex, jj) -
        symldl_v2_lfrag_st_row(fragIndex, jSt));
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
__global__ void symldl_v2_scatter_lpanel_fragment_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_lpanel_fragment_dev(
        iSt, jSt, gemmBuff, LDgemmBuff, rowPanel, fragIndex, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_lpanel_fragment(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_lpanel_fragment_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff, rowPanel, fragIndex, dA);
    gpuErrchk(cudaGetLastError());
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_part_update(
    xLUstruct_t<Ftype> *lu,
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    const std::vector<int_t> &frag,
    int_t *frag_index, Ftype *frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty() ||
        frag.empty() || frag_index == NULL || frag_val == NULL)
        return 0;
    if (frag.size() < LPANEL_HEADER_SIZE)
        ABORT("SymFact V2 L-fragment GEMM has truncated metadata.");

    int_t frag_nblocks = frag[0];
    int_t frag_lda = frag[1];
    if (frag_nblocks <= 0 || frag_lda <= 0)
        return 0;
    if (frag[3] != lu->supersize(k))
        ABORT("SymFact V2 L-fragment GEMM has inconsistent panel width.");

    iSt = SUPERLU_MAX((int_t) 0, iSt);
    jSt = SUPERLU_MAX((int_t) 0, jSt);
    iEnd = SUPERLU_MIN(iEnd, lpanel.nblocks());
    jEnd = SUPERLU_MIN(jEnd, frag_nblocks);
    if (iSt >= iEnd || jSt >= jEnd)
        return 0;

    int starts = LPANEL_HEADER_SIZE + frag_nblocks;
    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = frag[starts + jEnd] - frag[starts + jSt];
    int gemm_k = lu->supersize(k);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    cublasSetStream(handle, cuStream);
    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        frag_val + frag[starts + jSt], frag_lda, &beta,
                        gemmBuff, gemm_m);

    symldl_v2_scatter_lpanel_fragment<Ftype>(
        iSt, iEnd, jSt, jEnd, gemmBuff, gemm_m,
        lu->A_gpu.maxSuperSize, lu->ldt, lpanel.gpuPanel,
        frag_index, lu->dA_gpu, cuStream);
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_limited_update(
    xLUstruct_t<Ftype> *lu,
    int_t lStart, int_t lEnd,
    int_t fragStart, int_t fragEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    const std::vector<int_t> &frag,
    int_t *frag_index, Ftype *frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (lStart >= lEnd || fragStart >= fragEnd || lpanel.isEmpty() ||
        frag.empty() || frag_index == NULL || frag_val == NULL)
        return 0;
    if (frag.size() < LPANEL_HEADER_SIZE)
        ABORT("SymFact V2 L-fragment update has truncated metadata.");
    if (frag[3] != lu->supersize(k))
        ABORT("SymFact V2 L-fragment update has inconsistent panel width.");

    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    lStart = SUPERLU_MAX((int_t) 0, lStart);
    fragStart = SUPERLU_MAX((int_t) 0, fragStart);
    lEnd = SUPERLU_MIN(lEnd, nlb);
    fragEnd = SUPERLU_MIN(fragEnd, nfrag);
    if (lStart >= lEnd || fragStart >= fragEnd)
        return 0;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(lu->A_gpu.gemmBufferSize));
    int max_block_rows = 1;
    for (int_t ii = lStart; ii < lEnd; ++ii)
        max_block_rows = SUPERLU_MAX(max_block_rows,
                                     static_cast<int>(lpanel.nbrow(ii)));

    bool row_gid_sorted = true;
    bool frag_gid_sorted = true;
    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    if (envelope_requested)
    {
        for (int_t ii = lStart + 1; ii < lEnd; ++ii)
            if (lpanel.gid(ii) < lpanel.gid(ii - 1))
                row_gid_sorted = false;
        for (int_t jj = fragStart + 1; jj < fragEnd; ++jj)
            if (frag[LPANEL_HEADER_SIZE + jj] <
                frag[LPANEL_HEADER_SIZE + jj - 1])
                frag_gid_sorted = false;
    }
    const bool lower_envelope =
        envelope_requested && row_gid_sorted && frag_gid_sorted;

    const int_t starts = LPANEL_HEADER_SIZE + nfrag;
    int_t envelope_l_start = lStart;
    int_t jSt = fragStart;
    while (jSt < fragEnd)
    {
        if (lower_envelope)
        {
            int_t min_col_gid = frag[LPANEL_HEADER_SIZE + jSt];
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
        int remaining_cols = static_cast<int>(frag[starts + fragEnd] -
                                              frag[starts + jSt]);
        col_limit = SUPERLU_MAX(1, SUPERLU_MIN(col_limit,
                                               remaining_cols));

        int_t jNext = jSt + 1;
        while (jNext < fragEnd)
        {
            int candidate_cols = frag[starts + jNext + 1] -
                                 frag[starts + jSt];
            if (candidate_cols > col_limit)
                break;
            ++jNext;
        }

        int group_cols = frag[starts + jNext] - frag[starts + jSt];
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
            symldl_v2_l_fragment_part_update(
                lu, iSt, iEnd, jSt, jNext, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
        }
        jSt = jNext;
    }
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_host_find(
    const std::vector<int_t> &frag, int_t gid)
{
    if (frag.empty())
        return GLOBAL_BLOCK_NOT_FOUND;
    for (int_t i = 0; i < frag[0]; ++i)
        if (frag[LPANEL_HEADER_SIZE + i] == gid)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");

    const std::vector<int_t> &frag = lu->symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = symldl_v2_l_fragment_host_find<Ftype>(frag, laIdx);

    int_t *frag_index = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        symldl_v2_l_fragment_limited_update(
            lu, st_lb, nlb, laJLoc, laJLoc + 1, k, lpanel, frag,
            frag_index, frag_val, colHandle, colStream, colGemmBuff);

    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        if (laJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, 0, nfrag, k, lpanel, frag,
                frag_index, frag_val, rowHandle, rowStream, rowGemmBuff);
        }
        else
        {
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, 0, laJLoc, k, lpanel, frag,
                frag_index, frag_val, rowHandle, rowStream, rowGemmBuff);
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, laJLoc + 1, nfrag, k, lpanel,
                frag, frag_index, frag_val, rowHandle, rowStream,
                rowGemmBuff);
        }
    }

    return 0;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");

    const std::vector<int_t> &frag = lu->symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = symldl_v2_l_fragment_host_find<Ftype>(frag, ex);

    int_t *frag_index = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    auto update_i_range = [&](int_t ist, int_t iend)
    {
        if (ist >= iend)
            return;
        if (exJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, 0, nfrag, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
        }
        else
        {
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, 0, exJLoc, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, exJLoc + 1, nfrag, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
        }
    };

    if (exILoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        update_i_range(st_lb, nlb);
    }
    else
    {
        update_i_range(st_lb, exILoc);
        update_i_range(exILoc + 1, nlb);
    }

    return 0;
}

#endif
