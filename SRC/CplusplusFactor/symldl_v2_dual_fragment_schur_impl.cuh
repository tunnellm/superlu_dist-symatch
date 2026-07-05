#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"

#ifdef HAVE_CUDA

static inline bool symldl_v2_trace_fragment_schur()
{
    const char *env = std::getenv("GPU3DV2_TRACE_FRAGMENT_SCHUR");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

static __device__ int_t symldl_v2_frag_nblocks(const int_t *frag_index)
{
    return frag_index[0];
}

static __device__ int_t symldl_v2_frag_gid(const int_t *frag_index,
                                           int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static __device__ int_t symldl_v2_frag_st_row(const int_t *frag_index,
                                              int_t k)
{
    int_t nblocks = symldl_v2_frag_nblocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static __device__ int_t symldl_v2_frag_nbrow(const int_t *frag_index,
                                             int_t k)
{
    return symldl_v2_frag_st_row(frag_index, k + 1) -
           symldl_v2_frag_st_row(frag_index, k);
}

static __device__ int_t *symldl_v2_frag_row_list(int_t *frag_index,
                                                 int_t k)
{
    int_t nblocks = symldl_v2_frag_nblocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       frag_index[LPANEL_HEADER_SIZE + nblocks + k]];
}

static __device__ void symldl_v2_compute_indirect_map(
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
__device__ void symldl_v2_scatter_two_fragments_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = symldl_v2_frag_gid(rowFragIndex, ii);
    int_t gj = symldl_v2_frag_gid(colFragIndex, jj);
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

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    int nrows = static_cast<int>(symldl_v2_frag_nbrow(rowFragIndex, ii));
    int ncols = static_cast<int>(symldl_v2_frag_nbrow(colFragIndex, jj));

    symldl_v2_compute_indirect_map(
        row_src_to_dst, nrows, symldl_v2_frag_row_list(rowFragIndex, ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_compute_indirect_map(
        col_src_to_dst, ncols, symldl_v2_frag_row_list(colFragIndex, jj),
        dst_col_len, NULL, dst_idx);

    int nthreads = blockDim.x;
    int cols_per_thread_block = nrows > 0 ? nthreads / nrows : 1;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(
        symldl_v2_frag_st_row(rowFragIndex, ii) -
        symldl_v2_frag_st_row(rowFragIndex, iSt));
    int col_off = static_cast<int>(
        symldl_v2_frag_st_row(colFragIndex, jj) -
        symldl_v2_frag_st_row(colFragIndex, jSt));
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
__global__ void symldl_v2_scatter_two_fragments_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_two_fragments_dev(
        iSt, jSt, gemmBuff, LDgemmBuff,
        rowFragIndex, colFragIndex, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_two_fragments(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_two_fragments_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff,
            rowFragIndex, colFragIndex, dA);
    gpuErrchk(cudaGetLastError());
}

static inline int_t symldl_v2_frag_host_nblocks(
    const std::vector<int_t> &frag)
{
    return frag.empty() ? 0 : frag[0];
}

static inline int_t symldl_v2_frag_host_gid(
    const std::vector<int_t> &frag, int_t k)
{
    return frag[LPANEL_HEADER_SIZE + k];
}

static inline int_t symldl_v2_frag_host_st_row(
    const std::vector<int_t> &frag, int_t k)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    return frag[LPANEL_HEADER_SIZE + nblocks + k];
}

static inline int_t symldl_v2_frag_host_nbrow(
    const std::vector<int_t> &frag, int_t k)
{
    return symldl_v2_frag_host_st_row(frag, k + 1) -
           symldl_v2_frag_host_st_row(frag, k);
}

static inline int_t symldl_v2_frag_host_find(
    const std::vector<int_t> &frag, int_t gid)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    for (int_t i = 0; i < nblocks; ++i)
        if (symldl_v2_frag_host_gid(frag, i) == gid)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

static inline int_t symldl_v2_frag_host_end_block(
    const std::vector<int_t> &frag, int_t iSt, int maxRows)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    int_t base = symldl_v2_frag_host_st_row(frag, iSt);
    int_t iEnd = iSt + 1;
    while (iEnd < nblocks &&
           symldl_v2_frag_host_st_row(frag, iEnd + 1) - base <= maxRows)
        ++iEnd;
    return iEnd;
}

static inline int symldl_v2_frag_host_ranges_excluding(
    int_t begin, int_t end, int_t skip0, int_t skip1,
    int_t ranges[3][2])
{
    if (begin >= end)
        return 0;
    if (skip0 == skip1)
        skip1 = GLOBAL_BLOCK_NOT_FOUND;
    if (skip1 != GLOBAL_BLOCK_NOT_FOUND &&
        (skip0 == GLOBAL_BLOCK_NOT_FOUND || skip1 < skip0))
        std::swap(skip0, skip1);

    int nranges = 0;
    int_t cur = begin;
    int_t skips[2] = {skip0, skip1};
    for (int i = 0; i < 2; ++i)
    {
        int_t skip = skips[i];
        if (skip == GLOBAL_BLOCK_NOT_FOUND || skip < begin || skip >= end)
            continue;
        if (cur < skip)
        {
            ranges[nranges][0] = cur;
            ranges[nranges][1] = skip;
            ++nranges;
        }
        cur = skip + 1;
    }
    if (cur < end)
    {
        ranges[nranges][0] = cur;
        ranges[nranges][1] = end;
        ++nranges;
    }
    return nranges;
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_part_update(
    xLUstruct_t<Ftype> *lu,
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k,
    const std::vector<int_t> &row_frag,
    const std::vector<int_t> &col_frag,
    int_t *row_frag_index, Ftype *row_frag_val,
    int_t *col_frag_index, Ftype *col_frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd || row_frag.empty() ||
        col_frag.empty() || row_frag_index == NULL ||
        row_frag_val == NULL || col_frag_index == NULL ||
        col_frag_val == NULL)
        return 0;
    if (row_frag.size() < LPANEL_HEADER_SIZE ||
        col_frag.size() < LPANEL_HEADER_SIZE)
        ABORT("SymFact V2 Pc-fragment GEMM has truncated metadata.");

    int gemm_k = lu->supersize(k);
    if (row_frag[3] != gemm_k || col_frag[3] != gemm_k)
        ABORT("SymFact V2 Pc-fragment GEMM has inconsistent panel width.");

    int gemm_m = symldl_v2_frag_host_st_row(row_frag, iEnd) -
                 symldl_v2_frag_host_st_row(row_frag, iSt);
    int gemm_n = symldl_v2_frag_host_st_row(col_frag, jEnd) -
                 symldl_v2_frag_host_st_row(col_frag, jSt);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    int row_start = symldl_v2_frag_host_st_row(row_frag, iSt);
    int col_start = symldl_v2_frag_host_st_row(col_frag, jSt);
    int row_lda = row_frag[1];
    int col_lda = col_frag[1];
    if (row_lda <= 0 || col_lda <= 0)
        ABORT("SymFact V2 Pc-fragment GEMM has invalid fragment LDA.");

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    cublasSetStream(handle, cuStream);
    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        row_frag_val + row_start, row_lda,
                        col_frag_val + col_start, col_lda, &beta,
                        gemmBuff, gemm_m);

    symldl_v2_scatter_two_fragments<Ftype>(
        iSt, iEnd, jSt, jEnd, gemmBuff, gemm_m,
        lu->A_gpu.maxSuperSize, lu->ldt,
        row_frag_index, col_frag_index, lu->dA_gpu, cuStream);
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_limited_update(
    xLUstruct_t<Ftype> *lu,
    int_t rowStart, int_t rowEnd,
    int_t colStart, int_t colEnd,
    int_t k,
    const std::vector<int_t> &row_frag,
    const std::vector<int_t> &col_frag,
    int_t *row_frag_index, Ftype *row_frag_val,
    int_t *col_frag_index, Ftype *col_frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (row_frag.empty() || col_frag.empty() ||
        row_frag_index == NULL || row_frag_val == NULL ||
        col_frag_index == NULL || col_frag_val == NULL)
        return 0;
    if (row_frag.size() < LPANEL_HEADER_SIZE ||
        col_frag.size() < LPANEL_HEADER_SIZE)
        ABORT("SymFact V2 Pc-fragment update has truncated metadata.");
    if (row_frag[3] != lu->supersize(k) ||
        col_frag[3] != lu->supersize(k))
        ABORT("SymFact V2 Pc-fragment update has inconsistent panel width.");

    int_t nrow = symldl_v2_frag_host_nblocks(row_frag);
    int_t ncol = symldl_v2_frag_host_nblocks(col_frag);
    rowStart = SUPERLU_MAX((int_t) 0, rowStart);
    colStart = SUPERLU_MAX((int_t) 0, colStart);
    rowEnd = SUPERLU_MIN(rowEnd, nrow);
    colEnd = SUPERLU_MIN(colEnd, ncol);
    if (rowStart >= rowEnd || colStart >= colEnd)
        return 0;

    int nrows_total = symldl_v2_frag_host_st_row(row_frag, rowEnd) -
                      symldl_v2_frag_host_st_row(row_frag, rowStart);
    int ncols_total = symldl_v2_frag_host_st_row(col_frag, colEnd) -
                      symldl_v2_frag_host_st_row(col_frag, colStart);
    if (nrows_total <= 0 || ncols_total <= 0)
        return 0;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(lu->A_gpu.gemmBufferSize));
    int max_block_rows = 1;
    for (int_t ii = rowStart; ii < rowEnd; ++ii)
        max_block_rows = SUPERLU_MAX(
            max_block_rows,
            static_cast<int>(symldl_v2_frag_host_nbrow(row_frag, ii)));

    bool row_gid_sorted = true;
    bool col_gid_sorted = true;
    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    if (envelope_requested)
    {
        for (int_t ii = rowStart + 1; ii < rowEnd; ++ii)
            if (symldl_v2_frag_host_gid(row_frag, ii) <
                symldl_v2_frag_host_gid(row_frag, ii - 1))
                row_gid_sorted = false;
        for (int_t jj = colStart + 1; jj < colEnd; ++jj)
            if (symldl_v2_frag_host_gid(col_frag, jj) <
                symldl_v2_frag_host_gid(col_frag, jj - 1))
                col_gid_sorted = false;
    }
    const bool lower_envelope =
        envelope_requested && row_gid_sorted && col_gid_sorted;

    int_t envelope_row_start = rowStart;
    int_t jSt = colStart;
    while (jSt < colEnd)
    {
        if (lower_envelope)
        {
            int_t min_col_gid = symldl_v2_frag_host_gid(col_frag, jSt);
            while (envelope_row_start < rowEnd &&
                   symldl_v2_frag_host_gid(row_frag, envelope_row_start) <
                       min_col_gid)
                ++envelope_row_start;
            if (envelope_row_start >= rowEnd)
                break;
        }
        else
        {
            envelope_row_start = rowStart;
        }

        int group_nrows = symldl_v2_frag_host_st_row(row_frag, rowEnd) -
                          symldl_v2_frag_host_st_row(row_frag,
                                                     envelope_row_start);
        if (group_nrows <= 0)
            break;

        int col_limit = superlu_sym_v2_batch_schur_col_limit(
            group_nrows, gemm_capacity);
        int64_t hard_col_limit = gemm_capacity /
                                 static_cast<int64_t>(max_block_rows);
        if (hard_col_limit < 1)
            hard_col_limit = 1;
        col_limit = SUPERLU_MIN(
            col_limit,
            static_cast<int>(SUPERLU_MIN(
                hard_col_limit, static_cast<int64_t>(2147483647L))));
        int remaining_cols =
            symldl_v2_frag_host_st_row(col_frag, colEnd) -
            symldl_v2_frag_host_st_row(col_frag, jSt);
        col_limit = SUPERLU_MAX(1, SUPERLU_MIN(col_limit,
                                               remaining_cols));

        int_t jNext = jSt + 1;
        while (jNext < colEnd)
        {
            int candidate_cols =
                symldl_v2_frag_host_st_row(col_frag, jNext + 1) -
                symldl_v2_frag_host_st_row(col_frag, jSt);
            if (candidate_cols > col_limit)
                break;
            ++jNext;
        }

        int group_cols = symldl_v2_frag_host_st_row(col_frag, jNext) -
                         symldl_v2_frag_host_st_row(col_frag, jSt);
        int maxGemmRows = group_nrows;
        if (static_cast<int64_t>(group_nrows) * group_cols > gemm_capacity)
            maxGemmRows = static_cast<int>(
                gemm_capacity / static_cast<int64_t>(group_cols));
        maxGemmRows = SUPERLU_MAX(maxGemmRows, 1);

        int_t iEnd = envelope_row_start;
        while (iEnd < rowEnd)
        {
            int_t iSt = iEnd;
            iEnd = symldl_v2_frag_host_end_block(row_frag, iSt,
                                                 maxGemmRows);
            if (iEnd > rowEnd)
                iEnd = rowEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;
            symldl_v2_dual_fragment_part_update(
                lu, iSt, iEnd, jSt, jNext, k,
                row_frag, col_frag,
                row_frag_index, row_frag_val,
                col_frag_index, col_frag_val,
                handle, cuStream, gemmBuff);
        }
        jSt = jNext;
    }
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx)
{
    if (!lu->symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact V2 Pc-fragment lookahead called while disabled.");
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment metadata is missing.");
    if (static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment row metadata is missing.");

    const std::vector<int_t> &row_frag = lu->symV2RowFragRecvIndex[k];
    const std::vector<int_t> &col_frag = lu->symV2PartnerLRecvIndex[k];
    if (symldl_v2_trace_fragment_schur())
    {
        static int printed = 0;
        if (printed < 32)
        {
            std::fprintf(stderr,
                         "[symv2-frag-schur] rank %d lookahead k %d la %d row_blocks %d col_blocks %d\n",
                         lu->grid3d != NULL ? lu->grid3d->iam : -1,
                         static_cast<int>(k), static_cast<int>(laIdx),
                         row_frag.empty() ? 0 : static_cast<int>(row_frag[0]),
                         col_frag.empty() ? 0 : static_cast<int>(col_frag[0]));
            ++printed;
        }
    }
    if (row_frag.empty() || col_frag.empty())
        return 0;

    int_t *row_idx = lu->A_gpu.symV2RowFragIdxRecvBufs[streamId];
    Ftype *row_val = lu->A_gpu.symV2RowFragValRecvBufs[streamId];
    int_t *col_idx = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *col_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (row_idx == NULL || row_val == NULL ||
        col_idx == NULL || col_val == NULL)
        ABORT("SymFact V2 Pc-fragment GPU buffers are missing.");

    int_t nrow = symldl_v2_frag_host_nblocks(row_frag);
    int_t ncol = symldl_v2_frag_host_nblocks(col_frag);
    int_t rowLoc = symldl_v2_frag_host_find(row_frag, laIdx);
    int_t colLoc = symldl_v2_frag_host_find(col_frag, laIdx);
    int_t diagRowLoc = symldl_v2_frag_host_find(row_frag, k);
    int_t diagColLoc = symldl_v2_frag_host_find(col_frag, k);

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];
    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    int_t rowRanges[3][2];
    int nRowRanges = symldl_v2_frag_host_ranges_excluding(
        0, nrow, diagRowLoc, GLOBAL_BLOCK_NOT_FOUND, rowRanges);
    if (colLoc != GLOBAL_BLOCK_NOT_FOUND && colLoc != diagColLoc)
    {
        for (int ir = 0; ir < nRowRanges; ++ir)
            symldl_v2_dual_fragment_limited_update(
                lu, rowRanges[ir][0], rowRanges[ir][1],
                colLoc, colLoc + 1, k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                colHandle, colStream, colGemmBuff);
    }

    if (rowLoc != GLOBAL_BLOCK_NOT_FOUND && rowLoc != diagRowLoc)
    {
        int_t colRanges[3][2];
        int nColRanges = symldl_v2_frag_host_ranges_excluding(
            0, ncol, colLoc, diagColLoc, colRanges);
        for (int ic = 0; ic < nColRanges; ++ic)
            symldl_v2_dual_fragment_limited_update(
                lu, rowLoc, rowLoc + 1,
                colRanges[ic][0], colRanges[ic][1], k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                rowHandle, rowStream, rowGemmBuff);
    }
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex)
{
    if (!lu->symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact V2 Pc-fragment exclude called while disabled.");
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment metadata is missing.");
    if (static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment row metadata is missing.");

    const std::vector<int_t> &row_frag = lu->symV2RowFragRecvIndex[k];
    const std::vector<int_t> &col_frag = lu->symV2PartnerLRecvIndex[k];
    if (symldl_v2_trace_fragment_schur())
    {
        static int printed = 0;
        if (printed < 32)
        {
            std::fprintf(stderr,
                         "[symv2-frag-schur] rank %d exclude k %d ex %d row_blocks %d col_blocks %d\n",
                         lu->grid3d != NULL ? lu->grid3d->iam : -1,
                         static_cast<int>(k), static_cast<int>(ex),
                         row_frag.empty() ? 0 : static_cast<int>(row_frag[0]),
                         col_frag.empty() ? 0 : static_cast<int>(col_frag[0]));
            ++printed;
        }
    }
    if (row_frag.empty() || col_frag.empty())
        return 0;

    int_t *row_idx = lu->A_gpu.symV2RowFragIdxRecvBufs[streamId];
    Ftype *row_val = lu->A_gpu.symV2RowFragValRecvBufs[streamId];
    int_t *col_idx = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *col_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (row_idx == NULL || row_val == NULL ||
        col_idx == NULL || col_val == NULL)
        ABORT("SymFact V2 Pc-fragment GPU buffers are missing.");

    int_t nrow = symldl_v2_frag_host_nblocks(row_frag);
    int_t ncol = symldl_v2_frag_host_nblocks(col_frag);
    int_t rowLoc = symldl_v2_frag_host_find(row_frag, ex);
    int_t colLoc = symldl_v2_frag_host_find(col_frag, ex);
    int_t diagRowLoc = symldl_v2_frag_host_find(row_frag, k);
    int_t diagColLoc = symldl_v2_frag_host_find(col_frag, k);

    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    int_t rowRanges[3][2];
    int_t colRanges[3][2];
    int nRowRanges = symldl_v2_frag_host_ranges_excluding(
        0, nrow, rowLoc, diagRowLoc, rowRanges);
    int nColRanges = symldl_v2_frag_host_ranges_excluding(
        0, ncol, colLoc, diagColLoc, colRanges);
    for (int ir = 0; ir < nRowRanges; ++ir)
    {
        for (int ic = 0; ic < nColRanges; ++ic)
            symldl_v2_dual_fragment_limited_update(
                lu, rowRanges[ir][0], rowRanges[ir][1],
                colRanges[ic][0], colRanges[ic][1], k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                handle, cuStream, gemmBuff);
    }
    return 0;
}

#include "symldl_v2_l_fragment_schur_impl.cuh"
#include "symldl_v2_ll_schur_impl.cuh"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LookAheadUpdateGPU(
    int streamId, int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel)
{
    if (Pr == 1)
        return symldl_v2_ll_lookahead(this, streamId, k, laIdx, lpanel);
    if (symV2UsePcFragmentSchurPanel(k))
        return symldl_v2_dual_fragment_lookahead(
            this, streamId, k, laIdx);
    return symldl_v2_l_fragment_lookahead(
        this, streamId, k, laIdx, lpanel);
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2SchurCompUpdateExcludeOneGPU(
    int streamId, int_t k, int_t ex, xlpanel_t<Ftype> &lpanel)
{
    if (Pr == 1)
        return symldl_v2_ll_exclude(this, streamId, k, ex, lpanel);
    if (symV2UsePcFragmentSchurPanel(k))
        return symldl_v2_dual_fragment_exclude(
            this, streamId, k, ex);
    return symldl_v2_l_fragment_exclude(
        this, streamId, k, ex, lpanel);
}

#endif
