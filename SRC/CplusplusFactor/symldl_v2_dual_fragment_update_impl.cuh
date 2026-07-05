#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_dual_fragment_scatter_impl.cuh"

#ifdef HAVE_CUDA

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

#endif
