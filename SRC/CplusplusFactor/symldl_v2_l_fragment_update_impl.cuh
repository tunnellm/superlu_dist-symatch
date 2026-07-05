#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_l_fragment_scatter_impl.cuh"

#ifdef HAVE_CUDA

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

#endif
