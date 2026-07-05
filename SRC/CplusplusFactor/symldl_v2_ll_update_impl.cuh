#pragma once

#include <algorithm>
#include <cstdint>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_ll_scatter_impl.cuh"

#ifdef HAVE_CUDA

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

#endif
