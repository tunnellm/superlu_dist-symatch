#pragma once

#include <cstring>

#include "xlupanels.hpp"
#include "symldl_v2_workspace_impl.hpp"

template <typename Ftype>
static void symldl_v2_ensure_factor_work(xLUstruct_t<Ftype> *lu,
                                         int64_t requested)
{
    if (requested <= lu->symFactWorkSize)
        return;
    if (lu->symV2UsesCpuFactor())
    {
        if (lu->symV2CpuFactorLoopActive)
            ++lu->symV2CpuRuntimeAllocations;
        ABORT("SymFact V2 CPU factor workspace is undersized.");
    }
    Ftype *work = (Ftype *) SUPERLU_MALLOC(
        symldl_v2_checked_product((size_t) requested, sizeof(Ftype),
                                  "SymFact V2 workspace resize overflows."));
    if (work == NULL)
        ABORT("Malloc fails for SymFact V2 resized workspace.");
    if (lu->symFactWork != NULL)
        SUPERLU_FREE(lu->symFactWork);
    lu->symFactWork = work;
    lu->symFactWorkSize = requested;
}

#ifdef SLU_HAVE_LAPACK

static void symldl_v2_factor_invert_diag_owner(
    xLUstruct_t<double> *lu, int_t k, int_t ksupc,
    int_t handle_offset, double *invDiag)
{
    int_t lk = lu->symV2PanelIndex(k);
    if (lk < 0 || lk >= lu->symV2PanelCount())
        ABORT("SymFact V2 diagonal owner has no local panel.");
    xlpanel_t<double> &lpanel = lu->lPanelVec[lk];
    if (lpanel.isEmpty() || !lpanel.haveDiag() ||
        lpanel.gid(0) != k || lpanel.nbrow(0) != ksupc)
        ABORT("SymFact V2 diagonal owner has an invalid L-panel diagonal block.");

    double *diag = lpanel.blkPtr(0);
    int_t ldd = lpanel.LDA();
    if (lu->symFactWork == NULL || lu->symFactIPIV == NULL)
        ABORT("SymFact V2 factor workspace is not allocated.");

#ifdef HAVE_CUDA
    if (lu->superlu_acc_offload)
    {
        int stream_id = (handle_offset >= 0 &&
                         handle_offset < lu->A_gpu.numCudaStreams)
                            ? handle_offset
                            : 0;
        cudaStream_t stream = lu->A_gpu.cuStreams[stream_id];
        gpuErrchk(cudaMemcpy2DAsync(diag, ldd * sizeof(double),
                                    lpanel.blkPtrGPU(0),
                                    ldd * sizeof(double),
                                    ksupc * sizeof(double), ksupc,
                                    cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
#endif

    if (lu->symV2DiagBlocks.size() != (size_t) lu->nsupers)
        ABORT("SymFact V2 diagonal block vector has invalid size.");
    if (lu->symV2DiagBlocks[k] == NULL)
    {
        if (lu->symV2UsesCpuFactor() && lu->symV2CpuFactorLoopActive)
        {
            ++lu->symV2CpuRuntimeAllocations;
            ABORT("SymFact V2 CPU diagonal workspace was not preallocated.");
        }
        lu->symV2DiagBlocks[k] = (double *) SUPERLU_MALLOC(
            symldl_v2_checked_product(
                symldl_v2_checked_product((size_t) ksupc, (size_t) ksupc,
                                          "SymFact V2 diagonal block allocation overflows."),
                sizeof(double),
                "SymFact V2 diagonal block allocation overflows."));
        if (lu->symV2DiagBlocks[k] == NULL)
            ABORT("Malloc fails for SymFact V2 diagonal block.");
    }

    for (int_t j = 0; j < ksupc; ++j)
        std::memcpy(&lu->symV2DiagBlocks[k][j * ksupc],
                    &diag[j * ldd], ksupc * sizeof(double));
    for (int_t j = 0; j < ksupc; ++j)
        for (int_t i = 0; i < j; ++i)
            lu->symV2DiagBlocks[k][i + j * ksupc] =
                lu->symV2DiagBlocks[k][j + i * ksupc];

    char uplo = 'L';
    int n_i = (int) ksupc;
    int ldd_i = (int) ldd;
    int lwork = -1;
    int lapack_info = 0;
    int ntiny = 0;
    int n2x2 = 0;
    double thresh1 = lu->thresh / 10.0;
    double query = 0.0;

    if (lu->options->ReplaceTinyPivot == YES)
        dsytrf_mod_(&uplo, &n_i, diag, &ldd_i, &thresh1,
                    lu->symFactIPIV, &query, &lwork, &lapack_info,
                    &ntiny, &n2x2);
    else
        dsytrf_(&uplo, &n_i, diag, &ldd_i, lu->symFactIPIV,
                &query, &lwork, &lapack_info);

    int64_t requested_work = (int64_t) query;
    if (requested_work < 1)
        requested_work = 1;
    symldl_v2_ensure_factor_work(lu, requested_work);
    lwork = (int) requested_work;

    if (lu->options->ReplaceTinyPivot == YES)
    {
        ntiny = 0;
        n2x2 = 0;
        dsytrf_mod_(&uplo, &n_i, diag, &ldd_i, &thresh1,
                    lu->symFactIPIV, lu->symFactWork, &lwork,
                    &lapack_info, &ntiny, &n2x2);
        lu->stat->TinyPivots += ntiny;
        lu->stat->sytrf_2x2 += n2x2;
    }
    else
    {
        dsytrf_(&uplo, &n_i, diag, &ldd_i, lu->symFactIPIV,
                lu->symFactWork, &lwork, &lapack_info);
    }

    if (lapack_info != 0)
    {
        if (lapack_info > 0)
            *lu->info = lapack_info + lu->xsup[k];
        else
            *lu->info = lapack_info - lu->xsup[k];
    }

    int inertia[3];
    inertia_from_dsytrf(uplo, n_i, diag, ldd_i, lu->symFactIPIV,
                        1e-30, inertia);
    lu->stat->inertia[0] += inertia[0];
    lu->stat->inertia[1] += inertia[1];
    lu->stat->inertia[2] += inertia[2];

    dsytri_(&uplo, &n_i, diag, &ldd_i, lu->symFactIPIV,
            lu->symFactWork, &lapack_info);
    if (lapack_info != 0)
        ABORT("SymFact V2 dsytri failed.");

    for (int_t j = 0; j < ksupc; ++j)
        for (int_t i = j + 1; i < ksupc; ++i)
            diag[j + i * ldd] = diag[i + j * ldd];
    for (int_t j = 0; j < ksupc; ++j)
        std::memcpy(&invDiag[j * ksupc], &diag[j * ldd],
                    ksupc * sizeof(double));

    lu->stat->ops[FACT] += (flops_t) ksupc * ksupc * ksupc;
}

#endif
