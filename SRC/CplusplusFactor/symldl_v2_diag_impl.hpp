#pragma once

#include <climits>
#include <cstring>

#include "xlupanels.hpp"
#include "symldl_v2_diag_factor_impl.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymDiagFactorPanelSolve(
    int_t, int_t, int_t, diagFactBufs_type<Ftype> **)
{
    ABORT("SymFact GPU3DVERSION=2 requires double precision.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymDiagFactorPanelSolve(
    int_t k, int_t handle_offset, int_t buffer_offset,
    ddiagFactBufs_t **dFBufs)
{
    if (!useSymV2Solve())
        return dDiagFactorPanelSolve(k, buffer_offset, dFBufs);

#ifdef HAVE_CUDA
    if (superlu_acc_offload)
        dSymV2PrepackLFragmentsGPU(k, handle_offset);
#endif

#ifndef SLU_HAVE_LAPACK
    ABORT("SymFact GPU3DVERSION=2 requires LAPACK dsytrf/dsytri support.");
    return 0;
#else
    int_t ksupc = SuperSize(k);
    int_t sym_panel_root = symV2PanelRoot(k);
    int_t sym_diag_root = symV2DiagRoot(k);
    int sym_diag_proc = PNUM(sym_diag_root, sym_panel_root, grid);
    size_t diag_count = symldl_v2_checked_product((size_t) ksupc,
                                                  (size_t) ksupc,
                                                  "SymFact V2 diagonal block size overflows.");
    if (diag_count > (size_t) INT_MAX)
        ABORT("SymFact V2 diagonal block is too large for MPI count.");
    int diag_mpi_count = (int) diag_count;
    double t0 = SuperLU_timer_();

    if (buffer_offset < 0 || buffer_offset >= numDiagBufs)
        ABORT("SymFact V2 diagonal buffer offset is invalid.");
    if (dFBufs == NULL || dFBufs[buffer_offset] == NULL)
        ABORT("SymFact V2 diagonal buffers are not allocated.");

    double *invDiag = dFBufs[buffer_offset]->BlockUFactor;
    if (invDiag == NULL)
        ABORT("SymFact V2 inverse diagonal buffer is not allocated.");

    if (iam == sym_diag_proc)
        symldl_v2_factor_invert_diag_owner(this, k, ksupc,
                                           handle_offset, invDiag);

    if (mycol == sym_panel_root)
    {
        if (symV2DiagBlocks.size() != (size_t) nsupers)
            ABORT("SymFact V2 diagonal block vector has invalid size.");
        if (symV2DiagBlocks[k] == NULL)
        {
            symV2DiagBlocks[k] = (double *) SUPERLU_MALLOC(
                symldl_v2_checked_product(
                    symldl_v2_checked_product((size_t) ksupc, (size_t) ksupc,
                                              "SymFact V2 diagonal block allocation overflows."),
                    sizeof(double),
                    "SymFact V2 diagonal block allocation overflows."));
            if (symV2DiagBlocks[k] == NULL)
                ABORT("Malloc fails for SymFact V2 diagonal block.");
        }
        MPI_Bcast(symV2DiagBlocks[k], diag_mpi_count, MPI_DOUBLE,
                  (int) sym_diag_root, grid3d->cscp.comm);
        MPI_Bcast(invDiag, diag_mpi_count, MPI_DOUBLE,
                  (int) sym_diag_root, grid3d->cscp.comm);
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            if (symV2DiagBlocksGPU.size() != (size_t) nsupers)
                ABORT("SymFact V2 device diagonal block vector has invalid size.");
            if (symV2DiagBlocksGPU[k] == NULL)
                gpuErrchk(cudaMalloc(
                    (void **) &symV2DiagBlocksGPU[k],
                    symldl_v2_checked_product(
                        diag_count, sizeof(double),
                        "SymFact V2 device diagonal block allocation overflows.")));
            int stream_id = (buffer_offset >= 0 &&
                             buffer_offset < A_gpu.numCudaStreams)
                                ? buffer_offset
                                : 0;
            gpuErrchk(cudaMemcpyAsync(symV2DiagBlocksGPU[k],
                                      symV2DiagBlocks[k],
                                      diag_count * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      A_gpu.cuStreams[stream_id]));
        }
#endif
    }

    if (mycol == sym_panel_root)
    {
        int_t lk = symV2PanelIndex(k);
        if (lk < 0 || lk >= symV2PanelCount())
            return 0;
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (lpanel.isEmpty())
            return 0;
        if (lpanel.haveDiag())
        {
            if (lpanel.gid(0) != k || lpanel.nbrow(0) != ksupc)
                ABORT("SymFact V2 L-panel diagonal block is not first.");
            for (int_t j = 0; j < ksupc; ++j)
                std::memcpy(&lpanel.blkPtr(0)[j * lpanel.LDA()],
                            &invDiag[j * ksupc], ksupc * sizeof(double));
        }

#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            int stream_id = (handle_offset >= 0 &&
                             handle_offset < A_gpu.numCudaStreams)
                                ? handle_offset
                                : 0;
            cudaStream_t stream = A_gpu.cuStreams[stream_id];
            cublasHandle_t handle = A_gpu.cuHandles[stream_id];
            double *dInvDiag = A_gpu.dFBufs[buffer_offset];
            gpuErrchk(cudaMemcpyAsync(dInvDiag, invDiag,
                                      ksupc * ksupc * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
            if (lpanel.haveDiag())
                gpuErrchk(cudaMemcpy2DAsync(lpanel.blkPtrGPU(0),
                                            lpanel.LDA() * sizeof(double),
                                            dInvDiag, ksupc * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToDevice, stream));
            lpanel.panelSolveSymmetricGPU(handle, stream, ksupc, dInvDiag,
                                          ksupc,
                                          A_gpu.lookAheadLGemmBuffer[stream_id],
                                          lpanel.nzrows());
            gpuErrchk(cudaMemcpyAsync(dInvDiag, symV2DiagBlocks[k],
                                      diag_count * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[stream_id],
                                      stream));
            if (k >= 0 &&
                static_cast<size_t>(k) < symPanelReadyEventIds.size())
                symPanelReadyEventIds[k] = stream_id;
            bool local_singleton_panel =
                symV2IsCollapsedGrid() &&
                grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1;
            bool async_v2_panel = superlu_sym_v2_async_factor();
            if (!local_singleton_panel && !async_v2_panel)
                gpuErrchk(cudaStreamSynchronize(stream));
        }
        else
#endif
        {
            symldl_v2_ensure_factor_work(
                this, (int64_t) lpanel.nzrows() * (int64_t) ksupc);
            lpanel.panelSolveSymmetric(ksupc, invDiag, ksupc,
                                       symFactWork, lpanel.nzrows());
        }
    }

    SCT->tDiagFactorPanelSolve += (SuperLU_timer_() - t0);
    return 0;
#endif
}
