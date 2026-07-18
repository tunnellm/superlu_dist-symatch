#pragma once

#include <climits>
#include <cstring>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_diag_factor_impl.hpp"

static inline void symldl_v2_diag_bcast_chunks(
    double *buffer, size_t count, int root, MPI_Comm comm,
    uint64_t *oversized_chunks)
{
    size_t offset = 0;
    const size_t limit = static_cast<size_t>(INT_MAX);
    size_t chunks = count == 0 ? 0 : (count - 1) / limit + 1;
    if (chunks > 1 && oversized_chunks != NULL)
        *oversized_chunks += static_cast<uint64_t>(chunks);
    while (offset < count)
    {
        int chunk = static_cast<int>(SUPERLU_MIN(limit, count - offset));
        if (MPI_Bcast(buffer + offset, chunk, MPI_DOUBLE, root, comm) !=
            MPI_SUCCESS)
            ABORT("SymFact V2 diagonal broadcast failed.");
        offset += static_cast<size_t>(chunk);
    }
}

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
    double t0 = SuperLU_timer_();

    if (buffer_offset < 0 || buffer_offset >= numDiagBufs)
        ABORT("SymFact V2 diagonal buffer offset is invalid.");
    if (dFBufs == NULL || dFBufs[buffer_offset] == NULL)
        ABORT("SymFact V2 diagonal buffers are not allocated.");

    double *invDiag = dFBufs[buffer_offset]->BlockUFactor;
    if (invDiag == NULL)
        ABORT("SymFact V2 inverse diagonal buffer is not allocated.");

    if (iam == sym_diag_proc)
    {
        double diag_start = SuperLU_timer_();
        symldl_v2_factor_invert_diag_owner(this, k, ksupc,
                                           handle_offset, invDiag);
        if (symV2UsesCpuFactor())
            symV2CpuDiagFactorTime += SuperLU_timer_() - diag_start;
    }

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
        double comm_start = SuperLU_timer_();
        uint64_t *chunk_counter = symV2UsesCpuFactor()
            ? &symV2CpuOversizedMpiChunks : NULL;
        // D and inv(D) have the same root and lifetime, so send one packet.
        if (symV2UsesCpuFactor() && grid3d->cscp.Np > 1)
        {
            size_t packet_count = symldl_v2_checked_product(
                diag_count, (size_t) 2,
                "SymFact V2 diagonal broadcast size overflows.");
            if (packet_count > static_cast<size_t>(
                                   std::numeric_limits<int64_t>::max()))
                ABORT("SymFact V2 diagonal broadcast exceeds workspace limits.");
            symldl_v2_ensure_factor_work(this, (int64_t) packet_count);
            double *packet = symFactWork;
            if (iam == sym_diag_proc)
            {
                std::memcpy(packet, symV2DiagBlocks[k],
                            diag_count * sizeof(double));
                std::memcpy(packet + diag_count, invDiag,
                            diag_count * sizeof(double));
            }
            symldl_v2_diag_bcast_chunks(
                packet, packet_count, (int) sym_diag_root,
                grid3d->cscp.comm, chunk_counter);
            std::memcpy(symV2DiagBlocks[k], packet,
                        diag_count * sizeof(double));
            std::memcpy(invDiag, packet + diag_count,
                        diag_count * sizeof(double));
        }
        else if (!symV2UsesCpuFactor())
        {
            symldl_v2_diag_bcast_chunks(
                symV2DiagBlocks[k], diag_count, (int) sym_diag_root,
                grid3d->cscp.comm, chunk_counter);
            symldl_v2_diag_bcast_chunks(
                invDiag, diag_count, (int) sym_diag_root,
                grid3d->cscp.comm, chunk_counter);
        }
        if (symV2UsesCpuFactor())
        {
            symV2CpuInvDiagCommTime += SuperLU_timer_() - comm_start;
            if (myrow == sym_diag_root && grid3d->cscp.Np > 1)
                symV2CpuInvDiagBytes += static_cast<uint64_t>(
                    diag_count * sizeof(double)) *
                    static_cast<uint64_t>(grid3d->cscp.Np - 1);
        }
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
            double transform_start = SuperLU_timer_();
            symldl_v2_ensure_factor_work(
                this, (int64_t) lpanel.nzrows() * (int64_t) ksupc);
            lpanel.panelSolveSymmetric(ksupc, invDiag, ksupc,
                                       symFactWork, lpanel.nzrows());
            if (symV2UsesCpuFactor())
                symV2CpuWTransformTime +=
                    SuperLU_timer_() - transform_start;
        }
    }

    SCT->tDiagFactorPanelSolve += (SuperLU_timer_() - t0);
    return 0;
#endif
}
