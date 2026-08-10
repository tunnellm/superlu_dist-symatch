#pragma once

#include "xlupanels.hpp"
#include "gpu_mpi_utils.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_fragment_exchange_common.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PanelBcastGPU(int_t k, int_t offset)
{
    if (!useSymV2Solve())
        ABORT("GPU3DVERSION=2 requires SymFact=YES.");

    double t0 = SuperLU_timer_();
    int_t sym_panel_root = symV2PanelRoot(k);
    bool pc_fragment_schur = symV2UsePcFragmentSchurPanel(k);
    symV2RouteProfileNotePanelBcast(pc_fragment_schur);
    xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);

    if (Pr > 1 || pc_fragment_schur)
        dSymV2LFragmentExchangeGPU(k, offset);

    bool local_singleton_panel =
        symV2IsCollapsedGrid() &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1;

    const bool pcfrag_async_exchange_panel_ready =
        pc_fragment_schur && Pc > 1 &&
        !superlu_cuda_aware_mpi() &&
        superlu_sym_v2_pc_fragment_ldl_native() &&
        superlu_sym_v2_row_l_plan_v2_exchange() &&
        superlu_sym_v2_row_l_direct_recv() &&
        superlu_sym_v2_row_l_compressed_plan() &&
        superlu_sym_v2_row_l_lazy_sendmap() &&
        (superlu_sym_v2_pcfrag_async_exchange() ||
         superlu_sym_v2_pcfrag_async_pipeline());

    if (superlu_sym_v2_async_factor() &&
        pcfrag_async_exchange_panel_ready &&
        mycol == sym_panel_root &&
        LidxSendCounts[k] > 0 && k >= 0 &&
        static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
        symPanelReadyEventIds[k] >= 0)
    {
        int event_id = symPanelReadyEventIds[k];
        if (event_id >= A_gpu.numCudaStreams)
            ABORT("SymFact V2 panel-ready event is invalid.");
        symPanelReadyEventIds[k] = -1;
    }
    else if (superlu_sym_v2_async_factor() &&
             !local_singleton_panel && mycol == sym_panel_root &&
             LidxSendCounts[k] > 0 && k >= 0 &&
             static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
             symPanelReadyEventIds[k] >= 0)
    {
        int event_id = symPanelReadyEventIds[k];
        if (event_id >= A_gpu.numCudaStreams)
            ABORT("SymFact V2 panel-ready event is invalid.");
        gpuErrchk(cudaEventSynchronize(A_gpu.panelReadyEvents[event_id]));
        symPanelReadyEventIds[k] = -1;
    }

    if (LidxSendCounts[k] > 0 && grid3d->rscp.Np > 1 && !pc_fragment_schur)
    {
        int lidx_count = symldl_v2_mpi_count(
            LidxSendCounts[k],
            "SymFact V2 L-panel index count exceeds MPI limit.");
        int lval_count = symldl_v2_mpi_count(
            LvalSendCounts[k],
            "SymFact V2 L-panel value count exceeds MPI limit.");
        superlu_gpu_mpi_bcast(k_lpanel.gpuPanel.index, k_lpanel.index,
                              sizeof(int_t), lidx_count, mpi_int_t,
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
        superlu_gpu_mpi_bcast(k_lpanel.gpuPanel.val, k_lpanel.val,
                              sizeof(Ftype), lval_count,
                              get_mpi_type<Ftype>(),
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
        if (superlu_cuda_aware_mpi())
        {
            gpuErrchk(cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index,
                                 sizeof(int_t) *
                                     static_cast<size_t>(lidx_count),
                                 cudaMemcpyDeviceToHost));
        }
    }

    if (symV2UseGpuPr1Specialization() &&
        LidxSendCounts[k] > 0)
    {
        int_t ksupc = SuperSize(k);
        if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers) ||
            symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers))
            ABORT("SymFact V2 diagonal block vector has invalid size.");
        if (mycol == sym_panel_root && symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 device diagonal block is missing.");
        if (symV2DiagBlocks[k] == NULL)
        {
            symV2DiagBlocks[k] = (Ftype *) SUPERLU_MALLOC(
                symldl_v2_square_bytes(
                    ksupc, sizeof(Ftype),
                    "SymFact V2 diagonal block allocation overflows."));
            if (symV2DiagBlocks[k] == NULL)
                ABORT("Malloc fails for SymFact V2 diagonal block.");
        }
        if (symV2DiagBlocksGPU[k] == NULL)
            gpuErrchk(cudaMalloc(
                (void **) &symV2DiagBlocksGPU[k],
                symldl_v2_square_bytes(
                    ksupc, sizeof(Ftype),
                    "SymFact V2 device diagonal block allocation overflows.")));

        int diag_count = symldl_v2_mpi_count(
            symldl_v2_square_count(ksupc,
                                   "SymFact V2 diagonal block count overflows."),
            "SymFact V2 diagonal block count exceeds MPI limit.");
        superlu_gpu_mpi_bcast(symV2DiagBlocksGPU[k], symV2DiagBlocks[k],
                              sizeof(Ftype), diag_count,
                              get_mpi_type<Ftype>(),
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
    }

    if (symV2IsPr1Fastpath() && !symV2IsPc1Fastpath())
        symV2RouteProfileNote(
            pc_fragment_schur
                ? SYM_V2_ROUTE_GPU_PR1_DUAL_FRAGMENT
                : SYM_V2_ROUTE_GPU_PR1_FULL_PANEL);

    SCT->tPanelBcast += (SuperLU_timer_() - t0);
    return 0;
}


#endif
