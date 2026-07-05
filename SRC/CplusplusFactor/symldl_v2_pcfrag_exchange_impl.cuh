#pragma once

#include <climits>

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

static inline int symldl_v2_mpi_count(size_t count, const char *what)
{
    if (count > static_cast<size_t>(INT_MAX))
        ABORT(what);
    return static_cast<int>(count);
}

static inline int symldl_v2_mpi_count(int_t count, const char *what)
{
    if (count < 0)
        ABORT(what);
    return symldl_v2_mpi_count(static_cast<size_t>(count), what);
}

static inline size_t symldl_v2_square_count(int_t n, const char *what)
{
    if (n < 0)
        ABORT(what);
    size_t dim = static_cast<size_t>(n);
    if (dim != 0 && dim > static_cast<size_t>(-1) / dim)
        ABORT(what);
    return dim * dim;
}

static inline size_t symldl_v2_square_bytes(int_t n, size_t elem_size,
                                            const char *what)
{
    size_t count = symldl_v2_square_count(n, what);
    if (elem_size != 0 && count > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return count * elem_size;
}

static __global__ void symldl_v2_lfrag_pack_kernel(const double *lpanel,
                                                   double *sendbuf,
                                                   const int_t *sendmap,
                                                   int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    sendbuf[idx] = (src < 0) ? (double) (-src - 1) : lpanel[src];
}

static __global__ void symldl_v2_row_down_pack_segments_kernel(
    const double *lpanel,
    double *sendbuf,
    const int_t *base_sendmap,
    const SymV2RowDownSendSegmentGPU *segments,
    int nsegments,
    int_t ksupc,
    int_t dst_lda)
{
    int seg_id = blockIdx.x;
    if (seg_id >= nsegments)
        return;

    SymV2RowDownSendSegmentGPU seg = segments[seg_id];
    int_t nrows = seg.nrows;
    if (nrows <= 0 || ksupc <= 0 || dst_lda <= 0)
        return;

    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        size_t map_pos = seg.map_offset +
                         static_cast<size_t>(row) +
                         static_cast<size_t>(col) *
                             static_cast<size_t>(nrows);
        int_t src = base_sendmap[map_pos];
        sendbuf[seg.dst_row_offset + row + col * dst_lda] =
            (src < 0) ? (double) (-src - 1) : lpanel[src];
    }
}

static __global__ void symldl_v2_lfrag_assemble_kernel(
    const double *stage,
    double *frag,
    const int_t *recv_map,
    int pieces,
    int_t ksupc,
    int_t frag_lda)
{
    int piece = blockIdx.x;
    if (piece >= pieces)
        return;

    int_t dst_offset = recv_map[3 * piece];
    int_t nrows = recv_map[3 * piece + 1];
    int_t src_offset = recv_map[3 * piece + 2];
    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        frag[dst_offset + row + col * frag_lda] =
            stage[src_offset + row + col * nrows];
    }
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PrepackLFragmentsGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment prepack is not implemented.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment exchange is not implemented.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PanelBcastGPU(int_t k, int_t offset)
{
    if (!useSymV2Solve())
        ABORT("GPU3DVERSION=2 requires SymFact=YES.");

    double t0 = SuperLU_timer_();
    int_t sym_panel_root = symV2PanelRoot(k);
    bool pc_fragment_schur = symV2UsePcFragmentSchurPanel(k);
    xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);

    if (Pr > 1)
        dSymV2LFragmentExchangeGPU(k, offset);

    bool local_singleton_panel =
        Pr == 1 && Pc == 1 &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1;

    const bool pcfrag_async_exchange_panel_ready =
        pc_fragment_schur && Pr > 1 && Pc > 1 &&
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

    if (Pr == 1 && Pc > 1 && LidxSendCounts[k] > 0)
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

    SCT->tPanelBcast += (SuperLU_timer_() - t0);
    return 0;
}

#endif
