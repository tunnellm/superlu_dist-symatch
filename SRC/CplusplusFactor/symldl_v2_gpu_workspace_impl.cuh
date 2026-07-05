#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static inline size_t symldl_v2_arena_align(size_t value)
{
    const size_t alignment = 256;
    const size_t mask = alignment - 1;
    if (value > static_cast<size_t>(-1) - mask)
        ABORT("SymFact V2 GPU arena alignment overflows.");
    return (value + mask) & ~mask;
}

static inline size_t symldl_v2_arena_advance(
    size_t offset, size_t count, size_t elem_size, const char *what)
{
    offset = symldl_v2_arena_align(offset);
    if (elem_size != 0 && count > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    size_t bytes = count * elem_size;
    if (offset > static_cast<size_t>(-1) - bytes)
        ABORT(what);
    return offset + bytes;
}

static inline size_t symldl_v2_cuda_bytes(int_t count, size_t elem_size,
                                          const char *what)
{
    if (count <= 0)
        return 0;
    size_t n = static_cast<size_t>(count);
    if (elem_size != 0 && n > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return n * elem_size;
}

static inline void symldl_v2_cuda_malloc_optional(void **ptr, int_t count,
                                                  size_t elem_size,
                                                  const char *what)
{
    size_t bytes = symldl_v2_cuda_bytes(count, elem_size, what);
    if (bytes == 0)
    {
        *ptr = NULL;
        return;
    }
    gpuErrchk(cudaMalloc(ptr, bytes));
}

template <typename Ftype>
static void symldl_v2_setup_raw_panel_ring(xLUstruct_t<Ftype> *lu,
                                           int nstreams)
{
    lu->symV2RawPanelNodes.clear();
    if (lu->useSymV2Solve() && superlu_sym_v2_wpanel_cache())
        lu->symV2RawPanelNodes.assign(static_cast<size_t>(nstreams),
                                      (int_t) -1);
}

template <typename Ftype>
static void symldl_v2_setup_gpu_fragment_stream_buffers(
    xLUstruct_t<Ftype> *lu, int stream);

template <typename Ftype>
static size_t symldl_v2_l_panel_arena_bytes(xlpanel_t<Ftype> &panel)
{
    if (panel.isEmpty())
        return 0;

    size_t offset = 0;
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(panel.indexSize()), sizeof(int_t),
        "SymFact V2 L-panel arena index size overflows.");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(panel.nzvalSize()), sizeof(Ftype),
        "SymFact V2 L-panel arena value size overflows.");
    return symldl_v2_arena_align(offset);
}

template <typename Ftype>
static xlpanelGPU_t<Ftype> symldl_v2_copy_l_panel_to_arena(
    xlpanel_t<Ftype> &panel, void *base_ptr, size_t panel_bytes)
{
    if (panel.isEmpty())
        return panel.gpuPanel;

    char *base = static_cast<char *>(base_ptr);
    size_t offset = 0;
    size_t begin = symldl_v2_arena_align(offset);
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(panel.indexSize()), sizeof(int_t),
        "SymFact V2 L-panel arena index size overflows.");
    if (offset > panel_bytes)
        ABORT("SymFact V2 L-panel arena index layout is invalid.");
    panel.gpuPanel.index = reinterpret_cast<int_t *>(base + begin);
    gpuErrchk(cudaMemcpy(panel.gpuPanel.index, panel.index,
                         sizeof(int_t) *
                             static_cast<size_t>(panel.indexSize()),
                         cudaMemcpyHostToDevice));

    begin = symldl_v2_arena_align(offset);
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(panel.nzvalSize()), sizeof(Ftype),
        "SymFact V2 L-panel arena value size overflows.");
    if (offset > panel_bytes)
        ABORT("SymFact V2 L-panel arena value layout is invalid.");
    panel.gpuPanel.val = reinterpret_cast<Ftype *>(base + begin);
    gpuErrchk(cudaMemcpy(panel.gpuPanel.val, panel.val,
                         sizeof(Ftype) *
                             static_cast<size_t>(panel.nzvalSize()),
                         cudaMemcpyHostToDevice));
    return panel.gpuPanel;
}

template <typename Ftype>
static void symldl_v2_copy_l_panels_to_gpu(
    xLUstruct_t<Ftype> *lu, xlpanelGPU_t<Ftype> *lpanel_gpu,
    int_t local_l_panel_count)
{
    if (!superlu_sym_v2_panel_arena_enabled() || !lu->useSymV2Solve())
    {
        for (int_t i = 0; i < local_l_panel_count; ++i)
        {
            int_t k0 = lu->useSymV2Solve() ? lu->symV2PanelGid(i)
                                           : i * lu->Pc + lu->mycol;
            if (k0 < lu->nsupers && lu->isNodeInMyGrid[k0] == 1)
                lpanel_gpu[i] = lu->lPanelVec[i].copyToGPU();
        }
        return;
    }

    size_t arena_bytes = 0;
    for (int_t i = 0; i < local_l_panel_count; ++i)
    {
        int_t gid = lu->symV2PanelGid(i);
        if (gid >= lu->nsupers || lu->isNodeInMyGrid[gid] != 1 ||
            lu->lPanelVec[i].isEmpty())
            continue;
        arena_bytes = symldl_v2_arena_align(arena_bytes);
        size_t panel_bytes =
            symldl_v2_l_panel_arena_bytes(lu->lPanelVec[i]);
        if (arena_bytes > static_cast<size_t>(-1) - panel_bytes)
            ABORT("SymFact V2 L-panel arena size overflows.");
        arena_bytes += panel_bytes;
    }
    arena_bytes = symldl_v2_arena_align(arena_bytes);
    if (arena_bytes > 0)
    {
        gpuErrchk(cudaMalloc(&lu->symV2LPanelArenaGPU, arena_bytes));
        lu->symV2LPanelArenaBytes = arena_bytes;
    }

    size_t arena_offset = 0;
    for (int_t i = 0; i < local_l_panel_count; ++i)
    {
        int_t gid = lu->symV2PanelGid(i);
        if (gid >= lu->nsupers || lu->isNodeInMyGrid[gid] != 1 ||
            lu->lPanelVec[i].isEmpty())
            continue;
        arena_offset = symldl_v2_arena_align(arena_offset);
        size_t panel_bytes =
            symldl_v2_l_panel_arena_bytes(lu->lPanelVec[i]);
        if (lu->symV2LPanelArenaGPU == NULL ||
            arena_offset + panel_bytes > lu->symV2LPanelArenaBytes ||
            arena_offset + panel_bytes < arena_offset)
            ABORT("SymFact V2 L-panel arena layout is invalid.");
        lpanel_gpu[i] = symldl_v2_copy_l_panel_to_arena(
            lu->lPanelVec[i],
            static_cast<char *>(lu->symV2LPanelArenaGPU) + arena_offset,
            panel_bytes);
        arena_offset += panel_bytes;
    }
}

template <typename Ftype>
struct SymV2GpuStreamWorkspaceSpec
{
    int_t u_val_count;
    int_t u_idx_count;
    int_t partner_stage_count;
    int_t raw_panel_count;
    int_t row_stage_count;
    int_t row_recv_val_count;
    int_t row_recv_idx_count;
    int_t row_send_map_count;
    int_t lookahead_u_count;
    int_t diag_work_count;
    bool need_diag_work;
    bool need_partner_send_stage;
    bool pc_fragment_schur;
};

template <typename Ftype>
static SymV2GpuStreamWorkspaceSpec<Ftype>
symldl_v2_make_stream_workspace_spec(xLUstruct_t<Ftype> *lu,
                                     int diag_work_count,
                                     bool need_diag_work)
{
    SymV2GpuStreamWorkspaceSpec<Ftype> spec;
    spec.u_val_count = lu->needsUPanelStorage() ? lu->maxUvalCount : 0;
    spec.u_idx_count = lu->needsUPanelStorage() ? lu->maxUidxCount : 0;
    spec.partner_stage_count = lu->maxSymPartnerLvalCount;
    if (lu->useSymV2Solve() && lu->Pr <= 1)
        spec.partner_stage_count =
            SUPERLU_MAX(spec.partner_stage_count, lu->maxLvalCount);
    spec.raw_panel_count =
        (lu->useSymV2Solve() && superlu_sym_v2_wpanel_cache())
            ? lu->maxLvalCount
            : 0;
    spec.pc_fragment_schur = symldl_v2_use_pc_fragment_schur(lu->grid3d);
    spec.need_partner_send_stage =
        (spec.pc_fragment_schur &&
         superlu_sym_v2_row_l_separate_send_staging()) ||
        (lu->useSymV2Solve() && lu->Pr > 1 &&
         lu->symL2LSendMapPoolCount > 0 &&
         lu->symV2PartnerLSendBufPoolCount == 0);
    spec.row_stage_count =
        spec.pc_fragment_schur ? lu->maxSymV2RowFragStageCount : 0;
    if (spec.pc_fragment_schur &&
        (superlu_sym_v2_row_l_pack_all_dest() ||
         superlu_sym_v2_row_l_plan_v2_exchange()))
        spec.row_stage_count =
            SUPERLU_MAX(spec.row_stage_count,
                        lu->maxSymV2RowFragValSendCount);
    spec.row_recv_val_count =
        spec.pc_fragment_schur ? lu->maxSymV2RowFragValRecvCount : 0;
    spec.row_recv_idx_count =
        spec.pc_fragment_schur ? lu->maxSymV2RowFragIdxRecvCount : 0;
    spec.row_send_map_count =
        spec.pc_fragment_schur ? spec.row_stage_count : 0;
    spec.lookahead_u_count = lu->maxUvalCount;
    if (lu->useSymV2Solve() && lu->Pr <= 1)
        spec.lookahead_u_count =
            SUPERLU_MAX(spec.lookahead_u_count, lu->maxLvalCount);
    spec.diag_work_count = diag_work_count;
    spec.need_diag_work = need_diag_work;
    return spec;
}

template <typename Ftype>
static size_t symldl_v2_stream_workspace_bytes(
    xLUstruct_t<Ftype> *lu,
    const SymV2GpuStreamWorkspaceSpec<Ftype> &spec)
{
    size_t offset = 0;
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLvalCount)),
        sizeof(Ftype), "SymFact V2 stream L receive values");
    if (spec.u_val_count > 0)
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(spec.u_val_count), sizeof(Ftype),
            "SymFact V2 stream U receive values");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                lu->maxSymPartnerLvalCount)),
        sizeof(Ftype), "SymFact V2 stream partner values");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                spec.partner_stage_count)),
        sizeof(Ftype), "SymFact V2 stream partner staging");
    if (spec.need_partner_send_stage)
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, lu->maxSymPartnerLSendStageCount)),
            sizeof(Ftype), "SymFact V2 stream partner send staging");
    if (spec.pc_fragment_schur)
    {
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.row_stage_count)),
            sizeof(Ftype), "SymFact V2 stream row fragment staging");
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.row_recv_val_count)),
            sizeof(Ftype), "SymFact V2 stream row fragment values");
    }
    if (spec.raw_panel_count > 0)
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.raw_panel_count)),
            sizeof(Ftype), "SymFact V2 stream W panel");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLidxCount)),
        sizeof(int_t), "SymFact V2 stream L receive indices");
    if (spec.u_idx_count > 0)
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(spec.u_idx_count), sizeof(int_t),
            "SymFact V2 stream U receive indices");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                lu->maxSymPartnerLidxCount)),
        sizeof(int_t), "SymFact V2 stream partner indices");
    if (spec.pc_fragment_schur)
    {
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.row_recv_idx_count)),
            sizeof(int_t), "SymFact V2 stream row fragment indices");
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.row_send_map_count)),
            sizeof(int_t), "SymFact V2 stream row fragment send maps");
    }
    if (spec.need_diag_work)
    {
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX(1, spec.diag_work_count)),
            sizeof(Ftype), "SymFact V2 stream diagonal work");
        offset = symldl_v2_arena_advance(
            offset, 1, sizeof(int), "SymFact V2 stream diagonal info");
    }
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLvalCount)),
        sizeof(Ftype), "SymFact V2 stream lookahead L");
    offset = symldl_v2_arena_advance(
        offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                spec.lookahead_u_count)),
        sizeof(Ftype), "SymFact V2 stream lookahead U");
    return symldl_v2_arena_align(offset);
}

template <typename Ftype>
static void symldl_v2_clear_gpu_stream_workspace(xLUstruct_t<Ftype> *lu,
                                                 int stream)
{
    lu->A_gpu.LvalRecvBufs[stream] = NULL;
    lu->A_gpu.UvalRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLvalRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLStageBufs[stream] = NULL;
    lu->A_gpu.symPartnerLSendStageBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragStageBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragValRecvBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragIdxRecvBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragSendMapStageBufs[stream] = NULL;
    lu->A_gpu.symV2RawPanelBufs[stream] = NULL;
    lu->A_gpu.symV2RawPanelReadyEvents[stream] = NULL;
    lu->A_gpu.LidxRecvBufs[stream] = NULL;
    lu->A_gpu.UidxRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLidxRecvBufs[stream] = NULL;
    lu->A_gpu.diagFactWork[stream] = NULL;
    lu->A_gpu.diagFactInfo[stream] = NULL;
    lu->A_gpu.lookAheadLGemmBuffer[stream] = NULL;
    lu->A_gpu.lookAheadUGemmBuffer[stream] = NULL;
}

template <typename Ftype>
static void symldl_v2_setup_gpu_stream_workspace(
    xLUstruct_t<Ftype> *lu, int stream,
    const SymV2GpuStreamWorkspaceSpec<Ftype> &spec)
{
    symldl_v2_clear_gpu_stream_workspace(lu, stream);

    if (lu->useSymV2Solve() && superlu_sym_v2_workspace_arena_enabled())
    {
        size_t stride = symldl_v2_stream_workspace_bytes(lu, spec);
        if (lu->symV2StreamArenaGPU == NULL)
        {
            size_t nstreams = static_cast<size_t>(lu->A_gpu.numCudaStreams);
            if (stride != 0 &&
                nstreams > static_cast<size_t>(-1) / stride)
                ABORT("SymFact V2 stream workspace arena size overflows.");
            lu->symV2StreamArenaBytes = stride * nstreams;
            gpuErrchk(cudaMalloc(&lu->symV2StreamArenaGPU,
                                 lu->symV2StreamArenaBytes));
        }

        char *stream_base =
            static_cast<char *>(lu->symV2StreamArenaGPU) +
            static_cast<size_t>(stream) * stride;
        size_t offset = 0;
        auto take = [&](size_t count, size_t elem_size,
                        const char *what) -> void *
        {
            size_t begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(offset, count, elem_size, what);
            if (offset > stride)
                ABORT("SymFact V2 stream workspace layout is invalid.");
            return stream_base + begin;
        };

        lu->A_gpu.LvalRecvBufs[stream] = static_cast<Ftype *>(take(
            static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLvalCount)),
            sizeof(Ftype), "SymFact V2 stream L receive values"));
        if (spec.u_val_count > 0)
            lu->A_gpu.UvalRecvBufs[stream] = static_cast<Ftype *>(take(
                static_cast<size_t>(spec.u_val_count), sizeof(Ftype),
                "SymFact V2 stream U receive values"));
        lu->A_gpu.symPartnerLvalRecvBufs[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxSymPartnerLvalCount)),
                sizeof(Ftype), "SymFact V2 stream partner values"));
        lu->A_gpu.symPartnerLStageBufs[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, spec.partner_stage_count)),
                sizeof(Ftype), "SymFact V2 stream partner staging"));
        if (spec.need_partner_send_stage)
            lu->A_gpu.symPartnerLSendStageBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, lu->maxSymPartnerLSendStageCount)),
                    sizeof(Ftype),
                    "SymFact V2 stream partner send staging"));
        if (spec.pc_fragment_schur)
        {
            lu->A_gpu.symV2RowFragStageBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_stage_count)),
                    sizeof(Ftype),
                    "SymFact V2 stream row fragment staging"));
            lu->A_gpu.symV2RowFragValRecvBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_recv_val_count)),
                    sizeof(Ftype),
                    "SymFact V2 stream row fragment values"));
        }
        if (spec.raw_panel_count > 0)
            lu->A_gpu.symV2RawPanelBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.raw_panel_count)),
                    sizeof(Ftype), "SymFact V2 stream W panel"));
        lu->A_gpu.LidxRecvBufs[stream] = static_cast<int_t *>(take(
            static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLidxCount)),
            sizeof(int_t), "SymFact V2 stream L receive indices"));
        if (spec.u_idx_count > 0)
            lu->A_gpu.UidxRecvBufs[stream] = static_cast<int_t *>(take(
                static_cast<size_t>(spec.u_idx_count), sizeof(int_t),
                "SymFact V2 stream U receive indices"));
        lu->A_gpu.symPartnerLidxRecvBufs[stream] =
            static_cast<int_t *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxSymPartnerLidxCount)),
                sizeof(int_t), "SymFact V2 stream partner indices"));
        if (spec.pc_fragment_schur)
        {
            lu->A_gpu.symV2RowFragIdxRecvBufs[stream] =
                static_cast<int_t *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_recv_idx_count)),
                    sizeof(int_t),
                    "SymFact V2 stream row fragment indices"));
            lu->A_gpu.symV2RowFragSendMapStageBufs[stream] =
                static_cast<int_t *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_send_map_count)),
                    sizeof(int_t),
                    "SymFact V2 stream row fragment send maps"));
        }
        if (spec.need_diag_work)
        {
            lu->A_gpu.diagFactWork[stream] = static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(1, spec.diag_work_count)),
                sizeof(Ftype), "SymFact V2 stream diagonal work"));
            lu->A_gpu.diagFactInfo[stream] = static_cast<int *>(take(
                1, sizeof(int), "SymFact V2 stream diagonal info"));
        }
        lu->A_gpu.lookAheadLGemmBuffer[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxLvalCount)),
                sizeof(Ftype), "SymFact V2 stream lookahead L"));
        lu->A_gpu.lookAheadUGemmBuffer[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, spec.lookahead_u_count)),
                sizeof(Ftype), "SymFact V2 stream lookahead U"));
        return;
    }

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.LvalRecvBufs[stream], lu->maxLvalCount,
        sizeof(Ftype), "L value receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.UvalRecvBufs[stream], spec.u_val_count,
        sizeof(Ftype), "U value receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.LidxRecvBufs[stream], lu->maxLidxCount,
        sizeof(int_t), "L index receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.UidxRecvBufs[stream], spec.u_idx_count,
        sizeof(int_t), "U index receive buffer allocation overflows.");
    symldl_v2_setup_gpu_fragment_stream_buffers(lu, stream);
    if (spec.need_diag_work)
    {
        gpuErrchk(cudaMalloc(&lu->A_gpu.diagFactWork[stream],
                             sizeof(Ftype) *
                                 static_cast<size_t>(spec.diag_work_count)));
        gpuErrchk(cudaMalloc(&lu->A_gpu.diagFactInfo[stream],
                             sizeof(int)));
    }
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.lookAheadLGemmBuffer[stream],
        lu->maxLvalCount, sizeof(Ftype),
        "Lookahead L buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.lookAheadUGemmBuffer[stream],
        spec.lookahead_u_count, sizeof(Ftype),
        "Lookahead U buffer allocation overflows.");
}

template <typename Ftype>
static void symldl_v2_setup_gemm_workspace(
    xLUstruct_t<Ftype> *lu, int num_bufs, size_t dfbuf_elems,
    size_t *sum_diag_size, size_t *sum_gemm_size)
{
    if (lu->useSymV2Solve() && superlu_sym_v2_workspace_arena_enabled())
    {
        size_t stride = 0;
        stride = symldl_v2_arena_advance(
            stride, dfbuf_elems, sizeof(Ftype),
            "SymFact V2 diagonal buffer arena overflows.");
        stride = symldl_v2_arena_advance(
            stride, static_cast<size_t>(lu->A_gpu.gemmBufferSize),
            sizeof(Ftype), "SymFact V2 GEMM buffer arena overflows.");
        stride = symldl_v2_arena_align(stride);
        size_t nbufs = static_cast<size_t>(num_bufs);
        if (stride != 0 && nbufs > static_cast<size_t>(-1) / stride)
            ABORT("SymFact V2 GEMM workspace arena size overflows.");
        lu->symV2GemmArenaBytes = stride * nbufs;
        if (lu->symV2GemmArenaBytes > 0)
            gpuErrchk(cudaMalloc(&lu->symV2GemmArenaGPU,
                                 lu->symV2GemmArenaBytes));

        for (int i = 0; i < num_bufs; ++i)
        {
            char *base = static_cast<char *>(lu->symV2GemmArenaGPU) +
                         static_cast<size_t>(i) * stride;
            size_t offset = 0;
            size_t begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(
                offset, dfbuf_elems, sizeof(Ftype),
                "SymFact V2 diagonal buffer arena overflows.");
            lu->A_gpu.dFBufs[i] = reinterpret_cast<Ftype *>(base + begin);
            begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(
                offset, static_cast<size_t>(lu->A_gpu.gemmBufferSize),
                sizeof(Ftype), "SymFact V2 GEMM buffer arena overflows.");
            if (offset > stride)
                ABORT("SymFact V2 GEMM workspace layout is invalid.");
            lu->A_gpu.gpuGemmBuffs[i] =
                reinterpret_cast<Ftype *>(base + begin);
            *sum_diag_size += dfbuf_elems;
            *sum_gemm_size += lu->A_gpu.gemmBufferSize;
        }
        return;
    }

    for (int i = 0; i < num_bufs; ++i)
    {
        gpuErrchk(cudaMalloc(&(lu->A_gpu.dFBufs[i]),
                             dfbuf_elems * sizeof(Ftype)));
        gpuErrchk(cudaMalloc(&(lu->A_gpu.gpuGemmBuffs[i]),
                             lu->A_gpu.gemmBufferSize * sizeof(Ftype)));
        *sum_diag_size += dfbuf_elems;
        *sum_gemm_size += lu->A_gpu.gemmBufferSize;
    }
}

template <typename Ftype>
static void symldl_v2_setup_gpu_fragment_stream_buffers(
    xLUstruct_t<Ftype> *lu, int stream)
{
    lu->A_gpu.symV2RawPanelBufs[stream] = NULL;
    if (!lu->useSymV2Solve())
    {
        lu->A_gpu.symPartnerLvalRecvBufs[stream] = NULL;
        lu->A_gpu.symPartnerLStageBufs[stream] = NULL;
        lu->A_gpu.symPartnerLSendStageBufs[stream] = NULL;
        lu->A_gpu.symPartnerLidxRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragStageBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragValRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragIdxRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragSendMapStageBufs[stream] = NULL;
        return;
    }

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLvalRecvBufs[stream],
        lu->maxSymPartnerLvalCount, sizeof(Ftype),
        "SymFact V2 partner receive buffer allocation overflows.");
    int_t partner_stage_count = lu->maxSymPartnerLvalCount;
    if (lu->Pr <= 1)
        partner_stage_count =
            SUPERLU_MAX(partner_stage_count, lu->maxLvalCount);
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLStageBufs[stream],
        partner_stage_count, sizeof(Ftype),
        "SymFact V2 partner staging buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLSendStageBufs[stream],
        lu->maxSymPartnerLSendStageCount, sizeof(Ftype),
        "SymFact V2 partner send staging allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLidxRecvBufs[stream],
        lu->maxSymPartnerLidxCount, sizeof(int_t),
        "SymFact V2 partner index buffer allocation overflows.");

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragStageBufs[stream],
        lu->maxSymV2RowFragStageCount, sizeof(Ftype),
        "SymFact V2 row staging buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragValRecvBufs[stream],
        lu->maxSymV2RowFragValRecvCount, sizeof(Ftype),
        "SymFact V2 row receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragIdxRecvBufs[stream],
        lu->maxSymV2RowFragIdxRecvCount, sizeof(int_t),
        "SymFact V2 row index buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragSendMapStageBufs[stream],
        lu->maxSymV2RowFragValSendCount, sizeof(int_t),
        "SymFact V2 row send-map staging allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RawPanelBufs[stream],
        superlu_sym_v2_wpanel_cache() ? lu->maxLvalCount : 0,
        sizeof(Ftype),
        "SymFact V2 W-panel cache allocation overflows.");
}

template <typename Ftype>
static void symldl_v2_setup_gpu_panel_index(xLUstruct_t<Ftype> *lu)
{
    lu->A_gpu.useSymV2PanelIndex = 0;
    lu->A_gpu.symV2PanelLocalIndex = NULL;
}

template <>
inline void symldl_v2_setup_gpu_panel_index<double>(xLUstruct_t<double> *lu)
{
    lu->A_gpu.useSymV2PanelIndex = lu->useSymV2Solve() ? 1 : 0;
    lu->A_gpu.symV2PanelLocalIndex = NULL;
    if (!lu->useSymV2Solve())
        return;
    if (lu->trf3Dpartition == NULL ||
        lu->trf3Dpartition->symV2PanelLocalIndex == NULL)
        ABORT("SymFact V2 panel index metadata is missing.");
    gpuErrchk(cudaMalloc(
        (void **) &lu->A_gpu.symV2PanelLocalIndex,
        sizeof(int_t) * static_cast<size_t>(lu->nsupers)));
    gpuErrchk(cudaMemcpy(lu->A_gpu.symV2PanelLocalIndex,
                         lu->trf3Dpartition->symV2PanelLocalIndex,
                         sizeof(int_t) * static_cast<size_t>(lu->nsupers),
                         cudaMemcpyHostToDevice));
}

#endif
