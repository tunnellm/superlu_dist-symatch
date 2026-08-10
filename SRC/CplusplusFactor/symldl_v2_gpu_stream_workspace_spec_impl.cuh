#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"

#ifdef HAVE_CUDA

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
        (lu->useSymV2Solve() &&
         symldl_v2_use_wpanel_cache(lu->grid3d))
            ? lu->maxLvalCount
            : 0;
    spec.pc_fragment_schur =
        lu->symV2UsesGpuFactor() &&
        symldl_v2_use_gpu_pc_fragment_schur(lu->grid3d);
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
    if (lu->useSymV2Solve() && lu->Pr > 1 &&
        lu->symL2LSendMapPoolCount > 0 &&
        lu->symL2LSendMapPoolGPU == NULL)
        spec.row_send_map_count =
            SUPERLU_MAX(spec.row_send_map_count,
                        lu->maxSymPartnerLSendStageCount);
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
    }
    if (spec.row_send_map_count > 0)
    {
        offset = symldl_v2_arena_advance(
            offset, static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                    spec.row_send_map_count)),
            sizeof(int_t), "SymFact V2 stream L-fragment send maps");
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

#endif
