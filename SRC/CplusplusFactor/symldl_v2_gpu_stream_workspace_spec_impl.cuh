#pragma once

#include "xlupanels.hpp"
#include "dsymldl_v2_workspace_size.h"
#include "symldl_v2_config.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"

#ifdef HAVE_CUDA

static_assert(MAX_CUDA_STREAMS == DSYMLDL_V2_MAX_GPU_STREAMS,
              "SymLDL GPU stream limits are inconsistent.");
static_assert(LPANEL_HEADER_SIZE == DSYMLDL_V2_PANEL_HEADER_ENTRIES,
              "SymLDL panel header sizes are inconsistent.");

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
    dSymLDLV2StreamWorkspaceCounts counts = {};
    counts.l_value_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, lu->maxLvalCount));
    counts.u_value_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.u_val_count));
    counts.partner_value_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, lu->maxSymPartnerLvalCount));
    counts.partner_stage_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.partner_stage_count));
    counts.partner_send_stage_count = spec.need_partner_send_stage
        ? static_cast<size_t>(SUPERLU_MAX(
              (int_t) 1, lu->maxSymPartnerLSendStageCount))
        : 0;
    counts.row_stage_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.row_stage_count));
    counts.row_receive_value_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.row_recv_val_count));
    counts.raw_panel_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.raw_panel_count));
    counts.l_index_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, lu->maxLidxCount));
    counts.u_index_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.u_idx_count));
    counts.partner_index_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, lu->maxSymPartnerLidxCount));
    counts.row_receive_index_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.row_recv_idx_count));
    counts.row_send_map_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.row_send_map_count));
    counts.diagonal_work_count = static_cast<size_t>(
        SUPERLU_MAX(0, spec.diag_work_count));
    counts.lookahead_l_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, lu->maxLvalCount));
    counts.lookahead_row_count = static_cast<size_t>(
        SUPERLU_MAX((int_t) 0, spec.lookahead_u_count));
    counts.pc_fragment_schur = spec.pc_fragment_schur;
    counts.need_diagonal_work = spec.need_diag_work;

    size_t bytes = 0;
    if (!dSymLDLV2StreamWorkspaceBytes(
            &counts, sizeof(Ftype), sizeof(int_t), sizeof(int), &bytes))
        ABORT("SymFact V2 stream workspace size overflows.");
    return bytes;
}

#endif
