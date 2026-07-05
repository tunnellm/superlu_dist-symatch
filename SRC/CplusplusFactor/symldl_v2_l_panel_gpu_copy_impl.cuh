#pragma once

#include "symldl_v2_config.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"

#ifdef HAVE_CUDA

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

#endif
