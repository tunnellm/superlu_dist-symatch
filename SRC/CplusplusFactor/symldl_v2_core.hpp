#pragma once

#include "superlu_ddefs.h"
#include "symldl_v2_config.hpp"

static inline bool symldl_v2_is_collapsed_grid(const gridinfo3d_t *grid3d)
{
    return grid3d == NULL || (grid3d->nprow <= 1 && grid3d->npcol <= 1);
}

static inline bool symldl_v2_is_pc1_fastpath(const gridinfo3d_t *grid3d)
{
    return grid3d != NULL && grid3d->npcol <= 1;
}

static inline bool symldl_v2_is_pr1_fastpath(const gridinfo3d_t *grid3d)
{
    return grid3d != NULL && grid3d->nprow <= 1;
}

static inline bool symldl_v2_use_pc_fragment_schur(
    const gridinfo3d_t *grid3d)
{
    return grid3d != NULL &&
           grid3d->nprow > 1 &&
           grid3d->npcol > 1 &&
           superlu_sym_v2_pc_fragment_schur() &&
           superlu_sym_v2_pc_fragment_ldl_native();
}

static inline int symldl_v2_panel_root(
    const dtrf3Dpartition_t *partition, int_t k, const gridinfo_t *grid)
{
    if (partition != NULL && partition->symV2PanelRoot != NULL)
        return partition->symV2PanelRoot[k];
    return PCOL(k, grid);
}

static inline int symldl_v2_diag_root(
    const dtrf3Dpartition_t *partition, int_t k, const gridinfo_t *grid)
{
    if (partition != NULL && partition->symV2DiagRoot != NULL)
        return partition->symV2DiagRoot[k];
    return PROW(k, grid);
}

static inline int symldl_v2_owner_2d(
    const dtrf3Dpartition_t *partition, int_t k, const gridinfo_t *grid)
{
    return PNUM(symldl_v2_diag_root(partition, k, grid),
                symldl_v2_panel_root(partition, k, grid),
                grid);
}

static inline int symldl_v2_diag_owner_rank(
    const dtrf3Dpartition_t *partition, int_t k, const gridinfo_t *grid)
{
    if (partition != NULL && partition->symV2DiagOwner != NULL)
        return partition->symV2DiagOwner[k];
    return symldl_v2_owner_2d(partition, k, grid);
}

static inline int_t symldl_v2_panel_local_index(
    const dtrf3Dpartition_t *partition, int_t k)
{
    if (partition == NULL || partition->symV2PanelLocalIndex == NULL)
        return -1;
    return partition->symV2PanelLocalIndex[k];
}

static inline int_t symldl_v2_row_local_index(
    const dtrf3Dpartition_t *partition, int_t k)
{
    if (partition == NULL || partition->symV2RowLocalIndex == NULL)
        return -1;
    return partition->symV2RowLocalIndex[k];
}

static inline bool symldl_v2_has_local_panel(
    const dtrf3Dpartition_t *partition, int_t k)
{
    return symldl_v2_panel_local_index(partition, k) >= 0;
}

static inline bool symldl_v2_has_local_row(
    const dtrf3Dpartition_t *partition, int_t k)
{
    return symldl_v2_row_local_index(partition, k) >= 0;
}

static inline void symldl_v2_validate_async_pcfrag_config()
{
    if (!superlu_sym_v2_pcfrag_async_requested()) return;
    if (!superlu_sym_v2_pcfrag_async_prereqs_enabled())
        ABORT("SymFact V2 Pc-fragment async exchange requires LDL-native lazy row-down exchange.");
#ifdef HAVE_CUDA
    superlu_sym_v2_fail_if_cuda_aware_pcfrag_async();
#endif
}
