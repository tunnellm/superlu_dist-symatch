#pragma once

#ifdef HAVE_CUDA

extern "C" int pdgstrf3d_symv2_diag_panel_cuda_bridge(
    void *handle, int_t k, int_t handle_offset, int_t buffer_offset,
    ddiagFactBufs_t **diag_buffers);

extern "C" int pdgstrf3d_symv2_factor_forest_cuda_bridge(
    void *handle, sForest_t *forest, ddiagFactBufs_t **diag_buffers,
    gEtreeInfo_t *etree, int tag_ub);

extern "C" int pdgstrf3d_symv2_ancestor_cuda_bridge(
    void *handle, int_t level, int_t *node_count, int_t **tree_perm);

#endif
