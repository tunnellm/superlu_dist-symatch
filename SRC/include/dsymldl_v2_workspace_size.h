#ifndef DSYMLDL_V2_WORKSPACE_SIZE_H
#define DSYMLDL_V2_WORKSPACE_SIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DSYMLDL_V2_MAX_GPU_STREAMS 64
#define DSYMLDL_V2_PANEL_HEADER_ENTRIES 4
#define DSYMLDL_V2_GPU_ARENA_ALIGNMENT 256

#if DSYMLDL_V2_GPU_ARENA_ALIGNMENT < 1 || \
    (DSYMLDL_V2_GPU_ARENA_ALIGNMENT & \
     (DSYMLDL_V2_GPU_ARENA_ALIGNMENT - 1)) != 0
#error "SymLDL GPU arena alignment must be a power of two."
#endif

typedef struct {
    size_t l_value_count;
    size_t u_value_count;
    size_t partner_value_count;
    size_t partner_stage_count;
    size_t partner_send_stage_count;
    size_t row_stage_count;
    size_t row_receive_value_count;
    size_t raw_panel_count;
    size_t l_index_count;
    size_t u_index_count;
    size_t partner_index_count;
    size_t row_receive_index_count;
    size_t row_send_map_count;
    size_t diagonal_work_count;
    size_t lookahead_l_count;
    size_t lookahead_row_count;
    int pc_fragment_schur;
    int need_diagonal_work;
} dSymLDLV2StreamWorkspaceCounts;

int dSymLDLV2StreamWorkspaceBytes(
    const dSymLDLV2StreamWorkspaceCounts *counts,
    size_t value_size, size_t index_size, size_t diagonal_info_size,
    size_t *bytes);

int dSymLDLV2GemmWorkspaceBytes(
    size_t diagonal_value_count, size_t gemm_value_count,
    size_t value_size, size_t *bytes);

#ifdef __cplusplus
}
#endif

#endif
