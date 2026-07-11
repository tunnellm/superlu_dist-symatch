#include "dsymldl_v2_workspace_size.h"

#include <stdint.h>

static int dSymLDLV2ArenaAlign(size_t value, size_t *aligned)
{
    size_t mask = (size_t) DSYMLDL_V2_GPU_ARENA_ALIGNMENT - 1;
    if (value > SIZE_MAX - mask)
        return 0;
    *aligned = (value + mask) & ~mask;
    return 1;
}

static int dSymLDLV2ArenaAdvance(size_t *offset, size_t count,
                                  size_t element_size)
{
    size_t aligned;
    if (!dSymLDLV2ArenaAlign(*offset, &aligned) ||
        (element_size != 0 && count > SIZE_MAX / element_size))
        return 0;
    size_t allocation = count * element_size;
    if (aligned > SIZE_MAX - allocation)
        return 0;
    *offset = aligned + allocation;
    return 1;
}

static size_t dSymLDLV2AtLeastOne(size_t count)
{
    return count > 0 ? count : 1;
}

int dSymLDLV2StreamWorkspaceBytes(
    const dSymLDLV2StreamWorkspaceCounts *counts,
    size_t value_size, size_t index_size, size_t diagonal_info_size,
    size_t *bytes)
{
    size_t offset = 0;
    if (counts == NULL || bytes == NULL || value_size == 0 ||
        index_size == 0 || diagonal_info_size == 0)
        return 0;

    if (!dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->l_value_count),
            value_size) ||
        (counts->u_value_count > 0 &&
         !dSymLDLV2ArenaAdvance(
             &offset, counts->u_value_count, value_size)) ||
        !dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->partner_value_count),
            value_size) ||
        !dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->partner_stage_count),
            value_size) ||
        (counts->partner_send_stage_count > 0 &&
         !dSymLDLV2ArenaAdvance(
             &offset, counts->partner_send_stage_count, value_size)))
        return 0;

    if (counts->pc_fragment_schur &&
        (!dSymLDLV2ArenaAdvance(
             &offset, dSymLDLV2AtLeastOne(counts->row_stage_count),
             value_size) ||
         !dSymLDLV2ArenaAdvance(
             &offset,
             dSymLDLV2AtLeastOne(counts->row_receive_value_count),
             value_size)))
        return 0;
    if (counts->raw_panel_count > 0 &&
        !dSymLDLV2ArenaAdvance(
            &offset, counts->raw_panel_count, value_size))
        return 0;

    if (!dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->l_index_count),
            index_size) ||
        (counts->u_index_count > 0 &&
         !dSymLDLV2ArenaAdvance(
             &offset, counts->u_index_count, index_size)) ||
        !dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->partner_index_count),
            index_size))
        return 0;
    if (counts->pc_fragment_schur &&
        !dSymLDLV2ArenaAdvance(
            &offset,
            dSymLDLV2AtLeastOne(counts->row_receive_index_count),
            index_size))
        return 0;
    if (counts->row_send_map_count > 0 &&
        !dSymLDLV2ArenaAdvance(
            &offset, counts->row_send_map_count, index_size))
        return 0;

    if (counts->need_diagonal_work &&
        (!dSymLDLV2ArenaAdvance(
             &offset, dSymLDLV2AtLeastOne(counts->diagonal_work_count),
             value_size) ||
         !dSymLDLV2ArenaAdvance(
             &offset, 1, diagonal_info_size)))
        return 0;
    if (!dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->lookahead_l_count),
            value_size) ||
        !dSymLDLV2ArenaAdvance(
            &offset, dSymLDLV2AtLeastOne(counts->lookahead_row_count),
            value_size) ||
        !dSymLDLV2ArenaAlign(offset, bytes))
        return 0;
    return 1;
}

int dSymLDLV2GemmWorkspaceBytes(
    size_t diagonal_value_count, size_t gemm_value_count,
    size_t value_size, size_t *bytes)
{
    size_t offset = 0;
    if (bytes == NULL || value_size == 0 ||
        !dSymLDLV2ArenaAdvance(
            &offset, diagonal_value_count, value_size) ||
        !dSymLDLV2ArenaAdvance(
            &offset, gemm_value_count, value_size) ||
        !dSymLDLV2ArenaAlign(offset, bytes))
        return 0;
    return 1;
}
