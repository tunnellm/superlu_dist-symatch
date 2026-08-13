#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_window_update_impl.hpp"

template <typename Ftype>
static void symldl_v2_cpu_hybrid_launch_rectangle(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel,
    int_t row_begin, int_t row_end, int_t column_begin, int_t column_end,
    bool force_inline)
{
    if (row_begin >= row_end || column_begin >= column_end)
        return;
    bool lookahead = column_panel.gid(column_begin) == parent;
    for (int_t column = column_begin; column < column_end; ++column)
    {
        if ((column_panel.gid(column) == parent) != lookahead)
            ABORT("SymFact V2 CPU hybrid rectangle mixes scheduling modes.");
        symldl_v2_cpu_note_task_pending(
            lu, column_panel.gid(column), slot, 1);
    }

    int_t m = row_panel.stRow(row_end) - row_panel.stRow(row_begin);
    int_t n = column_panel.stRow(column_end) -
              column_panel.stRow(column_begin);
    uint64_t work = static_cast<uint64_t>(SUPERLU_MAX((int_t) 0, m)) *
                    static_cast<uint64_t>(SUPERLU_MAX((int_t) 0, n)) *
                    static_cast<uint64_t>(SUPERLU_MAX(
                        (int_t) 0, row_panel.ncols()));
    bool deferred = !force_inline && lu->symV2CpuWorkerCount > 1 &&
                    work >= symldl_v2_cpu_min_deferred_work();
    uint64_t generation = symldl_v2_cpu_slot_generation(
        lu, slot, source_k);
    ++lu->symV2CpuHybridRectangles;
    ++lu->symV2CpuSchurTasks;
    if (lookahead)
        ++lu->symV2CpuLookaheadTasks;
    else
        ++lu->symV2CpuExcludeTasks;

#ifdef _OPENMP
    if (deferred)
    {
        ++lu->symV2CpuHybridDeferredRectangles;
#pragma omp atomic update
        ++lu->symV2CpuDeferredTasksActive;
#pragma omp task firstprivate(lu, source_k, slot, row_panel, column_panel,       \
                              row_begin, row_end, column_begin, column_end,    \
                              lookahead, generation)
        {
            symldl_v2_cpu_assert_slot_generation(
                lu, slot, source_k, generation);
            symldl_v2_cpu_window_rectangle(
                lu, row_panel, column_panel, row_begin, row_end,
                column_begin, column_end, lookahead);
            for (int_t column = column_begin; column < column_end; ++column)
                symldl_v2_cpu_note_task_pending(
                    lu, column_panel.gid(column), slot, -1);
#pragma omp atomic update
            --lu->symV2CpuDeferredTasksActive;
        }
        return;
    }
#endif

    ++lu->symV2CpuHybridInlineRectangles;
    symldl_v2_cpu_assert_slot_generation(lu, slot, source_k, generation);
    symldl_v2_cpu_window_rectangle(
        lu, row_panel, column_panel, row_begin, row_end,
        column_begin, column_end, lookahead);
    for (int_t column = column_begin; column < column_end; ++column)
        symldl_v2_cpu_note_task_pending(
            lu, column_panel.gid(column), slot, -1);
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_limited_update(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel,
    int_t row_start, int_t row_end, int_t column_start, int_t column_end,
    bool force_inline)
{
    if (row_panel.isEmpty() || column_panel.isEmpty())
        return;
    row_start = SUPERLU_MAX((int_t) 0, row_start);
    column_start = SUPERLU_MAX((int_t) 0, column_start);
    row_end = SUPERLU_MIN(row_end, row_panel.nblocks());
    column_end = SUPERLU_MIN(column_end, column_panel.nblocks());
    if (row_start >= row_end || column_start >= column_end)
        return;

    bool row_gid_sorted = true;
    bool column_gid_sorted = true;
    if (superlu_sym_v2_lower_envelope_enabled())
    {
        for (int_t row = row_start + 1; row < row_end; ++row)
            row_gid_sorted = row_gid_sorted &&
                             row_panel.gid(row) >= row_panel.gid(row - 1);
        for (int_t column = column_start + 1;
             column < column_end; ++column)
            column_gid_sorted = column_gid_sorted &&
                column_panel.gid(column) >= column_panel.gid(column - 1);
    }
    bool lower_envelope = superlu_sym_v2_lower_envelope_enabled() &&
                          row_gid_sorted && column_gid_sorted;
    int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(lu->ldt) * lu->ldt);
    int_t max_block_rows = 1;
    for (int_t row = row_start; row < row_end; ++row)
        max_block_rows = SUPERLU_MAX(max_block_rows, row_panel.nbrow(row));

    int_t envelope_row = row_start;
    for (int_t column_group = column_start;
         column_group < column_end;)
    {
        if (lower_envelope)
            while (envelope_row < row_end &&
                   row_panel.gid(envelope_row) <
                       column_panel.gid(column_group))
                ++envelope_row;
        else
            envelope_row = row_start;
        if (envelope_row >= row_end)
            break;

        int_t group_rows = row_panel.stRow(row_end) -
                           row_panel.stRow(envelope_row);
        int column_limit = superlu_sym_v2_batch_schur_col_limit(
            group_rows, gemm_capacity);
        int64_t hard_column_limit = SUPERLU_MAX(
            static_cast<int64_t>(1), gemm_capacity / max_block_rows);
        column_limit = SUPERLU_MAX(
            1, SUPERLU_MIN(
                column_limit,
                static_cast<int>(SUPERLU_MIN(
                    hard_column_limit,
                    static_cast<int64_t>(std::numeric_limits<int>::max())))));

        int_t column_next = column_group + 1;
        while (column_next < column_end &&
               column_panel.stRow(column_next + 1) -
                       column_panel.stRow(column_group) <= column_limit)
            ++column_next;
        int_t group_columns = column_panel.stRow(column_next) -
                              column_panel.stRow(column_group);
        int_t max_rows = group_rows;
        if (static_cast<int64_t>(group_rows) * group_columns > gemm_capacity)
            max_rows = static_cast<int_t>(gemm_capacity / group_columns);
        max_rows = SUPERLU_MAX((int_t) 1, max_rows);
        for (int_t row_group = envelope_row; row_group < row_end;)
        {
            int_t row_next = row_group + 1;
            while (row_next < row_end &&
                   row_panel.stRow(row_next + 1) -
                           row_panel.stRow(row_group) <= max_rows)
                ++row_next;
            symldl_v2_cpu_hybrid_launch_rectangle(
                lu, source_k, parent, slot, row_panel, column_panel,
                row_group, row_next, column_group, column_next,
                force_inline);
            row_group = row_next;
        }
        column_group = column_next;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_submit_peer_ranges(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot, int pr,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel)
{
    size_t peer = static_cast<size_t>(k) * lu->Pr + pr;
    if (peer + 1 >= lu->symV2CpuPartnerPeerRangeOffsets.size())
        ABORT("SymFact V2 CPU hybrid peer range plan is missing.");
    int_t row_diag = symldl_v2_cpu_window_find(row_panel, k);
    int_t row_ranges[3][2];
    int row_range_count = symldl_v2_cpu_window_ranges_excluding(
        0, row_panel.nblocks(), row_diag, GLOBAL_BLOCK_NOT_FOUND,
        row_ranges);
    size_t begin = lu->symV2CpuPartnerPeerRangeOffsets[peer];
    size_t end = lu->symV2CpuPartnerPeerRangeOffsets[peer + 1];
    int_t parent_block = symldl_v2_cpu_window_find(column_panel, parent);
    int_t column_diag = symldl_v2_cpu_window_find(column_panel, k);
    for (size_t item = begin; item < end; ++item)
    {
        SymLDLV2CpuBlockRange range =
            lu->symV2CpuPartnerPeerRanges[item];
        int_t pieces[3][2];
        int piece_count = symldl_v2_cpu_window_ranges_excluding(
            range.begin, range.end, parent_block,
            column_diag, pieces);
        if (parent_block >= range.begin && parent_block < range.end)
            for (int row_range = 0; row_range < row_range_count; ++row_range)
                symldl_v2_cpu_hybrid_limited_update(
                    lu, k, parent, slot, row_panel, column_panel,
                    row_ranges[row_range][0], row_ranges[row_range][1],
                    parent_block, parent_block + 1, true);
        for (int piece = 0; piece < piece_count; ++piece)
            for (int row_range = 0; row_range < row_range_count; ++row_range)
                symldl_v2_cpu_hybrid_limited_update(
                    lu, k, parent, slot, row_panel, column_panel,
                    row_ranges[row_range][0], row_ranges[row_range][1],
                    pieces[piece][0], pieces[piece][1], false);
    }
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_submit_collapsed(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel)
{
    int_t row_diag = symldl_v2_cpu_window_find(row_panel, k);
    int_t column_diag = symldl_v2_cpu_window_find(column_panel, k);
    int_t parent_block = symldl_v2_cpu_window_find(column_panel, parent);
    int_t row_ranges[3][2];
    int_t column_ranges[3][2];
    int row_count = symldl_v2_cpu_window_ranges_excluding(
        0, row_panel.nblocks(), row_diag, GLOBAL_BLOCK_NOT_FOUND,
        row_ranges);
    int column_count = symldl_v2_cpu_window_ranges_excluding(
        0, column_panel.nblocks(), column_diag, parent_block,
        column_ranges);
    if (parent_block != GLOBAL_BLOCK_NOT_FOUND &&
        parent_block != column_diag)
        for (int row = 0; row < row_count; ++row)
            symldl_v2_cpu_hybrid_limited_update(
                lu, k, parent, slot, row_panel, column_panel,
                row_ranges[row][0], row_ranges[row][1],
                parent_block, parent_block + 1, true);
    for (int column = 0; column < column_count; ++column)
        for (int row = 0; row < row_count; ++row)
            symldl_v2_cpu_hybrid_limited_update(
                lu, k, parent, slot, row_panel, column_panel,
                row_ranges[row][0], row_ranges[row][1],
                column_ranges[column][0], column_ranges[column][1], false);
}
