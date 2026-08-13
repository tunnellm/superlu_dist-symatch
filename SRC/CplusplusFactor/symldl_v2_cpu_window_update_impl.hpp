#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_update_impl.hpp"

template <typename Ftype>
static int_t symldl_v2_cpu_window_find(
    xlpanel_t<Ftype> &panel, int_t gid)
{
    if (panel.isEmpty())
        return GLOBAL_BLOCK_NOT_FOUND;
    for (int_t block = 0; block < panel.nblocks(); ++block)
        if (panel.gid(block) == gid)
            return block;
    return GLOBAL_BLOCK_NOT_FOUND;
}

static inline int symldl_v2_cpu_window_ranges_excluding(
    int_t begin, int_t end, int_t skip0, int_t skip1,
    int_t ranges[3][2])
{
    if (begin >= end)
        return 0;
    if (skip0 == skip1)
        skip1 = GLOBAL_BLOCK_NOT_FOUND;
    if (skip1 != GLOBAL_BLOCK_NOT_FOUND &&
        (skip0 == GLOBAL_BLOCK_NOT_FOUND || skip1 < skip0))
        std::swap(skip0, skip1);
    int count = 0;
    int_t current = begin;
    int_t skips[2] = {skip0, skip1};
    for (int item = 0; item < 2; ++item)
    {
        int_t skip = skips[item];
        if (skip == GLOBAL_BLOCK_NOT_FOUND || skip < begin || skip >= end)
            continue;
        if (current < skip)
        {
            ranges[count][0] = current;
            ranges[count][1] = skip;
            ++count;
        }
        current = skip + 1;
    }
    if (current < end)
    {
        ranges[count][0] = current;
        ranges[count][1] = end;
        ++count;
    }
    return count;
}

template <typename Ftype>
static void symldl_v2_cpu_window_rectangle(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> row_panel,
    xlpanel_t<Ftype> column_panel, int_t row_begin, int_t row_end,
    int_t column_begin, int_t column_end, bool lookahead)
{
    if (row_begin >= row_end || column_begin >= column_end)
        return;
    double update_start = lu->symV2CpuProfileEnabled
                              ? SuperLU_timer_()
                              : 0.0;
    int_t m = row_panel.stRow(row_end) - row_panel.stRow(row_begin);
    int_t n = column_panel.stRow(column_end) -
              column_panel.stRow(column_begin);
    int_t k = row_panel.ncols();
    if (m <= 0 || n <= 0 || k <= 0 || column_panel.ncols() != k ||
        static_cast<uint64_t>(m) * n >
            static_cast<uint64_t>(lu->ldt) * lu->ldt)
        ABORT("SymFact V2 CPU window GEMM exceeds workspace.");
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Ftype *update = lu->bigV +
        static_cast<size_t>(thread) * lu->ldt * lu->ldt;
    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    double gemm_start = lu->symV2CpuProfileEnabled ?
                        SuperLU_timer_() : 0.0;
    symldl_v2_cpu_gemm<Ftype>(
        "N", "T", m, n, k, alpha,
        row_panel.val + row_panel.stRow(row_begin), row_panel.LDA(),
        column_panel.val + column_panel.stRow(column_begin),
        column_panel.LDA(), beta, update, m);
    double gemm_time = lu->symV2CpuProfileEnabled ?
                       SuperLU_timer_() - gemm_start : 0.0;
    symldl_v2_cpu_note_gemm_shape(lu, m, n, k, gemm_time);
    if (lu->symV2CpuProfileEnabled)
    {
#ifdef _OPENMP
#pragma omp atomic update
#endif
        lu->symV2CpuGemmTime += gemm_time;
        if (lookahead)
        {
#ifdef _OPENMP
#pragma omp atomic update
#endif
            lu->symV2CpuLookaheadGemmTime += gemm_time;
        }
        else
        {
#ifdef _OPENMP
#pragma omp atomic update
#endif
            lu->symV2CpuExcludeGemmTime += gemm_time;
        }
#ifdef _OPENMP
#pragma omp atomic update
#endif
        ++lu->symV2CpuGroupedGemms;
#ifdef _OPENMP
#pragma omp atomic update
#endif
        lu->symV2CpuGemmFlops += symldl_v2_cpu_gemm_flops(m, n, k);
    }

    for (int_t column = column_begin; column < column_end; ++column)
        for (int_t row = row_begin; row < row_end; ++row)
            symldl_v2_cpu_scatter_dual_block(
                lu, row_panel, row, column_panel, column,
                update + row_panel.stRow(row) -
                             row_panel.stRow(row_begin) +
                         (column_panel.stRow(column) -
                          column_panel.stRow(column_begin)) * m,
                m);
    if (lu->symV2CpuProfileEnabled)
    {
        double elapsed = SuperLU_timer_() - update_start;
#ifdef _OPENMP
#pragma omp atomic update
#endif
        lu->symV2CpuSchurTime += elapsed;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_window_launch_rectangle(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> row_panel,
    xlpanel_t<Ftype> column_panel, int_t row_begin, int_t row_end,
    int_t column_begin, int_t column_end, bool lookahead)
{
    int_t m = row_panel.stRow(row_end) - row_panel.stRow(row_begin);
    int_t n = column_panel.stRow(column_end) -
              column_panel.stRow(column_begin);
    uint64_t work = static_cast<uint64_t>(SUPERLU_MAX((int_t) 0, m)) *
                    static_cast<uint64_t>(SUPERLU_MAX((int_t) 0, n)) *
                    static_cast<uint64_t>(SUPERLU_MAX((int_t) 0,
                                                      row_panel.ncols()));
    bool deferred = lu->symV2CpuWorkerCount > 1 &&
                    work >= symldl_v2_cpu_min_deferred_work();
    ++lu->symV2CpuWindowRectangles;
    lu->symV2CpuWindowRowBlocks +=
        static_cast<uint64_t>(row_end - row_begin);
    lu->symV2CpuWindowColumnBlocks +=
        static_cast<uint64_t>(column_end - column_begin);
    if (lookahead)
        ++lu->symV2CpuLookaheadTasks;
    else
        ++lu->symV2CpuExcludeTasks;
    ++lu->symV2CpuSchurTasks;
    if (deferred)
    {
        ++lu->symV2CpuWindowDeferredRectangles;
#ifdef _OPENMP
#pragma omp task firstprivate(lu, row_panel, column_panel, row_begin, row_end, column_begin, column_end, lookahead)
#endif
        symldl_v2_cpu_window_rectangle(
            lu, row_panel, column_panel, row_begin, row_end,
            column_begin, column_end, lookahead);
    }
    else
    {
        ++lu->symV2CpuWindowInlineRectangles;
        symldl_v2_cpu_window_rectangle(
            lu, row_panel, column_panel, row_begin, row_end,
            column_begin, column_end, lookahead);
    }
}

template <typename Ftype>
static void symldl_v2_cpu_window_limited_update(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> row_panel,
    xlpanel_t<Ftype> column_panel, int_t row_start, int_t row_end,
    int_t column_start, int_t column_end, bool lookahead)
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
    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    if (envelope_requested)
    {
        for (int_t row = row_start + 1; row < row_end; ++row)
            if (row_panel.gid(row) < row_panel.gid(row - 1))
                row_gid_sorted = false;
        for (int_t column = column_start + 1; column < column_end; ++column)
            if (column_panel.gid(column) < column_panel.gid(column - 1))
                column_gid_sorted = false;
    }
    const bool lower_envelope =
        envelope_requested && row_gid_sorted && column_gid_sorted;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(lu->ldt) * lu->ldt);
    int_t max_block_rows = 1;
    for (int_t row = row_start; row < row_end; ++row)
        max_block_rows = SUPERLU_MAX(max_block_rows,
                                     row_panel.nbrow(row));

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
        if (group_rows <= 0)
            break;
        int column_limit = superlu_sym_v2_batch_schur_col_limit(
            group_rows, gemm_capacity);
        int64_t hard_column_limit = gemm_capacity / max_block_rows;
        hard_column_limit = SUPERLU_MAX(static_cast<int64_t>(1),
                                        hard_column_limit);
        column_limit = SUPERLU_MIN(
            column_limit,
            static_cast<int>(SUPERLU_MIN(
                hard_column_limit,
                static_cast<int64_t>(std::numeric_limits<int>::max()))));
        int_t remaining_columns = column_panel.stRow(column_end) -
                                  column_panel.stRow(column_group);
        column_limit = SUPERLU_MAX(
            1, SUPERLU_MIN(column_limit,
                           static_cast<int>(remaining_columns)));

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
            symldl_v2_cpu_window_launch_rectangle(
                lu, row_panel, column_panel, row_group, row_next,
                column_group, column_next, lookahead);
            row_group = row_next;
        }
        column_group = column_next;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_window_lookahead(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel)
{
    if (row_panel.isEmpty() || column_panel.isEmpty())
        return;
    int_t row_parent = symldl_v2_cpu_window_find(row_panel, parent);
    int_t column_parent = symldl_v2_cpu_window_find(column_panel, parent);
    int_t row_diag = symldl_v2_cpu_window_find(row_panel, k);
    int_t column_diag = symldl_v2_cpu_window_find(column_panel, k);
    int_t row_ranges[3][2];
    int row_range_count = symldl_v2_cpu_window_ranges_excluding(
        0, row_panel.nblocks(), row_diag, GLOBAL_BLOCK_NOT_FOUND,
        row_ranges);
    if (column_parent != GLOBAL_BLOCK_NOT_FOUND &&
        column_parent != column_diag)
        for (int range = 0; range < row_range_count; ++range)
            symldl_v2_cpu_window_limited_update(
                lu, row_panel, column_panel,
                row_ranges[range][0], row_ranges[range][1],
                column_parent, column_parent + 1, true);

    if (row_parent != GLOBAL_BLOCK_NOT_FOUND && row_parent != row_diag)
    {
        int_t column_ranges[3][2];
        int column_range_count = symldl_v2_cpu_window_ranges_excluding(
            0, column_panel.nblocks(), column_parent, column_diag,
            column_ranges);
        for (int range = 0; range < column_range_count; ++range)
            symldl_v2_cpu_window_limited_update(
                lu, row_panel, column_panel, row_parent, row_parent + 1,
                column_ranges[range][0], column_ranges[range][1], true);
    }
}

template <typename Ftype>
static void symldl_v2_cpu_window_exclude(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent,
    xlpanel_t<Ftype> row_panel, xlpanel_t<Ftype> column_panel)
{
    if (row_panel.isEmpty() || column_panel.isEmpty())
        return;
    int_t row_parent = symldl_v2_cpu_window_find(row_panel, parent);
    int_t column_parent = symldl_v2_cpu_window_find(column_panel, parent);
    int_t row_diag = symldl_v2_cpu_window_find(row_panel, k);
    int_t column_diag = symldl_v2_cpu_window_find(column_panel, k);
    int_t row_ranges[3][2];
    int_t column_ranges[3][2];
    int row_range_count = symldl_v2_cpu_window_ranges_excluding(
        0, row_panel.nblocks(), row_parent, row_diag, row_ranges);
    int column_range_count = symldl_v2_cpu_window_ranges_excluding(
        0, column_panel.nblocks(), column_parent, column_diag,
        column_ranges);
    for (int row_range = 0; row_range < row_range_count; ++row_range)
        for (int column_range = 0; column_range < column_range_count;
             ++column_range)
            symldl_v2_cpu_window_limited_update(
                lu, row_panel, column_panel,
                row_ranges[row_range][0], row_ranges[row_range][1],
                column_ranges[column_range][0],
                column_ranges[column_range][1], false);
}
