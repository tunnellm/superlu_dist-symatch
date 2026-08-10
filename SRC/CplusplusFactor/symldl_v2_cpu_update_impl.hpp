#pragma once

#include <algorithm>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_blas.hpp"

template <typename Ftype>
static int_t symldl_v2_cpu_panel_find(
    xlpanel_t<Ftype> &panel, int_t gid)
{
    int_t begin = 0;
    int_t end = panel.nblocks();
    while (begin < end)
    {
        int_t middle = begin + (end - begin) / 2;
        if (panel.gid(middle) < gid)
            begin = middle + 1;
        else
            end = middle;
    }
    if (begin < panel.nblocks() && panel.gid(begin) == gid)
        return begin;
    return GLOBAL_BLOCK_NOT_FOUND;
}

template <typename Ftype>
static int_t symldl_v2_cpu_panel_lower_bound(
    xlpanel_t<Ftype> &panel, int_t gid, int_t begin, int_t end)
{
    if (begin < 0 || end < begin || end > panel.nblocks())
        ABORT("SymFact V2 CPU panel search range is invalid.");
    while (begin < end)
    {
        int_t middle = begin + (end - begin) / 2;
        if (panel.gid(middle) < gid)
            begin = middle + 1;
        else
            end = middle;
    }
    return begin;
}

template <typename Ftype>
static size_t symldl_v2_cpu_output_lock_id(
    xLUstruct_t<Ftype> *lu, int_t local_panel, int_t local_block)
{
    if (local_panel < 0 || local_block < 0 ||
        static_cast<size_t>(local_panel + 1) >=
            lu->symV2CpuOutputLockOffsets.size())
        ABORT("SymFact V2 CPU output lock is missing.");
    size_t begin = lu->symV2CpuOutputLockOffsets[
        static_cast<size_t>(local_panel)];
    size_t end = lu->symV2CpuOutputLockOffsets[
        static_cast<size_t>(local_panel + 1)];
    size_t lock_id = begin + static_cast<size_t>(local_block);
    if (lock_id >= end)
        ABORT("SymFact V2 CPU output lock is invalid.");
    return lock_id;
}

template <typename Ftype>
static void symldl_v2_cpu_lock_output(
    xLUstruct_t<Ftype> *lu, size_t lock_id)
{
#ifdef _OPENMP
    if (lu->symV2CpuOutputLocks == NULL)
        ABORT("SymFact V2 CPU output lock table is missing.");
    omp_lock_t *locks =
        static_cast<omp_lock_t *>(lu->symV2CpuOutputLocks);
    if (!lu->symV2CpuProfileEnabled)
    {
        omp_set_lock(&locks[lock_id]);
        return;
    }
#pragma omp atomic update
    ++lu->symV2CpuOutputLockAttempts;
    double wait_start = SuperLU_timer_();
    if (!omp_test_lock(&locks[lock_id]))
    {
#pragma omp atomic update
        ++lu->symV2CpuOutputLockConflicts;
        omp_set_lock(&locks[lock_id]);
    }
    double wait = SuperLU_timer_() - wait_start;
#pragma omp atomic update
    lu->symV2CpuScatterLockWaitTime += wait;
#else
    (void) lu;
    (void) lock_id;
#endif
}

template <typename Ftype>
static void symldl_v2_cpu_unlock_output(
    xLUstruct_t<Ftype> *lu, size_t lock_id)
{
#ifdef _OPENMP
    omp_lock_t *locks =
        static_cast<omp_lock_t *>(lu->symV2CpuOutputLocks);
    omp_unset_lock(&locks[lock_id]);
#else
    (void) lu;
    (void) lock_id;
#endif
}

template <typename Ftype>
static void symldl_v2_cpu_assert_panel_unfactored(
    xLUstruct_t<Ftype> *lu, int_t local_panel)
{
    if (!symldl_v2_cpu_ownership_check_enabled())
        return;
    if (local_panel < 0 ||
        static_cast<size_t>(local_panel) >=
            lu->symV2CpuPanelFactorStarted.size())
        ABORT("SymFact V2 CPU panel ownership check is invalid.");
    int started = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    started = lu->symV2CpuPanelFactorStarted[
        static_cast<size_t>(local_panel)];
    if (started != 0)
        ABORT("SymFact V2 CPU update targets a factored panel.");
}

template <typename Ftype>
static void symldl_v2_cpu_mark_panel_factor_started(
    xLUstruct_t<Ftype> *lu, int_t local_panel)
{
    if (!symldl_v2_cpu_ownership_check_enabled())
        return;
    if (local_panel < 0 ||
        static_cast<size_t>(local_panel + 1) >=
            lu->symV2CpuOutputLockOffsets.size() ||
        static_cast<size_t>(local_panel) >=
            lu->symV2CpuPanelFactorStarted.size())
        ABORT("SymFact V2 CPU factor ownership check is invalid.");

    int started = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    started = lu->symV2CpuPanelFactorStarted[
        static_cast<size_t>(local_panel)];
    if (started != 0)
        ABORT("SymFact V2 CPU panel factorization started twice.");

#ifdef _OPENMP
    if (symldl_v2_cpu_output_lock_check_enabled())
    {
        if (lu->symV2CpuOutputLocks == NULL &&
            lu->symV2CpuOutputLockOffsets.back() != 0)
            ABORT("SymFact V2 CPU output lock table is missing.");
        omp_lock_t *locks =
            static_cast<omp_lock_t *>(lu->symV2CpuOutputLocks);
        size_t begin = lu->symV2CpuOutputLockOffsets[
            static_cast<size_t>(local_panel)];
        size_t end = lu->symV2CpuOutputLockOffsets[
            static_cast<size_t>(local_panel + 1)];
        for (size_t lock = begin; lock < end; ++lock)
        {
            if (!omp_test_lock(&locks[lock]))
                ABORT(
                    "SymFact V2 CPU panel factorization overlaps an update lock.");
            omp_unset_lock(&locks[lock]);
        }
    }
#endif

#ifdef _OPENMP
#pragma omp atomic write
#endif
    lu->symV2CpuPanelFactorStarted[static_cast<size_t>(local_panel)] = 1;
}

template <typename Ftype>
static uint64_t symldl_v2_cpu_slot_generation(
    xLUstruct_t<Ftype> *lu, int slot, int_t source_k)
{
    if (!symldl_v2_cpu_ownership_check_enabled())
        return 0;
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotGeneration.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotOwner.size())
        ABORT("SymFact V2 CPU slot ownership check is invalid.");
    uint64_t generation = 0;
    int_t owner = -1;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    generation = lu->symV2CpuSlotGeneration[static_cast<size_t>(slot)];
#ifdef _OPENMP
#pragma omp atomic read
#endif
    owner = lu->symV2CpuSlotOwner[static_cast<size_t>(slot)];
    if (generation == 0 || owner != source_k)
        ABORT("SymFact V2 CPU task observes a reused source slot.");
    return generation;
}

template <typename Ftype>
static void symldl_v2_cpu_assert_slot_generation(
    xLUstruct_t<Ftype> *lu, int slot, int_t source_k,
    uint64_t expected_generation)
{
    if (!symldl_v2_cpu_ownership_check_enabled())
        return;
    uint64_t generation = symldl_v2_cpu_slot_generation(
        lu, slot, source_k);
    if (generation != expected_generation)
        ABORT("SymFact V2 CPU task source slot changed generation.");
}

static inline uint64_t symldl_v2_cpu_gemm_flops(
    int_t m, int_t n, int_t k)
{
    uint64_t result = 2;
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    const uint64_t values[] = {
        static_cast<uint64_t>(m), static_cast<uint64_t>(n),
        static_cast<uint64_t>(k)
    };
    for (size_t i = 0; i < 3; ++i)
    {
        if (values[i] != 0 && result > limit / values[i])
            return limit;
        result *= values[i];
    }
    return result;
}

static inline uint64_t symldl_v2_cpu_saturating_add(
    uint64_t left, uint64_t right)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return left > limit - right ? limit : left + right;
}

template <typename Ftype>
static uint64_t symldl_v2_cpu_column_work(
    xlpanel_t<Ftype> &row_panel, int_t first_row_block,
    xlpanel_t<Ftype> &column_panel, int_t source_j)
{
    int_t gj = column_panel.gid(source_j);
    int_t source_i = first_row_block;
    while (source_i < row_panel.nblocks() &&
           row_panel.gid(source_i) < gj)
        ++source_i;
    if (source_i >= row_panel.nblocks())
        return 0;

    int_t last = row_panel.nblocks() - 1;
    int_t rows = row_panel.stRow(last) + row_panel.nbrow(last) -
                 row_panel.stRow(source_i);
    int_t columns = column_panel.nbrow(source_j);
    int_t inner = row_panel.ncols();
    if (rows <= 0 || columns <= 0 || inner <= 0)
        return 0;
    uint64_t gemm = symldl_v2_cpu_gemm_flops(rows, columns, inner);
    uint64_t scatter = symldl_v2_cpu_gemm_flops(rows, columns, 2);
    return symldl_v2_cpu_saturating_add(gemm, scatter);
}

static inline uint64_t symldl_v2_cpu_min_deferred_work()
{
    // Below this grain, OpenMP task bookkeeping costs more than the small
    // BLAS/scatter operation on ordinary CPU cores.  This is an operation
    // count, independent of matrix identity and process-grid topology.
    return UINT64_C(32768);
}

template <typename Ftype>
static void symldl_v2_cpu_note_gemm_shape(
    xLUstruct_t<Ftype> *lu, int_t m, int_t n, int_t k,
    double elapsed)
{
    if (!lu->symV2CpuProfileEnabled)
        return;
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    if (thread < 0 ||
        static_cast<size_t>(thread) >= lu->symV2CpuThreadProfiles.size())
        ABORT("SymFact V2 CPU worker profile does not cover the OpenMP team.");
    SymLDLV2CpuThreadProfile &profile =
        lu->symV2CpuThreadProfiles[static_cast<size_t>(thread)];
    uint64_t flops = symldl_v2_cpu_gemm_flops(m, n, k);
    ++profile.gemms;
    if (flops < symldl_v2_cpu_min_deferred_work())
    {
        ++profile.small_gemms;
        profile.small_gemm_time += elapsed;
    }
    else if (flops < UINT64_C(1048576))
    {
        ++profile.medium_gemms;
        profile.medium_gemm_time += elapsed;
    }
    else
    {
        ++profile.large_gemms;
        profile.large_gemm_time += elapsed;
    }
    profile.m_sum = symldl_v2_cpu_saturating_add(
        profile.m_sum, static_cast<uint64_t>(m));
    profile.n_sum = symldl_v2_cpu_saturating_add(
        profile.n_sum, static_cast<uint64_t>(n));
    profile.k_sum = symldl_v2_cpu_saturating_add(
        profile.k_sum, static_cast<uint64_t>(k));
    profile.max_m = SUPERLU_MAX(profile.max_m, static_cast<uint64_t>(m));
    profile.max_n = SUPERLU_MAX(profile.max_n, static_cast<uint64_t>(n));
    profile.max_k = SUPERLU_MAX(profile.max_k, static_cast<uint64_t>(k));
}

static bool symldl_v2_cpu_build_row_map(
    const int_t *source_row_list, int_t source_rows,
    const int_t *destination_row_list, int_t destination_rows,
    int_t *destination_permutation, int_t *row_map, int_t map_capacity,
    const SymLDLV2CpuRowLookup *destination_lookup,
    const int_t *destination_lookup_pool, bool *sorted_rows_out,
    unsigned char *lookup_kind_out)
{
    if (source_rows <= 0 || destination_rows <= 0 ||
        source_rows > map_capacity || destination_rows > map_capacity)
        ABORT("SymFact V2 CPU row map exceeds workspace bounds.");

    bool sorted_rows = true;
    for (int_t row = 1; row < source_rows; ++row)
        sorted_rows = sorted_rows &&
                      source_row_list[row - 1] < source_row_list[row];
    for (int_t row = 1; row < destination_rows; ++row)
        sorted_rows = sorted_rows &&
                      destination_row_list[row - 1] <
                          destination_row_list[row];

    if (sorted_rows)
    {
        if (lookup_kind_out != NULL)
            *lookup_kind_out = 0;
        int_t destination_row = 0;
        for (int_t source_row = 0; source_row < source_rows; ++source_row)
        {
            while (destination_row < destination_rows &&
                   destination_row_list[destination_row] <
                       source_row_list[source_row])
                ++destination_row;
            if (destination_row >= destination_rows ||
                destination_row_list[destination_row] !=
                    source_row_list[source_row])
                ABORT("SymFact V2 CPU source row is absent from destination.");
            row_map[source_row] = destination_row;
        }
    }
    else if (destination_lookup != NULL &&
             destination_lookup_pool != NULL)
    {
        if (lookup_kind_out != NULL)
            *lookup_kind_out = destination_lookup->dense ? 1 : 2;
        const int_t *entries =
            destination_lookup_pool + destination_lookup->offset;
        if (destination_lookup->dense)
        {
            for (int_t source_row = 0; source_row < source_rows;
                 ++source_row)
            {
                int_t gid = source_row_list[source_row];
                if (gid < 0 ||
                    static_cast<size_t>(gid) >=
                        destination_lookup->extent)
                    ABORT(
                        "SymFact V2 CPU source row is absent from destination.");
                int_t position = entries[gid];
                if (position < 0 || position >= destination_rows ||
                    destination_row_list[position] != gid)
                    ABORT(
                        "SymFact V2 CPU source row is absent from destination.");
                row_map[source_row] = position;
            }
        }
        else
        {
            for (int_t source_row = 0; source_row < source_rows;
                 ++source_row)
            {
                int_t gid = source_row_list[source_row];
                const int_t *position = std::lower_bound(
                    entries, entries + destination_lookup->extent, gid,
                    [destination_row_list](int_t row, int_t value)
                    {
                        return destination_row_list[row] < value;
                    });
                if (position == entries + destination_lookup->extent ||
                    destination_row_list[*position] != gid)
                    ABORT(
                        "SymFact V2 CPU source row is absent from destination.");
                row_map[source_row] = *position;
            }
        }
    }
    else
    {
        if (lookup_kind_out != NULL)
            *lookup_kind_out = 3;
        bool bounded_ids = true;
        for (int_t row = 0; row < source_rows; ++row)
            bounded_ids = bounded_ids && source_row_list[row] >= 0 &&
                          source_row_list[row] < map_capacity;
        for (int_t row = 0; row < destination_rows; ++row)
            bounded_ids = bounded_ids && destination_row_list[row] >= 0 &&
                          destination_row_list[row] < map_capacity;

        if (bounded_ids)
        {
            // Clear every key this map may read before reusing the worker
            // scratch.  This retains the linear indirect-map path without
            // relying on values left by an earlier update.
            for (int_t row = 0; row < source_rows; ++row)
                destination_permutation[source_row_list[row]] = -1;
            for (int_t row = 0; row < destination_rows; ++row)
                destination_permutation[destination_row_list[row]] = -1;
            for (int_t row = 0; row < destination_rows; ++row)
            {
                int_t gid = destination_row_list[row];
                if (destination_permutation[gid] != -1)
                    ABORT("SymFact V2 CPU destination rows are duplicated.");
                destination_permutation[gid] = row;
            }
            for (int_t row = 0; row < source_rows; ++row)
            {
                int_t gid = source_row_list[row];
                int_t position = destination_permutation[gid];
                if (position < 0 || position >= destination_rows ||
                    destination_row_list[position] != gid)
                    ABORT(
                        "SymFact V2 CPU source row is absent from destination.");
                row_map[row] = position;
            }
        }
        else
        {
            for (int_t row = 0; row < destination_rows; ++row)
                destination_permutation[row] = row;
            std::sort(
                destination_permutation,
                destination_permutation + destination_rows,
                [destination_row_list](int_t left, int_t right)
                {
                    int_t left_gid = destination_row_list[left];
                    int_t right_gid = destination_row_list[right];
                    return left_gid < right_gid ||
                           (left_gid == right_gid && left < right);
                });
            for (int_t row = 1; row < destination_rows; ++row)
                if (destination_row_list[
                        destination_permutation[row - 1]] ==
                    destination_row_list[destination_permutation[row]])
                    ABORT("SymFact V2 CPU destination rows are duplicated.");

            for (int_t source_row = 0; source_row < source_rows; ++source_row)
            {
                int_t gid = source_row_list[source_row];
                int_t *position = std::lower_bound(
                    destination_permutation,
                    destination_permutation + destination_rows, gid,
                    [destination_row_list](int_t row, int_t value)
                    {
                        return destination_row_list[row] < value;
                    });
                if (position == destination_permutation + destination_rows ||
                    destination_row_list[*position] != gid)
                    ABORT(
                        "SymFact V2 CPU source row is absent from destination.");
                row_map[source_row] = *position;
            }
        }
    }

    bool row_contiguous = true;
    for (int_t row = 1; row < source_rows; ++row)
        row_contiguous = row_contiguous &&
                         row_map[row] == row_map[0] + row;
    if (sorted_rows_out != NULL)
        *sorted_rows_out = sorted_rows;
    return row_contiguous;
}

template <typename Ftype>
static const SymLDLV2CpuRowLookup *symldl_v2_cpu_destination_row_lookup(
    xLUstruct_t<Ftype> *lu, int_t local_panel, int_t local_block)
{
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        ABORT("SymFact V2 CPU destination row lookup panel is invalid.");
    size_t panel_index = static_cast<size_t>(local_panel);
    if (panel_index + 1 >=
            lu->symV2CpuRowLookupPanelOffsets.size())
        ABORT("SymFact V2 CPU destination row lookup panel is invalid.");
    size_t begin = lu->symV2CpuRowLookupPanelOffsets[
        panel_index];
    size_t end = lu->symV2CpuRowLookupPanelOffsets[
        panel_index + 1];
    if (local_block < 0 ||
        static_cast<size_t>(local_block) >= end - begin ||
        begin + static_cast<size_t>(local_block) >=
            lu->symV2CpuRowLookups.size())
        ABORT("SymFact V2 CPU destination row lookup block is invalid.");
    const SymLDLV2CpuRowLookup *lookup = &lu->symV2CpuRowLookups[
        begin + static_cast<size_t>(local_block)];
    if (lookup->offset > lu->symV2CpuRowLookupPool.size() ||
        lookup->extent >
            lu->symV2CpuRowLookupPool.size() - lookup->offset)
        ABORT("SymFact V2 CPU destination row lookup range is invalid.");
    return lookup;
}

template <typename Ftype>
static void symldl_v2_cpu_scatter_dual_block(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &row_panel,
    int_t source_i, xlpanel_t<Ftype> &column_panel, int_t source_j,
    Ftype *update, int_t ld_update)
{
    int_t gi = row_panel.gid(source_i);
    int_t gj = column_panel.gid(source_j);
    if (gi < gj || lu->symV2DiagRoot(gi) != lu->myrow ||
        lu->symV2PanelRoot(gj) != lu->mycol)
        return;

    int_t local_panel = lu->symV2PanelIndex(gj);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        return;
    xlpanel_t<Ftype> &destination_panel = lu->lPanelVec[local_panel];
    if (destination_panel.isEmpty())
        return;
    int_t local_block = symldl_v2_cpu_panel_find(destination_panel, gi);
    if (local_block == GLOBAL_BLOCK_NOT_FOUND)
        return;
    symldl_v2_cpu_assert_panel_unfactored(lu, local_panel);

    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    int_t *destination_index = lu->indirect + thread_id * lu->ldt;
    int_t *row_map = lu->indirectRow + thread_id * lu->ldt;

    int_t source_rows = row_panel.nbrow(source_i);
    int_t source_columns = column_panel.nbrow(source_j);
    int_t destination_rows = destination_panel.nbrow(local_block);
    int_t destination_columns = lu->supersize(gj);
    int_t *source_row_list = row_panel.rowList(source_i);
    int_t *source_column_list = column_panel.rowList(source_j);
    int_t *destination_row_list = destination_panel.rowList(local_block);

    bool sorted_rows = false;
    unsigned char lookup_kind = 0;
    double row_map_start =
        lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    bool row_contiguous = symldl_v2_cpu_build_row_map(
        source_row_list, source_rows, destination_row_list,
        destination_rows, destination_index, row_map, lu->ldt,
        symldl_v2_cpu_destination_row_lookup(
            lu, local_panel, local_block),
        lu->symV2CpuRowLookupPool.data(), &sorted_rows, &lookup_kind);
    double row_map_time = 0.0;
    if (lu->symV2CpuProfileEnabled)
    {
        row_map_time = SuperLU_timer_() - row_map_start;
#ifdef _OPENMP
#pragma omp atomic update
#endif
        lu->symV2CpuRowMapTime += row_map_time;
    }

    if (lu->symV2CpuProfileEnabled)
    {
        bool row_exact = source_rows == destination_rows;
        if (row_exact)
            for (int_t i = 0; i < source_rows; ++i)
                row_exact = row_exact &&
                            source_row_list[i] == destination_row_list[i];
        bool column_full = source_columns == destination_columns;
        bool column_contiguous = source_columns > 0;
        for (int_t j = 0; j < source_columns; ++j)
        {
            column_full = column_full && source_column_list[j] == j;
            column_contiguous = column_contiguous &&
                                source_column_list[j] ==
                                    source_column_list[0] + j;
        }
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        if (thread < 0 || static_cast<size_t>(thread) >=
                              lu->symV2CpuThreadProfiles.size())
            ABORT(
                "SymFact V2 CPU scatter profile does not cover the OpenMP team.");
        SymLDLV2CpuThreadProfile &profile =
            lu->symV2CpuThreadProfiles[static_cast<size_t>(thread)];
        profile.mapped_scatter_values +=
            static_cast<uint64_t>(source_rows) *
            static_cast<uint64_t>(source_columns);
        profile.mapped_row_exact += row_exact ? 1 : 0;
        profile.mapped_row_contiguous += row_contiguous ? 1 : 0;
        profile.mapped_column_full += column_full ? 1 : 0;
        profile.mapped_column_contiguous += column_contiguous ? 1 : 0;
        profile.mapped_rectangular +=
            row_contiguous && column_contiguous ? 1 : 0;
        profile.mapped_sorted_rows += sorted_rows ? 1 : 0;
        bool destination_full = destination_rows == lu->supersize(gi);
        if (destination_full)
            for (int_t i = 0; i < destination_rows; ++i)
                destination_full = destination_full &&
                                   destination_row_list[i] == i;
        profile.mapped_destination_full += destination_full ? 1 : 0;
        uint64_t values = static_cast<uint64_t>(source_rows) *
                          static_cast<uint64_t>(source_columns);
        profile.mapped_row_contiguous_values +=
            row_contiguous ? values : 0;
        profile.mapped_rectangular_values +=
            row_contiguous && column_contiguous ? values : 0;
        profile.row_map_sorted += lookup_kind == 0 ? 1 : 0;
        profile.row_map_dense_lookup += lookup_kind == 1 ? 1 : 0;
        profile.row_map_sparse_lookup += lookup_kind == 2 ? 1 : 0;
        profile.row_map_sorted_time += lookup_kind == 0 ? row_map_time : 0.0;
        profile.row_map_dense_lookup_time +=
            lookup_kind == 1 ? row_map_time : 0.0;
        profile.row_map_sparse_lookup_time +=
            lookup_kind == 2 ? row_map_time : 0.0;
        if (column_full)
        {
            uint64_t source_values =
                static_cast<uint64_t>(source_rows) *
                static_cast<uint64_t>(source_columns);
            uint64_t destination_values =
                static_cast<uint64_t>(destination_rows) *
                static_cast<uint64_t>(destination_columns);
            ++profile.padded_candidate_scatters;
            profile.padded_candidate_source_values += source_values;
            profile.padded_candidate_destination_values += destination_values;
            uint64_t source_row_count = static_cast<uint64_t>(source_rows);
            uint64_t destination_row_count =
                static_cast<uint64_t>(destination_rows);
            if (4 * destination_row_count <= 5 * source_row_count)
            {
                profile.padded_le_125_source_values += source_values;
                profile.padded_le_125_destination_values +=
                    destination_values;
            }
            if (2 * destination_row_count <= 3 * source_row_count)
            {
                profile.padded_le_150_source_values += source_values;
                profile.padded_le_150_destination_values +=
                    destination_values;
            }
            if (destination_row_count <= 2 * source_row_count)
            {
                profile.padded_le_200_source_values += source_values;
                profile.padded_le_200_destination_values +=
                    destination_values;
            }
            if (destination_row_count <= 4 * source_row_count)
            {
                profile.padded_le_400_source_values += source_values;
                profile.padded_le_400_destination_values +=
                    destination_values;
            }
        }
    }

    Ftype *destination = destination_panel.blkPtr(local_block);
    int_t ld_destination = destination_panel.LDA();
    size_t lock_id =
        symldl_v2_cpu_output_lock_id(lu, local_panel, local_block);
    double scatter_start = lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    symldl_v2_cpu_lock_output(lu, lock_id);
    if (row_contiguous)
    {
        for (int_t j = 0; j < source_columns; ++j)
        {
            int_t dj = source_column_list[j];
            if (dj < 0 || dj >= destination_columns)
                ABORT("SymFact V2 CPU scatter column is invalid.");
            Ftype *destination_column =
                destination + row_map[0] + ld_destination * dj;
            Ftype *update_column = update + ld_update * j;
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int_t i = 0; i < source_rows; ++i)
                destination_column[i] -= update_column[i];
        }
    }
    else
    {
        for (int_t j = 0; j < source_columns; ++j)
        {
            int_t dj = source_column_list[j];
            if (dj < 0 || dj >= destination_columns)
                ABORT("SymFact V2 CPU scatter column is invalid.");
            for (int_t i = 0; i < source_rows; ++i)
            {
                int_t di = row_map[i];
                if (di < 0 || di >= destination_rows)
                    ABORT("SymFact V2 CPU scatter row is invalid.");
                destination[di + ld_destination * dj] -=
                    update[i + ld_update * j];
            }
        }
    }
    symldl_v2_cpu_unlock_output(lu, lock_id);
    if (!lu->symV2CpuProfileEnabled)
        return;
    double scatter_time = SuperLU_timer_() - scatter_start;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    lu->symV2CpuMappedScatterTime += scatter_time;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    ++lu->symV2CpuMappedScatters;
    SymLDLV2CpuThreadProfile &profile =
        lu->symV2CpuThreadProfiles[static_cast<size_t>(thread_id)];
    if (row_contiguous)
        profile.contiguous_scatter_time += scatter_time;
    else
        profile.irregular_scatter_time += scatter_time;
}

template <typename Ftype>
static void symldl_v2_cpu_scatter_block(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &source_panel,
    int_t source_i, int_t source_j, Ftype *update, int_t ld_update)
{
    symldl_v2_cpu_scatter_dual_block(
        lu, source_panel, source_i, source_panel, source_j,
        update, ld_update);
}

template <typename Ftype>
static void symldl_v2_cpu_update_block(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &panel,
    const Ftype *raw_values, int_t source_i, int_t source_j)
{
    int_t gi = panel.gid(source_i);
    int_t gj = panel.gid(source_j);
    if (gi < gj || lu->symV2DiagRoot(gi) != lu->myrow ||
        lu->symV2PanelRoot(gj) != lu->mycol)
        return;

    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    Ftype *update = lu->bigV +
                    static_cast<size_t>(thread_id) * lu->ldt * lu->ldt;
    int_t m = panel.nbrow(source_i);
    int_t n = panel.nbrow(source_j);
    int_t k = panel.ncols();
    if (m <= 0 || n <= 0 || k <= 0 || m > lu->ldt || n > lu->ldt)
        ABORT("SymFact V2 CPU update block exceeds workspace bounds.");

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    symldl_v2_cpu_gemm<Ftype>(
        "N", "T", m, n, k, alpha, panel.blkPtr(source_i), panel.LDA(),
        const_cast<Ftype *>(raw_values) + panel.blkPtrOffset(source_j),
        panel.LDA(), beta, update, m);
    symldl_v2_cpu_scatter_block(lu, panel, source_i, source_j,
                                update, m);
}

template <typename Ftype>
static void symldl_v2_cpu_dual_update_block(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &row_panel,
    int_t source_i, xlpanel_t<Ftype> &column_panel, int_t source_j)
{
    int_t gi = row_panel.gid(source_i);
    int_t gj = column_panel.gid(source_j);
    if (gi < gj || lu->symV2DiagRoot(gi) != lu->myrow ||
        lu->symV2PanelRoot(gj) != lu->mycol)
        return;

    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    Ftype *update = lu->bigV +
                    static_cast<size_t>(thread_id) * lu->ldt * lu->ldt;
    int_t m = row_panel.nbrow(source_i);
    int_t n = column_panel.nbrow(source_j);
    int_t k = row_panel.ncols();
    if (column_panel.ncols() != k || m <= 0 || n <= 0 || k <= 0 ||
        m > lu->ldt || n > lu->ldt)
        ABORT("SymFact V2 CPU fragment update exceeds workspace bounds.");

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    symldl_v2_cpu_gemm<Ftype>(
        "N", "T", m, n, k, alpha,
        row_panel.blkPtr(source_i), row_panel.LDA(),
        column_panel.blkPtr(source_j), column_panel.LDA(),
        beta, update, m);
    symldl_v2_cpu_scatter_dual_block(
        lu, row_panel, source_i, column_panel, source_j, update, m);
}

template <typename Ftype>
static bool symldl_v2_cpu_find_group_destination(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &row_panel,
    int_t first_row_block, int_t last_row_block,
    xlpanel_t<Ftype> &column_panel,
    int_t source_j, bool allow_row_padding, int_t *local_panel_out,
    int_t *first_block_out, int_t *last_block_out,
    int_t *destination_rows_out)
{
    int_t gj = column_panel.gid(source_j);
    if (lu->symV2PanelRoot(gj) != lu->mycol)
        return false;
    int_t n = column_panel.nbrow(source_j);
    if (n != lu->supersize(gj))
        return false;
    int_t *column_rows = column_panel.rowList(source_j);
    for (int_t column = 0; column < n; ++column)
        if (column_rows[column] != column)
            return false;

    int_t local_panel = lu->symV2PanelIndex(gj);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        return false;
    xlpanel_t<Ftype> &destination = lu->lPanelVec[local_panel];
    if (destination.isEmpty())
        return false;

    int_t first_destination_block = -1;
    int_t previous_destination_block = -1;
    int_t source_row_begin = row_panel.stRow(first_row_block);
    int_t destination_row_begin = -1;
    for (int_t source_block = first_row_block;
         source_block <= last_row_block; ++source_block)
    {
        int_t gi = row_panel.gid(source_block);
        if (gi < gj || lu->symV2DiagRoot(gi) != lu->myrow)
            return false;
        int_t destination_block =
            symldl_v2_cpu_panel_find(destination, gi);
        if (destination_block == GLOBAL_BLOCK_NOT_FOUND ||
            (previous_destination_block >= 0 &&
             destination_block != previous_destination_block + 1))
            return false;
        if (first_destination_block < 0)
        {
            first_destination_block = destination_block;
            destination_row_begin = destination.stRow(destination_block);
        }
        int_t rows = row_panel.nbrow(source_block);
        if (!allow_row_padding)
        {
            if (destination.nbrow(destination_block) != rows ||
                destination.stRow(destination_block) -
                        destination_row_begin !=
                    row_panel.stRow(source_block) - source_row_begin)
                return false;
            int_t *source_rows = row_panel.rowList(source_block);
            int_t *destination_rows = destination.rowList(destination_block);
            for (int_t row = 0; row < rows; ++row)
                if (source_rows[row] != destination_rows[row])
                    return false;
        }
        previous_destination_block = destination_block;
    }
    if (first_destination_block < 0)
        return false;
    *local_panel_out = local_panel;
    *first_block_out = first_destination_block;
    *last_block_out = previous_destination_block;
    *destination_rows_out =
        destination.stRow(previous_destination_block) +
        destination.nbrow(previous_destination_block) -
        destination_row_begin;
    return true;
}

template <typename Ftype>
static void symldl_v2_cpu_pack_padded_group(
    xLUstruct_t<Ftype> *lu, int_t destination_local_panel,
    xlpanel_t<Ftype> &row_panel, int_t first_row_block,
    int_t last_row_block, xlpanel_t<Ftype> &destination,
    int_t first_destination_block, int_t destination_rows,
    int_t *destination_index, int_t *row_map, int_t map_capacity,
    Ftype *padded)
{
    int_t source_row_begin = row_panel.stRow(first_row_block);
    int_t destination_row_begin =
        destination.stRow(first_destination_block);
    int_t k = row_panel.ncols();
    std::fill(padded,
              padded + static_cast<size_t>(destination_rows) * k,
              zeroT<Ftype>());

    for (int_t source_block = first_row_block;
         source_block <= last_row_block; ++source_block)
    {
        int_t destination_block =
            first_destination_block + source_block - first_row_block;
        int_t source_rows = row_panel.nbrow(source_block);
        int_t destination_block_rows =
            destination.nbrow(destination_block);
        int_t *source_row_list = row_panel.rowList(source_block);
        int_t *destination_row_list =
            destination.rowList(destination_block);
        int_t source_offset =
            row_panel.stRow(source_block) - source_row_begin;
        int_t destination_offset =
            destination.stRow(destination_block) - destination_row_begin;

        symldl_v2_cpu_build_row_map(
            source_row_list, source_rows, destination_row_list,
            destination_block_rows, destination_index, row_map,
            map_capacity, symldl_v2_cpu_destination_row_lookup(
                lu, destination_local_panel, destination_block),
            lu->symV2CpuRowLookupPool.data(), NULL, NULL);

        for (int_t column = 0; column < k; ++column)
        {
            Ftype *destination_column =
                padded + destination_offset +
                static_cast<size_t>(column) * destination_rows;
            Ftype *source_column =
                row_panel.val + source_row_begin + source_offset +
                static_cast<size_t>(column) * row_panel.LDA();
            for (int_t source_row = 0; source_row < source_rows;
                 ++source_row)
                destination_column[row_map[source_row]] =
                    source_column[source_row];
        }
    }
}

template <typename Ftype>
static long long symldl_v2_cpu_grouped_update_column(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &row_panel,
    int_t first_row_block, xlpanel_t<Ftype> &column_panel,
    int_t source_j, const Ftype *column_values, bool lookahead)
{
    int_t gj = column_panel.gid(source_j);
    int_t source_i = symldl_v2_cpu_panel_lower_bound(
        row_panel, gj, first_row_block, row_panel.nblocks());
    if (source_i >= row_panel.nblocks())
        return 0;

    int_t n = column_panel.nbrow(source_j);
    int_t k = row_panel.ncols();
    if (column_panel.ncols() != k || n <= 0 || k <= 0 || n > lu->ldt)
        ABORT("SymFact V2 CPU grouped update exceeds workspace bounds.");
    for (int_t block = source_i + 1; block < row_panel.nblocks(); ++block)
        if (row_panel.gid(block - 1) >= row_panel.gid(block))
            ABORT("SymFact V2 CPU row fragment is not ordered.");

    long long completed = 0;
    while (source_i < row_panel.nblocks())
    {
        int_t row_begin = row_panel.stRow(source_i);
        int_t group_last = source_i;
        while (group_last + 1 < row_panel.nblocks())
        {
            int_t candidate = group_last + 1;
            int_t candidate_end = row_panel.stRow(candidate) +
                                  row_panel.nbrow(candidate);
            if (candidate_end - row_begin > lu->ldt)
                break;
            group_last = candidate;
        }
        int_t row_end = row_panel.stRow(group_last) +
                        row_panel.nbrow(group_last);
        int_t m = row_end - row_begin;
        if (m <= 0 || m > lu->ldt)
            ABORT("SymFact V2 CPU grouped row range exceeds workspace bounds.");

        int_t direct_local_panel = -1;
        int_t direct_first_block = -1;
        int_t direct_last_block = -1;
        int_t direct_destination_rows = 0;
        if (symldl_v2_cpu_find_group_destination(
                lu, row_panel, source_i, group_last, column_panel, source_j,
                false, &direct_local_panel, &direct_first_block,
                &direct_last_block, &direct_destination_rows))
        {
            symldl_v2_cpu_assert_panel_unfactored(
                lu, direct_local_panel);
            xlpanel_t<Ftype> &destination =
                lu->lPanelVec[direct_local_panel];
            for (int_t block = direct_first_block;
                 block <= direct_last_block; ++block)
                symldl_v2_cpu_lock_output(
                    lu, symldl_v2_cpu_output_lock_id(
                            lu, direct_local_panel, block));
            Ftype alpha = -one<Ftype>();
            Ftype beta = one<Ftype>();
            double direct_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            symldl_v2_cpu_gemm<Ftype>(
                "N", "T", m, n, k, alpha,
                row_panel.val + row_begin, row_panel.LDA(),
                const_cast<Ftype *>(column_values) +
                    column_panel.blkPtrOffset(source_j),
                column_panel.LDA(), beta,
                destination.blkPtr(direct_first_block), destination.LDA());
            double direct_time = lu->symV2CpuProfileEnabled ?
                SuperLU_timer_() - direct_start : 0.0;
            symldl_v2_cpu_note_gemm_shape(
                lu, m, n, k, direct_time);
            for (int_t block = direct_last_block;
                 block >= direct_first_block; --block)
                symldl_v2_cpu_unlock_output(
                    lu, symldl_v2_cpu_output_lock_id(
                            lu, direct_local_panel, block));
            if (lu->symV2CpuProfileEnabled)
            {
#ifdef _OPENMP
#pragma omp atomic update
#endif
                lu->symV2CpuGemmTime += direct_time;
                if (lookahead)
                {
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    lu->symV2CpuLookaheadGemmTime += direct_time;
                }
                else
                {
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    lu->symV2CpuExcludeGemmTime += direct_time;
                }
#ifdef _OPENMP
#pragma omp atomic update
#endif
                lu->symV2CpuDirectScatterTime += direct_time;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                ++lu->symV2CpuGroupedGemms;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                ++lu->symV2CpuDirectScatters;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                lu->symV2CpuGemmFlops +=
                    symldl_v2_cpu_gemm_flops(m, n, k);
            }
        }
        else if (symldl_v2_cpu_padded_direct_enabled() &&
                 symldl_v2_cpu_find_group_destination(
                     lu, row_panel, source_i, group_last, column_panel,
                     source_j, true, &direct_local_panel,
                     &direct_first_block, &direct_last_block,
                     &direct_destination_rows) &&
                 direct_destination_rows <= lu->ldt &&
                 static_cast<uint64_t>(direct_destination_rows) * 4 <=
                     static_cast<uint64_t>(m) * 5)
        {
            symldl_v2_cpu_assert_panel_unfactored(
                lu, direct_local_panel);
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            Ftype *padded = lu->bigV +
                static_cast<size_t>(thread_id) * lu->ldt * lu->ldt;
            int_t *destination_index =
                lu->indirect + static_cast<size_t>(thread_id) * lu->ldt;
            int_t *row_map =
                lu->indirectRow + static_cast<size_t>(thread_id) * lu->ldt;
            xlpanel_t<Ftype> &destination =
                lu->lPanelVec[direct_local_panel];
            double pack_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            symldl_v2_cpu_pack_padded_group(
                lu, direct_local_panel, row_panel, source_i, group_last,
                destination, direct_first_block, direct_destination_rows,
                destination_index, row_map, lu->ldt, padded);
            double pack_time = lu->symV2CpuProfileEnabled ?
                SuperLU_timer_() - pack_start : 0.0;
            for (int_t block = direct_first_block;
                 block <= direct_last_block; ++block)
                symldl_v2_cpu_lock_output(
                    lu, symldl_v2_cpu_output_lock_id(
                            lu, direct_local_panel, block));
            Ftype alpha = -one<Ftype>();
            Ftype beta = one<Ftype>();
            double gemm_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            symldl_v2_cpu_gemm<Ftype>(
                "N", "T", direct_destination_rows, n, k, alpha,
                padded, direct_destination_rows,
                const_cast<Ftype *>(column_values) +
                    column_panel.blkPtrOffset(source_j),
                column_panel.LDA(), beta,
                destination.blkPtr(direct_first_block), destination.LDA());
            double gemm_time = lu->symV2CpuProfileEnabled ?
                SuperLU_timer_() - gemm_start : 0.0;
            for (int_t block = direct_last_block;
                 block >= direct_first_block; --block)
                symldl_v2_cpu_unlock_output(
                    lu, symldl_v2_cpu_output_lock_id(
                            lu, direct_local_panel, block));
            if (lu->symV2CpuProfileEnabled)
            {
                symldl_v2_cpu_note_gemm_shape(
                    lu, direct_destination_rows, n, k, gemm_time);
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
                lu->symV2CpuDirectScatterTime += gemm_time;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                lu->symV2CpuPaddedPackTime += pack_time;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                ++lu->symV2CpuGroupedGemms;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                ++lu->symV2CpuDirectScatters;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                lu->symV2CpuGemmFlops += symldl_v2_cpu_gemm_flops(
                    direct_destination_rows, n, k);
                if (thread_id < 0 ||
                    static_cast<size_t>(thread_id) >=
                        lu->symV2CpuThreadProfiles.size())
                    ABORT(
                        "SymFact V2 CPU padded profile does not cover the OpenMP team.");
                SymLDLV2CpuThreadProfile &profile =
                    lu->symV2CpuThreadProfiles[
                        static_cast<size_t>(thread_id)];
                ++profile.padded_direct_groups;
                profile.padded_direct_source_values +=
                    static_cast<uint64_t>(m) * n;
                profile.padded_direct_destination_values +=
                    static_cast<uint64_t>(direct_destination_rows) * n;
            }
        }
        else
        {
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            Ftype *update = lu->bigV +
                static_cast<size_t>(thread_id) * lu->ldt * lu->ldt;
            Ftype alpha = one<Ftype>();
            Ftype beta = zeroT<Ftype>();
            double gemm_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            symldl_v2_cpu_gemm<Ftype>(
                "N", "T", m, n, k, alpha,
                row_panel.val + row_begin, row_panel.LDA(),
                const_cast<Ftype *>(column_values) +
                    column_panel.blkPtrOffset(source_j),
                column_panel.LDA(), beta, update, m);
            double gemm_time = lu->symV2CpuProfileEnabled ?
                SuperLU_timer_() - gemm_start : 0.0;
            symldl_v2_cpu_note_gemm_shape(
                lu, m, n, k, gemm_time);
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
                lu->symV2CpuGemmFlops +=
                    symldl_v2_cpu_gemm_flops(m, n, k);
            }

            for (int_t block = source_i; block <= group_last; ++block)
                symldl_v2_cpu_scatter_dual_block(
                    lu, row_panel, block, column_panel, source_j,
                    update + row_panel.stRow(block) - row_begin, m);
        }
        completed += static_cast<long long>(group_last - source_i + 1);
        source_i = group_last + 1;
    }
    return completed;
}

template <typename Ftype>
static void symldl_v2_cpu_note_task_pending(
    xLUstruct_t<Ftype> *lu, int_t output_gid, int slot, int delta)
{
    if (lu->symV2PanelRoot(output_gid) != lu->mycol)
        ABORT("SymFact V2 CPU task has a nonlocal output panel.");
    int_t local_panel = lu->symV2PanelIndex(output_gid);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
        lu->symV2PanelGid(local_panel) != output_gid)
        ABORT("SymFact V2 CPU task output panel is invalid.");
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuRawPanelBufs.size())
        ABORT("SymFact V2 CPU task slot is invalid.");
    symldl_v2_cpu_assert_panel_unfactored(lu, local_panel);
#ifdef _OPENMP
#pragma omp atomic update
#endif
    lu->symV2CpuPanelPending[local_panel] += delta;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    lu->symV2CpuSlotPending[slot] += delta;
}

template <typename Ftype>
static void symldl_v2_cpu_submit_grouped_task_range(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    int_t *row_index, Ftype *row_values, int_t first_row_block,
    int_t *column_index, Ftype *column_values, int_t source_j_begin,
    int_t source_j_end, bool lookahead,
    uint64_t known_work = std::numeric_limits<uint64_t>::max())
{
    if (source_j_begin >= source_j_end)
        return;
    xlpanel_t<Ftype> column_panel(column_index, column_values);
    for (int_t source_j = source_j_begin; source_j < source_j_end;
         ++source_j)
    {
        int_t output_gid = column_panel.gid(source_j);
        if ((output_gid == parent) != lookahead)
            ABORT("SymFact V2 CPU task batch mixes scheduling modes.");
        symldl_v2_cpu_note_task_pending(lu, output_gid, slot, 1);
        if (lu->symV2CpuProfileEnabled)
        {
            if (lookahead)
                ++lu->symV2CpuLookaheadTasks;
            else
                ++lu->symV2CpuExcludeTasks;
        }
    }
    bool defer = false;
    uint64_t expected_generation =
        symldl_v2_cpu_slot_generation(lu, slot, source_k);
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads() > 1)
    {
        uint64_t range_work = known_work;
        if (range_work == std::numeric_limits<uint64_t>::max())
        {
            range_work = 0;
            xlpanel_t<Ftype> row_panel(row_index, row_values);
            for (int_t source_j = source_j_begin;
                 source_j < source_j_end; ++source_j)
                range_work = symldl_v2_cpu_saturating_add(
                    range_work, symldl_v2_cpu_column_work(
                        row_panel, first_row_block, column_panel, source_j));
        }
        defer = range_work >= symldl_v2_cpu_min_deferred_work();
        // Release the parent before exclude tasks occupy the worker queue.
        if (lookahead && (lu->Pr > 1 || lu->Pc > 1))
            defer = false;
    }
#endif
    if (lu->symV2CpuProfileEnabled)
    {
        ++lu->symV2CpuTaskBatches;
        if (defer)
            ++lu->symV2CpuDeferredTaskBatches;
        else
            ++lu->symV2CpuInlineTaskBatches;
    }
#ifdef _OPENMP
    if (defer)
    {
        int priority_value = lookahead ? 100 : 0;
#pragma omp atomic update
        ++lu->symV2CpuDeferredTasksActive;
#pragma omp task firstprivate(source_k, slot, row_index, row_values,             \
                              first_row_block, column_index, column_values,     \
                              source_j_begin, source_j_end, parent, lookahead,  \
                              priority_value, expected_generation)             \
                 shared(lu) priority(priority_value)
        {
            symldl_v2_cpu_assert_slot_generation(
                lu, slot, source_k, expected_generation);
            xlpanel_t<Ftype> task_row(row_index, row_values);
            xlpanel_t<Ftype> task_column(column_index, column_values);
            double task_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            long long completed = 0;
            for (int_t source_j = source_j_begin;
                 source_j < source_j_end; ++source_j)
            {
                symldl_v2_cpu_assert_slot_generation(
                    lu, slot, source_k, expected_generation);
                completed += symldl_v2_cpu_grouped_update_column(
                    lu, task_row, first_row_block, task_column, source_j,
                    column_values, lookahead);
                symldl_v2_cpu_note_task_pending(
                    lu, task_column.gid(source_j), slot, -1);
            }
            if (lu->symV2CpuProfileEnabled)
            {
                double task_elapsed = SuperLU_timer_() - task_start;
#pragma omp atomic update
                lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed);
#pragma omp atomic update
                lu->symV2CpuSchurTime += task_elapsed;
            }
#pragma omp atomic update
            --lu->symV2CpuDeferredTasksActive;
        }
        return;
    }
#endif
    xlpanel_t<Ftype> row_panel(row_index, row_values);
    symldl_v2_cpu_assert_slot_generation(
        lu, slot, source_k, expected_generation);
    double task_start = lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    long long completed = 0;
    for (int_t source_j = source_j_begin; source_j < source_j_end; ++source_j)
    {
        symldl_v2_cpu_assert_slot_generation(
            lu, slot, source_k, expected_generation);
        completed += symldl_v2_cpu_grouped_update_column(
            lu, row_panel, first_row_block, column_panel, source_j,
            column_values, lookahead);
        symldl_v2_cpu_note_task_pending(
            lu, column_panel.gid(source_j), slot, -1);
    }
    if (lu->symV2CpuProfileEnabled)
    {
        lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed);
        lu->symV2CpuSchurTime += SuperLU_timer_() - task_start;
    }
    (void) source_k;
}

template <typename Ftype>
static void symldl_v2_cpu_submit_exclude_task_batches(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    int_t *row_index, Ftype *row_values, int_t first_row_block,
    int_t *column_index, Ftype *column_values, int_t source_j_begin,
    int_t source_j_end)
{
    if (source_j_begin >= source_j_end)
        return;
    int workers = 1;
#ifdef _OPENMP
    workers = SUPERLU_MAX(1, omp_get_max_threads());
#endif
    if (workers == 1)
    {
        symldl_v2_cpu_submit_grouped_task_range(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values,
            source_j_begin, source_j_end, false, 0);
        return;
    }
    xlpanel_t<Ftype> row_panel(row_index, row_values);
    xlpanel_t<Ftype> column_panel(column_index, column_values);
    uint64_t total_work = 0;
    for (int_t source_j = source_j_begin; source_j < source_j_end; ++source_j)
        total_work = symldl_v2_cpu_saturating_add(
            total_work, symldl_v2_cpu_column_work(
                row_panel, first_row_block, column_panel, source_j));
    uint64_t target_batches = static_cast<uint64_t>(workers) * 4;
    uint64_t target_work = total_work == 0 ? 1 :
        (total_work - 1) / target_batches + 1;
    target_work = SUPERLU_MAX(target_work,
                              symldl_v2_cpu_min_deferred_work());

    int_t begin = source_j_begin;
    uint64_t batch_work = 0;
    for (int_t source_j = source_j_begin; source_j < source_j_end; ++source_j)
    {
        batch_work = symldl_v2_cpu_saturating_add(
            batch_work, symldl_v2_cpu_column_work(
                row_panel, first_row_block, column_panel, source_j));
        bool close_batch = batch_work >= target_work ||
                           source_j + 1 == source_j_end;
        if (!close_batch)
            continue;
        symldl_v2_cpu_submit_grouped_task_range(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values, begin,
            source_j + 1, false, batch_work);
        begin = source_j + 1;
        batch_work = 0;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_submit_schur_task_set(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    int_t *row_index, Ftype *row_values, int_t first_row_block,
    int_t *column_index, Ftype *column_values, int_t source_j_begin,
    int_t source_j_end)
{
    xlpanel_t<Ftype> column_panel(column_index, column_values);
    int_t parent_j = -1;
    for (int_t source_j = source_j_begin; source_j < source_j_end;
         ++source_j)
        if (column_panel.gid(source_j) == parent)
        {
            parent_j = source_j;
            break;
        }
    if (parent_j >= 0)
        symldl_v2_cpu_submit_grouped_task_range(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values,
            parent_j, parent_j + 1, true);
    symldl_v2_cpu_submit_exclude_task_batches(
        lu, source_k, parent, slot, row_index, row_values,
        first_row_block, column_index, column_values,
        source_j_begin, parent_j >= 0 ? parent_j : source_j_end);
    if (parent_j >= 0)
        symldl_v2_cpu_submit_exclude_task_batches(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values,
            parent_j + 1, source_j_end);
}

template <typename Ftype>
static void symldl_v2_cpu_submit_collapsed_schur_tasks(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    xlpanel_t<Ftype> &panel, Ftype *raw_values)
{
    if (panel.isEmpty())
        return;
    int_t first = panel.haveDiag() ? 1 : 0;
    symldl_v2_cpu_submit_schur_task_set(
        lu, k, parent, slot, panel.index, panel.val, first,
        panel.index, raw_values, first, panel.nblocks());
}

template <typename Ftype>
static void symldl_v2_cpu_submit_dual_schur_tasks(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    xlpanel_t<Ftype> &row_panel, xlpanel_t<Ftype> &column_panel)
{
    if (row_panel.isEmpty() || column_panel.isEmpty())
        return;
    symldl_v2_cpu_submit_schur_task_set(
        lu, k, parent, slot, row_panel.index, row_panel.val, 0,
        column_panel.index, column_panel.val, 0, column_panel.nblocks());
}

template <typename Ftype>
static void symldl_v2_cpu_submit_selected_task_range(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    int_t *row_index, Ftype *row_values, int_t first_row_block,
    int_t *column_index, Ftype *column_values, const int_t *selected,
    int_t selected_begin, int_t selected_end, bool lookahead,
    uint64_t known_work = std::numeric_limits<uint64_t>::max())
{
    if (selected_begin >= selected_end)
        return;
    xlpanel_t<Ftype> column_panel(column_index, column_values);
    for (int_t item = selected_begin; item < selected_end; ++item)
    {
        int_t source_j = selected[item];
        if (source_j < 0 || source_j >= column_panel.nblocks())
            ABORT("SymFact V2 CPU selected task is invalid.");
        int_t output_gid = column_panel.gid(source_j);
        if ((output_gid == parent) != lookahead)
            ABORT("SymFact V2 CPU selected task batch mixes scheduling modes.");
        symldl_v2_cpu_note_task_pending(lu, output_gid, slot, 1);
        if (lu->symV2CpuProfileEnabled)
        {
            if (lookahead)
                ++lu->symV2CpuLookaheadTasks;
            else
                ++lu->symV2CpuExcludeTasks;
        }
    }

    bool defer = false;
    uint64_t expected_generation =
        symldl_v2_cpu_slot_generation(lu, slot, source_k);
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads() > 1)
    {
        uint64_t range_work = known_work;
        if (range_work == std::numeric_limits<uint64_t>::max())
        {
            range_work = 0;
            xlpanel_t<Ftype> row_panel(row_index, row_values);
            for (int_t item = selected_begin; item < selected_end; ++item)
                range_work = symldl_v2_cpu_saturating_add(
                    range_work, symldl_v2_cpu_column_work(
                        row_panel, first_row_block, column_panel,
                        selected[item]));
        }
        defer = range_work >= symldl_v2_cpu_min_deferred_work();
        if (lookahead && (lu->Pr > 1 || lu->Pc > 1))
            defer = false;
    }
#endif
    if (lu->symV2CpuProfileEnabled)
    {
        ++lu->symV2CpuTaskBatches;
        if (defer)
            ++lu->symV2CpuDeferredTaskBatches;
        else
            ++lu->symV2CpuInlineTaskBatches;
    }
#ifdef _OPENMP
    if (defer)
    {
        int priority_value = lookahead ? 100 : 0;
#pragma omp atomic update
        ++lu->symV2CpuDeferredTasksActive;
#pragma omp task firstprivate(source_k, slot, row_index, row_values,             \
                              first_row_block, column_index, column_values,     \
                              selected, selected_begin, selected_end, parent,   \
                              lookahead, priority_value, expected_generation)   \
                 shared(lu) priority(priority_value)
        {
            symldl_v2_cpu_assert_slot_generation(
                lu, slot, source_k, expected_generation);
            xlpanel_t<Ftype> task_row(row_index, row_values);
            xlpanel_t<Ftype> task_column(column_index, column_values);
            double task_start =
                lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
            long long completed = 0;
            for (int_t item = selected_begin; item < selected_end; ++item)
            {
                int_t source_j = selected[item];
                symldl_v2_cpu_assert_slot_generation(
                    lu, slot, source_k, expected_generation);
                completed += symldl_v2_cpu_grouped_update_column(
                    lu, task_row, first_row_block, task_column, source_j,
                    column_values, lookahead);
                symldl_v2_cpu_note_task_pending(
                    lu, task_column.gid(source_j), slot, -1);
            }
            if (lu->symV2CpuProfileEnabled)
            {
                double elapsed = SuperLU_timer_() - task_start;
#pragma omp atomic update
                lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed);
#pragma omp atomic update
                lu->symV2CpuSchurTime += elapsed;
            }
#pragma omp atomic update
            --lu->symV2CpuDeferredTasksActive;
        }
        return;
    }
#endif
    xlpanel_t<Ftype> row_panel(row_index, row_values);
    symldl_v2_cpu_assert_slot_generation(
        lu, slot, source_k, expected_generation);
    double task_start = lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    long long completed = 0;
    for (int_t item = selected_begin; item < selected_end; ++item)
    {
        int_t source_j = selected[item];
        completed += symldl_v2_cpu_grouped_update_column(
            lu, row_panel, first_row_block, column_panel, source_j,
            column_values, lookahead);
        symldl_v2_cpu_note_task_pending(
            lu, column_panel.gid(source_j), slot, -1);
    }
    if (lu->symV2CpuProfileEnabled)
    {
        lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed);
        lu->symV2CpuSchurTime += SuperLU_timer_() - task_start;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_submit_selected_exclude_batches(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    int_t *row_index, Ftype *row_values, int_t first_row_block,
    int_t *column_index, Ftype *column_values, const int_t *selected,
    int_t selected_begin, int_t selected_end)
{
    if (selected_begin >= selected_end)
        return;
    int workers = 1;
#ifdef _OPENMP
    workers = SUPERLU_MAX(1, omp_get_max_threads());
#endif
    if (workers == 1)
    {
        symldl_v2_cpu_submit_selected_task_range(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values, selected,
            selected_begin, selected_end, false, 0);
        return;
    }
    xlpanel_t<Ftype> row_panel(row_index, row_values);
    xlpanel_t<Ftype> column_panel(column_index, column_values);
    uint64_t total_work = 0;
    for (int_t item = selected_begin; item < selected_end; ++item)
        total_work = symldl_v2_cpu_saturating_add(
            total_work, symldl_v2_cpu_column_work(
                row_panel, first_row_block, column_panel, selected[item]));
    uint64_t target_batches = static_cast<uint64_t>(workers) * 4;
    uint64_t target_work = total_work == 0 ? 1 :
        (total_work - 1) / target_batches + 1;
    target_work = SUPERLU_MAX(target_work,
                              symldl_v2_cpu_min_deferred_work());

    int_t begin = selected_begin;
    uint64_t batch_work = 0;
    for (int_t item = selected_begin; item < selected_end; ++item)
    {
        batch_work = symldl_v2_cpu_saturating_add(
            batch_work, symldl_v2_cpu_column_work(
                row_panel, first_row_block, column_panel, selected[item]));
        if (batch_work < target_work && item + 1 != selected_end)
            continue;
        symldl_v2_cpu_submit_selected_task_range(
            lu, source_k, parent, slot, row_index, row_values,
            first_row_block, column_index, column_values, selected,
            begin, item + 1, false, batch_work);
        begin = item + 1;
        batch_work = 0;
    }
}

template <typename Ftype>
static void symldl_v2_cpu_submit_selected_schur_tasks(
    xLUstruct_t<Ftype> *lu, int_t source_k, int_t parent, int slot,
    xlpanel_t<Ftype> &panel, Ftype *raw_values, const int_t *selected,
    int_t selected_count)
{
    if (panel.isEmpty() || selected_count <= 0)
        return;
    int_t parent_item = -1;
    for (int_t item = 0; item < selected_count; ++item)
        if (panel.gid(selected[item]) == parent)
        {
            parent_item = item;
            break;
        }
    int_t first = panel.haveDiag() ? 1 : 0;
    if (parent_item >= 0)
        symldl_v2_cpu_submit_selected_task_range(
            lu, source_k, parent, slot, panel.index, panel.val, first,
            panel.index, raw_values, selected, parent_item,
            parent_item + 1, true);
    symldl_v2_cpu_submit_selected_exclude_batches(
        lu, source_k, parent, slot, panel.index, panel.val, first,
        panel.index, raw_values, selected, 0,
        parent_item >= 0 ? parent_item : selected_count);
    if (parent_item >= 0)
        symldl_v2_cpu_submit_selected_exclude_batches(
            lu, source_k, parent, slot, panel.index, panel.val, first,
            panel.index, raw_values, selected, parent_item + 1,
            selected_count);
}

template <typename Ftype>
static void symldl_v2_cpu_schur_update(
    xLUstruct_t<Ftype> *lu, int_t, xlpanel_t<Ftype> &panel,
    const Ftype *raw_values)
{
    if (panel.isEmpty())
        return;
    int_t first = panel.haveDiag() ? 1 : 0;
    int_t column_blocks = panel.nblocks() - first;
    if (column_blocks <= 0)
        return;

    double start = SuperLU_timer_();
    long long completed_tasks = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:completed_tasks)
    for (int_t source_j = first; source_j < panel.nblocks(); ++source_j)
    {
        completed_tasks += symldl_v2_cpu_grouped_update_column(
            lu, panel, first, panel, source_j, raw_values, false);
    }
    lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed_tasks);
    lu->symV2CpuSchurTime += SuperLU_timer_() - start;
}

template <typename Ftype>
static void symldl_v2_cpu_dual_schur_update(
    xLUstruct_t<Ftype> *lu, int_t, xlpanel_t<Ftype> &row_panel,
    xlpanel_t<Ftype> &column_panel)
{
    if (row_panel.isEmpty() || column_panel.isEmpty())
        return;
    int_t column_blocks = column_panel.nblocks();
    if (row_panel.nblocks() <= 0 || column_blocks <= 0)
        return;

    double start = SuperLU_timer_();
    long long completed_tasks = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:completed_tasks)
    for (int_t source_j = 0; source_j < column_blocks; ++source_j)
    {
        completed_tasks += symldl_v2_cpu_grouped_update_column(
            lu, row_panel, 0, column_panel, source_j, column_panel.val,
            false);
    }
    lu->symV2CpuSchurTasks += static_cast<uint64_t>(completed_tasks);
    lu->symV2CpuSchurTime += SuperLU_timer_() - start;
}
