#pragma once

#include <cstddef>
#include <limits>

static inline size_t symldl_v2_cpu_mpi_chunk_count(size_t count)
{
    const size_t limit =
        static_cast<size_t>(std::numeric_limits<int>::max());
    return count == 0 ? 0 : (count - 1) / limit + 1;
}

static const size_t SYM_LDL_V2_CPU_CONTIGUOUS_ROWS =
    std::numeric_limits<size_t>::max();

struct SymLDLV2CpuPackSegment
{
    int_t source_block;
    int_t row_count;
    int_t packed_row_offset;
    size_t row_permutation_offset;
};

struct SymLDLV2CpuThreadProfile
{
    uint64_t gemms = 0;
    uint64_t small_gemms = 0;
    uint64_t medium_gemms = 0;
    uint64_t large_gemms = 0;
    uint64_t m_sum = 0;
    uint64_t n_sum = 0;
    uint64_t k_sum = 0;
    uint64_t max_m = 0;
    uint64_t max_n = 0;
    uint64_t max_k = 0;
    uint64_t mapped_scatter_values = 0;
    uint64_t mapped_row_exact = 0;
    uint64_t mapped_row_contiguous = 0;
    uint64_t mapped_column_full = 0;
    uint64_t mapped_column_contiguous = 0;
    uint64_t mapped_rectangular = 0;
    uint64_t mapped_sorted_rows = 0;
    uint64_t mapped_destination_full = 0;
    uint64_t mapped_row_contiguous_values = 0;
    uint64_t mapped_rectangular_values = 0;
};
