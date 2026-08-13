#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>

#include "symldl_v2_env_config.hpp"

static inline bool symldl_v2_cpu_padded_direct_enabled()
{
    static const bool enabled = []() {
        const char *value = std::getenv("GPU3DV2_CPU_PADDED_DIRECT");
        return value == NULL || std::atoi(value) != 0;
    }();
    return enabled;
}

static inline bool symldl_v2_cpu_2d_padded_direct_enabled()
{
    static const bool enabled = superlu_sym_v2_env_bool_flag(
        "GPU3DV2_CPU_2D_PADDED_DIRECT", 0);
    return enabled;
}

static inline int symldl_v2_cpu_2d_padded_direct_max_percent()
{
    static const int percent = []() {
        const char *env = std::getenv(
            "GPU3DV2_CPU_2D_PADDED_DIRECT_MAX_PERCENT");
        if (env == NULL || env[0] == '\0')
            return 125;
        char *end = NULL;
        long value = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || value < 100 || value > 400)
            ABORT(
                "GPU3DV2_CPU_2D_PADDED_DIRECT_MAX_PERCENT must be between 100 and 400.");
        return static_cast<int>(value);
    }();
    return percent;
}

static inline bool symldl_v2_cpu_async_exchange_enabled()
{
    static const bool enabled = superlu_sym_v2_env_bool_flag(
        "GPU3DV2_CPU_ASYNC_EXCHANGE", 1);
    return enabled;
}

static inline bool symldl_v2_cpu_grid_specializations_enabled()
{
    static const bool enabled = superlu_sym_v2_env_bool_flag(
        "GPU3DV2_CPU_GRID_SPECIALIZATIONS", 1);
    return enabled;
}

enum SymLDLV2CpuExchangeRoute
{
    SYM_LDL_V2_CPU_ROUTE_COLLAPSED = 0,
    SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY = 1,
    SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT = 2
};

template <typename Ftype>
static inline SymLDLV2CpuExchangeRoute symldl_v2_cpu_exchange_route(
    const Ftype *lu)
{
    if (lu->Pr <= 1 && lu->Pc <= 1)
        return SYM_LDL_V2_CPU_ROUTE_COLLAPSED;
    if (!symldl_v2_cpu_grid_specializations_enabled())
        return SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT;
    if (lu->Pc <= 1)
        return SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY;
    return SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT;
}

enum SymLDLV2CpuRequestKind
{
    SYM_LDL_V2_CPU_REQUEST_NONE = 0,
    SYM_LDL_V2_CPU_REQUEST_PARTNER_RECV = 1,
    SYM_LDL_V2_CPU_REQUEST_ROW_RECV = 2,
    SYM_LDL_V2_CPU_REQUEST_SEND = 3
};

static inline bool symldl_v2_cpu_ownership_check_enabled()
{
    static const bool enabled = []() {
        const char *value = std::getenv("GPU3DV2_CPU_OWNERSHIP_CHECK");
        return value != NULL && std::atoi(value) != 0;
    }();
    return enabled;
}

static inline bool symldl_v2_cpu_output_lock_check_enabled()
{
    static const bool enabled = []() {
        const char *value = std::getenv("GPU3DV2_CPU_OUTPUT_LOCK_CHECK");
        return value != NULL && std::atoi(value) != 0;
    }();
    return enabled;
}

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

struct SymLDLV2CpuRowLookup
{
    size_t offset = 0;
    size_t extent = 0;
    unsigned char dense = 0;
};

struct SymLDLV2CpuExchangeState
{
    int_t k = -1;
    int_t parent = -1;
    int_t local_panel = -1;
    int source_pc = -1;
    int row_chunks_remaining = 0;
    size_t receive_request_count = 0;
    size_t pending_receive_chunks = 0;
    unsigned char updates_submitted = 0;
    unsigned char route = SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT;
    unsigned char active = 0;
};

struct SymLDLV2CpuThreadProfile
{
    uint64_t grouped_gemms = 0;
    uint64_t gemm_flops = 0;
    uint64_t direct_scatters = 0;
    uint64_t mapped_scatters = 0;
    uint64_t output_lock_attempts = 0;
    uint64_t output_lock_conflicts = 0;
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
    double small_gemm_time = 0.0;
    double medium_gemm_time = 0.0;
    double large_gemm_time = 0.0;
    double contiguous_scatter_time = 0.0;
    double irregular_scatter_time = 0.0;
    double gemm_time = 0.0;
    double lookahead_gemm_time = 0.0;
    double exclude_gemm_time = 0.0;
    double direct_time = 0.0;
    double mapped_time = 0.0;
    double padded_pack_time = 0.0;
    double row_map_time = 0.0;
    double output_lock_wait_time = 0.0;
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
    uint64_t row_map_sorted = 0;
    uint64_t row_map_dense_lookup = 0;
    uint64_t row_map_sparse_lookup = 0;
    double row_map_sorted_time = 0.0;
    double row_map_dense_lookup_time = 0.0;
    double row_map_sparse_lookup_time = 0.0;
    uint64_t padded_candidate_scatters = 0;
    uint64_t padded_candidate_source_values = 0;
    uint64_t padded_candidate_destination_values = 0;
    uint64_t padded_le_125_source_values = 0;
    uint64_t padded_le_125_destination_values = 0;
    uint64_t padded_le_150_source_values = 0;
    uint64_t padded_le_150_destination_values = 0;
    uint64_t padded_le_200_source_values = 0;
    uint64_t padded_le_200_destination_values = 0;
    uint64_t padded_le_400_source_values = 0;
    uint64_t padded_le_400_destination_values = 0;
    uint64_t padded_direct_groups = 0;
    uint64_t padded_direct_source_values = 0;
    uint64_t padded_direct_destination_values = 0;
    uint64_t padded_2d_candidates = 0;
    uint64_t padded_2d_source_values = 0;
    uint64_t padded_2d_destination_values = 0;
    uint64_t padded_2d_le_110_source_values = 0;
    uint64_t padded_2d_le_110_destination_values = 0;
    uint64_t padded_2d_le_125_source_values = 0;
    uint64_t padded_2d_le_125_destination_values = 0;
    uint64_t padded_2d_le_150_source_values = 0;
    uint64_t padded_2d_le_150_destination_values = 0;
    uint64_t padded_2d_le_200_source_values = 0;
    uint64_t padded_2d_le_200_destination_values = 0;
    uint64_t padded_2d_executed = 0;
    uint64_t padded_2d_row_and_column = 0;
    uint64_t padded_2d_executed_source_values = 0;
    uint64_t padded_2d_executed_destination_values = 0;
    uint64_t padded_2d_layout_rejects = 0;
    uint64_t padded_2d_workspace_rejects = 0;
    uint64_t padded_2d_expansion_rejects = 0;
    double padded_2d_row_pack_time = 0.0;
    double padded_2d_column_pack_time = 0.0;
};
