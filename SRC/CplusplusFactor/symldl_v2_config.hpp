#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "superlu_ddefs.h"
#include "symldl_v2_env_config.hpp"
#include "symldl_v2_gpu_mpi_helpers.hpp"

static inline bool superlu_sym_v2_async_factor()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ASYNC_FACTOR", 1);
}

static inline int superlu_sym_v2_gpu3d_version()
{
    const char *env = std::getenv("GPU3DVERSION");
    if (env == NULL || env[0] == '\0') return 0;

    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 2)
        ABORT("GPU3DVERSION must be one of 0, 1, or 2.");
    return static_cast<int>(value);
}

static inline bool superlu_sym_v2_pinned_staging()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PINNED_STAGING", 1);
}

static inline bool superlu_sym_v2_pinned_staging_pool()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PINNED_STAGING_POOL", 0);
}

static inline bool superlu_sym_v2_wpanel_cache()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_WPANEL_CACHE", 0);
}

static inline bool superlu_sym_v2_panel_arena_enabled()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PANEL_ARENA", 0);
}

static inline bool superlu_sym_v2_workspace_arena_enabled()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_WORKSPACE_ARENA", 0);
}

static inline bool superlu_sym_v2_trace_pcfrag()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_TRACE_PCFRAG", 0);
}

static inline bool superlu_sym_v2_route_profile()
{
    if (superlu_sym_v2_env_bool_flag("SYMLDL_V2_ROUTE_PROFILE", 0))
        return true;

    const char *profile = std::getenv("SYMLDL_V2_PROFILE");
    if (profile == NULL || profile[0] == '\0')
        return false;
    return !std::strcmp(profile, "route") ||
           !std::strcmp(profile, "factor") ||
           !std::strcmp(profile, "all");
}

static inline bool superlu_sym_v2_factor_comm_profile()
{
    return superlu_sym_v2_env_bool_flag("SUPERLU_FACTOR_COMM_PROFILE", 0);
}

static inline bool superlu_sym_v2_scoped_fragment_metadata()
{
    return superlu_sym_v2_env_bool_flag(
        "SYMLDL_V2_SCOPED_FRAGMENT_METADATA", 0);
}

static inline bool superlu_sym_v2_scoped_fragment_metadata_verify()
{
    return superlu_sym_v2_env_bool_flag(
        "SYMLDL_V2_SCOPED_FRAGMENT_METADATA_VERIFY", 0);
}

static inline bool superlu_sym_v2_plan_signature_components()
{
    return superlu_sym_v2_env_bool_flag(
        "SYMLDL_V2_PLAN_SIGNATURE_COMPONENTS", 0);
}

static inline bool superlu_sym_v2_pc_fragment_schur()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PC_FRAGMENT_SCHUR", 1);
}

static inline bool superlu_sym_v2_pc_fragment_ldl_native()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_PC_FRAGMENT_LDL_NATIVE",
        superlu_sym_v2_pc_fragment_schur() ? 1 : 0);
}

static inline bool superlu_sym_v2_row_l_direct_recv()
{
    if (superlu_sym_v2_pc_fragment_ldl_native()) return true;
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_DIRECT_RECV", 0);
}

static inline bool superlu_sym_v2_row_l_separate_send_staging()
{
    if (superlu_sym_v2_pc_fragment_ldl_native()) return true;
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_SEPARATE_SEND_STAGING", 0);
}

static inline bool superlu_sym_v2_row_l_pack_all_dest()
{
    if (superlu_sym_v2_pc_fragment_ldl_native()) return true;
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PACK_ALL_DEST", 0);
}

static inline bool superlu_sym_v2_row_l_plan_v2()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2",
        superlu_sym_v2_pc_fragment_ldl_native() ? 1 : 0);
}

static inline bool superlu_sym_v2_row_l_plan_v2_dryrun()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2_DRYRUN",
        superlu_sym_v2_pc_fragment_ldl_native() ? 0 : 1);
}

static inline bool superlu_sym_v2_row_l_plan_v2_block()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PLAN_V2_BLOCK", 1);
}

static inline bool superlu_sym_v2_row_l_plan_v2_verify()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PLAN_V2_VERIFY", 0);
}

static inline bool superlu_sym_v2_row_l_plan_v2_exchange()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2_EXCHANGE",
        (superlu_sym_v2_pc_fragment_ldl_native() &&
         superlu_sym_v2_row_l_plan_v2()) ? 1 : 0);
}

static inline bool superlu_sym_v2_row_l_plan_v2_aggregate_dest()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2_AGGREGATE_DEST", 1);
}

static inline bool superlu_sym_v2_row_l_plan_v2_compact()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2_COMPACT",
        (superlu_sym_v2_pc_fragment_ldl_native() &&
         superlu_sym_v2_row_l_plan_v2()) ? 1 : 0);
}

static inline bool superlu_sym_v2_recv_map_index()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_RECV_MAP_INDEX",
        superlu_sym_v2_pc_fragment_ldl_native() ? 1 : 0);
}

static inline bool superlu_sym_v2_recv_map_index_verify()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_RECV_MAP_INDEX_VERIFY", 0);
}

static inline bool superlu_sym_v2_row_l_compressed_plan()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_COMPRESSED_PLAN",
        superlu_sym_v2_pc_fragment_ldl_native() ? 1 : 0);
}

static inline bool superlu_sym_v2_row_l_lazy_sendmap()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_LAZY_SENDMAP",
        superlu_sym_v2_pc_fragment_ldl_native() ? 1 : 0);
}

static inline bool superlu_sym_v2_lower_envelope_enabled()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_LOWER_ENVELOPE", 1);
}

static inline int superlu_sym_v2_batch_schur_col_limit(
    int nrows, int64_t gemm_capacity)
{
    const char *env = std::getenv("GPU3DV2_BATCH_SCHUR_COLS");
    if (env != NULL && env[0] != '\0')
    {
        char *end = NULL;
        long value = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || value <= 0 ||
            value > 2147483647L)
            ABORT("Invalid GPU3DV2_BATCH_SCHUR_COLS value.");
        return static_cast<int>(value);
    }

    int limit = static_cast<int>(std::sqrt(static_cast<double>(
        SUPERLU_MAX(static_cast<int64_t>(1), gemm_capacity))));
    if (limit < 1)
        limit = 1;
    if (nrows > 0 && nrows < limit)
    {
        int64_t wide_limit = gemm_capacity / static_cast<int64_t>(nrows);
        if (wide_limit > limit)
            limit = static_cast<int>(SUPERLU_MIN(
                wide_limit, static_cast<int64_t>(2147483647L)));
    }
    return limit;
}

static inline bool superlu_sym_v2_pcfrag_async_exchange()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_ASYNC_EXCHANGE", 0);
}

static inline bool superlu_sym_v2_pcfrag_async_pipeline()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_ASYNC_PIPELINE", 0);
}

static inline bool superlu_sym_v2_rowfrag_destination_path()
{
    return superlu_sym_v2_pc_fragment_ldl_native() ||
           superlu_sym_v2_row_l_direct_recv() ||
           superlu_sym_v2_row_l_plan_v2_exchange();
}

static inline bool superlu_sym_v2_pcfrag_async_requested()
{
    return superlu_sym_v2_async_factor() &&
           (superlu_sym_v2_pcfrag_async_exchange() ||
            superlu_sym_v2_pcfrag_async_pipeline());
}

static inline bool superlu_sym_v2_pcfrag_async_prereqs_enabled()
{
    return superlu_sym_v2_pc_fragment_schur() &&
           superlu_sym_v2_pc_fragment_ldl_native() &&
           superlu_sym_v2_row_l_plan_v2_exchange() &&
           superlu_sym_v2_row_l_direct_recv() &&
           superlu_sym_v2_row_l_compressed_plan() &&
           superlu_sym_v2_row_l_lazy_sendmap();
}

static inline size_t superlu_sym_v2_staging_pool_bytes()
{
    return superlu_sym_v2_env_size_flag(
        "GPU3DV2_STAGING_POOL_BYTES", 0, 0);
}

#ifdef HAVE_CUDA
static inline void superlu_sym_v2_fail_if_cuda_aware_pcfrag_async()
{
    if (superlu_sym_v2_pcfrag_async_requested() && superlu_cuda_aware_mpi())
        ABORT("SymFact V2 Pc-fragment async exchange is fail-closed for CUDA-aware MPI.");
}
#endif
