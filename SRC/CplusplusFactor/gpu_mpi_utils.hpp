#pragma once

#include <cstdlib>
#include <cstring>
#include <limits>

#include "mpi.h"
#include "gpuCommon.hpp"

#ifdef HAVE_CUDA

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define SUPERLU_HAVE_MPI_EXT_HEADER 1
#endif
#endif

static inline int superlu_env_truthy(const char *value)
{
    if (!value || !value[0]) return -1;
    if (!std::strcmp(value, "1") || !std::strcmp(value, "true") ||
        !std::strcmp(value, "TRUE") || !std::strcmp(value, "yes") ||
        !std::strcmp(value, "YES") || !std::strcmp(value, "on") ||
        !std::strcmp(value, "ON")) {
        return 1;
    }
    if (!std::strcmp(value, "0") || !std::strcmp(value, "false") ||
        !std::strcmp(value, "FALSE") || !std::strcmp(value, "no") ||
        !std::strcmp(value, "NO") || !std::strcmp(value, "off") ||
        !std::strcmp(value, "OFF")) {
        return 0;
    }
    return -1;
}

static inline bool superlu_sym_v2_async_factor()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;

    const char *env = std::getenv("GPU3DV2_ASYNC_FACTOR");
    if (env == NULL || env[0] == '\0')
    {
        cached = 1;
        return true;
    }

    int enabled = superlu_env_truthy(env);
    if (enabled < 0)
        ABORT("GPU3DV2_ASYNC_FACTOR must be a boolean value.");
    cached = enabled > 0 ? 1 : 0;
    return cached != 0;
}

static inline bool superlu_sym_v2_batch_ancestor_reduce()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_BATCH_ANCESTOR_REDUCE");
    if (env == NULL || env[0] == '\0')
    {
        cached = 1;
        return true;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_BATCH_ANCESTOR_REDUCE must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline size_t superlu_sym_v2_ancestor_batch_bytes()
{
    static size_t cached = 0;
    if (cached != 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_ANCESTOR_BATCH_BYTES");
    if (env == NULL || env[0] == '\0')
    {
        cached = 256u * 1024u * 1024u;
        return cached;
    }
    char *end = NULL;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || value < 4096ull)
        ABORT("GPU3DV2_ANCESTOR_BATCH_BYTES must be an integer >= 4096.");
    cached = static_cast<size_t>(value);
    return cached;
}

static inline bool superlu_sym_v2_pinned_staging()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_PINNED_STAGING");
    if (env == NULL || env[0] == '\0')
    {
        cached = 1;
        return true;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_PINNED_STAGING must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_pinned_staging_pool()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_PINNED_STAGING_POOL");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_PINNED_STAGING_POOL must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_large_recv_pipeline()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_LARGE_RECV_PIPELINE");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_LARGE_RECV_PIPELINE must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline size_t superlu_sym_v2_large_recv_bytes()
{
    static size_t cached = 0;
    if (cached != 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_LARGE_RECV_BYTES");
    if (env == NULL || env[0] == '\0')
    {
        cached = 4u * 1024u * 1024u;
        return cached;
    }
    char *end = NULL;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || value < sizeof(double) ||
        value > static_cast<unsigned long long>(
            std::numeric_limits<size_t>::max()))
        ABORT("GPU3DV2_LARGE_RECV_BYTES must be a positive byte count.");
    cached = static_cast<size_t>(value);
    return cached;
}

static inline bool superlu_sym_v2_wpanel_cache()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_WPANEL_CACHE");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_WPANEL_CACHE must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_pc_fragment_schur()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_PC_FRAGMENT_SCHUR");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_PC_FRAGMENT_SCHUR must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_pc_fragment_ldl_native()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_PC_FRAGMENT_LDL_NATIVE");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_PC_FRAGMENT_LDL_NATIVE must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_hybrid_row_bcast()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_HYBRID_ROW_BCAST");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_HYBRID_ROW_BCAST must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_row_l_source_pack()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_ROW_L_SOURCE_PACK");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_ROW_L_SOURCE_PACK must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_row_l_direct_recv()
{
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return true;
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_ROW_L_DIRECT_RECV");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_ROW_L_DIRECT_RECV must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_row_l_postsolve_send()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_ROW_L_POSTSOLVE_SEND");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_ROW_L_POSTSOLVE_SEND must be a boolean value.");
    if (parsed != 0)
        ABORT("GPU3DV2_ROW_L_POSTSOLVE_SEND was removed: the postsolve row-L transport experiment regressed memory and factor time.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_rowfrag_dest_pack()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_ROWFRAG_DEST_PACK");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_ROWFRAG_DEST_PACK must be a boolean value.");
    cached = parsed;
    return cached != 0;
}


// SYM_V2_PC2_PHASE1_FLAGS_BEGIN
static inline bool superlu_sym_v2_env_bool_flag(const char *name, int fallback)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0')
        return fallback != 0;
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("Invalid boolean GPU3DV2 row-L environment value.");
    return parsed != 0;
}

static inline double superlu_sym_v2_env_double_flag(
    const char *name, double fallback)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0')
        return fallback;
    char *end = NULL;
    double value = std::strtod(env, &end);
    if (end == env || *end != '\0' || value <= 0.0)
        ABORT("Invalid positive GPU3DV2 row-L floating-point environment value.");
    return value;
}

static inline bool superlu_sym_v2_row_l_separate_send_staging()
{
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return true;
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_SEPARATE_SEND_STAGING", 0);
}

static inline bool superlu_sym_v2_row_l_pack_all_dest()
{
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return true;
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PACK_ALL_DEST", 0);
}

static inline bool superlu_sym_v2_row_l_one_sync()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_ONE_SYNC", 0);
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
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PLAN_V2_AGGREGATE_DEST", 1);
}

static inline bool superlu_sym_v2_row_l_plan_v2_compact()
{
    return superlu_sym_v2_env_bool_flag(
        "GPU3DV2_ROW_L_PLAN_V2_COMPACT",
        (superlu_sym_v2_pc_fragment_ldl_native() &&
         superlu_sym_v2_row_l_plan_v2()) ? 1 : 0);
}

static inline bool superlu_sym_v2_row_hybrid_cost()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_HYBRID_COST", 0);
}

static inline bool superlu_sym_v2_row_hybrid_trace()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_HYBRID_TRACE", 0);
}

static inline double superlu_sym_v2_row_hybrid_margin()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_ROW_HYBRID_MARGIN", 0.75);
}

static inline double superlu_sym_v2_cost_lat_us()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_LAT_US", 3.0);
}

static inline double superlu_sym_v2_cost_net_gbps()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_NET_GBPS", 25.0);
}

static inline double superlu_sym_v2_cost_pcie_gbps()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_PCIE_GBPS", 20.0);
}

static inline double superlu_sym_v2_cost_pack_gbps()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_PACK_GBPS", 250.0);
}

static inline double superlu_sym_v2_cost_asm_gbps()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_ASM_GBPS", 250.0);
}

static inline double superlu_sym_v2_cost_kernel_us()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_KERNEL_US", 5.0);
}

static inline double superlu_sym_v2_cost_setup_weight()
{
    return superlu_sym_v2_env_double_flag("GPU3DV2_COST_SETUP_WEIGHT", 1.0);
}

static inline bool superlu_sym_v2_row_l_exchange_issue_complete()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_EXCHANGE_ISSUE_COMPLETE", 0);
}

static inline bool superlu_sym_v2_pcfrag_async_experiment()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_ASYNC_EXPERIMENT", 0);
}

static inline bool superlu_sym_v2_pcfrag_async_exchange()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_ASYNC_EXCHANGE", 0);
}

static inline bool superlu_sym_v2_pcfrag_async_pipeline()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_ASYNC_PIPELINE", 0);
}

static inline bool superlu_sym_v2_pcfrag_cuda_aware_experiment()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_CUDA_AWARE_EXPERIMENT", 0);
}

static inline bool superlu_sym_v2_pcfrag_taskflow()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_TASKFLOW", 0);
}

static inline bool superlu_sym_v2_pcfrag_taskflow_strict()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_TASKFLOW_STRICT", 1);
}

static inline bool superlu_sym_v2_pcfrag_taskflow_validate()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_TASKFLOW_VALIDATE", 0);
}

static inline bool superlu_sym_v2_pcfrag_taskflow_scheduler()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_TASKFLOW_SCHEDULER", 0);
}

static inline int superlu_sym_v2_pcfrag_taskflow_progress_budget()
{
    static int cached = -1;
    if (cached > 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_PCFRAG_TASKFLOW_PROGRESS_BUDGET");
    if (env == NULL || env[0] == '\0')
    {
        cached = 64;
        return cached;
    }
    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 1 || value > 1048576L)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_PROGRESS_BUDGET must be an integer in [1,1048576].");
    cached = static_cast<int>(value);
    return cached;
}

static inline int superlu_sym_v2_pcfrag_taskflow_producer_task_limit()
{
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *env =
        std::getenv("GPU3DV2_PCFRAG_TASKFLOW_PRODUCER_TASK_LIMIT");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return cached;
    }
    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 1048576L)
        ABORT("GPU3DV2_PCFRAG_TASKFLOW_PRODUCER_TASK_LIMIT must be an integer in [0,1048576]; 0 disables the cap.");
    cached = static_cast<int>(value);
    return cached;
}

static inline bool superlu_sym_v2_pcfrag_taskflow_eager()
{
    if (superlu_sym_v2_pcfrag_taskflow_scheduler())
        return false;
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_TASKFLOW_EAGER", 1);
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
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_COMPRESSED_PLAN", 0);
}

static inline bool superlu_sym_v2_row_l_parallel_sendmap()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_PARALLEL_SENDMAP", 0);
}

static inline bool superlu_sym_v2_row_l_skip_legacy_recv_map()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_SKIP_LEGACY_RECV_MAP", 0);
}

static inline bool superlu_sym_v2_row_l_lazy_sendmap()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_LAZY_SENDMAP", 0);
}

static inline bool superlu_sym_v2_pcfrag_setup_opt()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_PCFRAG_SETUP_OPT", 0);
}

static inline bool superlu_sym_v2_row_l_lazy_warp_pack()
{
    return superlu_sym_v2_env_bool_flag("GPU3DV2_ROW_L_LAZY_WARP_PACK", 0);
}
// SYM_V2_PC2_PHASE1_FLAGS_END
static inline bool superlu_sym_v2_rowfrag_destination_path()
{
// SYM_V2_PC2_PHASE4_DEST_PATH_FLAG_BEGIN
    return superlu_sym_v2_pc_fragment_ldl_native() ||
           superlu_sym_v2_rowfrag_dest_pack() ||
           superlu_sym_v2_row_l_source_pack() ||
           superlu_sym_v2_row_l_direct_recv() ||
           superlu_sym_v2_row_l_postsolve_send() ||
           superlu_sym_v2_row_l_plan_v2_exchange();
// SYM_V2_PC2_PHASE4_DEST_PATH_FLAG_END
}

static inline bool superlu_sym_v2_exact_fragment_demand()
{
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return true;
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_EXACT_FRAGMENT_DEMAND");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_EXACT_FRAGMENT_DEMAND must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_exact_partner_fragment_demand()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_EXACT_PARTNER_FRAGMENT_DEMAND");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_EXACT_PARTNER_FRAGMENT_DEMAND must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_exact_row_fragment_demand()
{
    if (superlu_sym_v2_pc_fragment_ldl_native())
        return true;
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_EXACT_ROW_FRAGMENT_DEMAND");
    if (env == NULL || env[0] == '\0')
    {
        cached = 1;
        return true;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_EXACT_ROW_FRAGMENT_DEMAND must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_sym_v2_exact_map_index()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_EXACT_MAP_INDEX");
    if (env == NULL || env[0] == '\0')
    {
        cached = 1;
        return true;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_EXACT_MAP_INDEX must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline bool superlu_cuda_aware_mpi()
{
    static int cached = -1;
    if (cached >= 0) return cached != 0;

    const int forced = superlu_env_truthy(std::getenv("SUPERLU_CUDA_AWARE_MPI"));
    if (forced >= 0) {
        cached = forced;
        return cached != 0;
    }

    const int mpich_gpu =
        superlu_env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED"));
    if (mpich_gpu >= 0) {
        cached = mpich_gpu;
        return cached != 0;
    }

#if defined(SUPERLU_HAVE_MPI_EXT_HEADER) && defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cached = MPIX_Query_cuda_support() != 0;
#else
    cached = 0;
#endif
    return cached != 0;
}

static inline void superlu_gpu_mpi_send(const void *device_buf, void *host_stage,
                                        size_t elem_size, int count,
                                        MPI_Datatype dtype, int dest, int tag,
                                        MPI_Comm comm)
{
    if (count <= 0) return;
    if (superlu_cuda_aware_mpi()) {
        MPI_Send(const_cast<void *>(device_buf), count, dtype, dest, tag, comm);
        return;
    }

    gpuErrchk(cudaMemcpy(host_stage, device_buf, elem_size * (size_t)count,
                         cudaMemcpyDeviceToHost));
    MPI_Send(host_stage, count, dtype, dest, tag, comm);
}

static inline void superlu_gpu_mpi_recv(void *device_buf, void *host_stage,
                                        size_t elem_size, int count,
                                        MPI_Datatype dtype, int src, int tag,
                                        MPI_Comm comm, MPI_Status *status)
{
    if (count <= 0) return;
    if (superlu_cuda_aware_mpi()) {
        MPI_Recv(device_buf, count, dtype, src, tag, comm, status);
        return;
    }

    MPI_Recv(host_stage, count, dtype, src, tag, comm, status);
    gpuErrchk(cudaMemcpy(device_buf, host_stage, elem_size * (size_t)count,
                         cudaMemcpyHostToDevice));
}

static inline void superlu_gpu_mpi_bcast(void *device_buf, void *host_stage,
                                         size_t elem_size, int count,
                                         MPI_Datatype dtype, int root,
                                         MPI_Comm comm)
{
    if (count <= 0) return;
    if (superlu_cuda_aware_mpi()) {
        MPI_Bcast(device_buf, count, dtype, root, comm);
        return;
    }

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        gpuErrchk(cudaMemcpy(host_stage, device_buf, elem_size * (size_t)count,
                             cudaMemcpyDeviceToHost));
    }
    MPI_Bcast(host_stage, count, dtype, root, comm);
    if (rank != root) {
        gpuErrchk(cudaMemcpy(device_buf, host_stage, elem_size * (size_t)count,
                             cudaMemcpyHostToDevice));
    }
}

#endif /* HAVE_CUDA */
