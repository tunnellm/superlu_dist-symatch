#pragma once 
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>
#include <cassert>
#include "superlu_defs.h"
#include "luAuxStructTemplated.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#include "cublas_cusolver_wrappers.hpp"
#include "gpu_mpi_utils.hpp"
#endif
#include "lupanels.hpp"  //unneeded??
#include "xlupanels.hpp"
#include "superlu_blas.hpp"

#ifdef HAVE_CUDA
template <>
int_t xLUstruct_t<double>::dSymStartL2UGPU(int_t k, int_t stream_offset);
template <>
int_t xLUstruct_t<double>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset);
#endif

#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
static inline void xlu_sym_gpu3d_trace(gridinfo3d_t *grid3d, const char *msg)
{
    std::printf("[sym-gpu3d-trace] rank %d: %s\n",
                (grid3d != NULL) ? grid3d->iam : -1, msg);
    std::fflush(stdout);
}
#else
static inline void xlu_sym_gpu3d_trace(gridinfo3d_t *grid3d, const char *msg)
{
    (void)grid3d;
    (void)msg;
}
#endif

static inline size_t xlu_checked_product(size_t a, size_t b, const char *what)
{
    (void) what;
    if (a != 0 && b > static_cast<size_t>(-1) / a)
        ABORT("Workspace size overflows allocation size.");
    return a * b;
}

static inline size_t xlu_checked_alloc_bytes(int_t count, size_t elem_size,
                                             const char *what)
{
    if (count < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(count);
    if (static_cast<int_t>(n) != count)
        ABORT("Allocation size overflows int_t.");
    return xlu_checked_product(n, elem_size, what);
}

static inline size_t xlu_checked_square_alloc_bytes(int_t dim, size_t elem_size,
                                                    const char *what)
{
    if (dim < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(dim);
    if (static_cast<int_t>(n) != dim)
        ABORT("Allocation size overflows int_t.");
    size_t count = xlu_checked_product(n, n, what);
    return xlu_checked_product(count, elem_size, what);
}

static inline size_t xlu_checked_bigv_alloc_bytes(int_t ldt, int_t num_threads,
                                                  size_t elem_size,
                                                  const char *what)
{
    if (ldt < 0 || num_threads < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(ldt);
    size_t nt = static_cast<size_t>(num_threads);
    if (static_cast<int_t>(n) != ldt || static_cast<int_t>(nt) != num_threads)
        ABORT("Allocation size overflows int_t.");
    size_t count = xlu_checked_product(8, n, what);
    count = xlu_checked_product(count, n, what);
    count = xlu_checked_product(count, nt, what);
    return xlu_checked_product(count, elem_size, what);
}

static inline int xlu_gpu3d_contract()
{
    const char *env = std::getenv("GPU3DCONTRACT");
    if (env == NULL || env[0] == '\0')
        return 0;

    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 3)
        ABORT("GPU3DCONTRACT must be one of 0, 1, 2, or 3.");
    return (int)value;
}

static inline int xlu_gpu3d_version()
{
    const char *env = std::getenv("GPU3DVERSION");
    if (env == NULL || env[0] == '\0')
        return 0;

    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 2)
        ABORT("GPU3DVERSION must be one of 0, 1, or 2.");
    return (int)value;
}

static inline int xlu_env_bool(const char *name)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0')
        return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "TRUE") == 0 || std::strcmp(env, "yes") == 0 ||
        std::strcmp(env, "YES") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "ON") == 0)
        return 1;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0 || std::strcmp(env, "no") == 0 ||
        std::strcmp(env, "NO") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "OFF") == 0)
        return 0;
    ABORT("Invalid boolean environment value.");
    return 0;
}

static inline double xlu_env_double(const char *name, double fallback)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0')
        return fallback;

    char *end = NULL;
    double value = std::strtod(env, &end);
    if (end == env || *end != '\0' || !std::isfinite(value) || value <= 0.0)
        ABORT("Invalid positive floating-point environment value.");
    return value;
}

static inline int xlu_sytrf_count_2x2(const int *ipiv, int n)
{
    int n2x2 = 0;
    for (int i = 0; i < n;)
    {
        if (ipiv[i] > 0)
        {
            ++i;
        }
        else
        {
            ++n2x2;
            i += 2;
        }
    }
    return n2x2;
}

static inline double xlu_sym_inverse_scaled_residual(const double *a, int lda,
                                                     const double *ainv, int n)
{
    double a_norm = 0.0;
    double inv_norm = 0.0;
    double err_norm = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double row_sum = 0.0;
        double inv_row_sum = 0.0;
        double err_row_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            const double aij = (i >= j) ? a[i + (size_t)j * lda]
                                        : a[j + (size_t)i * lda];
            row_sum += std::fabs(aij);
            inv_row_sum += std::fabs(ainv[i + (size_t)j * n]);

            double prod = 0.0;
            for (int kk = 0; kk < n; ++kk)
            {
                const double aik = (i >= kk) ? a[i + (size_t)kk * lda]
                                             : a[kk + (size_t)i * lda];
                prod += aik * ainv[kk + (size_t)j * n];
            }
            if (!std::isfinite(prod))
                return DBL_MAX;
            const double target = (i == j) ? 1.0 : 0.0;
            err_row_sum += std::fabs(prod - target);
        }

        a_norm = SUPERLU_MAX(a_norm, row_sum);
        inv_norm = SUPERLU_MAX(inv_norm, inv_row_sum);
        err_norm = SUPERLU_MAX(err_norm, err_row_sum);
    }

    if (!std::isfinite(a_norm) || !std::isfinite(inv_norm) ||
        !std::isfinite(err_norm) || inv_norm == 0.0)
        return DBL_MAX;

    double denom = a_norm * inv_norm * DBL_EPSILON * (double)SUPERLU_MAX(n, 1);
    if (denom < DBL_EPSILON)
        denom = DBL_EPSILON;
    return err_norm / denom;
}

template <typename Ftype>
void xLUstruct_t<Ftype>::printSymV2SetupProfile()
{
    static const char *labels[SYM_V2_SETUP_COUNT] = {
        "node_mask",
        "panel_vec_build",
        "send_count_exchange",
        "lfrag_scratch_size",
        "cpu_workspace_alloc",
        "recv_buffer_alloc",
        "diag_buffer_alloc",
        "init_sym_workspace",
        "sym_cpu_workspace",
        "lfrag_send_map_build",
        "lfrag_send_gpu_alloc_copy",
        "lfrag_recv_count_allreduce",
        "lfrag_meta_allgather",
        "lfrag_recv_map_build",
        "exact_demand_build",
        "exact_send_map_index",
        "exact_send_map_build",
        "set_gpu_total",
        "gpu_mem_estimate",
        "copy_l_panels_to_gpu",
        "sym_v2_index_copy",
        "gpu_panel_struct_copy",
        "gpu_diag_factor_setup",
        "per_stream_buffer_alloc",
        "dfbuf_gemmbuf_alloc",
        "stream_handle_create",
        "diag_prefetch_alloc",
        "device_struct_copy"
    };

    if (!symV2SetupProfileActive() || symV2SetupProfilePrinted ||
        grid3d == NULL)
        return;
    symV2SetupProfilePrinted = 1;

    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (!mpi_initialized || mpi_finalized)
        return;

    double sum_time[SYM_V2_SETUP_COUNT] = {};
    double max_time[SYM_V2_SETUP_COUNT] = {};
    long long sum_count[SYM_V2_SETUP_COUNT] = {};
    struct { double val; int rank; } local_max_time[SYM_V2_SETUP_COUNT];
    struct { double val; int rank; } global_max_time[SYM_V2_SETUP_COUNT];
    int nranks = 1;

    for (int i = 0; i < SYM_V2_SETUP_COUNT; ++i)
    {
        local_max_time[i].val = symV2SetupProfileTime[i];
        local_max_time[i].rank = grid3d->iam;
    }

    MPI_Comm_size(grid3d->comm, &nranks);
    MPI_Reduce(symV2SetupProfileTime, sum_time, SYM_V2_SETUP_COUNT,
               MPI_DOUBLE, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symV2SetupProfileTime, max_time, SYM_V2_SETUP_COUNT,
               MPI_DOUBLE, MPI_MAX, 0, grid3d->comm);
    MPI_Reduce(local_max_time, global_max_time, SYM_V2_SETUP_COUNT,
               MPI_DOUBLE_INT, MPI_MAXLOC, 0, grid3d->comm);
    MPI_Reduce(symV2SetupProfileCount, sum_count, SYM_V2_SETUP_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    printf("SymFact GPU3D V2 setup profile (GPU3DV2_PROFILE=1):\n");
    printf("  %-32s %12s %12s %12s %9s %12s\n",
           "phase", "sum(s)", "avg_rank(s)", "max_rank(s)", "rank", "calls");
    for (int i = 0; i < SYM_V2_SETUP_COUNT; ++i)
    {
        if (sum_count[i] == 0 && sum_time[i] == 0.0 && max_time[i] == 0.0)
            continue;
        double avg = (nranks > 0) ? sum_time[i] / (double)nranks : 0.0;
        printf("  %-32s %12.6f %12.6f %12.6f %9d %12lld\n",
               labels[i], sum_time[i], avg, max_time[i],
               global_max_time[i].rank, sum_count[i]);
    }
    fflush(stdout);
}

template <typename Ftype>
void xLUstruct_t<Ftype>::printSymV2FactorProfile()
{
    static const char *labels[SYM_V2_FACTOR_COUNT] = {
        "tree_wall",
        "initial_factor_dispatch",
        "initial_panel_bcast",
        "sched_lookahead_dispatch",
        "lookahead_update",
        "lookahead_sync",
        "sched_factor_dispatch",
        "parent_factor",
        "exclude_update",
        "bcast_advance",
        "final_sync",
        "diag_panel_solve",
        "panel_bcast",
        "partner_l_exchange"
    };
    static const char *payload_labels[SYM_V2_PAYLOAD_COUNT] = {
        "panel_call",
        "panel_mpi",
        "partner_call",
        "partner_mpi_send",
        "partner_mpi_recv",
        "partner_self",
        "rowfrag_call",
        "rowfrag_mpi_send",
        "rowfrag_mpi_recv",
        "rowfrag_host_staging",
        "rowfrag_self"
    };
    static const char *payload_bin_labels[SYM_V2_PAYLOAD_BIN_COUNT] = {
        "0",
        "<=1K",
        "<=4K",
        "<=16K",
        "<=64K",
        "<=256K",
        "<=1M",
        "<=4M",
        "<=16M",
        ">16M"
    };

    if (!symV2FactorProfileActive() || symV2FactorProfilePrinted ||
        grid3d == NULL)
        return;
    symV2FactorProfilePrinted = 1;

    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (!mpi_initialized || mpi_finalized)
        return;

    double sum_time[SYM_V2_FACTOR_COUNT] = {};
    double max_time[SYM_V2_FACTOR_COUNT] = {};
    long long sum_count[SYM_V2_FACTOR_COUNT] = {};
    long long payload_sum_count[SYM_V2_PAYLOAD_COUNT]
        [SYM_V2_PAYLOAD_BIN_COUNT] = {};
    long long payload_sum_bytes[SYM_V2_PAYLOAD_COUNT]
        [SYM_V2_PAYLOAD_BIN_COUNT] = {};
    long long payload_max_bytes[SYM_V2_PAYLOAD_COUNT]
        [SYM_V2_PAYLOAD_BIN_COUNT] = {};
    struct { double val; int rank; } local_max_time[SYM_V2_FACTOR_COUNT];
    struct { double val; int rank; } global_max_time[SYM_V2_FACTOR_COUNT];
    int nranks = 1;
    int payload_slots =
        SYM_V2_PAYLOAD_COUNT * SYM_V2_PAYLOAD_BIN_COUNT;

    for (int i = 0; i < SYM_V2_FACTOR_COUNT; ++i)
    {
        local_max_time[i].val = symV2FactorProfileTime[i];
        local_max_time[i].rank = grid3d->iam;
    }

    MPI_Comm_size(grid3d->comm, &nranks);
    MPI_Reduce(symV2FactorProfileTime, sum_time, SYM_V2_FACTOR_COUNT,
               MPI_DOUBLE, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symV2FactorProfileTime, max_time, SYM_V2_FACTOR_COUNT,
               MPI_DOUBLE, MPI_MAX, 0, grid3d->comm);
    MPI_Reduce(local_max_time, global_max_time, SYM_V2_FACTOR_COUNT,
               MPI_DOUBLE_INT, MPI_MAXLOC, 0, grid3d->comm);
    MPI_Reduce(symV2FactorProfileCount, sum_count, SYM_V2_FACTOR_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(&symV2PayloadProfileCount[0][0], &payload_sum_count[0][0],
               payload_slots, MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(&symV2PayloadProfileBytes[0][0], &payload_sum_bytes[0][0],
               payload_slots, MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(&symV2PayloadProfileMaxBytes[0][0], &payload_max_bytes[0][0],
               payload_slots, MPI_LONG_LONG_INT, MPI_MAX, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    printf("SymFact GPU3D V2 factor-loop profile (GPU3DV2_FACTOR_PROFILE=1):\n");
    printf("  %-28s %12s %12s %12s %9s %12s\n",
           "phase", "sum(s)", "avg_rank(s)", "max_rank(s)", "rank", "calls");
    for (int i = 0; i < SYM_V2_FACTOR_COUNT; ++i)
    {
        if (sum_count[i] == 0 && sum_time[i] == 0.0 && max_time[i] == 0.0)
            continue;
        double avg = (nranks > 0) ? sum_time[i] / (double)nranks : 0.0;
        printf("  %-28s %12.6f %12.6f %12.6f %9d %12lld\n",
               labels[i], sum_time[i], avg, max_time[i],
               global_max_time[i].rank, sum_count[i]);
    }
    printf("SymFact GPU3D V2 payload histogram:\n");
    printf("  %-18s %-8s %12s %12s %12s %12s\n",
           "payload", "bin", "messages", "total_MB", "avg_KB", "max_MB");
    for (int i = 0; i < SYM_V2_PAYLOAD_COUNT; ++i)
    {
        for (int b = 0; b < SYM_V2_PAYLOAD_BIN_COUNT; ++b)
        {
            long long count = payload_sum_count[i][b];
            if (count == 0)
                continue;
            double total_mb =
                (double)payload_sum_bytes[i][b] / (1024.0 * 1024.0);
            double avg_kb =
                count > 0
                    ? (double)payload_sum_bytes[i][b] /
                          (double)count / 1024.0
                    : 0.0;
            double max_mb =
                (double)payload_max_bytes[i][b] / (1024.0 * 1024.0);
            printf("  %-18s %-8s %12lld %12.3f %12.3f %12.3f\n",
                   payload_labels[i], payload_bin_labels[b], count,
                   total_mb, avg_kb, max_mb);
        }
    }
    fflush(stdout);
}

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
template <typename Ftype>
void xLUstruct_t<Ftype>::printSymGPU3DTiming()
{
    static const char *labels_v1[SYM_GPU3D_T_COUNT] = {
        "l2u_start",
        "diag_d2h",
        "diag_d2h_copy",
        "diag_d2h_wait",
        "diag_prefetch_issue",
        "diag_prefetch_wait",
        "cpu_sytrf",
        "gpu_sytri",
        "gpu_sytri_validate",
        "cpu_sytri",
        "diag_pack",
        "diag_bcast",
        "inv_h2d",
        "ldiag_d2d",
        "lpanel_transform",
        "l2u_finish",
        "lfrag_exchange_total",
        "lfrag_pack_issue",
        "lfrag_d2h_stage_issue",
        "lfrag_pack_stage_sync",
        "lfrag_recv_post",
        "lfrag_mpi_recv_wait",
        "lfrag_h2d_stage_issue",
        "lfrag_assemble_issue",
        "lfrag_send_post",
        "lfrag_send_wait",
        "lfrag_stream_sync",
        "panel_bcast",
        "panel_bcast_mpi",
        "panel_index_d2h",
        "panel_bcast_singleton",
        "schur_update",
        "schur_sync",
        "lookahead_update",
        "lookahead_sync",
        "exclude_update",
        "sched_lookahead_dispatch",
        "sched_prefetch_ready",
        "sched_factor_dispatch",
        "sched_bcast_advance",
        "sched_final_sync",
        "sched_lookahead_bookkeep",
        "sched_factor_bookkeep",
        "sched_bcast_bookkeep",
        "sched_final_sync_bookkeep",
        "initial_factor_dispatch",
        "initial_panel_bcast",
        "factor_tree_wall"
    };
    static const char *labels_v2[SYM_GPU3D_T_COUNT] = {
        "lfrag_start",
        "diag_d2h",
        "diag_d2h_copy",
        "diag_d2h_wait",
        "diag_prefetch_issue",
        "diag_prefetch_wait",
        "cpu_sytrf",
        "gpu_sytri",
        "gpu_sytri_validate",
        "cpu_sytri",
        "diag_pack",
        "diag_bcast",
        "inv_h2d",
        "ldiag_d2d",
        "lpanel_transform",
        "lfrag_finish",
        "lfrag_exchange_total",
        "lfrag_pack_issue",
        "lfrag_d2h_stage_issue",
        "lfrag_pack_stage_sync",
        "lfrag_recv_post",
        "lfrag_mpi_recv_wait",
        "lfrag_h2d_stage_issue",
        "lfrag_assemble_issue",
        "lfrag_send_post",
        "lfrag_send_wait",
        "lfrag_stream_sync",
        "panel_bcast",
        "panel_bcast_mpi",
        "panel_index_d2h",
        "panel_bcast_singleton",
        "schur_update",
        "schur_sync",
        "lookahead_update",
        "lookahead_sync",
        "exclude_update",
        "sched_lookahead_dispatch",
        "sched_prefetch_ready",
        "sched_factor_dispatch",
        "sched_bcast_advance",
        "sched_final_sync",
        "sched_lookahead_bookkeep",
        "sched_factor_bookkeep",
        "sched_bcast_bookkeep",
        "sched_final_sync_bookkeep",
        "initial_factor_dispatch",
        "initial_panel_bcast",
        "factor_tree_wall"
    };
    static const char *stat_labels_v1[SYM_GPU3D_S_COUNT] = {
        "factor_trees",
        "factor_nodes",
        "initial_factor_nodes",
        "parent_factor_nodes",
        "lookahead_updates",
        "exclude_updates",
        "panel_bcasts",
        "panel_bcast_bytes",
        "panel_bcast_mpi_bytes",
        "panel_index_d2h_bytes",
        "l2u_local_bytes",
        "l2u_send_bytes",
        "l2u_recv_bytes",
        "l2u_host_staging_bytes",
        "l2u_cuda_aware_send_bytes",
        "l2u_recv_requests",
        "l2u_send_requests",
        "diag_d2h_bytes",
        "diag_prefetch_hits",
        "diag_prefetch_misses",
        "diag_prefetch_issues",
        "sched_windows",
        "sched_window_nodes",
        "sched_ready_bcasts",
        "sched_max_window",
        "sched_max_num_la"
    };
    static const char *stat_labels_v2[SYM_GPU3D_S_COUNT] = {
        "factor_trees",
        "factor_nodes",
        "initial_factor_nodes",
        "parent_factor_nodes",
        "lookahead_updates",
        "exclude_updates",
        "panel_bcasts",
        "panel_bcast_bytes",
        "panel_bcast_mpi_bytes",
        "panel_index_d2h_bytes",
        "lfrag_local_bytes",
        "lfrag_send_bytes",
        "lfrag_recv_bytes",
        "lfrag_host_staging_bytes",
        "lfrag_cuda_aware_send_bytes",
        "lfrag_recv_requests",
        "lfrag_send_requests",
        "diag_d2h_bytes",
        "diag_prefetch_hits",
        "diag_prefetch_misses",
        "diag_prefetch_issues",
        "sched_windows",
        "sched_window_nodes",
        "sched_ready_bcasts",
        "sched_max_window",
        "sched_max_num_la"
    };

    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (!mpi_initialized || mpi_finalized || grid3d == NULL)
        return;

    double sum_time[SYM_GPU3D_T_COUNT] = {};
    double max_time[SYM_GPU3D_T_COUNT] = {};
    long long sum_count[SYM_GPU3D_T_COUNT] = {};
    long long sum_stat[SYM_GPU3D_S_COUNT] = {};
    long long min_stat[SYM_GPU3D_S_COUNT] = {};
    long long max_stat[SYM_GPU3D_S_COUNT] = {};
    struct { double val; int rank; } local_max_time[SYM_GPU3D_T_COUNT];
    struct { double val; int rank; } global_max_time[SYM_GPU3D_T_COUNT];
    int nranks = 1;

    for (int i = 0; i < SYM_GPU3D_T_COUNT; ++i)
    {
        local_max_time[i].val = symGPU3DTime[i];
        local_max_time[i].rank = grid3d->iam;
    }

    MPI_Comm_size(grid3d->comm, &nranks);
    MPI_Reduce(symGPU3DTime, sum_time, SYM_GPU3D_T_COUNT, MPI_DOUBLE,
               MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DTime, max_time, SYM_GPU3D_T_COUNT, MPI_DOUBLE,
               MPI_MAX, 0, grid3d->comm);
    MPI_Reduce(local_max_time, global_max_time, SYM_GPU3D_T_COUNT,
               MPI_DOUBLE_INT, MPI_MAXLOC, 0, grid3d->comm);
    MPI_Reduce(symGPU3DCount, sum_count, SYM_GPU3D_T_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, sum_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, min_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_MIN, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, max_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_MAX, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    const char **labels = useSymV2Solve() ? labels_v2 : labels_v1;
    const char **stat_labels =
        useSymV2Solve() ? stat_labels_v2 : stat_labels_v1;
    printf("** SymFact GPU3D timing debug (SLU_ENABLE_SYM_GPU3D_TIMING) **\n");
    printf("   %-22s %12s %12s %9s %12s\n",
           "phase", "sum(s)", "max_rank(s)", "rank", "calls");
    for (int i = 0; i < SYM_GPU3D_T_COUNT; ++i)
    {
        if (sum_count[i] == 0 && sum_time[i] == 0.0 && max_time[i] == 0.0)
            continue;
        printf("   %-22s %12.6f %12.6f %9d %12lld\n",
               labels[i], sum_time[i], max_time[i],
               global_max_time[i].rank, sum_count[i]);
    }
    printf("** SymFact GPU3D rank stats (SLU_ENABLE_SYM_GPU3D_TIMING) **\n");
    printf("   %-28s %16s %16s %16s %16s\n",
           "stat", "sum", "avg_rank", "min_rank", "max_rank");
    for (int i = 0; i < SYM_GPU3D_S_COUNT; ++i)
    {
        if (sum_stat[i] == 0 && min_stat[i] == 0 && max_stat[i] == 0)
            continue;
        double avg = (nranks > 0) ? ((double)sum_stat[i] / (double)nranks) : 0.0;
        printf("   %-28s %16lld %16.2f %16lld %16lld\n",
               stat_labels[i], sum_stat[i], avg, min_stat[i], max_stat[i]);
    }
    fflush(stdout);
}
#endif

template <typename Ftype>
diagFactBufs_type<Ftype> **xLUstruct_t<Ftype>::initDiagFactBufsArr(int_t num_bufs, int_t ldt)
{

    // diagFactBufs_type<Ftype> **dFBufs = new diagFactBufs_type<Ftype> *[num_bufs]; // use SuperLU_MALLOC instead
    size_t ptr_bytes = xlu_checked_alloc_bytes(num_bufs,
                                               sizeof(diagFactBufs_type<Ftype> *),
                                               "diagonal factor buffer array");
    size_t block_bytes = xlu_checked_square_alloc_bytes(ldt, sizeof(Ftype),
                                                        "diagonal factor block");
    if (num_bufs == 0)
        return NULL;
    diagFactBufs_type<Ftype> **dFBufs = (diagFactBufs_type<Ftype> **)SUPERLU_MALLOC(ptr_bytes);
    if (dFBufs == NULL)
        ABORT("Malloc fails for diagonal factor buffer array.");
    for (int_t i = 0; i < num_bufs; i++)
    {
        // dFBufs[i] = new diagFactBufs_type<Ftype>; // use SuperLU_MALLOC instead
        dFBufs[i] = (diagFactBufs_type<Ftype> *)SUPERLU_MALLOC(sizeof(diagFactBufs_type<Ftype>));
        if (dFBufs[i] == NULL)
            ABORT("Malloc fails for diagonal factor buffers.");
        dFBufs[i]->BlockUFactor = (Ftype *)SUPERLU_MALLOC(block_bytes);
        dFBufs[i]->BlockLFactor = (Ftype *)SUPERLU_MALLOC(block_bytes);
        if (dFBufs[i]->BlockUFactor == NULL || dFBufs[i]->BlockLFactor == NULL)
            ABORT("Malloc fails for diagonal factor buffers.");
    }
    return dFBufs;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::freeDiagFactBufsArr(int_t num_bufs, diagFactBufs_type<Ftype> ** dFBufs)
{
    for (int i = 0; i < num_bufs; i++)
    {
        SUPERLU_FREE(dFBufs[i]->BlockUFactor);
        SUPERLU_FREE(dFBufs[i]->BlockLFactor);
        SUPERLU_FREE(dFBufs[i]);
    }
    /* Sherry fix:
     * mxLeafNode can be 0 for the replicated layers of the processes ?? */
    if ( num_bufs ) SUPERLU_FREE(dFBufs);

    return 0;
}


#ifdef HAVE_CUDA
template <typename Ftype>
xupanel_t<Ftype> xLUstruct_t<Ftype>::getKUpanel(int_t k, int_t offset)
{
    if (!needsUPanelStorage())
        ABORT("SymFact GPU3DVERSION=2 does not materialize U panels.");
    return (
        myrow == krow(k) ? 
        uPanelVec[g2lRow(k)] : 
        xupanel_t<Ftype>(UidxRecvBufs[offset], UvalRecvBufs[offset],
            A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset])
    );
}

template <typename Ftype>
xlpanel_t<Ftype> xLUstruct_t<Ftype>::getKLpanel(int_t k, int_t offset)
{ 
    int_t panel_root = symV2PanelRoot(k);
    return (
        mycol == panel_root ?
        lPanelVec[symV2PanelIndex(k)] :
        xlpanel_t<Ftype>(LidxRecvBufs[offset], LvalRecvBufs[offset],
            A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset])
    );
}

#endif

template <typename Ftype>
Ftype* getBigV(int_t ldt, int_t num_threads)
{
    Ftype *bigV;
    size_t bigv_bytes = xlu_checked_bigv_alloc_bytes(ldt, num_threads,
                                                     sizeof(Ftype),
                                                     "dgemm buffV");
    if (!(bigV = (Ftype*) SUPERLU_MALLOC (bigv_bytes)))
        ABORT ("Malloc failed for dgemm buffV");
    return bigV;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2ComputePartnerScratchSize(
    LUStruct_type<Ftype> *LUstruct)
{
    (void)LUstruct;
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2ComputePartnerScratchSize(
    LUStruct_type<double> *LUstruct)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    maxSymV2RowFragStageCount = 0;
    maxSymV2RowFragValRecvCount = 0;
    maxSymV2RowFragIdxRecvCount = 0;
    maxSymV2RowFragValSendCount = 0;
    if (Pr <= 1)
    {
        /* No partner-row fragment exchange is needed with one process row, but
           the LL Schur path still reuses this buffer as raw-panel workspace. */
        maxSymPartnerLvalCount = maxLvalCount;
        maxSymPartnerLidxCount = 0;
        return 0;
    }
    SymV2SetupProfileScope profile_scope(
        this, SYM_V2_SETUP_PARTNER_SCRATCH_SIZE);

    size_t partner_count_size = xlu_checked_product(
        static_cast<size_t>(nsupers), static_cast<size_t>(Pc),
        "SymFact V2 partner-L count table");
    if (partner_count_size >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 partner-L count table is too large for MPI.");
    size_t row_source_count_size = xlu_checked_product(
        static_cast<size_t>(nsupers), static_cast<size_t>(Pr),
        "SymFact V2 row-fragment source count table");
    if (row_source_count_size >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 row-fragment source count table is too large for MPI.");

    std::vector<long long> local_partner_val(partner_count_size, 0);
    std::vector<long long> global_partner_val(partner_count_size, 0);
    std::vector<long long> local_partner_meta(partner_count_size, 0);
    std::vector<long long> global_partner_meta(partner_count_size, 0);
    std::vector<long long> local_row_val(row_source_count_size, 0);
    std::vector<long long> global_row_val(row_source_count_size, 0);
    std::vector<long long> local_row_meta(row_source_count_size, 0);
    std::vector<long long> global_row_meta(row_source_count_size, 0);
    long long local_row_send_val = 0;
    long long max_row_send_val = 0;
    (void) LUstruct;

    for (int_t i = 0; i < symV2PanelCount(); ++i)
    {
        int_t k0 = symV2PanelGid(i);
        if (k0 >= nsupers || isNodeInMyGrid[k0] != 1)
            continue;

        int_t lk = symV2PanelIndex(k0);
        if (lk < 0)
            continue;
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (lpanel.isEmpty())
            continue;

        int_t knsupc = SuperSize(k0);
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
        for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = symV2PanelRoot(ik);
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;
            long long block_values =
                static_cast<long long>(len) *
                static_cast<long long>(knsupc);
            size_t col_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(Pc) +
                             static_cast<size_t>(ikcol);
            size_t row_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(Pr) +
                             static_cast<size_t>(myrow);
            local_partner_val[col_pos] += block_values;
            local_row_val[row_pos] += block_values;
            local_partner_meta[col_pos] += static_cast<long long>(len) + 2;
            local_row_meta[row_pos] += static_cast<long long>(len) + 2;
            local_row_send_val = SUPERLU_MAX(local_row_send_val,
                                             block_values);
        }
    }

    MPI_Allreduce(local_partner_val.data(), global_partner_val.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, grid->comm);
    MPI_Allreduce(local_partner_meta.data(), global_partner_meta.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, grid->comm);
    MPI_Allreduce(local_row_val.data(), global_row_val.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, grid->comm);
    MPI_Allreduce(local_row_meta.data(), global_row_meta.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, grid->comm);
    MPI_Allreduce(&local_row_send_val, &max_row_send_val, 1,
                  MPI_LONG_LONG, MPI_MAX, grid->comm);

    long long max_partner_val = 0;
    long long max_partner_meta = 0;
    long long max_row_val = 0;
    long long max_row_meta = 0;
    for (int_t k0 = 0; k0 < nsupers; ++k0)
    {
        for (int pc = 0; pc < Pc; ++pc)
        {
            size_t pos = static_cast<size_t>(k0) *
                             static_cast<size_t>(Pc) +
                         static_cast<size_t>(pc);
            max_partner_val =
                SUPERLU_MAX(max_partner_val, global_partner_val[pos]);
            max_partner_meta =
                SUPERLU_MAX(max_partner_meta, global_partner_meta[pos]);
        }
        if (superlu_sym_v2_pc_fragment_schur() && Pc > 1)
        {
            for (int pr = 0; pr < Pr; ++pr)
            {
                size_t pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(Pr) +
                             static_cast<size_t>(pr);
                max_row_val =
                    SUPERLU_MAX(max_row_val, global_row_val[pos]);
                max_row_meta =
                    SUPERLU_MAX(max_row_meta, global_row_meta[pos]);
            }
        }
    }
    long long max_partner_idx = (max_partner_meta > 0)
                                    ? max_partner_meta + LPANEL_HEADER_SIZE + 1
                                    : 0;
    long long max_row_idx = (max_row_meta > 0)
                                ? max_row_meta + LPANEL_HEADER_SIZE + 1
                                : 0;
    if (max_partner_val >
            static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_partner_idx >
            static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_val >
            static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_idx >
            static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_send_val >
            static_cast<long long>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 partner-L scratch size exceeds int_t range.");

    maxSymPartnerLvalCount = static_cast<int_t>(max_partner_val);
    maxSymPartnerLidxCount = static_cast<int_t>(max_partner_idx);
    maxSymV2RowFragStageCount = static_cast<int_t>(max_row_val);
    maxSymV2RowFragValRecvCount = static_cast<int_t>(max_row_val);
    maxSymV2RowFragIdxRecvCount = static_cast<int_t>(max_row_idx);
    maxSymV2RowFragValSendCount = static_cast<int_t>(max_row_send_val);
    return 0;
}

/* Constructor */
template <typename Ftype>
xLUstruct_t<Ftype>::xLUstruct_t(int_t nsupers_, int_t ldt_,
                             trf3dpartitionType<Ftype> *trf3Dpartition_, 
                             LUStruct_type<Ftype> *LUstruct,
                             gridinfo3d_t *grid3d_in,
                             SCT_t *SCT_, superlu_dist_options_t *options_,
                             SuperLUStat_t *stat_, 
                             threshPivValType<Ftype> thresh_, int *info_) :
                             nsupers(nsupers_), trf3Dpartition(trf3Dpartition_),
                             ldt(ldt_), /* maximum supernode size */
				     grid3d(grid3d_in), SCT(SCT_),
				     options(options_), stat(stat_),
	                             LUstructPtr(LUstruct), symL2UOrders(NULL),
	                             symFactWork(NULL), symFactIPIV(NULL),
	                             symFactWorkSize(0), symFactTagUb(0),
	                             thresh(thresh_), info(info_), anc25d(grid3d_in)
{
    xlu_sym_gpu3d_trace(grid3d, "enter xLUstruct_t constructor");
#ifdef HAVE_CUDA
    symV2PartnerLSendBufPoolGPU = NULL;
    symL2LSendMapPoolGPU = NULL;
    symV2PartnerLExactSendBufPoolGPU = NULL;
    symV2PartnerLExactSendMapPoolGPU = NULL;
    symV2RowFragExactSendBufPoolGPU = NULL;
    symV2RowFragExactSendMapPoolGPU = NULL;
    symV2PartnerLRecvMapPoolGPU = NULL;
    symV2RowFragRecvMapPoolGPU = NULL;
    symV2PartnerLSendBufPoolCount = 0;
    symL2LSendMapPoolCount = 0;
    symV2PartnerLExactSendBufPoolCount = 0;
    symV2PartnerLExactSendMapPoolCount = 0;
    symV2RowFragExactSendBufPoolCount = 0;
    symV2RowFragExactSendMapPoolCount = 0;
    symV2PartnerLRecvMapPoolCount = 0;
    symV2RowFragRecvMapPoolCount = 0;
#endif
    grid = &(grid3d->grid2d);
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    mycol = MYCOL(iam, grid);
    symGPU3DVersion = (options->SymFact == YES) ? xlu_gpu3d_version() : 0;
    {
        const char *profile_env = std::getenv("GPU3DV2_PROFILE");
        const char *factor_profile_env =
            std::getenv("GPU3DV2_FACTOR_PROFILE");
        int profile_on =
            (profile_env != NULL && profile_env[0] != '\0' &&
             profile_env[0] != '0');
        int factor_profile_on =
            (factor_profile_env != NULL && factor_profile_env[0] != '\0')
                ? (factor_profile_env[0] != '0')
                : profile_on;
        symV2SetupProfileEnabled =
            (options->SymFact == YES && symGPU3DVersion == 2 &&
             profile_on);
        symV2FactorProfileEnabled =
            (options->SymFact == YES && symGPU3DVersion == 2 &&
             factor_profile_on);
    }
    maxLvl = symV2ForestLevelCount();
    double tSetupNodeMask = SuperLU_timer_();
    if (symV2ScheduleActive())
    {
        isNodeInMyGrid = int32Calloc_dist((int) nsupers);
        if (isNodeInMyGrid == NULL)
            ABORT("Calloc fails for SymFact V2 local node mask.");
        for (int_t k = 0; k < nsupers; ++k)
        {
            if (trf3Dpartition->superGridMap != NULL &&
                trf3Dpartition->superGridMap[k] != NOT_IN_GRID)
                isNodeInMyGrid[k] = 1;
        }
    }
    else
    {
        isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl,
                                           trf3Dpartition->myNodeCount,
                                           trf3Dpartition->treePerm);
    }
    symV2SetupProfileAdd(SYM_V2_SETUP_NODE_MASK,
                         SuperLU_timer_() - tSetupNodeMask);
    superlu_acc_offload = sp_ienv_dist(10, options); // get_acc_offload();
    xlu_sym_gpu3d_trace(grid3d, "constructor after isNodeInMyGrid");

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Enter xLUstruct_t constructor");
#endif
    if (options->SymFact == YES)
    {
        symFactTagUb = set_tag_ub();
        if (symFactTagUb <= 0)
            ABORT("Invalid MPI tag upper bound for SymFact communication.");
        if (symGPU3DVersion != 2)
        {
            if (options->CommL != YES)
                ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        }
        else
        {
            if (options->batchCount > 0)
                ABORT("SymFact GPU3DVERSION=2 does not support batchCount>0 until LDL-native batch sizing is implemented.");
            symV2DiagBlocks.assign(nsupers, NULL);
#ifdef HAVE_CUDA
            symV2DiagBlocksGPU.assign(nsupers, NULL);
#endif
        }
    }
    xsup = LUstruct->Glu_persist->xsup;
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    Ftype **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;
    Ftype **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    int_t localPanelCount = symV2PanelCount();
    int_t localRowCount = symV2RowCount();
    int_t panelVecCount = SUPERLU_MAX((int_t)1, localPanelCount);
    int_t rowVecCount = SUPERLU_MAX((int_t)1, localRowCount);
    bool sym_v2_mode = (options->SymFact == YES && symGPU3DVersion == 2);
    bool need_u_panel_storage = needsUPanelStorage();
    double tSetupPanelVec = SuperLU_timer_();
    lPanelVec = new xlpanel_t<Ftype>[panelVecCount];
    uPanelVec = need_u_panel_storage ? new xupanel_t<Ftype>[rowVecCount] : NULL;
    xlu_sym_gpu3d_trace(grid3d, "constructor after panel vector allocation");
    // create the lvectors
    maxLvalCount = 0;
    maxLidxCount = 0;
    maxSymPartnerLvalCount = 0;
    maxSymPartnerLidxCount = 0;
    maxSymV2RowFragStageCount = 0;
    maxSymV2RowFragValRecvCount = 0;
    maxSymV2RowFragIdxRecvCount = 0;
    maxSymV2RowFragValSendCount = 0;
    maxUvalCount = 0;
    maxUidxCount = 0;

    std::vector<int_t> localLvalSendCounts(panelVecCount, 0);
    std::vector<int_t> localUvalSendCounts(rowVecCount, 0);
    std::vector<int_t> localLidxSendCounts(panelVecCount, 0);
    std::vector<int_t> localUidxSendCounts(rowVecCount, 0);
    for (int_t i = 0; i < localPanelCount; ++i)
    {
        int_t k0 = symV2PanelGid(i);
        int_t *lsub = NULL;
        Ftype *lval = NULL;
        int_t src_lk = sym_v2_mode ? i : LBj(k0, grid);
        lsub = Lrowind_bc_ptr[src_lk];
        lval = Lnzval_bc_ptr[src_lk];
        if (lsub != NULL && isNodeInMyGrid[k0] == 1)
        {
            int_t isDiagIncluded = 0;

            if (myrow == symV2DiagRoot(k0))
                isDiagIncluded = 1;
            xlpanel_t<Ftype> lpanel(k0, lsub, lval, xsup, isDiagIncluded);
            lPanelVec[i] = lpanel;
            maxLvalCount = std::max(lPanelVec[i].nzvalSize(), maxLvalCount);
            maxLidxCount = std::max(lPanelVec[i].indexSize(), maxLidxCount);
            localLvalSendCounts[i] = lPanelVec[i].nzvalSize();
            localLidxSendCounts[i] = lPanelVec[i].indexSize();
        }
    }

    // create the vectors
    for (int_t i = 0; i < localRowCount; ++i)
    {
        int_t globalId = symV2RowGid(i);
        if (need_u_panel_storage &&
            Ufstnz_br_ptr != NULL && Unzval_br_ptr != NULL &&
            Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[globalId] == 1)
        {
            xupanel_t<Ftype> upanel(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
            uPanelVec[i] = upanel;
            maxUvalCount = std::max(uPanelVec[i].nzvalSize(), maxUvalCount);
            maxUidxCount = std::max(uPanelVec[i].indexSize(), maxUidxCount);
            localUvalSendCounts[i] = uPanelVec[i].nzvalSize();
            localUidxSendCounts[i] = uPanelVec[i].indexSize();
        }
    }
    symV2SetupProfileAdd(SYM_V2_SETUP_PANEL_VEC_BUILD,
                         SuperLU_timer_() - tSetupPanelVec);

    // compute the send sizes
    // send and recv count for 2d comm
    LvalSendCounts.resize(nsupers);
    UvalSendCounts.resize(nsupers);
    LidxSendCounts.resize(nsupers);
    UidxSendCounts.resize(nsupers);

    std::vector<int_t> recvBuf(std::max(rowVecCount, panelVecCount), 0);

    double tSetupSendCounts = SuperLU_timer_();
    if (!sym_v2_mode)
    {
        for (int pr = 0; pr < Pr; pr++)
        {
            int npr = CEILING(nsupers, Pr);
            std::copy(localUvalSendCounts.begin(), localUvalSendCounts.end(), recvBuf.begin());
            // Send the value counts ;
            MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
            for (int i = 0; i * Pr + pr < nsupers; i++)
            {
                UvalSendCounts[i * Pr + pr] = recvBuf[i];
            }

            std::copy(localUidxSendCounts.begin(), localUidxSendCounts.end(), recvBuf.begin());
            // send the index count
            MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
            for (int i = 0; i * Pr + pr < nsupers; i++)
            {
                UidxSendCounts[i * Pr + pr] = recvBuf[i];
            }
        }
    }

    if (sym_v2_mode)
    {
        std::vector<int_t> localLvalBySuper(nsupers, 0);
        std::vector<int_t> localLidxBySuper(nsupers, 0);
        for (int_t i = 0; i < localPanelCount; ++i)
        {
            int_t k0 = symV2PanelGid(i);
            if (!lPanelVec[i].isEmpty())
            {
                localLvalBySuper[k0] = lPanelVec[i].nzvalSize();
                localLidxBySuper[k0] = lPanelVec[i].indexSize();
            }
        }
        MPI_Allreduce(localLvalBySuper.data(), LvalSendCounts.data(),
                      nsupers, mpi_int_t, MPI_SUM, grid3d->rscp.comm);
        MPI_Allreduce(localLidxBySuper.data(), LidxSendCounts.data(),
                      nsupers, mpi_int_t, MPI_SUM, grid3d->rscp.comm);
    }
    else
    {
    for (int pc = 0; pc < Pc; pc++)
    {
        int npc = CEILING(nsupers, Pc);
        std::copy(localLvalSendCounts.begin(), localLvalSendCounts.end(), recvBuf.begin());
        // Send the value counts ;
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LvalSendCounts[i * Pc + pc] = recvBuf[i];
        }

        std::copy(localLidxSendCounts.begin(), localLidxSendCounts.end(), recvBuf.begin());
        // send the index count
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LidxSendCounts[i * Pc + pc] = recvBuf[i];
        }
    }
    }

    maxUvalCount = sym_v2_mode ? 0 : *std::max_element(UvalSendCounts.begin(), UvalSendCounts.end());
    maxUidxCount = sym_v2_mode ? 0 : *std::max_element(UidxSendCounts.begin(), UidxSendCounts.end());
    maxLvalCount = *std::max_element(LvalSendCounts.begin(), LvalSendCounts.end());
    maxLidxCount = *std::max_element(LidxSendCounts.begin(), LidxSendCounts.end());
    maxSymPartnerLvalCount = sym_v2_mode ? 0 : maxLvalCount;
    maxSymPartnerLidxCount = sym_v2_mode ? 0 : maxLidxCount;
    dSymV2ComputePartnerScratchSize(LUstruct);
    symV2SetupProfileAdd(SYM_V2_SETUP_SEND_COUNT_EXCHANGE,
                         SuperLU_timer_() - tSetupSendCounts);
#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
    std::printf("[sym-gpu3d-trace] rank %d: constructor counts nsupers=%lld Pr=%lld Pc=%lld numLA=%d maxLval=%lld maxUval=%lld maxLidx=%lld maxUidx=%lld maxSymPartnerLval=%lld maxSymPartnerLidx=%lld maxRowFragStage=%lld maxRowFragVal=%lld maxRowFragIdx=%lld\n",
                (grid3d != NULL) ? grid3d->iam : -1,
                (long long)nsupers, (long long)Pr, (long long)Pc,
                options->num_lookaheads,
                (long long)maxLvalCount, (long long)maxUvalCount,
                (long long)maxLidxCount, (long long)maxUidxCount,
                (long long)maxSymPartnerLvalCount,
                (long long)maxSymPartnerLidxCount,
                (long long)maxSymV2RowFragStageCount,
                (long long)maxSymV2RowFragValRecvCount,
                (long long)maxSymV2RowFragIdxRecvCount);
    std::fflush(stdout);
#endif

    // Allocate bigV, indirect
    double tSetupCPUWorkspace = SuperLU_timer_();
    nThreads = getNumThreads(iam);
    // bigV = dgetBigV(ldt, nThreads);
    bigV = getBigV<Ftype>(ldt, nThreads);
    if (nThreads < 0 || ldt < 0)
        ABORT("Negative allocation size.");
    size_t indirect_count = xlu_checked_product(static_cast<size_t>(nThreads),
                                                static_cast<size_t>(ldt),
                                                "panel indirect workspace");
    size_t indirect_bytes = xlu_checked_product(indirect_count, sizeof(int_t),
                                                "panel indirect workspace");
    indirect = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    indirectRow = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    indirectCol = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    if (indirect == NULL || indirectRow == NULL || indirectCol == NULL)
        ABORT("Malloc fails for panel indirect workspace.");
    symV2SetupProfileAdd(SYM_V2_SETUP_CPU_WORKSPACE_ALLOC,
                         SuperLU_timer_() - tSetupCPUWorkspace);
    xlu_sym_gpu3d_trace(grid3d, "constructor after indirect workspace allocation");

    // allocating communication buffers
    double tSetupRecvBuffers = SuperLU_timer_();
    LvalRecvBufs.resize(options->num_lookaheads);
    UvalRecvBufs.resize(options->num_lookaheads);
    symPartnerLvalRecvBufs.resize(options->num_lookaheads);
    symV2RowFragHostRecvBufs.resize(options->num_lookaheads);
    LidxRecvBufs.resize(options->num_lookaheads);
    UidxRecvBufs.resize(options->num_lookaheads);
    symPartnerLidxRecvBufs.resize(options->num_lookaheads);
    // bcastLval.resize(options->num_lookaheads);
    // bcastUval.resize(options->num_lookaheads);
    // bcastLidx.resize(options->num_lookaheads);
    // bcastUidx.resize(options->num_lookaheads);

    int_t u_recv_val_count = sym_v2_mode ? 0 : maxUvalCount;
    int_t u_recv_idx_count = sym_v2_mode ? 0 : maxUidxCount;
    const bool sym_v2_pc_fragment_schur =
        sym_v2_mode && superlu_sym_v2_pc_fragment_schur() &&
        Pr > 1 && Pc > 1;

#ifdef HAVE_CUDA
    const bool use_sym_v2_pooled_pinned_staging =
        sym_v2_mode && superlu_acc_offload &&
        !superlu_cuda_aware_mpi() &&
        superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool();
#else
    const bool use_sym_v2_pooled_pinned_staging = false;
#endif
#ifdef HAVE_CUDA
    if (use_sym_v2_pooled_pinned_staging &&
        maxSymPartnerLvalCount > 0 && options->num_lookaheads > 0)
    {
        const size_t pooled_recv_bytes = xlu_checked_alloc_bytes(
            maxSymPartnerLvalCount, sizeof(Ftype),
            "SymFact V2 pooled pinned receive staging");
        gpuErrchk(cudaMallocHost(
            (void **)&symV2PartnerLHostRecvPoolPinned,
            pooled_recv_bytes));
        symV2PartnerLHostRecvPoolPinnedCount =
            static_cast<size_t>(maxSymPartnerLvalCount);
        symV2PartnerLHostRecvPinned = 1;
    }
    if (use_sym_v2_pooled_pinned_staging &&
        sym_v2_pc_fragment_schur &&
        maxSymV2RowFragStageCount > 0 &&
        options->num_lookaheads > 0)
    {
        const size_t pooled_row_recv_bytes = xlu_checked_alloc_bytes(
            maxSymV2RowFragStageCount, sizeof(Ftype),
            "SymFact V2 pooled pinned row-fragment receive staging");
        gpuErrchk(cudaMallocHost(
            (void **)&symV2RowFragHostRecvPoolPinned,
            pooled_row_recv_bytes));
        symV2RowFragHostRecvPoolPinnedCount =
            static_cast<size_t>(maxSymV2RowFragStageCount);
        symV2RowFragHostRecvPinned = 1;
    }
#endif

    for (int i = 0; i < options->num_lookaheads; i++)
    {
        size_t lval_bytes = xlu_checked_alloc_bytes(maxLvalCount, sizeof(Ftype),
                                                    "L value receive buffer");
        size_t uval_bytes = xlu_checked_alloc_bytes(u_recv_val_count, sizeof(Ftype),
                                                    "U value receive buffer");
        size_t sym_partner_lval_bytes =
            xlu_checked_alloc_bytes(maxSymPartnerLvalCount, sizeof(Ftype),
                                    "SymFact V2 partner-L value receive buffer");
        size_t sym_row_frag_lval_bytes =
            xlu_checked_alloc_bytes(maxSymV2RowFragStageCount, sizeof(Ftype),
                                    "SymFact V2 row-fragment host receive buffer");
        size_t lidx_bytes = xlu_checked_alloc_bytes(maxLidxCount, sizeof(int_t),
                                                    "L index receive buffer");
        size_t uidx_bytes = xlu_checked_alloc_bytes(u_recv_idx_count, sizeof(int_t),
                                                    "U index receive buffer");
        size_t sym_partner_lidx_bytes =
            xlu_checked_alloc_bytes(maxSymPartnerLidxCount, sizeof(int_t),
                                    "SymFact V2 partner-L index receive buffer");
        LvalRecvBufs[i] = lval_bytes ? (Ftype *)SUPERLU_MALLOC(lval_bytes) : NULL;
        UvalRecvBufs[i] = uval_bytes ? (Ftype *)SUPERLU_MALLOC(uval_bytes) : NULL;
        symPartnerLvalRecvBufs[i] = NULL;
        symV2RowFragHostRecvBufs[i] = NULL;
#ifdef HAVE_CUDA
        if (symV2PartnerLHostRecvPoolPinned != NULL)
        {
            symPartnerLvalRecvBufs[i] = symV2PartnerLHostRecvPoolPinned;
        }
        else if (sym_partner_lval_bytes && sym_v2_mode &&
                 superlu_acc_offload &&
                 superlu_sym_v2_pinned_staging())
        {
            gpuErrchk(cudaMallocHost(
                (void **)&symPartnerLvalRecvBufs[i], sym_partner_lval_bytes));
            symV2PartnerLHostRecvPinned = 1;
        }
        else
#endif
        if (sym_partner_lval_bytes)
        {
            symPartnerLvalRecvBufs[i] =
                (Ftype *)SUPERLU_MALLOC(sym_partner_lval_bytes);
        }
#ifdef HAVE_CUDA
        if (symV2RowFragHostRecvPoolPinned != NULL)
            symV2RowFragHostRecvBufs[i] = symV2RowFragHostRecvPoolPinned;
        else if (sym_row_frag_lval_bytes && sym_v2_pc_fragment_schur &&
                 superlu_acc_offload &&
                 superlu_sym_v2_pinned_staging())
        {
            gpuErrchk(cudaMallocHost(
                (void **)&symV2RowFragHostRecvBufs[i],
                sym_row_frag_lval_bytes));
            symV2RowFragHostRecvPinned = 1;
        }
        else
#endif
        if (sym_row_frag_lval_bytes && sym_v2_pc_fragment_schur)
        {
            symV2RowFragHostRecvBufs[i] =
                (Ftype *)SUPERLU_MALLOC(sym_row_frag_lval_bytes);
        }
        LidxRecvBufs[i] = lidx_bytes ? (int_t *)SUPERLU_MALLOC(lidx_bytes) : NULL;
        UidxRecvBufs[i] = uidx_bytes ? (int_t *)SUPERLU_MALLOC(uidx_bytes) : NULL;
        symPartnerLidxRecvBufs[i] =
            sym_partner_lidx_bytes ? (int_t *)SUPERLU_MALLOC(sym_partner_lidx_bytes) : NULL;
        if ((lval_bytes != 0 && LvalRecvBufs[i] == NULL) ||
            (uval_bytes != 0 && UvalRecvBufs[i] == NULL) ||
            (sym_partner_lval_bytes != 0 &&
             symPartnerLvalRecvBufs[i] == NULL) ||
            (lidx_bytes != 0 && LidxRecvBufs[i] == NULL) ||
            (uidx_bytes != 0 && UidxRecvBufs[i] == NULL) ||
            (sym_partner_lidx_bytes != 0 &&
             symPartnerLidxRecvBufs[i] == NULL) ||
            (sym_v2_pc_fragment_schur &&
             sym_row_frag_lval_bytes != 0 &&
             symV2RowFragHostRecvBufs[i] == NULL))
            ABORT("Malloc fails for panel receive buffers.");

        //TODO: check if setup correctly
        #pragma warning disabling bcaststruct 
        #if 0
        bcastStruct bcLval(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        bcastLval[i] = bcLval;
        bcastStruct bcUval(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        bcastUval[i] = bcUval;
        bcastStruct bcLidx(grid3d->rscp.comm, mpi_int_t, SYNC);
        bcastLidx[i] = bcLidx;
        bcastStruct bcUidx(grid3d->cscp.comm, mpi_int_t, SYNC);
        bcastUidx[i] = bcUidx;
        #endif
    }
    symV2SetupProfileAdd(SYM_V2_SETUP_RECV_BUFFER_ALLOC,
                         SuperLU_timer_() - tSetupRecvBuffers);
    xlu_sym_gpu3d_trace(grid3d, "constructor after panel receive buffer allocation");

    double tSetupDiagBuffers = SuperLU_timer_();
    numDiagBufs = 2*options->num_lookaheads;
    diagFactBufs.resize(numDiagBufs);  /* Sherry?? numDiagBufs == 32 hard-coded */
    // bcastDiagRow.resize(numDiagBufs);
    // bcastDiagCol.resize(numDiagBufs);

    int_t diagBufDim = ldt;
    if (sym_v2_mode && useSymV2Solve())
    {
        diagBufDim = 1;
        for (int_t k = 0; k < nsupers; ++k)
        {
            if (isNodeInMyGrid[k] == 1 && symV2PanelRoot(k) == mycol)
                diagBufDim = SUPERLU_MAX(diagBufDim, SuperSize(k));
        }
    }

    for (int i = 0; i < numDiagBufs; i++) /* Sherry?? these strcutures not used */
    {
        diagFactBufs[i] = (Ftype *)SUPERLU_MALLOC(
            xlu_checked_square_alloc_bytes(diagBufDim, sizeof(Ftype), "diagonal factor buffer"));
        if (diagFactBufs[i] == NULL)
            ABORT("Malloc fails for diagonal factor buffer.");
        // bcastStruct bcDiagRow(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        // bcastDiagRow[i] = bcDiagRow;
        // bcastStruct bcDiagCol(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        // bcastDiagCol[i] = bcDiagCol;
    }
    xlu_sym_gpu3d_trace(grid3d, "constructor after diagonal buffer allocation");

    int mxLeafNode = 0;
    if (sym_v2_mode && symV2ScheduleActive())
    {
        mxLeafNode = trf3Dpartition->mxLeafNode;
    }
    else
    {
        int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
        sForest_t **sForests = trf3Dpartition->sForests;
        for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
        {
            if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
                mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
        }
    }
    //Yang: how is dFBufs being used in the c++ factorization code? Shall we call dinitDiagFactBufsArrMod instead to save memory? 
    dFBufs = initDiagFactBufsArr(numDiagBufs, diagBufDim);
    maxLeafNodes = mxLeafNode;
    symV2SetupProfileAdd(SYM_V2_SETUP_DIAG_BUFFER_ALLOC,
                         SuperLU_timer_() - tSetupDiagBuffers);
    xlu_sym_gpu3d_trace(grid3d, "constructor after dFBufs allocation");

    xlu_sym_gpu3d_trace(grid3d, "constructor before initSymFactWorkspace");
    double tSetupSymWorkspace = SuperLU_timer_();
    initSymFactWorkspace();
    symV2SetupProfileAdd(SYM_V2_SETUP_INIT_SYM_WORKSPACE,
                         SuperLU_timer_() - tSetupSymWorkspace);
    xlu_sym_gpu3d_trace(grid3d, "constructor after initSymFactWorkspace");
    
    double tGPU = SuperLU_timer_();
    if(superlu_acc_offload)
    {
    #ifdef HAVE_CUDA
        xlu_sym_gpu3d_trace(grid3d, "constructor before setLUstruct_GPU");
        double tSetupGPU = SuperLU_timer_();
        setLUstruct_GPU();  /* Set up LU structure and buffers on GPU */
        symV2SetupProfileAdd(SYM_V2_SETUP_SET_GPU_TOTAL,
                             SuperLU_timer_() - tSetupGPU);
        xlu_sym_gpu3d_trace(grid3d, "constructor after setLUstruct_GPU");

        // TODO: remove it, checking is very slow
        if(0)
            checkGPU();     
    #endif
    }
        
    tGPU = SuperLU_timer_() -tGPU;
#if ( PRNTlevel >= 1 )    
    printf("Time to intialize GPU DS= %g\n",tGPU );
#endif

    // if (superluAccOffload)

    // for(int pc=0;pc<Pc; pc++)
    // {
    //     MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    //     ...
    // }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Exit xLUstruct_t constructor");
#endif
    
} /* constructor xLUstruct_t */

template <typename Ftype>
int xLUstruct_t<Ftype>::initSymFactWorkspace()
{
    return 0;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::freeSymFactWorkspace()
{
    return 0;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::ensureSymFactWorkSize(int64_t minSize)
{
    return 0;
}

template <>
inline int xLUstruct_t<double>::initSymFactWorkspace()
{
    if (options->SymFact != YES)
        return 0;

    xlu_sym_gpu3d_trace(grid3d, "enter initSymFactWorkspace");
    int profile_setup = symV2SetupProfileActive() ? 1 : 0;
    symGPU3DContract = xlu_gpu3d_contract();
    symContractValidateTol = (symGPU3DContract == 1)
        ? xlu_env_double("GPU3DCONTRACT_VALIDATE_TOL", 1.0e8)
        : 1.0e8;
    symContract1Accepted = 0;
    symContract1Fallbacks = 0;
    symContract1MaxResid = 0.0;
    bool need_l2u_workspace = !useSymV2Solve();

    if (ldt < 0 || maxLvalCount < 0)
        ABORT("Negative SymFact workspace size.");
    double tSymCPUWorkspace = SuperLU_timer_();
    size_t diag_work_count = xlu_checked_product(static_cast<size_t>(ldt),
                                                 static_cast<size_t>(ldt),
                                                 "SymFact work");
    int64_t diag_work_size = (int64_t)diag_work_count;
    if (diag_work_size < 0 || (size_t)diag_work_size != diag_work_count)
        ABORT("SymFact workspace size overflows int64_t.");
    int64_t panel_work_size = (int64_t)maxLvalCount;
    int64_t workspace_size = SUPERLU_MAX(diag_work_size, panel_work_size);
    if (workspace_size <= 0)
        workspace_size = 1;

    symFactWorkSize = workspace_size;
    size_t work_count = (size_t)workspace_size;
    if ((int64_t)work_count != workspace_size)
        ABORT("SymFact workspace size overflows allocation size.");
    symFactWork = (double *)SUPERLU_MALLOC(
        xlu_checked_product(work_count, sizeof(double), "SymFact work"));
    if (symFactWork == NULL)
        ABORT("Malloc fails for SymFact work[].");

    size_t ipiv_count = (size_t)ldt;
    if ((int_t)ipiv_count != ldt)
        ABORT("SymFact IPIV size overflows allocation size.");
    symFactIPIV = (int *)SUPERLU_MALLOC(
        xlu_checked_product(ipiv_count, sizeof(int), "SymFact IPIV"));
    if (symFactIPIV == NULL)
        ABORT("Malloc fails for SymFact IPIV[].");

    if (need_l2u_workspace)
    {
        size_t order_count = xlu_checked_product(2, ipiv_count, "SymFact L2U order");
        symL2UOrders = (int *)SUPERLU_MALLOC(
            xlu_checked_product(order_count, sizeof(int), "SymFact L2U order"));
        if (symL2UOrders == NULL)
            ABORT("Malloc fails for SymFact L2U order workspace.");
    }
    symV2SetupProfileAdd(SYM_V2_SETUP_SYM_CPU_WORKSPACE,
                         SuperLU_timer_() - tSymCPUWorkspace);
    xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after CPU workspace allocation");

#ifdef HAVE_CUDA
    if (superlu_acc_offload)
    {
#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
        std::printf("[sym-gpu3d-trace] rank %d: initSymFactWorkspace GPU setup local_cols=%lld local_rows=%lld Pc=%lld Pr=%lld contract=%d\n",
                    (grid3d != NULL) ? grid3d->iam : -1,
                    (long long)symV2PanelCount(),
                    (long long)symV2RowCount(),
                    (long long)Pc, (long long)Pr, symGPU3DContract);
        std::fflush(stdout);
#endif
        int_t local_cols = symV2PanelCount();
        size_t l2u_slots = xlu_checked_product(static_cast<size_t>(local_cols),
                                               static_cast<size_t>(Pc),
                                               "SymFact GPU L2U buffers");
        if (need_l2u_workspace)
        {
            symL2USendBufsGPU.assign(l2u_slots, NULL);
            symL2USendMapsGPU.assign(l2u_slots, NULL);
            symL2ULocalMapsGPU.assign(CEILING(nsupers, Pr), NULL);
        }
        symV2PartnerLSendBufsGPU.assign(l2u_slots, NULL);
        symL2LSendMapsGPU.assign(l2u_slots, NULL);
        symL2LSendMeta.assign(l2u_slots, std::vector<int_t>());
        symV2PartnerLHostSendBufs.assign(l2u_slots, std::vector<double>());
        symV2PartnerLHostSendBufsPinned.assign(l2u_slots, NULL);
        symV2PartnerLHostSendScratchOffsets.assign(l2u_slots, 0);
        symV2ExchangeSendSizesScratch.assign(static_cast<size_t>(Pc), 0);
        symV2ExchangeRecvSizesScratch.assign(static_cast<size_t>(Pr), 0);
        symV2ExchangeRecvOffsetsScratch.assign(static_cast<size_t>(Pr), -1);
        symV2ExchangeRecvReqsScratch.clear();
        symV2ExchangeRecvReqsScratch.reserve(static_cast<size_t>(Pr));
        symV2ExchangeSendReqsScratch.clear();
        symV2ExchangeSendReqsScratch.reserve(xlu_checked_product(
            static_cast<size_t>(Pr), static_cast<size_t>(Pc),
            "SymFact V2 pooled MPI request scratch"));
        symV2ExchangeRecvPeersScratch.clear();
        symV2ExchangeRecvPeersScratch.reserve(static_cast<size_t>(Pr));
        symV2ExchangeWaitIndicesScratch.assign(static_cast<size_t>(Pr), 0);
        symV2ExchangeWaitStatusesScratch.resize(static_cast<size_t>(Pr));
        symV2PartnerLSendSizes.assign(l2u_slots, 0);
        symV2PartnerLSendRowActive.assign(
            xlu_checked_product(l2u_slots, static_cast<size_t>(Pr),
                                "SymFact V2 partner-L send row activity"),
            0);
        symV2RowFragSendActive.assign(
            xlu_checked_product(l2u_slots, static_cast<size_t>(Pc),
                                "SymFact V2 row-fragment send activity"),
            0);
        size_t partner_exact_slots = xlu_checked_product(
            l2u_slots, static_cast<size_t>(Pr),
            "SymFact V2 exact partner-L send slots");
        size_t row_exact_slots = xlu_checked_product(
            l2u_slots, static_cast<size_t>(Pc),
            "SymFact V2 exact row-fragment send slots");
        symV2PartnerLExactSendSizes.assign(partner_exact_slots, 0);
        symV2PartnerLExactSendBufsGPU.assign(partner_exact_slots, NULL);
        symV2PartnerLExactSendMapsGPU.assign(partner_exact_slots, NULL);
        symV2PartnerLExactHostSendBufs.assign(
            partner_exact_slots, std::vector<double>());
        symV2PartnerLExactHostSendBufsPinned.assign(partner_exact_slots, NULL);
        symV2RowFragExactSendSizes.assign(row_exact_slots, 0);
        symV2RowFragExactSendBufsGPU.assign(row_exact_slots, NULL);
        symV2RowFragExactSendMapsGPU.assign(row_exact_slots, NULL);
        symV2RowFragExactHostSendBufs.assign(
            row_exact_slots, std::vector<double>());
        symV2RowFragExactHostSendBufsPinned.assign(row_exact_slots, NULL);
        symV2RowFragExactSendMapsHost.clear();
        symV2RowFragExactSendMapOffsets.assign(row_exact_slots, 0);
        symV2RowDirectSendSizes.assign(l2u_slots, 0);
        symV2RowDirectSendMapOffsets.assign(l2u_slots, 0);
        symV2RowDirectSendMapsHost.clear();
        symV2PartnerLPrepacked.assign(static_cast<size_t>(local_cols), 0);
        symPanelReadyEventIds.assign(nsupers, -1);
        symDiagPrefetchEventIds.assign(nsupers, -1);
        std::vector<size_t> symV2PartnerLMapOffsets(l2u_slots, 0);
        std::vector<int_t> symV2PartnerLPackedMaps;
        struct SymV2ExactSendSegment
        {
            int_t gid;
            size_t map_offset;
            size_t map_count;
        };
        const bool exact_send_map_index_setup =
            symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
            superlu_sym_v2_pc_fragment_schur() &&
            superlu_sym_v2_exact_fragment_demand() &&
            superlu_sym_v2_exact_map_index();
        std::vector<std::vector<SymV2ExactSendSegment> >
            symV2ExactSendSegments;
        if (exact_send_map_index_setup)
            symV2ExactSendSegments.assign(l2u_slots,
                                          std::vector<SymV2ExactSendSegment>());

        dLocalLU_t *Llu = LUstructPtr->Llu;
        if (need_l2u_workspace)
        {
            int_t local_rows = symV2RowCount();
            for (int_t lk = 0; lk < local_rows; ++lk)
            {
                int_t k = symV2RowGid(lk);
                if (k >= nsupers || isNodeInMyGrid[k] != 1)
                    continue;
                if (mycol != kcol(k))
                    continue;

                xupanel_t<double> &upanel = uPanelVec[g2lRow(k)];
                xlpanel_t<double> &lpanel = lPanelVec[symV2PanelIndex(k)];
                int_t *usub = Llu->Ufstnz_br_ptr[lk];
                if (upanel.isEmpty() || lpanel.isEmpty() || usub == NULL)
                    continue;

                std::vector<int_t> local_map(upanel.nzvalSize(), -1);
                int_t ksupc = SuperSize(k);
                int_t klst = FstBlockC(k + 1);
                int_t usub_ptr = BR_HEADER;
                int_t dst_col = 0;
                int_t nub = usub[0];
                bool map_ok = true;

                for (int_t ub = 0; ub < nub; ++ub)
                {
                    int_t jb = usub[usub_ptr];
                    int_t gsupc = SuperSize(jb);
                    int_t lblock = lpanel.find(jb);
                    if (lblock == GLOBAL_BLOCK_NOT_FOUND)
                    {
                        map_ok = false;
                        break;
                    }

                    int_t *lrows = lpanel.rowList(lblock);
                    int_t n_lrows = lpanel.nbrow(lblock);
                    for (int_t col = 0; col < gsupc; ++col)
                    {
                        int_t segsize = klst - usub[usub_ptr + UB_DESCRIPTOR + col];
                        if (segsize <= 0)
                            continue;

                        int_t src_row = GLOBAL_BLOCK_NOT_FOUND;
                        for (int_t rr = 0; rr < n_lrows; ++rr)
                        {
                            if (lrows[rr] == col)
                            {
                                src_row = rr;
                                break;
                            }
                        }
                        if (src_row == GLOBAL_BLOCK_NOT_FOUND)
                        {
                            map_ok = false;
                            break;
                        }

                        for (int_t row = 0; row < ksupc; ++row)
                        {
                            int_t dst = dst_col * upanel.LDA() + row;
                            if (row >= ksupc - segsize)
                                local_map[dst] = lpanel.blkPtrOffset(lblock) +
                                                 src_row + row * lpanel.LDA();
                        }
                        ++dst_col;
                    }
                    if (!map_ok)
                        break;
                    usub_ptr += UB_DESCRIPTOR + gsupc;
                }

                if (!map_ok)
                    continue;
                if (dst_col != upanel.nzcols())
                    ABORT("SymFact local GPU L2U map has an invalid U column count.");

                gpuErrchk(cudaMalloc((void **)&symL2ULocalMapsGPU[lk],
                                     xlu_checked_product(static_cast<size_t>(upanel.nzvalSize()),
                                                         sizeof(int_t),
                                                         "SymFact local GPU L2U map")));
                gpuErrchk(cudaMemcpy(symL2ULocalMapsGPU[lk], local_map.data(),
                                     sizeof(int_t) * static_cast<size_t>(upanel.nzvalSize()),
                                     cudaMemcpyHostToDevice));
            }
            xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after local GPU L2U map setup");
        }

        if (symGPU3DVersion == 2 && Pr > 1)
        {
            double tPartnerSendMapBuild =
                profile_setup ? SuperLU_timer_() : 0.0;
            std::vector<size_t> map_counts(l2u_slots, 0);
            std::vector<size_t> meta_counts(l2u_slots, 0);

            for (int_t lk = 0; lk < local_cols; ++lk)
            {
                int_t *lsub = Llu->Lrowind_bc_ptr[lk];
                int_t *lloc = Llu->Lindval_loc_bc_ptr[lk];
                if (lsub == NULL || lloc == NULL || lsub[0] <= 0)
                    continue;

                int_t jb = symV2PanelGid(lk);
                if (jb >= nsupers)
                    continue;

                int_t nb;
                int_t idx_i;
                if (myrow == symV2DiagRoot(jb))
                {
                    nb = lsub[0] - 1;
                    idx_i = nb + 2;
                }
                else
                {
                    nb = lsub[0];
                    idx_i = nb;
                }
                if (nb <= 0)
                    continue;

                int_t knsupc = SuperSize(jb);
                for (int_t lb = 0; lb < nb; ++lb)
                {
                    int_t lptr_tmp = lloc[lb + idx_i];
                    int_t ik = lsub[lptr_tmp];
                    int ikcol = symV2PanelRoot(ik);
                    if (ikcol < 0 || ikcol >= Pc)
                        ABORT("SymFact V2 partner-L target process column is invalid.");
                    int_t len = lsub[lptr_tmp + 1];
                    if (len <= 0)
                        continue;

                    size_t flat = static_cast<size_t>(lk) *
                                      static_cast<size_t>(Pc) +
                                  static_cast<size_t>(ikcol);
                    size_t map_add = xlu_checked_product(
                        static_cast<size_t>(len),
                        static_cast<size_t>(knsupc),
                        "SymFact V2 packed partner-L send map");
                    if (map_counts[flat] >
                        std::numeric_limits<size_t>::max() - map_add)
                        ABORT("SymFact V2 packed partner-L send map size overflows.");
                    map_counts[flat] += map_add;

                    size_t meta_add = static_cast<size_t>(len) + 2;
                    if (meta_counts[flat] >
                        std::numeric_limits<size_t>::max() - meta_add)
                        ABORT("SymFact V2 packed partner-L metadata size overflows.");
                    meta_counts[flat] += meta_add;
                }
            }

            size_t total_partner_send = 0;
            size_t max_partner_host_send_scratch = 0;
            if (superlu_sym_v2_pinned_staging() &&
                superlu_sym_v2_pinned_staging_pool() &&
                !superlu_cuda_aware_mpi())
            {
                for (int_t lk = 0; lk < local_cols; ++lk)
                {
                    size_t panel_scratch_count = 0;
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        size_t flat = static_cast<size_t>(lk) *
                                          static_cast<size_t>(Pc) +
                                      static_cast<size_t>(pc);
                        symV2PartnerLHostSendScratchOffsets[flat] =
                            panel_scratch_count;
                        if (panel_scratch_count >
                            std::numeric_limits<size_t>::max() -
                                map_counts[flat])
                            ABORT("SymFact V2 pooled send staging size overflows.");
                        panel_scratch_count += map_counts[flat];
                    }
                    max_partner_host_send_scratch = SUPERLU_MAX(
                        max_partner_host_send_scratch,
                        panel_scratch_count);
                }
            }

            for (size_t flat = 0; flat < l2u_slots; ++flat)
            {
                if (map_counts[flat] >
                    static_cast<size_t>(std::numeric_limits<int>::max()))
                    ABORT("SymFact GPU L2L send map is too large for MPI.");
                if (meta_counts[flat] >
                    static_cast<size_t>(std::numeric_limits<int_t>::max()))
                    ABORT("SymFact V2 partner-L metadata is too large.");
                symV2PartnerLMapOffsets[flat] = total_partner_send;
                total_partner_send += map_counts[flat];
                if (total_partner_send < symV2PartnerLMapOffsets[flat])
                    ABORT("SymFact V2 packed partner-L send map size overflows.");

                symV2PartnerLSendSizes[flat] =
                    static_cast<int>(map_counts[flat]);
                if (map_counts[flat] > 0)
                {
                    if (superlu_sym_v2_pinned_staging() &&
                        !superlu_sym_v2_pinned_staging_pool())
                    {
                        /* Original R2 mode: one pinned allocation per chunk. */
                        gpuErrchk(cudaMallocHost(
                            (void **)&symV2PartnerLHostSendBufsPinned[flat],
                            xlu_checked_product(map_counts[flat], sizeof(double),
                                                "SymFact V2 pinned send staging")));
                    }
                    else if (!superlu_sym_v2_pinned_staging())
                    {
                        symV2PartnerLHostSendBufs[flat].resize(map_counts[flat]);
                    }
                }
                if (meta_counts[flat] > 0)
                    symL2LSendMeta[flat].resize(meta_counts[flat]);
            }
            symV2PartnerLPackedMaps.assign(total_partner_send, 0);
            if (superlu_sym_v2_pinned_staging() &&
                superlu_sym_v2_pinned_staging_pool() &&
                !superlu_cuda_aware_mpi() &&
                max_partner_host_send_scratch > 0)
            {
                gpuErrchk(cudaMallocHost(
                    (void **)&symV2PartnerLHostSendPoolPinned,
                    xlu_checked_product(max_partner_host_send_scratch,
                                        sizeof(double),
                                        "SymFact V2 pooled pinned send staging")));
                symV2PartnerLHostSendPoolPinnedCount =
                    max_partner_host_send_scratch;
                for (size_t flat = 0; flat < l2u_slots; ++flat)
                {
                    if (map_counts[flat] == 0)
                        continue;
                    size_t offset =
                        symV2PartnerLHostSendScratchOffsets[flat];
                    if (offset + map_counts[flat] >
                            symV2PartnerLHostSendPoolPinnedCount ||
                        offset + map_counts[flat] < offset)
                        ABORT("SymFact V2 pooled send staging map is invalid.");
                    symV2PartnerLHostSendBufsPinned[flat] =
                        symV2PartnerLHostSendPoolPinned + offset;
                }
            }

            std::vector<size_t> map_write_offsets =
                symV2PartnerLMapOffsets;
            std::vector<size_t> meta_write_offsets(l2u_slots, 0);
            std::vector<std::pair<int_t, int_t> > row_order;
            for (int_t lk = 0; lk < local_cols; ++lk)
            {
                int_t *lsub = Llu->Lrowind_bc_ptr[lk];
                int_t *lloc = Llu->Lindval_loc_bc_ptr[lk];
                if (lsub == NULL || lloc == NULL || lsub[0] <= 0)
                    continue;

                int_t jb = symV2PanelGid(lk);
                if (jb >= nsupers)
                    continue;

                int_t nb;
                int_t idx_i;
                int_t idx_v;
                if (myrow == symV2DiagRoot(jb))
                {
                    nb = lsub[0] - 1;
                    idx_i = nb + 2;
                    idx_v = 2 * nb + 3;
                }
                else
                {
                    nb = lsub[0];
                    idx_i = nb;
                    idx_v = 2 * nb;
                }
                if (nb <= 0)
                    continue;

                int_t knsupc = SuperSize(jb);
                int_t nsupr = lsub[1];
                for (int_t lb = 0; lb < nb; ++lb)
                {
                    int_t luptr_tmp = lloc[lb + idx_v];
                    int_t lptr_tmp = lloc[lb + idx_i];
                    int_t ik = lsub[lptr_tmp];
                    int ikcol = symV2PanelRoot(ik);
                    if (ikcol < 0 || ikcol >= Pc)
                        ABORT("SymFact V2 partner-L target process column is invalid.");
                    int_t len = lsub[lptr_tmp + 1];
                    if (len <= 0)
                        continue;
                    int_t fsupc = FstBlockC(ik);

                    row_order.clear();
                    row_order.reserve(static_cast<size_t>(len));
                    for (int_t i = 0; i < len; ++i)
                        row_order.push_back(std::make_pair(
                            lsub[lptr_tmp + 2 + i] - fsupc, i));
                    std::sort(row_order.begin(), row_order.end());

                    size_t flat = static_cast<size_t>(lk) *
                                      static_cast<size_t>(Pc) +
                                  static_cast<size_t>(ikcol);
                    std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t meta_pos = meta_write_offsets[flat];
                    if (meta_pos + static_cast<size_t>(len) + 2 >
                        meta.size())
                        ABORT("SymFact V2 packed partner-L metadata overrun.");
                    meta[meta_pos++] = ik;
                    meta[meta_pos++] = len;
                    for (int_t i = 0; i < len; ++i)
                        meta[meta_pos++] = row_order[i].first;
                    meta_write_offsets[flat] = meta_pos;

                    size_t map_pos = map_write_offsets[flat];
                    size_t map_end = symV2PartnerLMapOffsets[flat] +
                                     map_counts[flat];
                    size_t segment_map_offset = map_pos;
                    size_t segment_map_count = xlu_checked_product(
                        static_cast<size_t>(len),
                        static_cast<size_t>(knsupc),
                        "SymFact V2 exact send map index segment");
                    for (int_t j = 0; j < knsupc; ++j)
                    {
                        for (int_t i = 0; i < len; ++i)
                        {
                            if (map_pos >= map_end)
                                ABORT("SymFact V2 packed partner-L send map overrun.");
                            int_t src_row = row_order[i].second;
                            symV2PartnerLPackedMaps[map_pos++] =
                                luptr_tmp + src_row + j * nsupr;
                        }
                    }
                    if (map_pos != segment_map_offset + segment_map_count)
                        ABORT("SymFact V2 exact send map index segment size mismatch.");
                    if (!symV2ExactSendSegments.empty())
                    {
                        SymV2ExactSendSegment segment;
                        segment.gid = ik;
                        segment.map_offset = segment_map_offset;
                        segment.map_count = segment_map_count;
                        symV2ExactSendSegments[flat].push_back(segment);
                    }
                    map_write_offsets[flat] = map_pos;
                }
            }

            for (size_t flat = 0; flat < l2u_slots; ++flat)
            {
                if (map_write_offsets[flat] !=
                    symV2PartnerLMapOffsets[flat] + map_counts[flat])
                    ABORT("SymFact V2 packed partner-L send map size mismatch.");
                if (meta_write_offsets[flat] != meta_counts[flat])
                    ABORT("SymFact V2 packed partner-L metadata size mismatch.");
            }
            if (!symV2ExactSendSegments.empty())
            {
                double tExactSendMapIndex =
                    profile_setup ? SuperLU_timer_() : 0.0;
                for (size_t flat = 0; flat < symV2ExactSendSegments.size();
                     ++flat)
                {
                    std::vector<SymV2ExactSendSegment> &segments =
                        symV2ExactSendSegments[flat];
                    if (segments.empty())
                        continue;
                    std::sort(
                        segments.begin(), segments.end(),
                        [](const SymV2ExactSendSegment &a,
                           const SymV2ExactSendSegment &b)
                        {
                            if (a.gid != b.gid)
                                return a.gid < b.gid;
                            return a.map_offset < b.map_offset;
                        });
                }
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_EXACT_SEND_MAP_INDEX,
                        SuperLU_timer_() - tExactSendMapIndex);
            }

            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_SEND_MAP_BUILD,
                    SuperLU_timer_() - tPartnerSendMapBuild);
        }

        for (int_t lk = 0; lk < local_cols; ++lk)
        {
            bool have_comml_send =
                (need_l2u_workspace &&
                 Llu->Send_CommL != NULL &&
                 Llu->Send_CommL[lk].ComQuant != NULL);
            if (!have_comml_send)
                continue;

            int_t *lsub = Llu->Lrowind_bc_ptr[lk];
            int_t *lloc = Llu->Lindval_loc_bc_ptr[lk];
            if (lsub == NULL || lloc == NULL || lsub[0] <= 0)
                continue;

            int_t jb = symV2PanelGid(lk);
            if (jb >= nsupers)
                continue;

            int_t nb;
            int_t idx_i;
            int_t idx_v;
            if (myrow == symV2DiagRoot(jb))
            {
                nb = lsub[0] - 1;
                idx_i = nb + 2;
                idx_v = 2 * nb + 3;
            }
            else
            {
                nb = lsub[0];
                idx_i = nb;
                idx_v = 2 * nb;
            }
            if (nb <= 0)
                continue;

            double tPartnerSendMapBuild =
                profile_setup ? SuperLU_timer_() : 0.0;
            int_t knsupc = SuperSize(jb);
            int_t nsupr = lsub[1];
            std::vector<std::vector<int_t> > host_maps(Pc);
            for (int pc = 0; pc < Pc; ++pc)
            {
                int size = have_comml_send
                               ? Llu->Send_CommL[lk].ComQuant[pc].size
                               : 0;
                if (size > 0)
                {
                    host_maps[pc].reserve(size);
                }
            }

            for (int_t lb = 0; lb < nb; ++lb)
            {
                int_t luptr_tmp = lloc[lb + idx_v];
                int_t lptr_tmp = lloc[lb + idx_i];
                int_t ik = lsub[lptr_tmp];
                int ikcol = (symGPU3DVersion == 2)
                                ? symV2PanelRoot(ik)
                                : PCOL(ik, grid);
                int_t len = lsub[lptr_tmp + 1];
                int_t fsupc = FstBlockC(ik);

                std::vector<std::pair<int_t, int_t> > row_order;
                row_order.reserve(len);
                for (int_t i = 0; i < len; ++i)
                    row_order.push_back(std::make_pair(lsub[lptr_tmp + 2 + i] - fsupc, i));
                std::sort(row_order.begin(), row_order.end());

                if (have_comml_send)
                {
                    std::vector<int_t> &map = host_maps[ikcol];
                    map.push_back(-(ik + 1));
                    for (int_t i = 0; i < len; ++i)
                    {
                        int_t src_row = row_order[i].second;
                        for (int_t j = 0; j < knsupc; ++j)
                            map.push_back(luptr_tmp + src_row + j * nsupr);
                    }
                }
            }
            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_SEND_MAP_BUILD,
                    SuperLU_timer_() - tPartnerSendMapBuild);

            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);

                int l2u_size = have_comml_send
                                   ? Llu->Send_CommL[lk].ComQuant[pc].size
                                   : 0;
                if (l2u_size > 0)
                {
                    if (host_maps[pc].size() !=
                        static_cast<size_t>(l2u_size))
                        ABORT("SymFact GPU L2U send map size mismatch.");
                    gpuErrchk(cudaMalloc(
                        (void **)&symL2USendBufsGPU[flat],
                        xlu_checked_product(static_cast<size_t>(l2u_size),
                                            sizeof(double),
                                            "SymFact GPU L2U send buffer")));
                    gpuErrchk(cudaMalloc(
                        (void **)&symL2USendMapsGPU[flat],
                        xlu_checked_product(static_cast<size_t>(l2u_size),
                                            sizeof(int_t),
                                            "SymFact GPU L2U send map")));
                    gpuErrchk(cudaMemcpy(
                        symL2USendMapsGPU[flat], host_maps[pc].data(),
                        sizeof(int_t) * static_cast<size_t>(l2u_size),
                        cudaMemcpyHostToDevice));
                }
            }
        }
        if (symGPU3DVersion == 2 && Pr > 1)
        {
            double tPartnerSendGPU =
                profile_setup ? SuperLU_timer_() : 0.0;
            size_t total_partner_send = symV2PartnerLPackedMaps.size();
            for (size_t flat = 0; flat < l2u_slots; ++flat)
            {
                int size = symV2PartnerLSendSizes[flat];
                if (size < 0)
                    ABORT("SymFact V2 partner-L send size is invalid.");
                if (size <= 0)
                    continue;
                size_t offset = symV2PartnerLMapOffsets[flat];
                if (offset + static_cast<size_t>(size) > total_partner_send ||
                    offset + static_cast<size_t>(size) < offset)
                    ABORT("SymFact V2 partner-L host map size mismatch.");
            }

            symV2PartnerLSendBufPoolCount = total_partner_send;
            symL2LSendMapPoolCount = total_partner_send;
            if (total_partner_send > 0)
            {
                gpuErrchk(cudaMalloc(
                    (void **)&symV2PartnerLSendBufPoolGPU,
                    xlu_checked_product(total_partner_send, sizeof(double),
                                        "SymFact V2 partner-L send buffer pool")));
                gpuErrchk(cudaMalloc(
                    (void **)&symL2LSendMapPoolGPU,
                    xlu_checked_product(total_partner_send, sizeof(int_t),
                                        "SymFact GPU L2L send map pool")));

                for (size_t flat = 0; flat < l2u_slots; ++flat)
                {
                    int size = symV2PartnerLSendSizes[flat];
                    if (size <= 0)
                        continue;
                    size_t offset = symV2PartnerLMapOffsets[flat];
                    symV2PartnerLSendBufsGPU[flat] =
                        symV2PartnerLSendBufPoolGPU + offset;
                    symL2LSendMapsGPU[flat] =
                        symL2LSendMapPoolGPU + offset;
                }
                gpuErrchk(cudaMemcpy(
                    symL2LSendMapPoolGPU, symV2PartnerLPackedMaps.data(),
                    sizeof(int_t) * total_partner_send,
                    cudaMemcpyHostToDevice));
            }
            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_SEND_GPU_ALLOC_COPY,
                    SuperLU_timer_() - tPartnerSendGPU);
        }
        if (symGPU3DVersion == 2 && Pr > 1)
        {
            double tPartnerRecvCount =
                profile_setup ? SuperLU_timer_() : 0.0;
            size_t table_count = xlu_checked_product(
                xlu_checked_product(static_cast<size_t>(nsupers),
                                    static_cast<size_t>(Pc),
                                    "SymFact V2 partner-L receive count table"),
                static_cast<size_t>(Pr),
                "SymFact V2 partner-L receive count table");
            if (table_count >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 partner-L receive count table is too large for MPI.");

            std::vector<int> local_recv_sizes(table_count, 0);
            std::vector<int> global_recv_sizes(table_count, 0);

            for (int_t lk = 0; lk < local_cols; ++lk)
            {
                int_t k0 = symV2PanelGid(lk);
                if (k0 >= nsupers)
                    continue;

                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t flat = static_cast<size_t>(lk) *
                                      static_cast<size_t>(Pc) +
                                  static_cast<size_t>(pc);
                    int size = (flat < symV2PartnerLSendSizes.size())
                                   ? symV2PartnerLSendSizes[flat]
                                   : 0;
                    size_t pos =
                        (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                         static_cast<size_t>(pc)) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(myrow);
                    local_recv_sizes[pos] = size;
                }
            }

            MPI_Allreduce(local_recv_sizes.data(), global_recv_sizes.data(),
                          static_cast<int>(table_count), MPI_INT, MPI_SUM,
                          grid->comm);
            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_RECV_COUNT_ALLREDUCE,
                    SuperLU_timer_() - tPartnerRecvCount);

            double tPartnerMetaAllgather =
                profile_setup ? SuperLU_timer_() : 0.0;
            std::vector<int_t> local_meta_payload;
            for (int_t lk = 0; lk < local_cols; ++lk)
            {
                int_t k0 = symV2PanelGid(lk);
                if (k0 >= nsupers)
                    continue;

                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t flat = static_cast<size_t>(lk) *
                                      static_cast<size_t>(Pc) +
                                  static_cast<size_t>(pc);
                    if (flat >= symL2LSendMeta.size() ||
                        symL2LSendMeta[flat].empty())
                        continue;
                    if (symL2LSendMeta[flat].size() >
                        static_cast<size_t>(std::numeric_limits<int_t>::max()))
                        ABORT("SymFact V2 partner-L metadata is too large.");

                    local_meta_payload.push_back(pc);
                    local_meta_payload.push_back(k0);
                    local_meta_payload.push_back(
                        static_cast<int_t>(symL2LSendMeta[flat].size()));
                    local_meta_payload.insert(local_meta_payload.end(),
                                              symL2LSendMeta[flat].begin(),
                                              symL2LSendMeta[flat].end());
                }
            }
            if (local_meta_payload.size() >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 partner-L metadata payload is too large for MPI.");

            int comm_size = 0;
            MPI_Comm_size(grid->comm, &comm_size);
            int local_meta_count = static_cast<int>(local_meta_payload.size());
            std::vector<int> meta_counts(comm_size, 0);
            MPI_Allgather(&local_meta_count, 1, MPI_INT, meta_counts.data(),
                          1, MPI_INT, grid->comm);

            std::vector<int> meta_displs(comm_size, 0);
            long long total_meta_count = 0;
            for (int r = 0; r < comm_size; ++r)
            {
                if (meta_counts[r] < 0)
                    ABORT("SymFact V2 partner-L metadata count is invalid.");
                if (total_meta_count >
                    static_cast<long long>(std::numeric_limits<int>::max()))
                    ABORT("SymFact V2 partner-L metadata payload is too large for MPI.");
                meta_displs[r] = static_cast<int>(total_meta_count);
                total_meta_count += meta_counts[r];
            }
            if (total_meta_count >
                static_cast<long long>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 partner-L metadata payload is too large for MPI.");

            std::vector<int_t> all_meta_payload(
                static_cast<size_t>(total_meta_count));
            MPI_Allgatherv(local_meta_payload.empty()
                               ? NULL
                               : local_meta_payload.data(),
                           local_meta_count, mpi_int_t,
                           all_meta_payload.empty()
                               ? NULL
                               : all_meta_payload.data(),
                           meta_counts.data(), meta_displs.data(), mpi_int_t,
                           grid->comm);
            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_META_ALLGATHER,
                    SuperLU_timer_() - tPartnerMetaAllgather);

            size_t compact_count = xlu_checked_product(
                static_cast<size_t>(nsupers), static_cast<size_t>(Pr),
                "SymFact V2 partner-L compact receive count table");
            const bool pc_fragment_schur_setup =
                superlu_sym_v2_pc_fragment_schur() && Pr > 1 && Pc > 1;
            const bool exact_fragment_demand_setup =
                pc_fragment_schur_setup &&
                superlu_sym_v2_exact_fragment_demand();
            const bool exact_partner_fragment_demand_setup =
                exact_fragment_demand_setup &&
                superlu_sym_v2_exact_partner_fragment_demand();
            const bool exact_row_fragment_demand_setup =
                exact_fragment_demand_setup &&
                superlu_sym_v2_exact_row_fragment_demand();
            if (exact_row_fragment_demand_setup &&
                !superlu_sym_v2_rowfrag_destination_path())
                ABORT("GPU3DV2_EXACT_ROW_FRAGMENT_DEMAND requires GPU3DV2_ROW_L_SOURCE_PACK=1, GPU3DV2_ROW_L_DIRECT_RECV=1, or GPU3DV2_ROWFRAG_DEST_PACK=1.");
            double tExactDemandBuild =
                (profile_setup && exact_fragment_demand_setup)
                    ? SuperLU_timer_()
                    : 0.0;
            std::vector<unsigned char> local_receive_demand(compact_count, 0);
            std::vector<std::vector<int_t> >
                local_partner_block_demand(compact_count);
            std::vector<std::vector<int_t> > row_blocks_by_panel(nsupers);
            std::vector<std::vector<int_t> > needed_row_blocks_by_panel(nsupers);

            for (int r = 0; r < comm_size; ++r)
            {
                size_t meta_pos = static_cast<size_t>(meta_displs[r]);
                size_t rank_end =
                    meta_pos + static_cast<size_t>(meta_counts[r]);
                int source_pr = MYROW(r, grid);
                while (meta_pos < rank_end)
                {
                    if (meta_pos + 3 > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is truncated.");
                    int_t target_pc = all_meta_payload[meta_pos++];
                    int_t k0 = all_meta_payload[meta_pos++];
                    int_t meta_len = all_meta_payload[meta_pos++];
                    if (target_pc < 0 || target_pc >= Pc || k0 < 0 ||
                        k0 >= nsupers || meta_len < 0 ||
                        meta_pos + static_cast<size_t>(meta_len) > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is invalid.");

                    size_t block_pos = meta_pos;
                    size_t block_end =
                        meta_pos + static_cast<size_t>(meta_len);
                    if (source_pr == myrow)
                    {
                        std::vector<int_t> &row_blocks =
                            row_blocks_by_panel[k0];
                        while (block_pos < block_end)
                        {
                            if (block_pos + 2 > block_end)
                                ABORT("SymFact V2 partner-L metadata block is truncated.");
                            int_t gid = all_meta_payload[block_pos++];
                            int_t len = all_meta_payload[block_pos++];
                            if (len < 0 ||
                                block_pos + static_cast<size_t>(len) >
                                    block_end)
                                ABORT("SymFact V2 partner-L metadata block has invalid length.");
                            row_blocks.push_back(gid);
                            block_pos += static_cast<size_t>(len);
                        }
                    }
                    meta_pos += static_cast<size_t>(meta_len);
                }
            }

            for (int_t k0 = 0; k0 < nsupers; ++k0)
            {
                std::vector<int_t> &row_blocks = row_blocks_by_panel[k0];
                if (row_blocks.empty())
                    continue;
                std::sort(row_blocks.begin(), row_blocks.end());
                row_blocks.erase(std::unique(row_blocks.begin(),
                                             row_blocks.end()),
                                 row_blocks.end());
            }

            for (int r = 0; r < comm_size; ++r)
            {
                size_t meta_pos = static_cast<size_t>(meta_displs[r]);
                size_t rank_end =
                    meta_pos + static_cast<size_t>(meta_counts[r]);
                int source_pr = MYROW(r, grid);
                while (meta_pos < rank_end)
                {
                    if (meta_pos + 3 > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is truncated.");
                    int_t target_pc = all_meta_payload[meta_pos++];
                    int_t k0 = all_meta_payload[meta_pos++];
                    int_t meta_len = all_meta_payload[meta_pos++];
                    if (target_pc < 0 || target_pc >= Pc || k0 < 0 ||
                        k0 >= nsupers || meta_len < 0 ||
                        meta_pos + static_cast<size_t>(meta_len) > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is invalid.");

                    size_t block_pos = meta_pos;
                    size_t block_end =
                        meta_pos + static_cast<size_t>(meta_len);
                    size_t demand_pos =
                        static_cast<size_t>(k0) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(source_pr);
                    bool demand_chunk = false;
                    if (target_pc == mycol)
                    {
                        const std::vector<int_t> &row_blocks =
                            row_blocks_by_panel[k0];
                        while (block_pos < block_end)
                        {
                            if (block_pos + 2 > block_end)
                                ABORT("SymFact V2 partner-L metadata block is truncated.");
                            int_t gj = all_meta_payload[block_pos++];
                            int_t len = all_meta_payload[block_pos++];
                            if (len < 0 ||
                                block_pos + static_cast<size_t>(len) >
                                    block_end)
                                ABORT("SymFact V2 partner-L metadata block has invalid length.");
                            bool demand_block = false;
                            if (!row_blocks.empty() &&
                                symV2PanelRoot(gj) == mycol)
                            {
                                int_t lj = symV2PanelIndex(gj);
                                if (lj >= 0)
                                {
                                    xlpanel_t<double> &dst_panel =
                                        lPanelVec[lj];
                                    if (!dst_panel.isEmpty())
                                    {
                                        std::vector<int_t>::const_iterator it =
                                            std::lower_bound(row_blocks.begin(),
                                                             row_blocks.end(),
                                                             gj);
                                        for (; it != row_blocks.end(); ++it)
                                        {
                                            if (dst_panel.find(*it) !=
                                                GLOBAL_BLOCK_NOT_FOUND)
                                            {
                                                demand_block = true;
                                                demand_chunk = true;
                                                if (pc_fragment_schur_setup)
                                                    needed_row_blocks_by_panel[k0]
                                                        .push_back(*it);
                                            }
                                        }
                                    }
                                }
                            }
                            if (demand_block &&
                                exact_partner_fragment_demand_setup)
                                local_partner_block_demand[demand_pos]
                                    .push_back(gj);
                            block_pos += static_cast<size_t>(len);
                        }
                    }
                    if (demand_chunk)
                    {
                        local_receive_demand[demand_pos] = 1;
                    }
                    meta_pos = block_end;
                }
            }

            std::vector<unsigned char> local_row_chunk_demand(table_count, 0);
            std::vector<std::vector<int_t> > local_row_block_demand(table_count);
            for (int_t k0 = 0; k0 < nsupers; ++k0)
            {
                std::vector<int_t> &needed =
                    needed_row_blocks_by_panel[k0];
                if (needed.empty())
                    continue;
                std::sort(needed.begin(), needed.end());
                needed.erase(std::unique(needed.begin(), needed.end()),
                             needed.end());
                for (size_t ig = 0; ig < needed.size(); ++ig)
                {
                    int chunk_pc = symV2PanelRoot(needed[ig]);
                    if (chunk_pc < 0 || chunk_pc >= Pc)
                        ABORT("SymFact V2 row-fragment demand has invalid process column.");
                    size_t row_chunk_pos =
                        (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                         static_cast<size_t>(chunk_pc)) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(myrow);
                    local_row_chunk_demand[row_chunk_pos] = 1;
                    if (exact_row_fragment_demand_setup)
                        local_row_block_demand[row_chunk_pos]
                            .push_back(needed[ig]);
                }
            }
            if (exact_partner_fragment_demand_setup)
            {
                for (size_t pos = 0; pos < local_partner_block_demand.size(); ++pos)
                {
                    std::vector<int_t> &blocks =
                        local_partner_block_demand[pos];
                    if (blocks.empty())
                        continue;
                    std::sort(blocks.begin(), blocks.end());
                    blocks.erase(std::unique(blocks.begin(), blocks.end()),
                                 blocks.end());
                }
            }
            if (exact_row_fragment_demand_setup)
            {
                for (size_t pos = 0; pos < local_row_block_demand.size(); ++pos)
                {
                    std::vector<int_t> &blocks = local_row_block_demand[pos];
                    if (blocks.empty())
                        continue;
                    std::sort(blocks.begin(), blocks.end());
                    blocks.erase(std::unique(blocks.begin(), blocks.end()),
                                 blocks.end());
                }
            }
            std::vector<int_t> local_demand_payload;
            for (int_t k0 = 0; k0 < nsupers; ++k0)
            {
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t demand_pos =
                        static_cast<size_t>(k0) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    if (!local_receive_demand[demand_pos])
                        continue;
                    if (exact_partner_fragment_demand_setup &&
                        local_partner_block_demand[demand_pos].empty())
                        ABORT("SymFact V2 exact partner-L demand has no blocks.");
                    local_demand_payload.push_back(mycol);
                    local_demand_payload.push_back(myrow);
                    local_demand_payload.push_back(k0);
                    local_demand_payload.push_back(pr);
                    if (exact_partner_fragment_demand_setup)
                    {
                        const std::vector<int_t> &blocks =
                            local_partner_block_demand[demand_pos];
                        local_demand_payload.push_back(
                            static_cast<int_t>(blocks.size()));
                        local_demand_payload.insert(local_demand_payload.end(),
                                                    blocks.begin(),
                                                    blocks.end());
                    }
                }
            }
            if (local_demand_payload.size() >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 partner-L demand payload is too large for MPI.");

            int local_demand_count =
                static_cast<int>(local_demand_payload.size());
            std::vector<int> demand_counts(comm_size, 0);
            MPI_Allgather(&local_demand_count, 1, MPI_INT,
                          demand_counts.data(), 1, MPI_INT, grid->comm);

            std::vector<int> demand_displs(comm_size, 0);
            long long total_demand_count = 0;
            for (int r = 0; r < comm_size; ++r)
            {
                if (demand_counts[r] < 0)
                    ABORT("SymFact V2 partner-L demand count is invalid.");
                if (total_demand_count >
                    static_cast<long long>(std::numeric_limits<int>::max()))
                    ABORT("SymFact V2 partner-L demand payload is too large for MPI.");
                demand_displs[r] = static_cast<int>(total_demand_count);
                total_demand_count += demand_counts[r];
            }
            if (total_demand_count >
                static_cast<long long>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 partner-L demand payload is too large for MPI.");

            std::vector<int_t> all_demand_payload(
                static_cast<size_t>(total_demand_count));
            MPI_Allgatherv(local_demand_payload.empty()
                               ? NULL
                               : local_demand_payload.data(),
                           local_demand_count, mpi_int_t,
                           all_demand_payload.empty()
                               ? NULL
                               : all_demand_payload.data(),
                           demand_counts.data(), demand_displs.data(),
                           mpi_int_t, grid->comm);

            std::vector<std::vector<int_t> > partner_exact_send_block_gids(
                symV2PartnerLSendRowActive.size());
            for (size_t pos = 0; pos < all_demand_payload.size();)
            {
                if (pos + 4 > all_demand_payload.size())
                    ABORT("SymFact V2 partner-L demand payload is truncated.");
                int_t target_pc = all_demand_payload[pos++];
                int_t dest_pr = all_demand_payload[pos++];
                int_t k0 = all_demand_payload[pos++];
                int_t source_pr = all_demand_payload[pos++];
                int_t block_count = 0;
                size_t block_pos = pos;
                if (exact_partner_fragment_demand_setup)
                {
                    if (pos >= all_demand_payload.size())
                        ABORT("SymFact V2 exact partner-L demand payload is truncated.");
                    block_count = all_demand_payload[pos++];
                    if (block_count <= 0 ||
                        pos + static_cast<size_t>(block_count) >
                            all_demand_payload.size())
                        ABORT("SymFact V2 exact partner-L demand payload is invalid.");
                    block_pos = pos;
                    pos += static_cast<size_t>(block_count);
                }
                if (target_pc < 0 || target_pc >= Pc ||
                    dest_pr < 0 || dest_pr >= Pr ||
                    k0 < 0 || k0 >= nsupers ||
                    source_pr < 0 || source_pr >= Pr)
                    ABORT("SymFact V2 partner-L demand payload is invalid.");
                if (source_pr != myrow || symV2PanelRoot(k0) != mycol)
                    continue;
                int_t lk = symV2PanelIndex(k0);
                if (lk < 0)
                    continue;
                size_t flat =
                    static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(target_pc);
                size_t active_pos =
                    flat * static_cast<size_t>(Pr) +
                    static_cast<size_t>(dest_pr);
                if (active_pos >= symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner-L send demand index is invalid.");
                symV2PartnerLSendRowActive[active_pos] = 1;
                if (exact_partner_fragment_demand_setup)
                {
                    std::vector<int_t> &blocks =
                        partner_exact_send_block_gids[active_pos];
                    blocks.insert(blocks.end(),
                                  all_demand_payload.begin() + block_pos,
                                  all_demand_payload.begin() + block_pos +
                                      block_count);
                }
            }
            if (exact_partner_fragment_demand_setup)
            {
                for (size_t pos = 0;
                     pos < partner_exact_send_block_gids.size(); ++pos)
                {
                    std::vector<int_t> &blocks =
                        partner_exact_send_block_gids[pos];
                    if (blocks.empty())
                        continue;
                    std::sort(blocks.begin(), blocks.end());
                    blocks.erase(std::unique(blocks.begin(), blocks.end()),
                                 blocks.end());
                }
            }

            std::vector<int_t> local_row_demand_payload;
            for (int_t k0 = 0; k0 < nsupers; ++k0)
            {
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t row_chunk_pos =
                        (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                         static_cast<size_t>(pc)) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(myrow);
                    if (!local_row_chunk_demand[row_chunk_pos])
                        continue;
                    if (exact_row_fragment_demand_setup &&
                        local_row_block_demand[row_chunk_pos].empty())
                        ABORT("SymFact V2 exact row-fragment demand has no blocks.");
                    local_row_demand_payload.push_back(mycol);
                    local_row_demand_payload.push_back(myrow);
                    local_row_demand_payload.push_back(k0);
                    local_row_demand_payload.push_back(pc);
                    if (exact_row_fragment_demand_setup)
                    {
                        const std::vector<int_t> &blocks =
                            local_row_block_demand[row_chunk_pos];
                        local_row_demand_payload.push_back(
                            static_cast<int_t>(blocks.size()));
                        local_row_demand_payload.insert(
                            local_row_demand_payload.end(),
                            blocks.begin(), blocks.end());
                    }
                }
            }
            if (local_row_demand_payload.size() >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 row-fragment demand payload is too large for MPI.");

            int local_row_demand_count =
                static_cast<int>(local_row_demand_payload.size());
            std::vector<int> row_demand_counts(comm_size, 0);
            MPI_Allgather(&local_row_demand_count, 1, MPI_INT,
                          row_demand_counts.data(), 1, MPI_INT, grid->comm);

            std::vector<int> row_demand_displs(comm_size, 0);
            long long total_row_demand_count = 0;
            for (int r = 0; r < comm_size; ++r)
            {
                if (row_demand_counts[r] < 0)
                    ABORT("SymFact V2 row-fragment demand count is invalid.");
                if (total_row_demand_count >
                    static_cast<long long>(std::numeric_limits<int>::max()))
                    ABORT("SymFact V2 row-fragment demand payload is too large for MPI.");
                row_demand_displs[r] =
                    static_cast<int>(total_row_demand_count);
                total_row_demand_count += row_demand_counts[r];
            }
            if (total_row_demand_count >
                static_cast<long long>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 row-fragment demand payload is too large for MPI.");

            std::vector<int_t> all_row_demand_payload(
                static_cast<size_t>(total_row_demand_count));
            MPI_Allgatherv(local_row_demand_payload.empty()
                               ? NULL
                               : local_row_demand_payload.data(),
                           local_row_demand_count, mpi_int_t,
                           all_row_demand_payload.empty()
                               ? NULL
                               : all_row_demand_payload.data(),
                           row_demand_counts.data(),
                           row_demand_displs.data(), mpi_int_t, grid->comm);

            std::vector<std::vector<int_t> > row_exact_send_block_gids(
                symV2RowFragSendActive.size());
            for (size_t pos = 0; pos < all_row_demand_payload.size();)
            {
                if (pos + 4 > all_row_demand_payload.size())
                    ABORT("SymFact V2 row-fragment demand payload is truncated.");
                int_t dest_pc = all_row_demand_payload[pos++];
                int_t dest_pr = all_row_demand_payload[pos++];
                int_t k0 = all_row_demand_payload[pos++];
                int_t chunk_pc = all_row_demand_payload[pos++];
                int_t block_count = 0;
                size_t block_pos = pos;
                if (exact_row_fragment_demand_setup)
                {
                    if (pos >= all_row_demand_payload.size())
                        ABORT("SymFact V2 exact row-fragment demand payload is truncated.");
                    block_count = all_row_demand_payload[pos++];
                    if (block_count <= 0 ||
                        pos + static_cast<size_t>(block_count) >
                            all_row_demand_payload.size())
                        ABORT("SymFact V2 exact row-fragment demand payload is invalid.");
                    block_pos = pos;
                    pos += static_cast<size_t>(block_count);
                }
                if (dest_pc < 0 || dest_pc >= Pc ||
                    dest_pr < 0 || dest_pr >= Pr ||
                    k0 < 0 || k0 >= nsupers ||
                    chunk_pc < 0 || chunk_pc >= Pc)
                    ABORT("SymFact V2 row-fragment demand payload is invalid.");
                if (dest_pr != myrow || symV2PanelRoot(k0) != mycol)
                    continue;
                int_t lk = symV2PanelIndex(k0);
                if (lk < 0)
                    continue;
                size_t flat =
                    static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(chunk_pc);
                size_t active_pos =
                    flat * static_cast<size_t>(Pc) +
                    static_cast<size_t>(dest_pc);
                if (active_pos >= symV2RowFragSendActive.size())
                    ABORT("SymFact V2 row-fragment send demand index is invalid.");
                symV2RowFragSendActive[active_pos] = 1;
                if (exact_row_fragment_demand_setup)
                {
                    std::vector<int_t> &blocks =
                        row_exact_send_block_gids[active_pos];
                    blocks.insert(blocks.end(),
                                  all_row_demand_payload.begin() + block_pos,
                                  all_row_demand_payload.begin() + block_pos +
                                      block_count);
                }
            }
            if (exact_row_fragment_demand_setup)
            {
                for (size_t pos = 0; pos < row_exact_send_block_gids.size(); ++pos)
                {
                    std::vector<int_t> &blocks =
                        row_exact_send_block_gids[pos];
                    if (blocks.empty())
                        continue;
                    std::sort(blocks.begin(), blocks.end());
                    blocks.erase(std::unique(blocks.begin(), blocks.end()),
                                 blocks.end());
                }
            }
            if (profile_setup && exact_fragment_demand_setup)
                symV2SetupProfileAdd(SYM_V2_SETUP_EXACT_DEMAND_BUILD,
                                     SuperLU_timer_() - tExactDemandBuild);

            double tPartnerRecvMapBuild =
                profile_setup ? SuperLU_timer_() : 0.0;
            struct SymV2CachedPartnerBlock
            {
                int_t gid;
                std::vector<int_t> cols;
            };
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_partner_blocks(nsupers);
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_partner_recv_blocks(compact_count);
            size_t row_chunk_count = xlu_checked_product(
                static_cast<size_t>(nsupers), static_cast<size_t>(Pc),
                "SymFact V2 row-fragment receive table");
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_row_blocks(nsupers);
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_row_recv_blocks(row_chunk_count);

            for (int r = 0; r < comm_size; ++r)
            {
                size_t meta_pos = static_cast<size_t>(meta_displs[r]);
                size_t rank_end = meta_pos + static_cast<size_t>(meta_counts[r]);
                int source_pr = MYROW(r, grid);
                while (meta_pos < rank_end)
                {
                    if (meta_pos + 3 > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is truncated.");
                    int_t target_pc = all_meta_payload[meta_pos++];
                    int_t k0 = all_meta_payload[meta_pos++];
                    int_t meta_len = all_meta_payload[meta_pos++];
                    if (target_pc < 0 || target_pc >= Pc || k0 < 0 ||
                        k0 >= nsupers || meta_len < 0 ||
                        meta_pos + static_cast<size_t>(meta_len) > rank_end)
                        ABORT("SymFact V2 partner-L metadata payload is invalid.");

                    size_t block_pos = meta_pos;
                    size_t block_end =
                        meta_pos + static_cast<size_t>(meta_len);
                    size_t demand_pos =
                        static_cast<size_t>(k0) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(source_pr);
                    if (target_pc == mycol &&
                        local_receive_demand[demand_pos])
                    {
                        const std::vector<int_t> &exact_blocks =
                            local_partner_block_demand[demand_pos];
                        while (block_pos < block_end)
                        {
                            if (block_pos + 2 > block_end)
                                ABORT("SymFact V2 partner-L metadata block is truncated.");
                            SymV2CachedPartnerBlock block;
                            block.gid = all_meta_payload[block_pos++];
                            int_t len = all_meta_payload[block_pos++];
                            if (len < 0 ||
                                block_pos + static_cast<size_t>(len) >
                                    block_end)
                                ABORT("SymFact V2 partner-L metadata block has invalid length.");
                            block.cols.assign(
                                all_meta_payload.begin() + block_pos,
                                all_meta_payload.begin() + block_pos + len);
                            block_pos += static_cast<size_t>(len);
                            if (exact_partner_fragment_demand_setup &&
                                !std::binary_search(exact_blocks.begin(),
                                                    exact_blocks.end(),
                                                    block.gid))
                                continue;
                            cached_partner_blocks[k0].push_back(block);
                            size_t recv_pos = static_cast<size_t>(k0) *
                                                  static_cast<size_t>(Pr) +
                                              static_cast<size_t>(source_pr);
                            cached_partner_recv_blocks[recv_pos].push_back(block);
                        }
                    }
                    if (source_pr == myrow)
                    {
                        size_t row_chunk_pos =
                            (static_cast<size_t>(k0) *
                                 static_cast<size_t>(Pc) +
                             static_cast<size_t>(target_pc)) *
                                static_cast<size_t>(Pr) +
                            static_cast<size_t>(myrow);
                        if (local_row_chunk_demand[row_chunk_pos])
                        {
                            const std::vector<int_t> &exact_blocks =
                                local_row_block_demand[row_chunk_pos];
                            size_t row_block_pos = meta_pos;
                            while (row_block_pos < block_end)
                            {
                                if (row_block_pos + 2 > block_end)
                                    ABORT("SymFact V2 row-fragment metadata block is truncated.");
                                SymV2CachedPartnerBlock block;
                                block.gid = all_meta_payload[row_block_pos++];
                                int_t len = all_meta_payload[row_block_pos++];
                                if (len < 0 ||
                                    row_block_pos + static_cast<size_t>(len) >
                                        block_end)
                                    ABORT("SymFact V2 row-fragment metadata block has invalid length.");
                                block.cols.assign(
                                    all_meta_payload.begin() + row_block_pos,
                                    all_meta_payload.begin() + row_block_pos + len);
                                row_block_pos += static_cast<size_t>(len);
                                if (exact_row_fragment_demand_setup &&
                                    !std::binary_search(exact_blocks.begin(),
                                                        exact_blocks.end(),
                                                        block.gid))
                                    continue;
                                cached_row_blocks[k0].push_back(block);
                                size_t row_recv_pos =
                                    static_cast<size_t>(k0) *
                                        static_cast<size_t>(Pc) +
                                    static_cast<size_t>(target_pc);
                                cached_row_recv_blocks[row_recv_pos]
                                    .push_back(block);
                            }
                        }
                    }
                    meta_pos = block_end;
                }
            }

            symV2PartnerLRecvSizes.assign(compact_count, 0);
            symV2PartnerLRecvIndex.assign(nsupers, std::vector<int_t>());
            symV2PartnerLRecvIndexBySrc.assign(compact_count, std::vector<int_t>());
            symV2PartnerLRecvMap.assign(compact_count, std::vector<int_t>());
            symV2RowFragRecvSizes.assign(row_chunk_count, 0);
            symV2RowFragRecvIndex.assign(nsupers, std::vector<int_t>());
            symV2RowFragRecvMap.assign(row_chunk_count, std::vector<int_t>());
            std::vector<size_t> symV2PartnerLRecvMapOffsets(compact_count, 0);
            std::vector<size_t> symV2RowFragRecvMapOffsets(row_chunk_count, 0);
            for (int_t k0 = 0; k0 < nsupers; ++k0)
            {
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t src_pos =
                        (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                         static_cast<size_t>(mycol)) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(pr);
                    size_t dst_pos = static_cast<size_t>(k0) *
                                         static_cast<size_t>(Pr) +
                                     static_cast<size_t>(pr);
                    symV2PartnerLRecvSizes[dst_pos] =
                        (exact_partner_fragment_demand_setup ||
                         !local_receive_demand[dst_pos])
                            ? 0
                            : global_recv_sizes[src_pos];
                }
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t row_src_pos =
                        (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                         static_cast<size_t>(pc)) *
                            static_cast<size_t>(Pr) +
                        static_cast<size_t>(myrow);
                    size_t row_dst_pos =
                        static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                        static_cast<size_t>(pc);
                    symV2RowFragRecvSizes[row_dst_pos] =
                        (exact_row_fragment_demand_setup ||
                         !local_row_chunk_demand[row_src_pos])
                            ? 0
                            : global_recv_sizes[row_src_pos];
                }

                std::vector<SymV2CachedPartnerBlock> &blocks =
                    cached_partner_blocks[k0];
                if (blocks.empty())
                    continue;
                std::sort(blocks.begin(), blocks.end(),
                          [](const SymV2CachedPartnerBlock &a,
                             const SymV2CachedPartnerBlock &b)
                          {
                              return a.gid < b.gid;
                          });

                int_t partner_nblocks = static_cast<int_t>(blocks.size());
                int_t partner_nrows = 0;
                for (size_t ib = 0; ib < blocks.size(); ++ib)
                    partner_nrows += static_cast<int_t>(blocks[ib].cols.size());
                int_t partner_index_size =
                    LPANEL_HEADER_SIZE + 2 * partner_nblocks + 1 +
                    partner_nrows;
                if (partner_index_size > maxSymPartnerLidxCount)
                    ABORT("SymFact V2 partner-L cached index exceeds receive buffer.");
                if (static_cast<int64_t>(partner_nrows) *
                        static_cast<int64_t>(SuperSize(k0)) >
                    static_cast<int64_t>(maxSymPartnerLvalCount))
                    ABORT("SymFact V2 partner-L cached values exceed receive buffer.");

                std::vector<int_t> &index = symV2PartnerLRecvIndex[k0];
                index.assign(static_cast<size_t>(partner_index_size), 0);
                index[0] = partner_nblocks;
                index[1] = partner_nrows;
                index[2] = 0;
                index[3] = SuperSize(k0);
                int_t gid_ptr = LPANEL_HEADER_SIZE;
                int_t px_ptr = LPANEL_HEADER_SIZE + partner_nblocks;
                int_t row_ptr = LPANEL_HEADER_SIZE + 2 * partner_nblocks + 1;
                index[px_ptr] = 0;
                for (int_t ib = 0; ib < partner_nblocks; ++ib)
                {
                    index[gid_ptr + ib] = blocks[ib].gid;
                    index[px_ptr + ib + 1] =
                        index[px_ptr + ib] +
                        static_cast<int_t>(blocks[ib].cols.size());
                    for (size_t j = 0; j < blocks[ib].cols.size(); ++j)
                        index[row_ptr++] = blocks[ib].cols[j];
                }

                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t recv_pos = static_cast<size_t>(k0) *
                                          static_cast<size_t>(Pr) +
                                      static_cast<size_t>(pr);
                    std::vector<SymV2CachedPartnerBlock> &recv_blocks =
                        cached_partner_recv_blocks[recv_pos];
                    std::vector<int_t> &recv_map =
                        symV2PartnerLRecvMap[recv_pos];
                    std::vector<int_t> &src_index =
                        symV2PartnerLRecvIndexBySrc[recv_pos];
                    if (!recv_blocks.empty())
                    {
                        int_t src_nblocks =
                            static_cast<int_t>(recv_blocks.size());
                        int_t src_nrows = 0;
                        for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                            src_nrows +=
                                static_cast<int_t>(recv_blocks[rb].cols.size());
                        int_t src_index_size =
                            LPANEL_HEADER_SIZE + 2 * src_nblocks + 1 +
                            src_nrows;
                        if (src_index_size > maxSymPartnerLidxCount)
                            ABORT("SymFact V2 per-source fragment index exceeds receive buffer.");
                        src_index.assign(static_cast<size_t>(src_index_size), 0);
                        src_index[0] = src_nblocks;
                        src_index[1] = src_nrows;
                        src_index[2] = 0;
                        src_index[3] = SuperSize(k0);
                        int_t src_gid_ptr = LPANEL_HEADER_SIZE;
                        int_t src_px_ptr = LPANEL_HEADER_SIZE + src_nblocks;
                        int_t src_row_ptr =
                            LPANEL_HEADER_SIZE + 2 * src_nblocks + 1;
                        src_index[src_px_ptr] = 0;
                        for (int_t rb = 0; rb < src_nblocks; ++rb)
                        {
                            src_index[src_gid_ptr + rb] =
                                recv_blocks[static_cast<size_t>(rb)].gid;
                            src_index[src_px_ptr + rb + 1] =
                                src_index[src_px_ptr + rb] +
                                static_cast<int_t>(
                                    recv_blocks[static_cast<size_t>(rb)].cols.size());
                            for (size_t rc = 0;
                                 rc < recv_blocks[static_cast<size_t>(rb)].cols.size();
                                 ++rc)
                                src_index[src_row_ptr++] =
                                    recv_blocks[static_cast<size_t>(rb)].cols[rc];
                        }
                    }
                    long long expected_values = 0;
                    int_t src_offset = 0;
                    for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                    {
                        int_t ib = GLOBAL_BLOCK_NOT_FOUND;
                        for (int_t probe = 0; probe < partner_nblocks; ++probe)
                        {
                            if (blocks[probe].gid == recv_blocks[rb].gid)
                            {
                                ib = probe;
                                break;
                            }
                        }
                        if (ib == GLOBAL_BLOCK_NOT_FOUND)
                            ABORT("SymFact V2 partner-L receive map cannot find a block.");
                        int_t nrows =
                            static_cast<int_t>(recv_blocks[rb].cols.size());
                        recv_map.push_back(index[px_ptr + ib]);
                        recv_map.push_back(nrows);
                        recv_map.push_back(src_offset);
                        src_offset += nrows * SuperSize(k0);
                        expected_values +=
                            static_cast<long long>(nrows) *
                            static_cast<long long>(SuperSize(k0));
                    }
                    if (exact_partner_fragment_demand_setup)
                    {
                        if (expected_values >
                            static_cast<long long>(
                                std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 exact partner-L receive size is too large.");
                        symV2PartnerLRecvSizes[recv_pos] =
                            static_cast<int>(expected_values);
                    }
                    else if (expected_values !=
                             static_cast<long long>(
                                 symV2PartnerLRecvSizes[recv_pos]))
                    {
                        ABORT("SymFact V2 partner-L receive map size mismatch.");
                    }
                }

                std::vector<SymV2CachedPartnerBlock> &row_blocks =
                    cached_row_blocks[k0];
                if (row_blocks.empty())
                    continue;
                std::sort(row_blocks.begin(), row_blocks.end(),
                          [](const SymV2CachedPartnerBlock &a,
                             const SymV2CachedPartnerBlock &b)
                          {
                              return a.gid < b.gid;
                          });
                std::vector<SymV2CachedPartnerBlock> unique_row_blocks;
                unique_row_blocks.reserve(row_blocks.size());
                for (size_t rb = 0; rb < row_blocks.size(); ++rb)
                {
                    if (!unique_row_blocks.empty() &&
                        unique_row_blocks.back().gid == row_blocks[rb].gid)
                        continue;
                    unique_row_blocks.push_back(row_blocks[rb]);
                }
                row_blocks.swap(unique_row_blocks);

                int_t row_nblocks = static_cast<int_t>(row_blocks.size());
                int_t row_nrows = 0;
                for (size_t rb = 0; rb < row_blocks.size(); ++rb)
                    row_nrows +=
                        static_cast<int_t>(row_blocks[rb].cols.size());
                int_t row_index_size =
                    LPANEL_HEADER_SIZE + 2 * row_nblocks + 1 + row_nrows;
                if (row_index_size > maxSymV2RowFragIdxRecvCount)
                    ABORT("SymFact V2 row-fragment cached index exceeds receive buffer.");
                if (static_cast<int64_t>(row_nrows) *
                        static_cast<int64_t>(SuperSize(k0)) >
                    static_cast<int64_t>(maxSymV2RowFragValRecvCount))
                    ABORT("SymFact V2 row-fragment cached values exceed receive buffer.");

                std::vector<int_t> &row_index = symV2RowFragRecvIndex[k0];
                row_index.assign(static_cast<size_t>(row_index_size), 0);
                row_index[0] = row_nblocks;
                row_index[1] = row_nrows;
                row_index[2] = 0;
                row_index[3] = SuperSize(k0);
                int_t row_gid_ptr = LPANEL_HEADER_SIZE;
                int_t row_px_ptr = LPANEL_HEADER_SIZE + row_nblocks;
                int_t row_data_ptr = LPANEL_HEADER_SIZE + 2 * row_nblocks + 1;
                row_index[row_px_ptr] = 0;
                for (int_t rb = 0; rb < row_nblocks; ++rb)
                {
                    row_index[row_gid_ptr + rb] = row_blocks[rb].gid;
                    row_index[row_px_ptr + rb + 1] =
                        row_index[row_px_ptr + rb] +
                        static_cast<int_t>(row_blocks[rb].cols.size());
                    for (size_t rc = 0; rc < row_blocks[rb].cols.size(); ++rc)
                        row_index[row_data_ptr++] = row_blocks[rb].cols[rc];
                }

                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_t recv_pos =
                        static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                        static_cast<size_t>(pc);
                    std::vector<SymV2CachedPartnerBlock> &recv_blocks =
                        cached_row_recv_blocks[recv_pos];
                    std::vector<int_t> &recv_map =
                        symV2RowFragRecvMap[recv_pos];
                    long long expected_values = 0;
                    int_t src_offset = 0;
                    for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                    {
                        int_t ib = GLOBAL_BLOCK_NOT_FOUND;
                        for (int_t probe = 0; probe < row_nblocks; ++probe)
                        {
                            if (row_blocks[probe].gid == recv_blocks[rb].gid)
                            {
                                ib = probe;
                                break;
                            }
                        }
                        if (ib == GLOBAL_BLOCK_NOT_FOUND)
                            ABORT("SymFact V2 row-fragment receive map cannot find a block.");
                        int_t nrows =
                            static_cast<int_t>(recv_blocks[rb].cols.size());
                        recv_map.push_back(row_index[row_px_ptr + ib]);
                        recv_map.push_back(nrows);
                        recv_map.push_back(src_offset);
                        src_offset += nrows * SuperSize(k0);
                        expected_values +=
                            static_cast<long long>(nrows) *
                            static_cast<long long>(SuperSize(k0));
                    }
                    if (exact_row_fragment_demand_setup)
                    {
                        if (expected_values >
                            static_cast<long long>(
                                std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 exact row-fragment receive size is too large.");
                        symV2RowFragRecvSizes[recv_pos] =
                            static_cast<int>(expected_values);
                    }
                    else if (expected_values !=
                             static_cast<long long>(
                                 symV2RowFragRecvSizes[recv_pos]))
                    {
                        ABORT("SymFact V2 row-fragment receive map size mismatch.");
                    }
                }
            }
            if (exact_partner_fragment_demand_setup ||
                exact_row_fragment_demand_setup)
            {
                const bool use_exact_send_map_index =
                    exact_send_map_index_setup &&
                    !symV2ExactSendSegments.empty();
                auto append_exact_send_map_scan =
                    [&](size_t flat, const std::vector<int_t> &requested,
                        std::vector<int_t> &out)
                {
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size() ||
                        flat >= symV2PartnerLMapOffsets.size())
                        ABORT("SymFact V2 exact send map source is invalid.");
                    if (requested.empty())
                        ABORT("SymFact V2 exact send map has no requested blocks.");
                    int_t lk = static_cast<int_t>(
                        flat / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 exact send map has invalid panel.");
                    int_t ksupc = SuperSize(k0);
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    if (meta.empty())
                        ABORT("SymFact V2 exact send map source metadata is empty.");

                    size_t map_pos = symV2PartnerLMapOffsets[flat];
                    size_t map_end = map_pos +
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                    if (map_end > symV2PartnerLPackedMaps.size() ||
                        map_end < map_pos)
                        ABORT("SymFact V2 exact send map source size is invalid.");

                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 exact send metadata is truncated.");
                        int_t gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 ||
                            meta_pos + static_cast<size_t>(len) > meta.size())
                            ABORT("SymFact V2 exact send metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 exact send map segment");
                        if (map_pos + value_count > map_end ||
                            map_pos + value_count < map_pos)
                            ABORT("SymFact V2 exact send map segment is invalid.");
                        if (std::binary_search(requested.begin(),
                                               requested.end(), gid))
                        {
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() +
                                           map_pos,
                                       symV2PartnerLPackedMaps.begin() +
                                           map_pos + value_count);
                        }
                        map_pos += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (map_pos != map_end)
                        ABORT("SymFact V2 exact send map source size mismatch.");
                };

                auto append_exact_send_map_indexed =
                    [&](size_t flat, const std::vector<int_t> &requested,
                        std::vector<int_t> &out)
                {
                    if (flat >= symV2ExactSendSegments.size())
                        ABORT("SymFact V2 exact send map index source is invalid.");
                    if (requested.empty())
                        ABORT("SymFact V2 exact send map has no requested blocks.");
                    const std::vector<SymV2ExactSendSegment> &segments =
                        symV2ExactSendSegments[flat];
                    if (segments.empty())
                        ABORT("SymFact V2 exact send map index source is empty.");
                    for (size_t ri = 0; ri < requested.size(); ++ri)
                    {
                        SymV2ExactSendSegment key;
                        key.gid = requested[ri];
                        key.map_offset = 0;
                        key.map_count = 0;
                        std::vector<SymV2ExactSendSegment>::const_iterator it =
                            std::lower_bound(
                                segments.begin(), segments.end(), key,
                                [](const SymV2ExactSendSegment &a,
                                   const SymV2ExactSendSegment &b)
                                {
                                    return a.gid < b.gid;
                                });
                        bool found = false;
                        for (; it != segments.end() &&
                               it->gid == requested[ri]; ++it)
                        {
                            if (it->map_offset + it->map_count >
                                    symV2PartnerLPackedMaps.size() ||
                                it->map_offset + it->map_count <
                                    it->map_offset)
                                ABORT("SymFact V2 exact send map index segment is invalid.");
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() +
                                           it->map_offset,
                                       symV2PartnerLPackedMaps.begin() +
                                           it->map_offset + it->map_count);
                            found = true;
                        }
                        if (!found)
                            ABORT("SymFact V2 exact send map index cannot find a block.");
                    }
                };

                auto append_exact_send_map =
                    [&](size_t flat, const std::vector<int_t> &requested,
                        std::vector<int_t> &out)
                {
                    if (use_exact_send_map_index)
                        append_exact_send_map_indexed(flat, requested, out);
                    else
                        append_exact_send_map_scan(flat, requested, out);
                };

                auto build_exact_send_maps =
                    [&](const std::vector<std::vector<int_t> > &block_gids,
                        int dest_dim,
                        std::vector<int> &sizes,
                        std::vector<double *> &bufs_gpu,
                        std::vector<int_t *> &maps_gpu,
                        double **buf_pool_gpu,
                        int_t **map_pool_gpu,
                        size_t *buf_pool_count,
                        size_t *map_pool_count,
                        std::vector<std::vector<double> > &host_bufs,
                        std::vector<double *> &host_bufs_pinned,
                        const char *what)
                {
                    if (block_gids.size() != sizes.size() ||
                        block_gids.size() != bufs_gpu.size() ||
                        block_gids.size() != maps_gpu.size())
                        ABORT("SymFact V2 exact send slot table is invalid.");
                    std::vector<size_t> offsets(block_gids.size(), 0);
                    std::vector<int_t> packed_maps;
                    for (size_t slot = 0; slot < block_gids.size(); ++slot)
                    {
                        if (block_gids[slot].empty())
                            continue;
                        size_t flat = slot / static_cast<size_t>(dest_dim);
                        std::vector<int_t> exact_map;
                        append_exact_send_map(flat, block_gids[slot],
                                              exact_map);
                        if (exact_map.empty())
                            ABORT("SymFact V2 exact send demand produced no data.");
                        if (exact_map.size() >
                            static_cast<size_t>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 exact send map is too large for MPI.");
                        offsets[slot] = packed_maps.size();
                        sizes[slot] = static_cast<int>(exact_map.size());
                        packed_maps.insert(packed_maps.end(),
                                           exact_map.begin(), exact_map.end());
                    }

                    *buf_pool_count = packed_maps.size();
                    *map_pool_count = packed_maps.size();
                    if (!packed_maps.empty())
                    {
                        gpuErrchk(cudaMalloc(
                            (void **)buf_pool_gpu,
                            xlu_checked_product(packed_maps.size(),
                                                sizeof(double), what)));
                        gpuErrchk(cudaMalloc(
                            (void **)map_pool_gpu,
                            xlu_checked_product(packed_maps.size(),
                                                sizeof(int_t), what)));
                        gpuErrchk(cudaMemcpy(
                            *map_pool_gpu, packed_maps.data(),
                            sizeof(int_t) * packed_maps.size(),
                            cudaMemcpyHostToDevice));
                    }

                    bool use_pinned_host =
                        !superlu_cuda_aware_mpi() &&
                        superlu_sym_v2_pinned_staging();
                    for (size_t slot = 0; slot < sizes.size(); ++slot)
                    {
                        if (sizes[slot] <= 0)
                            continue;
                        bufs_gpu[slot] = *buf_pool_gpu + offsets[slot];
                        maps_gpu[slot] = *map_pool_gpu + offsets[slot];
                        if (superlu_cuda_aware_mpi())
                            continue;
                        if (use_pinned_host)
                        {
                            gpuErrchk(cudaMallocHost(
                                (void **)&host_bufs_pinned[slot],
                                xlu_checked_product(
                                    static_cast<size_t>(sizes[slot]),
                                    sizeof(double), what)));
                        }
                        else
                        {
                            host_bufs[slot].resize(
                                static_cast<size_t>(sizes[slot]));
                        }
                    }
                };

                auto build_exact_send_host_maps_slot_order =
                    [&](const std::vector<std::vector<int_t> > &block_gids,
                        int dest_dim, std::vector<int> &sizes,
                        std::vector<size_t> &offsets,
                        std::vector<int_t> &packed_maps)
                {
                    if (block_gids.size() != sizes.size())
                        ABORT("SymFact V2 exact send host slot table is invalid.");
                    offsets.assign(block_gids.size(), 0);
                    packed_maps.clear();
                    for (size_t slot = 0; slot < block_gids.size(); ++slot)
                    {
                        if (block_gids[slot].empty())
                            continue;
                        size_t flat = slot / static_cast<size_t>(dest_dim);
                        std::vector<int_t> exact_map;
                        append_exact_send_map(flat, block_gids[slot],
                                              exact_map);
                        if (exact_map.empty())
                            ABORT("SymFact V2 exact send demand produced no data.");
                        if (exact_map.size() >
                            static_cast<size_t>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 exact send map is too large for MPI.");
                        offsets[slot] = packed_maps.size();
                        sizes[slot] = static_cast<int>(exact_map.size());
                        packed_maps.insert(packed_maps.end(),
                                           exact_map.begin(), exact_map.end());
                    }
                };

                auto build_exact_row_send_host_maps =
                    [&](const std::vector<std::vector<int_t> > &block_gids,
                        std::vector<int> &sizes,
                        std::vector<size_t> &offsets,
                        std::vector<int_t> &packed_maps)
                {
                    if (block_gids.size() != sizes.size())
                        ABORT("SymFact V2 exact row send host slot table is invalid.");
                    offsets.assign(block_gids.size(), 0);
                    packed_maps.clear();
                    for (int_t lk = 0; lk < local_cols; ++lk)
                    {
                        for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                        {
                            for (int pc = 0; pc < Pc; ++pc)
                            {
                                size_t flat =
                                    static_cast<size_t>(lk) *
                                        static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc);
                                size_t slot =
                                    flat * static_cast<size_t>(Pc) +
                                    static_cast<size_t>(pc_dest);
                                if (slot >= block_gids.size())
                                    ABORT("SymFact V2 exact row send slot is invalid.");
                                if (block_gids[slot].empty())
                                    continue;
                                std::vector<int_t> exact_map;
                                append_exact_send_map(flat, block_gids[slot],
                                                      exact_map);
                                if (exact_map.empty())
                                    ABORT("SymFact V2 exact row send demand produced no data.");
                                if (exact_map.size() >
                                    static_cast<size_t>(std::numeric_limits<int>::max()))
                                    ABORT("SymFact V2 exact row send map is too large for MPI.");
                                offsets[slot] = packed_maps.size();
                                sizes[slot] =
                                    static_cast<int>(exact_map.size());
                                packed_maps.insert(packed_maps.end(),
                                                   exact_map.begin(),
                                                   exact_map.end());
                            }
                        }
                    }
                };

                double tExactSendMapBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
                if (exact_partner_fragment_demand_setup)
                    build_exact_send_maps(
                        partner_exact_send_block_gids, Pr,
                        symV2PartnerLExactSendSizes,
                        symV2PartnerLExactSendBufsGPU,
                        symV2PartnerLExactSendMapsGPU,
                        &symV2PartnerLExactSendBufPoolGPU,
                        &symV2PartnerLExactSendMapPoolGPU,
                        &symV2PartnerLExactSendBufPoolCount,
                        &symV2PartnerLExactSendMapPoolCount,
                        symV2PartnerLExactHostSendBufs,
                        symV2PartnerLExactHostSendBufsPinned,
                        "SymFact V2 exact partner-L send map");
                if (exact_row_fragment_demand_setup)
                {
                    if (superlu_sym_v2_rowfrag_destination_path() &&
                        superlu_sym_v2_exact_map_index())
                    {
                        build_exact_row_send_host_maps(
                            row_exact_send_block_gids,
                            symV2RowFragExactSendSizes,
                            symV2RowFragExactSendMapOffsets,
                            symV2RowFragExactSendMapsHost);
                    }
                    else
                    {
                        build_exact_send_host_maps_slot_order(
                        row_exact_send_block_gids, Pc,
                        symV2RowFragExactSendSizes,
                        symV2RowFragExactSendMapOffsets,
                        symV2RowFragExactSendMapsHost);
                    }
                    symV2RowFragExactSendBufPoolCount = 0;
                    symV2RowFragExactSendMapPoolCount =
                        symV2RowFragExactSendMapsHost.size();
                }
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_EXACT_SEND_MAP_BUILD,
                        SuperLU_timer_() - tExactSendMapBuild);
            }

            if (pc_fragment_schur_setup &&
                superlu_sym_v2_row_l_direct_recv())
            {
                struct SymV2DirectRowBlock
                {
                    int_t gid;
                    int chunk_pc;
                };
                auto append_all_block_gids_from_meta =
                    [&](size_t flat, std::vector<SymV2DirectRowBlock> &out)
                {
                    if (flat >= symL2LSendMeta.size())
                        ABORT("SymFact V2 direct row-L metadata source is invalid.");
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t meta_pos = 0;
                    int chunk_pc = static_cast<int>(
                        flat % static_cast<size_t>(Pc));
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 direct row-L metadata is truncated.");
                        SymV2DirectRowBlock block;
                        block.gid = meta[meta_pos++];
                        block.chunk_pc = chunk_pc;
                        int_t len = meta[meta_pos++];
                        if (len < 0 ||
                            meta_pos + static_cast<size_t>(len) >
                                meta.size())
                            ABORT("SymFact V2 direct row-L metadata block is invalid.");
                        out.push_back(block);
                        meta_pos += static_cast<size_t>(len);
                    }
                };

                auto append_source_map_for_gid =
                    [&](size_t flat, int_t gid, std::vector<int_t> &out)
                {
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size() ||
                        flat >= symV2PartnerLMapOffsets.size())
                        ABORT("SymFact V2 direct row-L source map is invalid.");
                    int_t lk = static_cast<int_t>(
                        flat / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 direct row-L source panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t map_pos = symV2PartnerLMapOffsets[flat];
                    size_t map_end = map_pos +
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                    if (map_end > symV2PartnerLPackedMaps.size() ||
                        map_end < map_pos)
                        ABORT("SymFact V2 direct row-L source map size is invalid.");
                    bool found = false;
                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 direct row-L metadata is truncated.");
                        int_t block_gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 ||
                            meta_pos + static_cast<size_t>(len) >
                                meta.size())
                            ABORT("SymFact V2 direct row-L metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 direct row-L send map segment");
                        if (map_pos + value_count > map_end ||
                            map_pos + value_count < map_pos)
                            ABORT("SymFact V2 direct row-L source map segment is invalid.");
                        if (block_gid == gid)
                        {
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() +
                                           map_pos,
                                       symV2PartnerLPackedMaps.begin() +
                                           map_pos + value_count);
                            found = true;
                        }
                        map_pos += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (map_pos != map_end)
                        ABORT("SymFact V2 direct row-L source map size mismatch.");
                    if (!found)
                        ABORT("SymFact V2 direct row-L source map cannot find a requested block.");
                };

                double tDirectRowMapBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
                std::fill(symV2RowDirectSendSizes.begin(),
                          symV2RowDirectSendSizes.end(), 0);
                std::fill(symV2RowDirectSendMapOffsets.begin(),
                          symV2RowDirectSendMapOffsets.end(), 0);
                symV2RowDirectSendMapsHost.clear();

                std::vector<SymV2DirectRowBlock> direct_blocks;
                for (int_t lk = 0; lk < local_cols; ++lk)
                {
                    for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                    {
                        direct_blocks.clear();
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            size_t active_pos =
                                flat * static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc_dest);
                            if (active_pos >= symV2RowFragSendActive.size())
                                ABORT("SymFact V2 direct row-L send mask is invalid.");
                            if (!symV2RowFragSendActive[active_pos])
                                continue;
                            if (exact_row_fragment_demand_setup)
                            {
                                if (active_pos >=
                                    row_exact_send_block_gids.size())
                                    ABORT("SymFact V2 direct row-L exact demand slot is invalid.");
                                const std::vector<int_t> &blocks =
                                    row_exact_send_block_gids[active_pos];
                                if (blocks.empty())
                                    ABORT("SymFact V2 direct row-L exact demand is empty.");
                                for (size_t bi = 0; bi < blocks.size(); ++bi)
                                {
                                    SymV2DirectRowBlock block;
                                    block.gid = blocks[bi];
                                    block.chunk_pc = pc;
                                    direct_blocks.push_back(block);
                                }
                            }
                            else
                            {
                                append_all_block_gids_from_meta(flat,
                                                                direct_blocks);
                            }
                        }
                        if (direct_blocks.empty())
                            continue;
                        std::sort(
                            direct_blocks.begin(), direct_blocks.end(),
                            [](const SymV2DirectRowBlock &a,
                               const SymV2DirectRowBlock &b)
                            {
                                if (a.gid != b.gid)
                                    return a.gid < b.gid;
                                return a.chunk_pc < b.chunk_pc;
                            });
                        std::vector<SymV2DirectRowBlock> unique_blocks;
                        unique_blocks.reserve(direct_blocks.size());
                        for (size_t bi = 0; bi < direct_blocks.size(); ++bi)
                        {
                            if (!unique_blocks.empty() &&
                                unique_blocks.back().gid ==
                                    direct_blocks[bi].gid)
                                continue;
                            unique_blocks.push_back(direct_blocks[bi]);
                        }

                        size_t slot =
                            static_cast<size_t>(lk) *
                                static_cast<size_t>(Pc) +
                            static_cast<size_t>(pc_dest);
                        if (slot >= symV2RowDirectSendSizes.size() ||
                            slot >= symV2RowDirectSendMapOffsets.size())
                            ABORT("SymFact V2 direct row-L send slot is invalid.");
                        symV2RowDirectSendMapOffsets[slot] =
                            symV2RowDirectSendMapsHost.size();
                        int_t k0 = symV2PanelGid(lk);
                        if (k0 < 0 || k0 >= nsupers)
                            ABORT("SymFact V2 direct row-L panel is invalid.");
                        int_t ksupc = SuperSize(k0);
                        std::vector<std::vector<int_t> > block_maps(
                            unique_blocks.size());
                        std::vector<int_t> block_nrows(unique_blocks.size(), 0);
                        for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                        {
                            size_t flat =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(
                                    unique_blocks[bi].chunk_pc);
                            append_source_map_for_gid(
                                flat, unique_blocks[bi].gid,
                                block_maps[bi]);
                            if (ksupc <= 0 ||
                                block_maps[bi].size() %
                                        static_cast<size_t>(ksupc) !=
                                    0)
                                ABORT("SymFact V2 direct row-L block map has invalid width.");
                            size_t nrows =
                                block_maps[bi].size() /
                                static_cast<size_t>(ksupc);
                            if (nrows >
                                static_cast<size_t>(
                                    std::numeric_limits<int_t>::max()))
                                ABORT("SymFact V2 direct row-L block is too large.");
                            block_nrows[bi] = static_cast<int_t>(nrows);
                        }
                        for (int_t col = 0; col < ksupc; ++col)
                        {
                            for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                            {
                                size_t src =
                                    static_cast<size_t>(col) *
                                    static_cast<size_t>(block_nrows[bi]);
                                symV2RowDirectSendMapsHost.insert(
                                    symV2RowDirectSendMapsHost.end(),
                                    block_maps[bi].begin() + src,
                                    block_maps[bi].begin() + src +
                                        block_nrows[bi]);
                            }
                        }
                        size_t map_count =
                            symV2RowDirectSendMapsHost.size() -
                            symV2RowDirectSendMapOffsets[slot];
                        if (map_count >
                            static_cast<size_t>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 direct row-L send map is too large for MPI.");
                        symV2RowDirectSendSizes[slot] =
                            static_cast<int>(map_count);
                    }
                }
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_EXACT_SEND_MAP_BUILD,
                        SuperLU_timer_() - tDirectRowMapBuild);
            }
            if (profile_setup)
                symV2SetupProfileAdd(
                    SYM_V2_SETUP_PARTNER_RECV_MAP_BUILD,
                    SuperLU_timer_() - tPartnerRecvMapBuild);

            size_t total_recv_map = 0;
            for (size_t pos = 0; pos < symV2PartnerLRecvMap.size(); ++pos)
            {
                symV2PartnerLRecvMapOffsets[pos] = total_recv_map;
                size_t map_size = symV2PartnerLRecvMap[pos].size();
                if (total_recv_map >
                    std::numeric_limits<size_t>::max() - map_size)
                    ABORT("SymFact V2 partner-L receive map size overflows.");
                total_recv_map += map_size;
            }
            symV2PartnerLRecvMapPoolCount = total_recv_map;
            symV2PartnerLRecvMapsGPU.assign(compact_count, NULL);
            if (total_recv_map > 0)
            {
                std::vector<int_t> packed_recv_maps(total_recv_map, 0);
                for (size_t pos = 0; pos < symV2PartnerLRecvMap.size(); ++pos)
                {
                    if (symV2PartnerLRecvMap[pos].empty())
                        continue;
                    std::copy(symV2PartnerLRecvMap[pos].begin(),
                              symV2PartnerLRecvMap[pos].end(),
                              packed_recv_maps.begin() +
                                  symV2PartnerLRecvMapOffsets[pos]);
                }
                gpuErrchk(cudaMalloc(
                    (void **)&symV2PartnerLRecvMapPoolGPU,
                    xlu_checked_product(total_recv_map, sizeof(int_t),
                                        "SymFact V2 partner-L receive map pool")));
                gpuErrchk(cudaMemcpy(
                    symV2PartnerLRecvMapPoolGPU, packed_recv_maps.data(),
                    sizeof(int_t) * total_recv_map,
                    cudaMemcpyHostToDevice));
                for (size_t pos = 0; pos < symV2PartnerLRecvMap.size(); ++pos)
                {
                    if (!symV2PartnerLRecvMap[pos].empty())
                        symV2PartnerLRecvMapsGPU[pos] =
                            symV2PartnerLRecvMapPoolGPU +
                            symV2PartnerLRecvMapOffsets[pos];
                }
            }

            size_t total_row_recv_map = 0;
            for (size_t pos = 0; pos < symV2RowFragRecvMap.size(); ++pos)
            {
                symV2RowFragRecvMapOffsets[pos] = total_row_recv_map;
                size_t map_size = symV2RowFragRecvMap[pos].size();
                if (total_row_recv_map >
                    std::numeric_limits<size_t>::max() - map_size)
                    ABORT("SymFact V2 row-fragment receive map size overflows.");
                total_row_recv_map += map_size;
            }
            symV2RowFragRecvMapPoolCount = total_row_recv_map;
            symV2RowFragRecvMapsGPU.assign(row_chunk_count, NULL);
            if (total_row_recv_map > 0)
            {
                std::vector<int_t> packed_row_recv_maps(total_row_recv_map, 0);
                for (size_t pos = 0; pos < symV2RowFragRecvMap.size(); ++pos)
                {
                    if (symV2RowFragRecvMap[pos].empty())
                        continue;
                    std::copy(symV2RowFragRecvMap[pos].begin(),
                              symV2RowFragRecvMap[pos].end(),
                              packed_row_recv_maps.begin() +
                                  symV2RowFragRecvMapOffsets[pos]);
                }
                gpuErrchk(cudaMalloc(
                    (void **)&symV2RowFragRecvMapPoolGPU,
                    xlu_checked_product(total_row_recv_map, sizeof(int_t),
                                        "SymFact V2 row-fragment receive map pool")));
                gpuErrchk(cudaMemcpy(
                    symV2RowFragRecvMapPoolGPU,
                    packed_row_recv_maps.data(),
                    sizeof(int_t) * total_row_recv_map,
                    cudaMemcpyHostToDevice));
                for (size_t pos = 0; pos < symV2RowFragRecvMap.size(); ++pos)
                {
                    if (!symV2RowFragRecvMap[pos].empty())
                        symV2RowFragRecvMapsGPU[pos] =
                            symV2RowFragRecvMapPoolGPU +
                            symV2RowFragRecvMapOffsets[pos];
                }
            }
        }
        xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after send GPU L2U map setup");
    }
#endif

    xlu_sym_gpu3d_trace(grid3d, "exit initSymFactWorkspace");
    return 0;
}

template <>
inline int xLUstruct_t<double>::freeSymFactWorkspace()
{
    if (symL2UOrders != NULL)
    {
        SUPERLU_FREE(symL2UOrders);
        symL2UOrders = NULL;
    }

    if (symFactWork != NULL)
    {
        SUPERLU_FREE(symFactWork);
        symFactWork = NULL;
    }

    if (symFactIPIV != NULL)
    {
        SUPERLU_FREE(symFactIPIV);
        symFactIPIV = NULL;
    }

#ifdef HAVE_CUDA
    for (size_t i = 0; i < symL2USendBufsGPU.size(); ++i)
        if (symL2USendBufsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2USendBufsGPU[i]));
    for (size_t i = 0; i < symL2USendMapsGPU.size(); ++i)
        if (symL2USendMapsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2USendMapsGPU[i]));
    if (symV2PartnerLSendBufPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2PartnerLSendBufPoolGPU));
        symV2PartnerLSendBufPoolGPU = NULL;
    }
    else
    {
        for (size_t i = 0; i < symV2PartnerLSendBufsGPU.size(); ++i)
            if (symV2PartnerLSendBufsGPU[i] != NULL)
                gpuErrchk(cudaFree(symV2PartnerLSendBufsGPU[i]));
    }
    if (symL2LSendMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symL2LSendMapPoolGPU));
        symL2LSendMapPoolGPU = NULL;
    }
    else
    {
        for (size_t i = 0; i < symL2LSendMapsGPU.size(); ++i)
            if (symL2LSendMapsGPU[i] != NULL)
                gpuErrchk(cudaFree(symL2LSendMapsGPU[i]));
    }
    if (symV2PartnerLExactSendBufPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2PartnerLExactSendBufPoolGPU));
        symV2PartnerLExactSendBufPoolGPU = NULL;
    }
    if (symV2PartnerLExactSendMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2PartnerLExactSendMapPoolGPU));
        symV2PartnerLExactSendMapPoolGPU = NULL;
    }
    if (symV2RowFragExactSendBufPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2RowFragExactSendBufPoolGPU));
        symV2RowFragExactSendBufPoolGPU = NULL;
    }
    if (symV2RowFragExactSendMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2RowFragExactSendMapPoolGPU));
        symV2RowFragExactSendMapPoolGPU = NULL;
    }
    if (symV2PartnerLRecvMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2PartnerLRecvMapPoolGPU));
        symV2PartnerLRecvMapPoolGPU = NULL;
    }
    if (symV2RowFragRecvMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2RowFragRecvMapPoolGPU));
        symV2RowFragRecvMapPoolGPU = NULL;
    }
    symV2PartnerLSendBufPoolCount = 0;
    symL2LSendMapPoolCount = 0;
    symV2PartnerLExactSendBufPoolCount = 0;
    symV2PartnerLExactSendMapPoolCount = 0;
    symV2RowFragExactSendBufPoolCount = 0;
    symV2RowFragExactSendMapPoolCount = 0;
    symV2PartnerLRecvMapPoolCount = 0;
    symV2RowFragRecvMapPoolCount = 0;
    for (size_t i = 0; i < symL2ULocalMapsGPU.size(); ++i)
        if (symL2ULocalMapsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2ULocalMapsGPU[i]));
    symL2USendBufsGPU.clear();
    symL2USendMapsGPU.clear();
    symV2PartnerLSendBufsGPU.clear();
    symL2LSendMapsGPU.clear();
    symV2PartnerLExactSendBufsGPU.clear();
    symV2PartnerLExactSendMapsGPU.clear();
    symV2RowFragExactSendBufsGPU.clear();
    symV2RowFragExactSendMapsGPU.clear();
    symL2LSendMeta.clear();
    if (symV2PartnerLHostSendPoolPinned != NULL)
    {
        gpuErrchk(cudaFreeHost(symV2PartnerLHostSendPoolPinned));
        symV2PartnerLHostSendPoolPinned = NULL;
    }
    else
    {
        for (size_t i = 0; i < symV2PartnerLHostSendBufsPinned.size(); ++i)
            if (symV2PartnerLHostSendBufsPinned[i] != NULL)
                gpuErrchk(cudaFreeHost(symV2PartnerLHostSendBufsPinned[i]));
    }
    symV2PartnerLHostSendPoolPinnedCount = 0;
    symV2PartnerLHostSendBufsPinned.clear();
    for (size_t i = 0; i < symV2PartnerLExactHostSendBufsPinned.size(); ++i)
        if (symV2PartnerLExactHostSendBufsPinned[i] != NULL)
            gpuErrchk(cudaFreeHost(symV2PartnerLExactHostSendBufsPinned[i]));
    for (size_t i = 0; i < symV2RowFragExactHostSendBufsPinned.size(); ++i)
        if (symV2RowFragExactHostSendBufsPinned[i] != NULL)
            gpuErrchk(cudaFreeHost(symV2RowFragExactHostSendBufsPinned[i]));
    symV2PartnerLExactHostSendBufsPinned.clear();
    symV2RowFragExactHostSendBufsPinned.clear();
    symV2PartnerLHostSendScratchOffsets.clear();
    symV2ExchangeSendSizesScratch.clear();
    symV2ExchangeRecvSizesScratch.clear();
    symV2ExchangeRecvOffsetsScratch.clear();
    symV2ExchangeRecvReqsScratch.clear();
    symV2ExchangeSendReqsScratch.clear();
    symV2PartnerLHostSendBufs.clear();
    symV2PartnerLExactHostSendBufs.clear();
    symV2RowFragExactHostSendBufs.clear();
    symV2RowFragExactSendMapsHost.clear();
    symV2RowFragExactSendMapOffsets.clear();
    symV2RowDirectSendSizes.clear();
    symV2RowDirectSendMapOffsets.clear();
    symV2RowDirectSendMapsHost.clear();
    symV2PartnerLSendSizes.clear();
    symV2PartnerLExactSendSizes.clear();
    symV2RowFragExactSendSizes.clear();
    symV2PartnerLSendRowActive.clear();
    symV2RowFragSendActive.clear();
    symV2PartnerLPrepacked.clear();
    symV2PartnerLRecvSizes.clear();
    symV2PartnerLRecvIndex.clear();
    symV2PartnerLRecvIndexBySrc.clear();
    symV2PartnerLRecvMap.clear();
    symV2PartnerLRecvMapsGPU.clear();
    symV2RowFragRecvSizes.clear();
    symV2RowFragRecvIndex.clear();
    symV2RowFragRecvMap.clear();
    symV2RowFragRecvMapsGPU.clear();
    symL2ULocalMapsGPU.clear();
    symPanelReadyEventIds.clear();
    symV2RawPanelNodes.clear();
    for (size_t i = 0; i < symDiagPrefetchBufs.size(); ++i)
        if (symDiagPrefetchBufs[i] != NULL)
            gpuErrchk(cudaFreeHost(symDiagPrefetchBufs[i]));
    for (size_t i = 0; i < symDiagPrefetchDoneEvents.size(); ++i)
        gpuErrchk(cudaEventDestroy(symDiagPrefetchDoneEvents[i]));
    symDiagPrefetchBufs.clear();
    symDiagPrefetchDoneEvents.clear();
    symDiagPrefetchEventIds.clear();
    symDiagPrefetchNodes.clear();
#endif

    symFactWorkSize = 0;
    return 0;
}

template <>
inline int xLUstruct_t<double>::ensureSymFactWorkSize(int64_t minSize)
{
    if (minSize <= symFactWorkSize)
        return 0;

    if (symFactWork != NULL)
        SUPERLU_FREE(symFactWork);

    symFactWorkSize = minSize;
    size_t work_count = (size_t)symFactWorkSize;
    if ((int64_t)work_count != symFactWorkSize)
        ABORT("SymFact workspace size overflows allocation size.");
    symFactWork = (double *)SUPERLU_MALLOC(
        xlu_checked_product(work_count, sizeof(double), "SymFact work"));
    if (symFactWork == NULL)
        ABORT("Malloc fails for SymFact work[].");

    return 0;
}

template <typename Ftype>
static int_t *xluSymCheckedIndirectMap(
    xLUstruct_t<Ftype> *A,
    typename xLUstruct_t<Ftype>::indirectMapType direction,
    int_t srcLen, int_t *srcVec,
    int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL)
        return srcVec;

    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = A->indirect + thread_id * A->ldt;
    for (int_t i = 0; i < srcLen; ++i)
        if (srcVec[i] >= 0 && srcVec[i] < A->ldt)
            dstIdx[srcVec[i]] = GLOBAL_BLOCK_NOT_FOUND;

    for (int_t i = 0; i < dstLen; ++i)
        if (dstVec[i] >= 0 && dstVec[i] < A->ldt)
            dstIdx[dstVec[i]] = i;

    int_t *RCmap = (direction == xLUstruct_t<Ftype>::ROW_MAP)
                       ? A->indirectRow
                       : A->indirectCol;
    RCmap += thread_id * A->ldt;
    for (int_t i = 0; i < srcLen; ++i)
        RCmap[i] = (srcVec[i] >= 0 && srcVec[i] < A->ldt)
                       ? dstIdx[srcVec[i]]
                       : GLOBAL_BLOCK_NOT_FOUND;

    return RCmap;
}

template <typename Ftype>
static int_t xluSymScatterLowerToL(
    xLUstruct_t<Ftype> *A,
    int_t m, int_t n,
    int_t gi, int_t gj,
    Ftype *Src, int_t ldsrc,
    int_t *srcRowList, int_t *srcColList)
{
    if (gi < gj)
        return 0;

    int_t lj = A->useSymV2Solve() ? A->symV2PanelIndex(gj) : A->g2lCol(gj);
    if (lj < 0)
        return 0;
    if (A->lPanelVec[lj].isEmpty())
        return 0;
    int_t li = A->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return 0;

    Ftype *Dst = A->lPanelVec[lj].blkPtr(li);
    int_t lddst = A->lPanelVec[lj].LDA();
    int_t dstRowLen = A->lPanelVec[lj].nbrow(li);
    int_t *dstRowList = A->lPanelVec[lj].rowList(li);
    int_t dstColLen = A->supersize(gj);
    int_t *dstColList = NULL;

    int_t *rowS2D = xluSymCheckedIndirectMap(
        A, xLUstruct_t<Ftype>::ROW_MAP, m, srcRowList,
        dstRowLen, dstRowList);
    int_t *colS2D = xluSymCheckedIndirectMap(
        A, xLUstruct_t<Ftype>::COL_MAP, n, srcColList,
        dstColLen, dstColList);

    for (int_t j = 0; j < n; ++j)
    {
        int_t dj = (colS2D != NULL) ? colS2D[j] : j;
        if (dj < 0 || dj >= dstColLen)
            continue;
        for (int_t i = 0; i < m; ++i)
        {
            int_t di = (rowS2D != NULL) ? rowS2D[i] : i;
            if (di < 0 || di >= dstRowLen)
                continue;
            Dst[di + lddst * dj] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeHost(
    int_t k, int_t offset, int raw_values)
{
    (void)offset;
    (void)raw_values;
    ABORT("SymFact GPU3D V2 host L-fragment exchange is implemented for double precision only.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2LFragmentExchangeHost(
    int_t k, int_t offset, int raw_values)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (k < 0 || k >= nsupers)
        return 0;
    if (offset < 0 ||
        static_cast<size_t>(offset) >= symPartnerLidxRecvBufs.size() ||
        static_cast<size_t>(offset) >= symPartnerLvalRecvBufs.size())
        ABORT("SymFact V2 host L-fragment exchange has an invalid buffer offset.");

    int_t kcol_ = symV2PanelRoot(k);
    int_t ksupc = SuperSize(k);
    int_t *frag_index = symPartnerLidxRecvBufs[offset];
    double *frag_val = symPartnerLvalRecvBufs[offset];
    std::vector<std::vector<int_t> > send_meta(Pc);
    std::vector<std::vector<double> > send_vals(Pc);

    if (mycol == kcol_)
    {
        double *diag = NULL;
        if (!raw_values)
        {
            if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers) ||
                symV2DiagBlocks[k] == NULL)
                ABORT("SymFact V2 host partner-L diagonal block is missing.");
            diag = symV2DiagBlocks[k];
        }

        xlpanel_t<double> &lpanel = lPanelVec[symV2PanelIndex(k)];
        if (!lpanel.isEmpty())
        {
            int_t st_lb = lpanel.haveDiag() ? 1 : 0;
            int_t nlb = lpanel.nblocks();
            for (int_t lb = st_lb; lb < nlb; ++lb)
            {
                int_t ik = lpanel.gid(lb);
                int_t len = lpanel.nbrow(lb);
                if (len <= 0)
                    continue;

                std::vector<std::pair<int_t, int_t> > row_order;
                row_order.reserve(static_cast<size_t>(len));
                int_t *rows = lpanel.rowList(lb);
                for (int_t i = 0; i < len; ++i)
                    row_order.push_back(std::make_pair(rows[i], i));
                std::sort(row_order.begin(), row_order.end());

                int pc = symV2PanelRoot(ik);
                if (pc < 0 || pc >= Pc)
                    ABORT("SymFact V2 partner-L target process column is invalid.");
                {
                    std::vector<int_t> &meta = send_meta[pc];
                    meta.push_back(ik);
                    meta.push_back(len);
                    for (int_t i = 0; i < len; ++i)
                        meta.push_back(row_order[static_cast<size_t>(i)].first);

                    std::vector<double> &vals = send_vals[pc];
                    size_t old_size = vals.size();
                    vals.resize(old_size +
                                static_cast<size_t>(len) *
                                    static_cast<size_t>(ksupc));
                    double *dst = vals.data() + old_size;
                    double *src = lpanel.blkPtr(lb);
                    int_t lda = lpanel.LDA();
                    for (int_t j = 0; j < ksupc; ++j)
                    {
                        for (int_t i = 0; i < len; ++i)
                        {
                            int_t src_row =
                                row_order[static_cast<size_t>(i)].second;
                            if (raw_values)
                            {
                                dst[i + j * len] = src[src_row + j * lda];
                            }
                            else
                            {
                                double sum = 0.0;
                                for (int_t p = 0; p < ksupc; ++p)
                                    sum += src[src_row + p * lda] *
                                           diag[p + j * ksupc];
                                dst[i + j * len] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int_t> local_meta_payload;
    for (int pc = 0; pc < Pc; ++pc)
    {
        if (send_meta[pc].empty())
            continue;
        local_meta_payload.push_back(pc);
        local_meta_payload.push_back(
            static_cast<int_t>(send_meta[pc].size()));
        local_meta_payload.insert(local_meta_payload.end(),
                                  send_meta[pc].begin(),
                                  send_meta[pc].end());
    }
    if (local_meta_payload.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 host partner-L metadata is too large for MPI.");

    int comm_size = 0;
    MPI_Comm_size(grid->comm, &comm_size);
    int local_meta_count =
        static_cast<int>(local_meta_payload.size());
    std::vector<int> meta_counts(comm_size, 0);
    MPI_Allgather(&local_meta_count, 1, MPI_INT, meta_counts.data(),
                  1, MPI_INT, grid->comm);

    std::vector<int> meta_displs(comm_size, 0);
    long long total_meta_count = 0;
    for (int r = 0; r < comm_size; ++r)
    {
        if (meta_counts[r] < 0)
            ABORT("SymFact V2 host partner-L metadata count is invalid.");
        meta_displs[r] = static_cast<int>(total_meta_count);
        total_meta_count += meta_counts[r];
        if (total_meta_count >
            static_cast<long long>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 host partner-L metadata is too large for MPI.");
    }

    std::vector<int_t> all_meta_payload(
        static_cast<size_t>(total_meta_count));
    MPI_Allgatherv(local_meta_payload.empty()
                       ? NULL
                       : local_meta_payload.data(),
                   local_meta_count, mpi_int_t,
                   all_meta_payload.empty()
                       ? NULL
                       : all_meta_payload.data(),
                   meta_counts.data(), meta_displs.data(), mpi_int_t,
                   grid->comm);

    struct SymV2HostPartnerBlock
    {
        int_t gid;
        std::vector<int_t> rows;
    };
    struct SymV2HostRecvPiece
    {
        int cid;
        int_t nrows;
    };
    std::vector<SymV2HostPartnerBlock> blocks;
    std::vector<std::vector<SymV2HostRecvPiece> > recv_pieces(Pr);
    std::vector<int> recv_sizes(Pr, 0);

    for (int r = 0; r < comm_size; ++r)
    {
        if (MYCOL(r, grid) != kcol_)
            continue;
        int source_pr = MYROW(r, grid);
        size_t meta_pos = static_cast<size_t>(meta_displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta_counts[r]);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 2 > rank_end)
                ABORT("SymFact V2 host partner-L metadata is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= Pc || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 host partner-L metadata is invalid.");
            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (target_pc == mycol)
            {
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 host partner-L block metadata is truncated.");
                    SymV2HostPartnerBlock block;
                    block.gid = all_meta_payload[block_pos++];
                    int_t len = all_meta_payload[block_pos++];
                    if (len < 0 ||
                        block_pos + static_cast<size_t>(len) > block_end)
                        ABORT("SymFact V2 host partner-L block metadata has invalid length.");
                    block.rows.assign(all_meta_payload.begin() + block_pos,
                                      all_meta_payload.begin() + block_pos + len);
                    block_pos += static_cast<size_t>(len);

                    int cid = GLOBAL_BLOCK_NOT_FOUND;
                    for (size_t probe = 0; probe < blocks.size(); ++probe)
                    {
                        if (blocks[probe].gid == block.gid)
                        {
                            cid = static_cast<int>(probe);
                            break;
                        }
                    }
                    if (cid != GLOBAL_BLOCK_NOT_FOUND)
                        ABORT("SymFact V2 host partner-L encountered duplicate block metadata.");
                    cid = static_cast<int>(blocks.size());
                    blocks.push_back(block);
                    recv_pieces[source_pr].push_back({cid, len});
                    recv_sizes[source_pr] += static_cast<int>(len * ksupc);
                }
            }
            meta_pos = block_end;
        }
    }

    std::vector<int> order(blocks.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&blocks](int a, int b)
              {
                  return blocks[static_cast<size_t>(a)].gid <
                         blocks[static_cast<size_t>(b)].gid;
              });

    std::vector<int_t> block_starts(blocks.size(), 0);
    int_t partner_nrows = 0;
    for (size_t oi = 0; oi < order.size(); ++oi)
    {
        int cid = order[oi];
        block_starts[static_cast<size_t>(cid)] = partner_nrows;
        partner_nrows += static_cast<int_t>(
            blocks[static_cast<size_t>(cid)].rows.size());
    }

    int_t partner_nblocks = static_cast<int_t>(blocks.size());
    int_t partner_index_size =
        LPANEL_HEADER_SIZE + 2 * partner_nblocks + 1 + partner_nrows;
	    if (partner_nblocks > 0 &&
	        (partner_index_size > maxSymPartnerLidxCount ||
	         static_cast<int64_t>(partner_nrows) *
	                 static_cast<int64_t>(ksupc) >
	             static_cast<int64_t>(maxSymPartnerLvalCount)))
	        ABORT("SymFact V2 host L-fragment buffer is too small.");

	    if (partner_nblocks > 0 && (frag_index == NULL || frag_val == NULL))
	        ABORT("SymFact V2 host L-fragment receive buffer is missing.");

	    if (frag_index != NULL)
	    {
	        frag_index[0] = partner_nblocks;
	        frag_index[1] = partner_nrows;
	        frag_index[2] = 0;
	        frag_index[3] = ksupc;
	        int_t gid_ptr = LPANEL_HEADER_SIZE;
	        int_t px_ptr = LPANEL_HEADER_SIZE + partner_nblocks;
	        int_t row_ptr = LPANEL_HEADER_SIZE + 2 * partner_nblocks + 1;
	        frag_index[px_ptr] = 0;
	        for (int_t ib = 0; ib < partner_nblocks; ++ib)
	        {
	            int cid = order[static_cast<size_t>(ib)];
	            const SymV2HostPartnerBlock &block =
	                blocks[static_cast<size_t>(cid)];
	            frag_index[gid_ptr + ib] = block.gid;
	            frag_index[px_ptr + ib + 1] =
	                frag_index[px_ptr + ib] +
	                static_cast<int_t>(block.rows.size());
	            for (size_t j = 0; j < block.rows.size(); ++j)
	                frag_index[row_ptr++] = block.rows[j];
	        }
	        if (frag_val != NULL && partner_nrows > 0)
	            std::memset(frag_val, 0,
	                        sizeof(double) *
	                            static_cast<size_t>(partner_nrows) *
	                            static_cast<size_t>(ksupc));
	    }

    std::vector<MPI_Request> recv_reqs;
    std::vector<MPI_Request> send_reqs;
    std::vector<std::vector<double> > recv_buffers(Pr);
    int tag_ub = symFactTagUb;
    for (int pr = 0; pr < Pr; ++pr)
    {
        if (recv_sizes[pr] <= 0)
            continue;
        recv_buffers[pr].resize(static_cast<size_t>(recv_sizes[pr]));
        MPI_Request req;
        MPI_Irecv(recv_buffers[pr].data(), recv_sizes[pr], MPI_DOUBLE,
                  PNUM(pr, kcol_, grid), SLU_MPI_TAG(5, k),
                  grid->comm, &req);
        recv_reqs.push_back(req);
    }

    if (mycol == kcol_)
    {
        for (int pc = 0; pc < Pc; ++pc)
        {
            int size = static_cast<int>(send_vals[pc].size());
            if (size <= 0)
                continue;
            for (int pr = 0; pr < Pr; ++pr)
            {
                MPI_Request req;
                MPI_Isend(send_vals[pc].data(), size, MPI_DOUBLE,
                          PNUM(pr, pc, grid), SLU_MPI_TAG(5, k),
                          grid->comm, &req);
                send_reqs.push_back(req);
            }
        }
    }

    if (!recv_reqs.empty())
        MPI_Waitall(static_cast<int>(recv_reqs.size()), recv_reqs.data(),
                    MPI_STATUSES_IGNORE);

	    if (frag_val != NULL && partner_nrows > 0)
	    {
	        for (int pr = 0; pr < Pr; ++pr)
	        {
            size_t pos = 0;
            for (size_t p = 0; p < recv_pieces[pr].size(); ++p)
            {
                int cid = recv_pieces[pr][p].cid;
                int_t nrows = recv_pieces[pr][p].nrows;
                int_t dst_offset =
                    block_starts[static_cast<size_t>(cid)];
	                size_t need =
	                    static_cast<size_t>(nrows) *
	                    static_cast<size_t>(ksupc);
	                if (pos + need > recv_buffers[pr].size())
	                    ABORT("SymFact V2 host L-fragment receive buffer is truncated.");
	                for (int_t j = 0; j < ksupc; ++j)
	                    std::memcpy(&frag_val[dst_offset +
	                                          j * partner_nrows],
	                                &recv_buffers[pr][pos +
	                                                   static_cast<size_t>(j) *
	                                                       static_cast<size_t>(nrows)],
                                sizeof(double) *
                                    static_cast<size_t>(nrows));
	                pos += need;
	            }
	            if (pos != recv_buffers[pr].size())
	                ABORT("SymFact V2 host L-fragment receive buffer has extra data.");
	        }
	    }

    if (!send_reqs.empty())
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurComplementUpdate(
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;
        blockUpdate(k, ii, jj, lpanel, upanel);
    }

    return 0;
}

// should be called from an openMP region
template <typename Ftype>
int_t *xLUstruct_t<Ftype>::computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        return srcVec;
    }
    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = indirect + thread_id * ldt;
    for (int_t i = 0; i < dstLen; i++)
    {
        // if(thread_id < dstLen)
        dstIdx[dstVec[i]] = i;
    }

    int_t *RCmap = (direction == ROW_MAP) ? indirectRow : indirectCol;
    RCmap += thread_id * ldt;

    for (int_t i = 0; i < srcLen; i++)
    {
        RCmap[i] = dstIdx[srcVec[i]];
    }

    return RCmap;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dScatter(int_t m, int_t n,
                              int_t gi, int_t gj,
                              Ftype *Src, int_t ldsrc,
                              int_t *srcRowList, int_t *srcColList)
{

    Ftype *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    if (gj > gi) // its in upanel
    {
        int li = g2lRow(gi);
        int lj = uPanelVec[li].find(gj);
        Dst = uPanelVec[li].blkPtr(lj);
        lddst = supersize(gi);
        dstRowLen = supersize(gi);
        dstRowList = NULL;
        dstColLen = uPanelVec[li].nbcol(lj);
        dstColList = uPanelVec[li].colList(lj);
        // std::cout<<li<<" "<<lj<<" Dst[0] is"<<Dst[0] << "\n";
    }
    else
    {
        int lj = g2lCol(gj);
        int li = lPanelVec[lj].find(gi);
        Dst = lPanelVec[lj].blkPtr(li);
        lddst = lPanelVec[lj].LDA();
        dstRowLen = lPanelVec[lj].nbrow(li);
        dstRowList = lPanelVec[lj].rowList(li);
        dstColLen = supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int_t *rowS2D = computeIndirectMap(ROW_MAP, m, srcRowList,
                                       dstRowLen, dstRowList);

    // compute source col to dest col mapping
    int_t *colS2D = computeIndirectMap(COL_MAP, n, srcColList,
                                       dstColLen, dstColList);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::packedU2skyline(LUStruct_type<Ftype> *LUstruct)
{

    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    Ftype **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;
    if (Ufstnz_br_ptr == NULL || Unzval_br_ptr == NULL)
        return 0;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            uPanelVec[i].packed2skyline(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
        }
    }

    return 0;
}

int numProcsPerNode(MPI_Comm baseCommunicator);
// int numProcsPerNode(MPI_Comm baseCommunicator)
// {
//     MPI_Comm sharedComm;
//     MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm);
//     int count = 0;
//     MPI_Comm_size(sharedComm, &count);
//     return count;
// }


template <typename Ftype>
int_t xLUstruct_t<Ftype>::lookAheadUpdate(
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t laILoc = lpanel.find(laIdx);
    int_t nub = upanel.nblocks();
    int_t laJLoc = upanel.find(laIdx);

#pragma omp parallel
    {
        /*Next lpanelUpdate*/
#pragma omp for nowait
        for (size_t ii = st_lb; ii < nlb; ii++)
        {
            int_t jj = laJLoc;
            if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }

        /*Next upanelUpdate*/
#pragma omp for nowait
        for (size_t jj = 0; jj < nub; jj++)
        {
            int_t ii = laILoc;
            if (laILoc != GLOBAL_BLOCK_NOT_FOUND && jj != laJLoc)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::blockUpdate(int_t k,
                                 int_t ii, int_t jj, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    int thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif

    Ftype *V = bigV + thread_id * ldt * ldt;

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    superlu_gemm<Ftype>("N", "N",
                  lpanel.nbrow(ii), upanel.nbcol(jj), supersize(k), alpha,
                  lpanel.blkPtr(ii), lpanel.LDA(),
                  upanel.blkPtr(jj), upanel.LDA(), beta,
                  V, lpanel.nbrow(ii));

    // now do the scatter
    int_t ib = lpanel.gid(ii);
    int_t jb = upanel.gid(jj);

    dScatter(lpanel.nbrow(ii), upanel.nbcol(jj),
             ib, jb, V, lpanel.nbrow(ii),
             lpanel.rowList(ii), upanel.colList(jj));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdateExcludeOne(
    int_t k, int_t ex, // suypernodes to be excluded
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;

        if (ii != exILoc && jj != exJLoc)
            blockUpdate(k, ii, jj, lpanel, upanel);
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdatePartLL(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty())
        return 0;
    if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers) ||
        symV2DiagBlocks[k] == NULL)
        ABORT("SymFact V2 host LL update diagonal block is missing.");

    int_t gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int_t gemm_n = lpanel.stRow(jEnd) - lpanel.stRow(jSt);
    int_t gemm_k = supersize(k);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    int64_t raw_count = static_cast<int64_t>(gemm_n) *
                        static_cast<int64_t>(gemm_k);
    int64_t gemm_count = static_cast<int64_t>(gemm_m) *
                         static_cast<int64_t>(gemm_n);
    ensureSymFactWorkSize(raw_count + gemm_count);
    Ftype *rawBlock = symFactWork;
    Ftype *gemmBuff = symFactWork + raw_count;
    Ftype *diag = symV2DiagBlocks[k];

    for (int_t jb = jSt; jb < jEnd; ++jb)
    {
        int_t row_offset = lpanel.stRow(jb) - lpanel.stRow(jSt);
        Ftype *src = lpanel.blkPtr(jb);
        int_t nbrow = lpanel.nbrow(jb);
        for (int_t j = 0; j < gemm_k; ++j)
        {
            for (int_t i = 0; i < nbrow; ++i)
            {
                Ftype sum = zeroT<Ftype>();
                for (int_t p = 0; p < gemm_k; ++p)
                    sum += src[i + p * lpanel.LDA()] *
                           diag[p + j * gemm_k];
                rawBlock[row_offset + i + j * gemm_n] = sum;
            }
        }
    }

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    superlu_gemm<Ftype>("N", "T",
                        gemm_m, gemm_n, gemm_k, alpha,
                        lpanel.blkPtr(iSt), lpanel.LDA(),
                        rawBlock, gemm_n, beta,
                        gemmBuff, gemm_m);

    for (int_t ii = iSt; ii < iEnd; ++ii)
    {
        int_t row_off = lpanel.stRow(ii) - lpanel.stRow(iSt);
        for (int_t jj = jSt; jj < jEnd; ++jj)
        {
            if (lpanel.gid(ii) < lpanel.gid(jj))
                continue;
            int_t col_off = lpanel.stRow(jj) - lpanel.stRow(jSt);
            xluSymScatterLowerToL(this,
                                  lpanel.nbrow(ii), lpanel.nbrow(jj),
                                  lpanel.gid(ii), lpanel.gid(jj),
                                  &gemmBuff[row_off + col_off * gemm_m],
                                  gemm_m,
                                  lpanel.rowList(ii), lpanel.rowList(jj));
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpLimitedMemLL(
    int_t lStart, int_t lEnd,
    int_t jStart, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel)
{
    if (lStart >= lEnd || jStart >= jEnd || lpanel.isEmpty())
        return 0;

    int_t max_gemm_rows = SUPERLU_MAX((int_t)1, ldt);
    for (int_t jSt = jStart; jSt < jEnd; ++jSt)
    {
        int_t jNext = jSt + 1;
        int_t iEnd = lStart;
        while (iEnd < lEnd)
        {
            int_t iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, max_gemm_rows);
            if (iEnd > lEnd)
                iEnd = lEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;
            dSymSchurCompUpdatePartLL(iSt, iEnd, jSt, jNext,
                                      k, lpanel);
        }
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymLookAheadUpdateLL(
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    int_t laLoc = lpanel.find(laIdx);
    if (laLoc == GLOBAL_BLOCK_NOT_FOUND)
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t laGid = lpanel.gid(laLoc);
    for (int_t ii = st_lb; ii < nlb; ++ii)
    {
        if (ii == laLoc || lpanel.gid(ii) >= laGid)
            dSymSchurCompUpdatePartLL(ii, ii + 1,
                                      laLoc, laLoc + 1,
                                      k, lpanel);
        else
            dSymSchurCompUpdatePartLL(laLoc, laLoc + 1,
                                      ii, ii + 1,
                                      k, lpanel);
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdateExcludeOneLL(
    int_t k, int_t ex, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t exLoc = lpanel.find(ex);
    if (exLoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        dSymSchurCompUpLimitedMemLL(st_lb, nlb, st_lb, nlb,
                                    k, lpanel);
    }
    else
    {
        dSymSchurCompUpLimitedMemLL(st_lb, exLoc, st_lb, exLoc,
                                    k, lpanel);
        dSymSchurCompUpLimitedMemLL(exLoc + 1, nlb, st_lb, exLoc,
                                    k, lpanel);
        dSymSchurCompUpLimitedMemLL(exLoc + 1, nlb, exLoc + 1, nlb,
                                    k, lpanel);
    }

    return 0;
}

static inline int_t xluSymLFragmentNBlocks(const int_t *frag_index)
{
    return frag_index == NULL ? 0 : frag_index[0];
}

static inline int_t xluSymLFragmentGid(const int_t *frag_index, int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static inline int_t xluSymLFragmentStRow(const int_t *frag_index, int_t k)
{
    int_t nblocks = xluSymLFragmentNBlocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static inline int_t xluSymLFragmentNbrow(const int_t *frag_index, int_t k)
{
    return xluSymLFragmentStRow(frag_index, k + 1) -
           xluSymLFragmentStRow(frag_index, k);
}

static inline int_t *xluSymLFragmentRowList(int_t *frag_index, int_t k)
{
    int_t nblocks = xluSymLFragmentNBlocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       xluSymLFragmentStRow(frag_index, k)];
}

static inline int_t xluSymLFragmentFind(const int_t *frag_index, int_t gid)
{
    int_t nblocks = xluSymLFragmentNBlocks(frag_index);
    for (int_t i = 0; i < nblocks; ++i)
        if (xluSymLFragmentGid(frag_index, i) == gid)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdatePartWithLFragments(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    int_t *frag_index, Ftype *frag_val)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty() ||
        frag_index == NULL || xluSymLFragmentNBlocks(frag_index) <= 0)
        return 0;
    if (frag_val == NULL)
        ABORT("SymFact V2 host L-fragment values are missing.");

    int_t gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int_t gemm_k = supersize(k);
    int_t frag_lda = frag_index[1];
    if (gemm_m <= 0 || gemm_k <= 0)
        return 0;
    if (frag_index[3] != gemm_k)
        ABORT("SymFact V2 host L-fragment column count does not match the panel.");

    for (int_t jj = jSt; jj < jEnd; ++jj)
    {
        int_t gemm_n = xluSymLFragmentNbrow(frag_index, jj);
        if (gemm_n <= 0)
            continue;

        int64_t gemm_count = static_cast<int64_t>(gemm_m) *
                             static_cast<int64_t>(gemm_n);
        ensureSymFactWorkSize(gemm_count);
        Ftype *gemmBuff = symFactWork;

        Ftype alpha = one<Ftype>();
        Ftype beta = zeroT<Ftype>();
        superlu_gemm<Ftype>("N", "T",
                            gemm_m, gemm_n, gemm_k, alpha,
                            lpanel.blkPtr(iSt), lpanel.LDA(),
                            &frag_val[xluSymLFragmentStRow(frag_index, jj)],
                            frag_lda, beta,
                            gemmBuff, gemm_m);

        for (int_t ii = iSt; ii < iEnd; ++ii)
        {
            if (lpanel.gid(ii) < xluSymLFragmentGid(frag_index, jj))
                continue;
            int_t row_off = lpanel.stRow(ii) - lpanel.stRow(iSt);
            xluSymScatterLowerToL(this,
                                  lpanel.nbrow(ii),
                                  gemm_n,
                                  lpanel.gid(ii),
                                  xluSymLFragmentGid(frag_index, jj),
                                  &gemmBuff[row_off],
                                  gemm_m,
                                  lpanel.rowList(ii),
                                  xluSymLFragmentRowList(frag_index, jj));
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpLimitedMemWithLFragments(
    int_t lStart, int_t lEnd,
    int_t fragStart, int_t fragEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    int_t *frag_index, Ftype *frag_val)
{
    if (lStart >= lEnd || fragStart >= fragEnd || lpanel.isEmpty() ||
        frag_index == NULL || xluSymLFragmentNBlocks(frag_index) <= 0)
        return 0;

    int_t nlb = lpanel.nblocks();
    int_t nfrag = xluSymLFragmentNBlocks(frag_index);
    lStart = SUPERLU_MAX((int_t)0, lStart);
    fragStart = SUPERLU_MAX((int_t)0, fragStart);
    lEnd = SUPERLU_MIN(lEnd, nlb);
    fragEnd = SUPERLU_MIN(fragEnd, nfrag);
    if (lStart >= lEnd || fragStart >= fragEnd)
        return 0;

    int_t max_gemm_rows = SUPERLU_MAX((int_t)1, ldt);
    for (int_t jSt = fragStart; jSt < fragEnd; ++jSt)
    {
        int_t iEnd = lStart;
        while (iEnd < lEnd)
        {
            int_t iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, max_gemm_rows);
            if (iEnd > lEnd)
                iEnd = lEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;
            dSymSchurCompUpdatePartWithLFragments(iSt, iEnd,
                                                  jSt, jSt + 1,
                                                  k, lpanel,
                                                  frag_index, frag_val);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymLookAheadUpdateWithLFragments(
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel,
    int_t *frag_index, Ftype *frag_val)
{
    if (lpanel.isEmpty() || frag_index == NULL ||
        xluSymLFragmentNBlocks(frag_index) <= 0)
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = xluSymLFragmentNBlocks(frag_index);
    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = xluSymLFragmentFind(frag_index, laIdx);

    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        dSymSchurCompUpLimitedMemWithLFragments(st_lb, nlb,
                                                laJLoc, laJLoc + 1,
                                                k, lpanel,
                                                frag_index, frag_val);

    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        if (laJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            dSymSchurCompUpLimitedMemWithLFragments(laILoc, laILoc + 1,
                                                    0, nfrag,
                                                    k, lpanel,
                                                    frag_index, frag_val);
        }
        else
        {
            dSymSchurCompUpLimitedMemWithLFragments(laILoc, laILoc + 1,
                                                    0, laJLoc,
                                                    k, lpanel,
                                                    frag_index, frag_val);
            dSymSchurCompUpLimitedMemWithLFragments(laILoc, laILoc + 1,
                                                    laJLoc + 1, nfrag,
                                                    k, lpanel,
                                                    frag_index, frag_val);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdateExcludeOneWithLFragments(
    int_t k, int_t ex, xlpanel_t<Ftype> &lpanel,
    int_t *frag_index, Ftype *frag_val)
{
    if (lpanel.isEmpty() || frag_index == NULL ||
        xluSymLFragmentNBlocks(frag_index) <= 0)
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = xluSymLFragmentNBlocks(frag_index);
    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = xluSymLFragmentFind(frag_index, ex);

    auto update_i_range = [&](int_t ist, int_t iend)
    {
        if (ist >= iend)
            return;
        if (exJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            dSymSchurCompUpLimitedMemWithLFragments(ist, iend, 0, nfrag,
                                                    k, lpanel,
                                                    frag_index, frag_val);
        }
        else
        {
            dSymSchurCompUpLimitedMemWithLFragments(ist, iend, 0, exJLoc,
                                                    k, lpanel,
                                                    frag_index, frag_val);
            dSymSchurCompUpLimitedMemWithLFragments(ist, iend,
                                                    exJLoc + 1, nfrag,
                                                    k, lpanel,
                                                    frag_index, frag_val);
        }
    };

    if (exILoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        update_i_range(st_lb, nlb);
    }
    else
    {
        update_i_range(st_lb, exILoc);
        update_i_range(exILoc + 1, nlb);
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDiagFactorPanelSolve(int_t k, int_t offset, diagFactBufs_type<Ftype>**dFBufs)
{
    if (options->SymFact == YES)
        return dSymDiagFactorPanelSolve(k, offset, offset, dFBufs);


    int_t ksupc = SuperSize(k);
    /*=======   Diagonal Factorization      ======*/
    if (iam == procIJ(k, k))
    {
        lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                        thresh, xsup, options, stat, info);
        lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
    }

    /*=======   Diagonal Broadcast          ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                  MPI_DOUBLE, kcol(k), (grid->rscp).comm);
    if (mycol == kcol(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                  MPI_DOUBLE, krow(k), (grid->cscp).comm);

    /*=======   Panel Update                ======*/
    if (myrow == krow(k))
        uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);

    if (mycol == kcol(k))
        lPanelVec[g2lCol(k)].panelSolve(ksupc, dFBufs[offset]->BlockUFactor, ksupc);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymDiagFactorPanelSolve(int_t k, int_t handle_offset,
                                                   int_t buffer_offset,
                                                   diagFactBufs_type<Ftype> **dFBufs)
{
    ABORT("LUv1 SymFact is implemented for double precision only.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymStartL2U(int_t k, int_t stream_offset)
{
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymFinishL2U(int_t k)
{
    return 0;
}

#ifdef HAVE_CUDA
template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymStartDiagPrefetch(int_t k, int_t stream_offset)
{
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymStartDiagPrefetch(int_t k,
                                                        int_t stream_offset)
{
    if (options->SymFact != YES || !superlu_acc_offload)
        return 0;
    if (!(Pr == 1 && Pc == 1 &&
          grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1))
        return 0;
    if (k < 0 || k >= nsupers || iam != symV2DiagProc(k))
        return 0;
    if (stream_offset < 0 ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchBufs.size()) ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchDoneEvents.size()) ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchNodes.size()))
        return 0;
    if (symDiagPrefetchBufs[stream_offset] == NULL)
        return 0;
    if (static_cast<size_t>(k) >= symDiagPrefetchEventIds.size())
        return 0;
    if (symDiagPrefetchEventIds[k] >= 0)
        return 0;
    if (symDiagPrefetchNodes[stream_offset] != -1 &&
        symDiagPrefetchNodes[stream_offset] != k)
        return 0;

    xlpanel_t<double> &lpanel = lPanelVec[symV2PanelIndex(k)];
    if (lpanel.isEmpty() || !lpanel.haveDiag())
        return 0;
    if (lpanel.gid(0) != k || lpanel.nbrow(0) != SuperSize(k))
        ABORT("SymFact V2 diagonal prefetch saw an invalid L-panel diagonal block.");

    int_t ksupc = SuperSize(k);
    cudaStream_t stream = A_gpu.lookAheadLStream[stream_offset];
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_prefetch_t = SuperLU_timer_();
#endif
    gpuErrchk(cudaMemcpy2DAsync(symDiagPrefetchBufs[stream_offset],
                                ldt * sizeof(double),
                                lpanel.blkPtrGPU(0),
                                lpanel.LDA() * sizeof(double),
                                ksupc * sizeof(double), ksupc,
                                cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaEventRecord(symDiagPrefetchDoneEvents[stream_offset],
                              stream));
    symDiagPrefetchNodes[stream_offset] = k;
    symDiagPrefetchEventIds[k] = stream_offset;
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_ISSUES);
    symStatAdd(SYM_GPU3D_S_DIAG_D2H_BYTES,
               static_cast<long long>(ksupc) * static_cast<long long>(ksupc) *
               static_cast<long long>(sizeof(double)));
    symTimingAdd(SYM_GPU3D_T_DIAG_PREFETCH_ISSUE,
                 SuperLU_timer_() - sym_prefetch_t);
#endif
    return 0;
}
#endif

template <>
inline int_t xLUstruct_t<double>::dSymStartL2U(int_t k, int_t stream_offset)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    SymTimingScope sym_timer(this, SYM_GPU3D_T_L2U_START);
#endif
    if (options->SymFact != YES || options->CommL != YES)
    {
        if (options->SymFact == YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        return 0;
    }

    if (symL2UOrders == NULL)
        ABORT("SymFact L2U order workspace is not allocated.");
    if (symFactTagUb <= 0)
        ABORT("Invalid MPI tag upper bound for LUv1 SymFact L2U communication.");

#ifdef HAVE_CUDA
    if (superlu_acc_offload)
        return dSymStartL2UGPU(k, stream_offset);
#else
    (void)stream_offset;
#endif

    dStartL2U_comm(k, grid, options, LUstructPtr, stat, info, SCT, symFactTagUb,
                   symL2UOrders, ldt);
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymFinishL2U(int_t k)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    SymTimingScope sym_timer(this, SYM_GPU3D_T_L2U_FINISH);
#endif
    if (options->SymFact != YES || options->CommL != YES)
    {
        if (options->SymFact == YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        return 0;
    }

#ifdef HAVE_CUDA
    if (superlu_acc_offload && Pr == 1 && Pc == 1)
        return 0;
#endif

    dLocalLU_t *Llu = LUstructPtr->Llu;
    dWaitL2U_recv(k, grid, options, LUstructPtr, stat, SCT);

    if (myrow == krow(k))
    {
        int_t lk = LBi(k, grid);
        if (Llu->Ufstnz_br_ptr[lk] != NULL && Llu->Unzval_br_ptr[lk] != NULL)
        {
            uPanelVec[g2lRow(k)].loadFromSkyline(k, Llu->Ufstnz_br_ptr[lk],
                                                 Llu->Unzval_br_ptr[lk], xsup);
#ifdef HAVE_CUDA
            if (superlu_acc_offload && symGPU3DVersion != 2)
                uPanelVec[g2lRow(k)].copyBackToGPU();
#endif
        }
    }

    dWaitL2U_send(k, grid, options, LUstructPtr, stat, info, SCT, symFactTagUb);
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymDiagFactorPanelSolve(int_t k, int_t handle_offset,
                                                           int_t buffer_offset,
                                                           ddiagFactBufs_t **dFBufs)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symStatAdd(SYM_GPU3D_S_FACTOR_NODES);
#endif
    if (symGPU3DVersion != 2)
        dSymStartL2U(k, handle_offset);
#ifdef HAVE_CUDA
    else if (superlu_acc_offload)
        dSymV2PrepackLFragmentsGPU(k, handle_offset);
#endif

    int_t ksupc = SuperSize(k);
    int_t sym_panel_root = symV2PanelRoot(k);
    int_t sym_diag_root = symV2DiagRoot(k);
    int_t sym_diag_proc = symV2DiagProc(k);
    double *invDiag = dFBufs[buffer_offset]->BlockUFactor;
    double *origDiag = dFBufs[buffer_offset]->BlockLFactor;
    int contract = symGPU3DContract;
    bool invDiagOnDevice = false;
    if (contract == 3)
        ABORT("GPU3DCONTRACT=3 is reserved for later SymFact diagonal experiments.");

#ifndef SLU_HAVE_LAPACK
    ABORT("LUv1 SymFact requires LAPACK dsytrf/dsytri support.");
#else
    if (iam == sym_diag_proc)
    {
        xlpanel_t<double> &lpanel = lPanelVec[symV2PanelIndex(k)];
        if (lpanel.isEmpty() || !lpanel.haveDiag() ||
            lpanel.gid(0) != k || lpanel.nbrow(0) != ksupc)
        {
            std::fprintf(stderr,
                         "SymFact V2 invalid diag panel rank=%d k=%lld have=%lld gid0=%lld nb0=%lld ksupc=%lld myrow=%lld mycol=%lld diagRoot=%lld panelRoot=%lld\n",
                         (grid3d != NULL) ? grid3d->iam : -1,
                         (long long)k,
                         (long long)(lpanel.isEmpty() ? -1 : lpanel.haveDiag()),
                         (long long)(lpanel.isEmpty() ? -1 : lpanel.gid(0)),
                         (long long)(lpanel.isEmpty() ? -1 : lpanel.nbrow(0)),
                         (long long)ksupc,
                         (long long)myrow,
                         (long long)mycol,
                         (long long)sym_diag_root,
                         (long long)sym_panel_root);
            ABORT("SymFact V2 diagonal owner has an invalid L-panel diagonal block.");
        }
        double *diag = lpanel.blkPtr(0);
        int_t panelLdd = lpanel.LDA();
        int_t ldd = panelLdd;
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            bool prefetched = false;
            if (Pr == 1 && Pc == 1 &&
                grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1 &&
                static_cast<size_t>(k) < symDiagPrefetchEventIds.size())
            {
                int event_id = symDiagPrefetchEventIds[k];
                if (event_id >= 0 &&
                    event_id < static_cast<int>(symDiagPrefetchBufs.size()) &&
                    event_id < static_cast<int>(symDiagPrefetchDoneEvents.size()) &&
                    event_id < static_cast<int>(symDiagPrefetchNodes.size()) &&
                    symDiagPrefetchNodes[event_id] == k)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double sym_prefetch_wait_t = SuperLU_timer_();
#endif
                    gpuErrchk(cudaEventSynchronize(symDiagPrefetchDoneEvents[event_id]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAdd(SYM_GPU3D_T_DIAG_PREFETCH_WAIT,
                                 SuperLU_timer_() - sym_prefetch_wait_t);
                    symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_HITS);
#endif
                    diag = symDiagPrefetchBufs[event_id];
                    ldd = ldt;
                    symDiagPrefetchEventIds[k] = -1;
                    symDiagPrefetchNodes[event_id] = -1;
                    prefetched = true;
                }
            }
            if (!prefetched)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_MISSES);
                symStatAdd(SYM_GPU3D_S_DIAG_D2H_BYTES,
                           static_cast<long long>(ksupc) * static_cast<long long>(ksupc) *
                           static_cast<long long>(sizeof(double)));
                double sym_t = SuperLU_timer_();
                cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
                cudaEvent_t start_event = A_gpu.diagD2HStartEvents[handle_offset];
                cudaEvent_t end_event = A_gpu.diagD2HEndEvents[handle_offset];
                gpuErrchk(cudaEventRecord(start_event, cuStream));
                gpuErrchk(cudaMemcpy2DAsync(diag, ldd * sizeof(double),
                                            lpanel.blkPtrGPU(0),
                                            panelLdd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaEventRecord(end_event, cuStream));
                gpuErrchk(cudaEventSynchronize(end_event));
                float copy_ms = 0.0f;
                gpuErrchk(cudaEventElapsedTime(&copy_ms, start_event, end_event));
                double total = SuperLU_timer_() - sym_t;
                double copy = (double)copy_ms * 1.0e-3;
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H, total);
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H_COPY, copy);
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H_WAIT, SUPERLU_MAX(0.0, total - copy));
#else
                gpuErrchk(cudaMemcpy2D(diag, ldd * sizeof(double),
                                       lpanel.blkPtrGPU(0),
                                       panelLdd * sizeof(double),
                                       ksupc * sizeof(double), ksupc,
                                       cudaMemcpyDeviceToHost));
#endif
            }
        }
#endif
        int nsupc_i = (int)ksupc;
        int ldd_i = (int)ldd;
        int panel_ldd_i = (int)panelLdd;
        int_t jfst = FstBlockC(k);
        int lwork = -1;
        int lapack_info = 0;
        char uplo = 'L';
        double *work = symFactWork;
        int *ipiv = symFactIPIV;
        bool diagonal_done = false;

        if (work == NULL || ipiv == NULL)
            ABORT("LUv1 SymFact workspace is not allocated.");

        if (symGPU3DVersion == 2)
        {
            if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers))
                ABORT("SymFact V2 diagonal block vector has invalid size.");
            if (symV2DiagBlocks[k] == NULL)
            {
                symV2DiagBlocks[k] = (double *)SUPERLU_MALLOC(
                    xlu_checked_square_alloc_bytes(ksupc, sizeof(double),
                                                   "SymFact V2 diagonal block"));
                if (symV2DiagBlocks[k] == NULL)
                    ABORT("Malloc fails for SymFact V2 diagonal block.");
            }
            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&symV2DiagBlocks[k][j * ksupc], &diag[j * ldd],
                       ksupc * sizeof(double));
            for (int_t j = 0; j < ksupc; ++j)
                for (int_t i = 0; i < j; ++i)
                    symV2DiagBlocks[k][i + j * ksupc] =
                        symV2DiagBlocks[k][j + i * ksupc];
#ifdef HAVE_CUDA
            if (superlu_acc_offload)
            {
                if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers))
                    ABORT("SymFact V2 device diagonal block vector has invalid size.");
                if (symV2DiagBlocksGPU[k] == NULL)
                {
                    gpuErrchk(cudaMalloc((void **)&symV2DiagBlocksGPU[k],
                                         xlu_checked_square_alloc_bytes(
                                             ksupc, sizeof(double),
                                             "SymFact V2 device diagonal block")));
                }
                int stream_id = handle_offset;
                if (stream_id < 0 || stream_id >= A_gpu.numCudaStreams)
                    stream_id = 0;
                gpuErrchk(cudaMemcpyAsync(symV2DiagBlocksGPU[k],
                                          symV2DiagBlocks[k],
                                          ksupc * ksupc * sizeof(double),
                                          cudaMemcpyHostToDevice,
                                          A_gpu.cuStreams[stream_id]));
            }
#endif
        }

        if (contract == 2)
            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&origDiag[j * ksupc], &diag[j * ldd],
                       ksupc * sizeof(double));

        double thresh1 = thresh / 10.0;

#ifdef HAVE_CUDA
        if (contract == 1)
        {
            if (!superlu_acc_offload)
                ABORT("GPU3DCONTRACT=1 requires GPU offload.");

            cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[handle_offset];
            cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
            double *dDiag = lpanel.blkPtrGPU(0);
            double *dWork = A_gpu.diagFactWork[handle_offset];
            int *dIpiv = A_gpu.diagFactIPIV[handle_offset];
            int *dInfo = A_gpu.diagFactInfo[handle_offset];
            int sytrf_lwork = 0;
            int sytri_lwork = 0;
            int gpu_info = 0;

            gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
            gpuCusolverErrchk(cusolverDnDsytrf_bufferSize(cusolverH, nsupc_i,
                                                          dDiag, panel_ldd_i,
                                                          &sytrf_lwork));
            gpuCusolverErrchk(cusolverDnDsytrf(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                               nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                               dWork, sytrf_lwork, dInfo));
            gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                      cudaMemcpyDeviceToHost, cuStream));
            gpuErrchk(cudaStreamSynchronize(cuStream));

            if (gpu_info == 0)
            {
                gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                            dDiag, panelLdd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaMemcpyAsync(ipiv, dIpiv,
                                          ksupc * sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));

                const double tol_inertia = 1e-30;
                int inertia[3];
                inertia_from_dsytrf(uplo, nsupc_i, work, nsupc_i, ipiv,
                                    tol_inertia, inertia);
                int n2x2 = xlu_sytrf_count_2x2(ipiv, nsupc_i);

                gpuCusolverErrchk(cusolverDnDsytri_bufferSize(cusolverH,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              nsupc_i, dDiag,
                                                              panel_ldd_i, dIpiv,
                                                              &sytri_lwork));
                gpuCusolverErrchk(cusolverDnDsytri(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                                   nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                                   dWork, sytri_lwork, dInfo));
                gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));

                if (gpu_info == 0)
                {
                    gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                                dDiag, panelLdd * sizeof(double),
                                                ksupc * sizeof(double), ksupc,
                                                cudaMemcpyDeviceToHost, cuStream));
                    gpuErrchk(cudaStreamSynchronize(cuStream));

                    for (int_t j = 0; j < ksupc; ++j)
                        for (int_t i = j + 1; i < ksupc; ++i)
                            work[j + i * ksupc] = work[i + j * ksupc];

                    double scaled_resid =
                        xlu_sym_inverse_scaled_residual(diag, ldd, work, nsupc_i);
                    if (scaled_resid <= symContractValidateTol)
                    {
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&invDiag[j * ksupc], &work[j * ksupc],
                                   ksupc * sizeof(double));
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&diag[j * ldd], &work[j * ksupc],
                                   ksupc * sizeof(double));

                        stat->sytrf_2x2 += n2x2;
                        stat->inertia[0] += inertia[0];
                        stat->inertia[1] += inertia[1];
                        stat->inertia[2] += inertia[2];
                        stat->ops[FACT] += (flops_t)ksupc * ksupc * ksupc;
                        *info = 0;
                        diagonal_done = true;
                        ++symContract1Accepted;
                        symContract1MaxResid =
                            SUPERLU_MAX(symContract1MaxResid, scaled_resid);
                    }
                }
            }

            if (!diagonal_done)
                ++symContract1Fallbacks;
        }
#else
        if (contract == 1)
            ABORT("GPU3DCONTRACT=1 requires CUDA support.");
#endif

        if (!diagonal_done)
        {
            int ntiny = 0;
            int n2x2 = 0;
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_sytrf_t = SuperLU_timer_();
#endif
            if (options->ReplaceTinyPivot == YES)
            {
                dsytrf_mod_(&uplo, &nsupc_i, diag, &ldd_i, &thresh1, ipiv,
                             work, &lwork, info, &ntiny, &n2x2);
                int64_t requested_work_size = (int64_t)work[0];
                ensureSymFactWorkSize(requested_work_size);
                work = symFactWork;
                lwork = (int)requested_work_size;
                dsytrf_mod_(&uplo, &nsupc_i, diag, &ldd_i, &thresh1, ipiv,
                             work, &lwork, info, &ntiny, &n2x2);
                stat->TinyPivots += ntiny;
                stat->sytrf_2x2 += n2x2;
            }
            else
            {
                dsytrf_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lwork, info);
                int64_t requested_work_size = (int64_t)work[0];
                ensureSymFactWorkSize(requested_work_size);
                work = symFactWork;
                lwork = (int)requested_work_size;
                dsytrf_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lwork, info);
                n2x2 = xlu_sytrf_count_2x2(ipiv, nsupc_i);
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_CPU_SYTRF, SuperLU_timer_() - sym_sytrf_t);
#endif

            if (*info > 0)
                *info += jfst;
            else if (*info < 0)
                *info -= jfst;

            const double tol_inertia = 1e-30;
            int inertia[3];
            inertia_from_dsytrf(uplo, nsupc_i, diag, ldd_i, ipiv,
                                tol_inertia, inertia);
            stat->inertia[0] += inertia[0];
            stat->inertia[1] += inertia[1];
            stat->inertia[2] += inertia[2];

            bool inverse_done = false;
#ifdef HAVE_CUDA
            if (contract == 2 && ntiny == 0 && n2x2 == 0)
            {
                if (!superlu_acc_offload)
                    ABORT("GPU3DCONTRACT=2 requires GPU offload.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_gpu_sytri_t = SuperLU_timer_();
#endif

                cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[handle_offset];
                cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
                double *dDiag = lpanel.blkPtrGPU(0);
                double *dWork = A_gpu.diagFactWork[handle_offset];
                int *dIpiv = A_gpu.diagFactIPIV[handle_offset];
                int *dInfo = A_gpu.diagFactInfo[handle_offset];
                int sytri_lwork = 0;
                int gpu_info = 0;

                gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
                gpuErrchk(cudaMemcpy2DAsync(dDiag, panelLdd * sizeof(double),
                                            diag, ldd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyHostToDevice, cuStream));
                gpuErrchk(cudaMemcpyAsync(dIpiv, ipiv,
                                          ksupc * sizeof(int),
                                          cudaMemcpyHostToDevice, cuStream));
                gpuCusolverErrchk(cusolverDnDsytri_bufferSize(cusolverH,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              nsupc_i, dDiag,
                                                              panel_ldd_i, dIpiv,
                                                              &sytri_lwork));
                gpuCusolverErrchk(cusolverDnDsytri(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                                   nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                                   dWork, sytri_lwork, dInfo));
                gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_GPU_SYTRI, SuperLU_timer_() - sym_gpu_sytri_t);
#endif

                if (gpu_info == 0)
                {
                    gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                                dDiag, panelLdd * sizeof(double),
                                                ksupc * sizeof(double), ksupc,
                                                cudaMemcpyDeviceToHost, cuStream));
                    gpuErrchk(cudaStreamSynchronize(cuStream));

                    for (int_t j = 0; j < ksupc; ++j)
                        for (int_t i = j + 1; i < ksupc; ++i)
                            work[j + i * ksupc] = work[i + j * ksupc];

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double sym_validate_t = SuperLU_timer_();
#endif
                    double scaled_resid =
                        xlu_sym_inverse_scaled_residual(origDiag, nsupc_i,
                                                        work, nsupc_i);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAdd(SYM_GPU3D_T_GPU_SYTRI_VALIDATE,
                                 SuperLU_timer_() - sym_validate_t);
#endif
                    if (scaled_resid <= symContractValidateTol)
                    {
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&diag[j * ldd], &work[j * ksupc],
                                   ksupc * sizeof(double));
                        inverse_done = true;
                    }
                }
            }
#else
            if (contract == 2)
                ABORT("GPU3DCONTRACT=2 requires CUDA support.");
#endif

            if (!inverse_done)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_cpu_sytri_t = SuperLU_timer_();
#endif
                dsytri_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lapack_info);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_CPU_SYTRI, SuperLU_timer_() - sym_cpu_sytri_t);
#endif
            }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_pack_t = SuperLU_timer_();
#endif
            for (int_t j = 0; j < ksupc; ++j)
                for (int_t i = j + 1; i < ksupc; ++i)
                    diag[j + i * ldd] = diag[i + j * ldd];

            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&invDiag[j * ksupc], &diag[j * ldd], ksupc * sizeof(double));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_DIAG_PACK, SuperLU_timer_() - sym_pack_t);
#endif

            stat->ops[FACT] += (flops_t)ksupc * ksupc * ksupc;
        }
    }

    if (symGPU3DVersion == 2 && mycol == sym_panel_root)
    {
        if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers))
            ABORT("SymFact V2 diagonal block vector has invalid size.");
        if (symV2DiagBlocks[k] == NULL)
        {
            symV2DiagBlocks[k] = (double *)SUPERLU_MALLOC(
                xlu_checked_square_alloc_bytes(ksupc, sizeof(double),
                                               "SymFact V2 diagonal block"));
            if (symV2DiagBlocks[k] == NULL)
                ABORT("Malloc fails for SymFact V2 diagonal block.");
        }
        MPI_Bcast((void *)symV2DiagBlocks[k], ksupc * ksupc, MPI_DOUBLE,
                  sym_diag_root, (grid->cscp).comm);
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers))
                ABORT("SymFact V2 device diagonal block vector has invalid size.");
            if (symV2DiagBlocksGPU[k] == NULL)
            {
                gpuErrchk(cudaMalloc((void **)&symV2DiagBlocksGPU[k],
                                     xlu_checked_square_alloc_bytes(
                                         ksupc, sizeof(double),
                                         "SymFact V2 device diagonal block")));
            }
            int stream_id = buffer_offset;
            if (stream_id < 0 || stream_id >= A_gpu.numCudaStreams)
                stream_id = 0;
            gpuErrchk(cudaMemcpyAsync(symV2DiagBlocksGPU[k],
                                      symV2DiagBlocks[k],
                                      ksupc * ksupc * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      A_gpu.cuStreams[stream_id]));
        }
#endif
    }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_bcast_t = SuperLU_timer_();
#endif
#ifdef HAVE_CUDA
    if (superlu_acc_offload && mycol == sym_panel_root && superlu_cuda_aware_mpi())
    {
        int stream_id = handle_offset;
        if (stream_id < 0 || stream_id >= A_gpu.numCudaStreams)
            stream_id = 0;
        cudaStream_t cuStream = A_gpu.cuStreams[stream_id];
        double *dInvDiag = A_gpu.dFBufs[buffer_offset];

        if (iam == sym_diag_proc)
        {
            gpuErrchk(cudaMemcpyAsync(dInvDiag, invDiag,
                                      ksupc * ksupc * sizeof(double),
                                      cudaMemcpyHostToDevice, cuStream));
            gpuErrchk(cudaStreamSynchronize(cuStream));
        }
        superlu_gpu_mpi_bcast(dInvDiag, invDiag, sizeof(double),
                              static_cast<int>(ksupc * ksupc), MPI_DOUBLE,
                              sym_diag_root, (grid->cscp).comm);
        invDiagOnDevice = true;
    }
    else
#endif
    {
        if (mycol == sym_panel_root)
            MPI_Bcast((void *)invDiag, ksupc * ksupc, MPI_DOUBLE,
                      sym_diag_root, (grid->cscp).comm);
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_DIAG_BCAST, SuperLU_timer_() - sym_bcast_t);
#endif

    if (mycol == sym_panel_root)
    {
        xlpanel_t<double> &lpanel = lPanelVec[symV2PanelIndex(k)];
        if (lpanel.isEmpty())
            return 0;
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            int64_t gpu_work_size = (int64_t)lpanel.nzrows() * (int64_t)ksupc;
            if (gpu_work_size > (int64_t)maxLvalCount)
                ABORT("SymFact GPU L-panel workspace is too small.");

            cublasHandle_t cubHandle = A_gpu.cuHandles[handle_offset];
            cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
            double *dInvDiag = A_gpu.dFBufs[buffer_offset];

            if (!invDiagOnDevice)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_inv_h2d_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpyAsync(dInvDiag, invDiag,
                                          ksupc * ksupc * sizeof(double),
                                          cudaMemcpyHostToDevice, cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_INV_H2D,
                             SuperLU_timer_() - sym_inv_h2d_t);
#endif
            }
            if (lpanel.haveDiag())
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_ldiag_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpy2DAsync(lpanel.blkPtrGPU(0),
                                            lpanel.LDA() * sizeof(double),
                                            dInvDiag, ksupc * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToDevice, cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_LDIAG_D2D,
                             SuperLU_timer_() - sym_ldiag_t);
#endif
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_lpanel_t = SuperLU_timer_();
#endif
            lpanel.panelSolveSymmetricGPU(cubHandle, cuStream,
                                          ksupc, dInvDiag, ksupc,
                                          A_gpu.lookAheadLGemmBuffer[handle_offset],
                                          lpanel.nzrows());
            if (symGPU3DVersion == 2)
            {
                gpuErrchk(cudaMemcpyAsync(dInvDiag, symV2DiagBlocks[k],
                                          ksupc * ksupc * sizeof(double),
                                          cudaMemcpyHostToDevice, cuStream));
            }
            gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[handle_offset],
                                      cuStream));
            if ((symGPU3DVersion == 2 || (Pr == 1 && Pc == 1)) &&
                k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size())
                symPanelReadyEventIds[k] = handle_offset;
            bool local_singleton_panel =
                (Pr == 1 && Pc == 1 &&
                 grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1);
            bool async_v2_panel =
                (symGPU3DVersion == 2 && superlu_sym_v2_async_factor());
            if (!local_singleton_panel && !async_v2_panel)
                gpuErrchk(cudaStreamSynchronize(cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_LPANEL_TRANSFORM,
                         SuperLU_timer_() - sym_lpanel_t);
#endif
        }
        else
#endif
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double sym_lpanel_t = SuperLU_timer_();
#endif
        if (lpanel.haveDiag())
        {
            if (lpanel.gid(0) != k || lpanel.nbrow(0) != ksupc)
                ABORT("SymFact V2 L-panel diagonal block is not first.");
            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&lpanel.blkPtr(0)[j * lpanel.LDA()],
                       &invDiag[j * ksupc],
                       ksupc * sizeof(double));
        }
        ensureSymFactWorkSize((int64_t)lpanel.nzrows() * (int64_t)ksupc);
        lpanel.panelSolveSymmetric(ksupc, invDiag, ksupc, symFactWork,
                                   lpanel.nzrows());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LPANEL_TRANSFORM,
                     SuperLU_timer_() - sym_lpanel_t);
#endif
        }
    }

#endif

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dPanelBcast(int_t k, int_t offset)
{
    if (options->SymFact == YES)
    {
        if (symGPU3DVersion != 2)
            dSymFinishL2U(k);
    }

    /*=======   Panel Broadcast             ======*/
    xupanel_t<Ftype> k_upanel;
    xlpanel_t<Ftype> k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset]);
    int_t sym_panel_root = (symGPU3DVersion == 2)
                               ? symV2PanelRoot(k)
                               : kcol(k);

    if (mycol == sym_panel_root)
        k_lpanel = lPanelVec[symV2PanelIndex(k)];

	    if (symGPU3DVersion == 2)
	    {
	        if (Pr > 1)
	            dSymV2LFragmentExchangeHost(k, offset);

	        if (LidxSendCounts[k] > 0)
	        {
            MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t,
                      sym_panel_root, grid3d->rscp.comm);
            MPI_Bcast(k_lpanel.val, LvalSendCounts[k],
                      get_mpi_type<Ftype>(), sym_panel_root,
                      grid3d->rscp.comm);
        }
        if (Pr == 1 && Pc > 1 && LidxSendCounts[k] > 0)
        {
            int_t ksupc = SuperSize(k);
            if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers))
                ABORT("SymFact V2 diagonal block vector has invalid size.");
            if (symV2DiagBlocks[k] == NULL)
            {
                symV2DiagBlocks[k] = (Ftype *)SUPERLU_MALLOC(
                    xlu_checked_square_alloc_bytes(ksupc, sizeof(Ftype),
                                                   "SymFact V2 diagonal block broadcast"));
                if (symV2DiagBlocks[k] == NULL)
                    ABORT("Malloc fails for SymFact V2 diagonal block broadcast.");
            }
            MPI_Bcast(symV2DiagBlocks[k], ksupc * ksupc,
                      get_mpi_type<Ftype>(), sym_panel_root,
                      grid3d->rscp.comm);
        }
        return 0;
    }

    k_upanel = xupanel_t<Ftype>(UidxRecvBufs[offset], UvalRecvBufs[offset]);
    if (myrow == krow(k))
        k_upanel = uPanelVec[g2lRow(k)];

    if (UidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
        MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
    }

    if (LidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, sym_panel_root, grid3d->rscp.comm);
        MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, sym_panel_root, grid3d->rscp.comm);
    }
    return 0;
}
