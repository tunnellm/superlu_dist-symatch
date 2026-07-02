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

static inline size_t xlu_checked_sum_size(size_t a, size_t b, const char *what)
{
    (void) what;
    if (b > static_cast<size_t>(-1) - a)
        ABORT("Workspace size overflows allocation size.");
    return a + b;
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
        "recv_cache_build",
        "partner_recv_index_build",
        "partner_recv_lookup_build",
        "row_recv_index_build",
        "row_recv_lookup_build",
        "exact_demand_build",
        "exact_send_map_index",
        "exact_send_map_build",
        "row_compact_demand_build",
        "row_compact_recv_meta_build",
        "row_compact_send_map_build",
        "row_compact_size_check",
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
    static const char *scalar_labels[SYM_V2_PROFILE_COUNT] = {
        "gpu_usable_bytes",
        "gpu_persistent_bytes",
        "gpu_delayed_metadata_bytes",
        "gpu_per_stream_base_bytes",
        "gpu_per_stream_bytes",
        "gpu_gemm_buffer_bytes",
        "gpu_gemm_shrink_bytes",
        "gpu_streams",
        "gpu_raw_w_cache_bytes",
        "gpu_partner_value_bytes",
        "gpu_partner_index_bytes",
        "gpu_row_stage_bytes",
        "gpu_row_recv_value_bytes",
        "gpu_row_index_bytes",
        "gpu_row_send_stage_bytes",
        "gpu_row_stage_reuse",
        "gpu_row_stage_chosen_bytes",
        "gpu_diag_bytes",
        "row_current_recv_values",
        "row_sparse_send_values",
        "row_sparse_recv_values",
        "row_saved_recv_values",
        "row_demand_records",
        "row_send_messages",
        "row_recv_messages"
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
    long long scalar_sum[SYM_V2_PROFILE_COUNT] = {};
    long long scalar_max[SYM_V2_PROFILE_COUNT] = {};
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
    MPI_Reduce(symV2ProfileScalar, scalar_sum, SYM_V2_PROFILE_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symV2ProfileScalar, scalar_max, SYM_V2_PROFILE_COUNT,
               MPI_LONG_LONG_INT, MPI_MAX, 0, grid3d->comm);

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
    printf("SymFact GPU3D V2 scalar profile:\n");
    printf("  %-32s %18s %18s\n", "metric", "sum", "max_rank");
    for (int i = 0; i < SYM_V2_PROFILE_COUNT; ++i)
    {
        if (scalar_sum[i] == 0 && scalar_max[i] == 0)
            continue;
        printf("  %-32s %18lld %18lld\n",
               scalar_labels[i], scalar_sum[i], scalar_max[i]);
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
        "partner_lfrag_pack_issue",
        "partner_lfrag_d2h_stage_issue",
        "partner_lfrag_pack_stage_sync",
        "partner_lfrag_recv_post",
        "partner_lfrag_mpi_recv_wait",
        "partner_lfrag_h2d_stage_issue",
        "partner_lfrag_assemble_issue",
        "partner_lfrag_send_post",
        "partner_lfrag_send_wait",
        "row_lfrag_pack_issue",
        "row_lfrag_d2h_stage_issue",
        "row_lfrag_pack_stage_sync",
        "row_lfrag_recv_post",
        "row_lfrag_mpi_recv_wait",
        "row_lfrag_h2d_stage_issue",
        "row_lfrag_assemble_issue",
        "row_lfrag_send_post",
        "row_lfrag_send_wait",
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
        "partner_lfrag_pack_issue",
        "partner_lfrag_d2h_stage_issue",
        "partner_lfrag_pack_stage_sync",
        "partner_lfrag_recv_post",
        "partner_lfrag_mpi_recv_wait",
        "partner_lfrag_h2d_stage_issue",
        "partner_lfrag_assemble_issue",
        "partner_lfrag_send_post",
        "partner_lfrag_send_wait",
        "row_lfrag_pack_issue",
        "row_lfrag_d2h_stage_issue",
        "row_lfrag_pack_stage_sync",
        "row_lfrag_recv_post",
        "row_lfrag_mpi_recv_wait",
        "row_lfrag_h2d_stage_issue",
        "row_lfrag_assemble_issue",
        "row_lfrag_send_post",
        "row_lfrag_send_wait",
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

// SYM_V2_PC2_PHASE1_AGG_SEND_CAPACITY_BEGIN
    /* Phase 1 fixes maxSymV2RowFragValSendCount to mean aggregate
       row-send staging capacity.  The base branch only tracked a single
       block; pack-all-dest and sparse row-down need room for all remote
       destination process columns for one panel. */
    if (superlu_sym_v2_pc_fragment_schur() && Pc > 1)
    {
        const long long row_send_multiplier =
            static_cast<long long>(Pc - 1);
        if (max_row_val > 0 &&
            row_send_multiplier >
                std::numeric_limits<long long>::max() / max_row_val)
            ABORT("SymFact V2 aggregate row-fragment send size overflows.");
        const long long aggregate_row_send_val =
            max_row_val * row_send_multiplier;
        max_row_send_val =
            SUPERLU_MAX(max_row_send_val, aggregate_row_send_val);
    }
// SYM_V2_PC2_PHASE1_AGG_SEND_CAPACITY_END
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
// SYM_V2_PC2_PHASE4_CTOR_ROW_DOWN_GPU_INIT_BEGIN
    symV2RowDownSendMapPoolGPU = NULL;
// SYM_V2_PC2_PHASE4_CTOR_ROW_DOWN_GPU_INIT_END

    symV2PartnerLSendBufPoolCount = 0;
    symL2LSendMapPoolCount = 0;
    symV2PartnerLExactSendBufPoolCount = 0;
    symV2PartnerLExactSendMapPoolCount = 0;
    symV2RowFragExactSendBufPoolCount = 0;
    symV2RowFragExactSendMapPoolCount = 0;
    symV2PartnerLRecvMapPoolCount = 0;
    symV2RowFragRecvMapPoolCount = 0;
// SYM_V2_PC2_PHASE4_FREE_ROW_DOWN_GPU_COUNT_BEGIN
    symV2RowDownSendMapPoolCount = 0;
    symV2RowDownSendMapsGPU.clear();
// SYM_V2_PC2_PHASE4_FREE_ROW_DOWN_GPU_COUNT_END

// SYM_V2_PC2_PHASE4_CTOR_ROW_DOWN_GPU_COUNT_INIT_BEGIN
    symV2RowDownSendMapPoolCount = 0;
// SYM_V2_PC2_PHASE4_CTOR_ROW_DOWN_GPU_COUNT_INIT_END

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
            if (superlu_sym_v2_pc_fragment_ldl_native() &&
                !superlu_sym_v2_pc_fragment_schur())
                ABORT("GPU3DV2_PC_FRAGMENT_LDL_NATIVE requires GPU3DV2_PC_FRAGMENT_SCHUR=1.");
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
    maxSymPartnerLSendStageCount = 0;
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
    maxSymPartnerLSendStageCount = 0;
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
// SYM_V2_PC2_PHASE1_INIT_SEND_RESIZE_BEGIN
    symV2RowFragHostSendBufs.resize(options->num_lookaheads);
// SYM_V2_PC2_PHASE1_INIT_SEND_RESIZE_END

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


// SYM_V2_PC2_PHASE1_INIT_SEND_POOL_ALLOC_BEGIN
#ifdef HAVE_CUDA
    if (use_sym_v2_pooled_pinned_staging &&
        sym_v2_pc_fragment_schur &&
        superlu_sym_v2_row_l_separate_send_staging() &&
        (maxSymV2RowFragStageCount > 0 ||
         maxSymV2RowFragValSendCount > 0) &&
        options->num_lookaheads > 0)
    {
        const int_t send_stage_count = SUPERLU_MAX(
            maxSymV2RowFragStageCount, maxSymV2RowFragValSendCount);
        const size_t pooled_row_send_bytes = xlu_checked_alloc_bytes(
            send_stage_count, sizeof(Ftype),
            "SymFact V2 pooled pinned row-fragment send staging");
        gpuErrchk(cudaMallocHost(
            (void **)&symV2RowFragHostSendPoolPinned,
            pooled_row_send_bytes));
        symV2RowFragHostSendPoolPinnedCount =
            static_cast<size_t>(send_stage_count);
        symV2RowFragHostSendPinned = 1;
    }
#endif
// SYM_V2_PC2_PHASE1_INIT_SEND_POOL_ALLOC_END
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
// SYM_V2_PC2_PHASE1_INIT_SEND_BYTES_BEGIN
        int_t sym_row_frag_send_stage_count = SUPERLU_MAX(
            maxSymV2RowFragStageCount, maxSymV2RowFragValSendCount);
        size_t sym_row_frag_send_lval_bytes =
            xlu_checked_alloc_bytes(sym_row_frag_send_stage_count, sizeof(Ftype),
                                    "SymFact V2 row-fragment host send buffer");
// SYM_V2_PC2_PHASE1_INIT_SEND_BYTES_END

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
// SYM_V2_PC2_PHASE1_INIT_SEND_NULL_BEGIN
        symV2RowFragHostSendBufs[i] = NULL;
// SYM_V2_PC2_PHASE1_INIT_SEND_NULL_END

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

// SYM_V2_PC2_PHASE1_INIT_SEND_ALLOC_BEGIN
#ifdef HAVE_CUDA
        if (symV2RowFragHostSendPoolPinned != NULL)
            symV2RowFragHostSendBufs[i] = symV2RowFragHostSendPoolPinned;
        else if (sym_row_frag_send_lval_bytes && sym_v2_pc_fragment_schur &&
                 superlu_sym_v2_row_l_separate_send_staging() &&
                 superlu_acc_offload &&
                 superlu_sym_v2_pinned_staging())
        {
            gpuErrchk(cudaMallocHost(
                (void **)&symV2RowFragHostSendBufs[i],
                sym_row_frag_send_lval_bytes));
            symV2RowFragHostSendPinned = 1;
        }
        else
#endif
        if (sym_row_frag_send_lval_bytes && sym_v2_pc_fragment_schur &&
            superlu_sym_v2_row_l_separate_send_staging())
        {
            symV2RowFragHostSendBufs[i] =
                (Ftype *)SUPERLU_MALLOC(sym_row_frag_send_lval_bytes);
        }
// SYM_V2_PC2_PHASE1_INIT_SEND_ALLOC_END
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
// SYM_V2_PC2_PHASE1_INIT_SEND_MALLOC_CHECK_BEGIN
            (sym_v2_pc_fragment_schur &&
             sym_row_frag_lval_bytes != 0 &&
             symV2RowFragHostRecvBufs[i] == NULL) ||
            (sym_v2_pc_fragment_schur &&
             superlu_sym_v2_row_l_separate_send_staging() &&
             sym_row_frag_send_lval_bytes != 0 &&
             symV2RowFragHostSendBufs[i] == NULL))
// SYM_V2_PC2_PHASE1_INIT_SEND_MALLOC_CHECK_END
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
// SYM_V2_PC2_SETUP_OPT_L2U_BEGIN
    const bool pcfrag_setup_opt =
        superlu_acc_offload &&
        symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
        superlu_sym_v2_pc_fragment_schur() &&
        superlu_sym_v2_pc_fragment_ldl_native() &&
        superlu_sym_v2_pcfrag_setup_opt();
    if (pcfrag_setup_opt)
        need_l2u_workspace = false;
// SYM_V2_PC2_SETUP_OPT_L2U_END

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
            symL2USendMapsHost.assign(l2u_slots, std::vector<int_t>());
            symL2ULocalMapsGPU.assign(CEILING(nsupers, Pr), NULL);
            symL2ULocalMapsHost.assign(CEILING(nsupers, Pr),
                                       std::vector<int_t>());
        }
// SYM_V2_PC2_SETUP_OPT_L2U_ASSERT_BEGIN
        else if (pcfrag_setup_opt)
        {
            symL2USendBufsGPU.clear();
            symL2USendMapsGPU.clear();
            symL2USendMapsHost.clear();
            symL2ULocalMapsGPU.clear();
            symL2ULocalMapsHost.clear();
        }
// SYM_V2_PC2_SETUP_OPT_L2U_ASSERT_END
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
        symV2PartnerLExactSendMapsHost.clear();
        symV2PartnerLExactSendMapOffsets.assign(partner_exact_slots, 0);
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
        symV2RowDirectSendBlocksHost.assign(
            l2u_slots, std::vector<int_t>());
        symV2RowDirectSendMapScratchHost.clear();
// SYM_V2_PC2_PHASE3_FREE_ROW_DOWN_PLAN_BEGIN
    symV2RowDownSendSizes.clear();
    symV2RowDownSendMapOffsets.clear();
    symV2RowDownSendMapsHost.clear();
// SYM_V2_PC2_LAZY_SENDMAP_CLEAR_BEGIN
    symV2RowDownSendSegsHost.clear();
    symV2RowDownSendSegOffsets.clear();
    symV2RowDownSendSegCounts.clear();
    symV2RowDownSendSegsGPU.clear();
    symV2RowDownSendSegPoolCount = 0;
// SYM_V2_PC2_LAZY_SENDMAP_CLEAR_END
    symV2RowDownSegOffsets.clear();
    symV2RowDownSegs.clear();
    symV2RowDownRecvSizes.clear();
    symV2RowDownPlanReady.clear();
    symV2RowDownSparseSendValues = 0;
    symV2RowDownSparseRecvValues = 0;
    symV2RowDownCurrentRecvValues = 0;
    symV2RowDownDemandRecords = 0;
    symV2RowDownSendMessages = 0;
    symV2RowDownRecvMessages = 0;
    symV2RowDownSetupSeconds = 0.0;
// SYM_V2_PC2_PHASE3_FREE_ROW_DOWN_PLAN_END

// SYM_V2_PC2_PHASE3_INIT_ROW_DOWN_VECTORS_BEGIN
        symV2RowDownSendSizes.assign(l2u_slots, 0);
        symV2RowDownSendMapOffsets.assign(l2u_slots, 0);
        symV2RowDownSendMapsHost.clear();
// SYM_V2_PC2_LAZY_SENDMAP_INIT_BEGIN
        symV2RowDownSendSegPoolCount = 0;
        symV2RowDownSendSegsHost.clear();
        symV2RowDownSendSegOffsets.assign(l2u_slots, 0);
        symV2RowDownSendSegCounts.assign(l2u_slots, 0);
        symV2RowDownSendSegsGPU.assign(l2u_slots, NULL);
// SYM_V2_PC2_LAZY_SENDMAP_INIT_END
        symV2RowDownSegOffsets.assign(l2u_slots + 1, 0);
        symV2RowDownSegs.clear();
        symV2RowDownRecvSizes.assign(
            xlu_checked_product(static_cast<size_t>(nsupers),
                                static_cast<size_t>(Pc),
                                "SymFact V2 row-down receive sizes"),
            0);
        symV2RowDownPlanReady.assign(static_cast<size_t>(nsupers), 0);
        symV2RowDownSparseSendValues = 0;
        symV2RowDownSparseRecvValues = 0;
        symV2RowDownCurrentRecvValues = 0;
        symV2RowDownDemandRecords = 0;
        symV2RowDownSendMessages = 0;
        symV2RowDownRecvMessages = 0;
        symV2RowDownSetupSeconds = 0.0;
// SYM_V2_PC2_PHASE3_INIT_ROW_DOWN_VECTORS_END

        symV2PartnerLPrepacked.assign(static_cast<size_t>(local_cols), 0);
        symPanelReadyEventIds.assign(nsupers, -1);
        symV2UsePcFragmentSchur.assign(
            static_cast<size_t>(nsupers),
            (symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
             superlu_sym_v2_pc_fragment_schur())
                ? 1
                : 0);
        symDiagPrefetchEventIds.assign(nsupers, -1);
        symV2PartnerLMapOffsets.assign(l2u_slots, 0);
        symV2PartnerLPackedMaps.clear();
        symV2PartnerLRecvMapOffsets.clear();
        symV2RowFragRecvMapOffsets.clear();
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

// SYM_V2_PC2_SEGMENT_SENDMAP_FAST_BEGIN
        const bool row_l_segment_sendmap_setup =
            symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
            superlu_sym_v2_pc_fragment_schur() &&
            superlu_sym_v2_pc_fragment_ldl_native() &&
            superlu_sym_v2_row_l_compressed_plan();
// SYM_V2_PC2_SEGMENT_SENDMAP_FAST_END

// SYM_V2_PC2_PARALLEL_SENDMAP_SETUP_BEGIN
        const bool row_l_parallel_sendmap_setup =
            symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
            superlu_sym_v2_pc_fragment_schur() &&
            superlu_sym_v2_pc_fragment_ldl_native() &&
            superlu_sym_v2_row_l_parallel_sendmap();
// SYM_V2_PC2_PARALLEL_SENDMAP_SETUP_END

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

                if (lk < 0 ||
                    static_cast<size_t>(lk) >= symL2ULocalMapsHost.size())
                    ABORT("SymFact local GPU L2U host map slot is invalid.");
                symL2ULocalMapsHost[static_cast<size_t>(lk)].swap(local_map);
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
            if (max_partner_host_send_scratch >
                static_cast<size_t>(std::numeric_limits<int_t>::max()))
                ABORT("SymFact V2 partner-L send staging size exceeds int_t range.");
            maxSymPartnerLSendStageCount =
                static_cast<int_t>(max_partner_host_send_scratch);

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
                        /* Allocated after stream sizing. */
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
                symV2PartnerLHostSendPoolPinnedCount =
                    max_partner_host_send_scratch;
            }

            std::vector<size_t> map_write_offsets =
                symV2PartnerLMapOffsets;
            std::vector<size_t> meta_write_offsets(l2u_slots, 0);
// SYM_V2_PC2_PARALLEL_SENDMAP_FILL_BEGIN
            if (row_l_parallel_sendmap_setup)
            {
                auto build_partner_l_send_maps_for_lk =
                    [&](int_t lk,
                        std::vector<std::pair<int_t, int_t> > &row_order,
                        std::vector<int_t> &row_pos_scratch,
                        std::vector<int_t> &row_sorted_scratch)
                {
                    int_t *lsub = Llu->Lrowind_bc_ptr[lk];
                    int_t *lloc = Llu->Lindval_loc_bc_ptr[lk];
                    if (lsub == NULL || lloc == NULL || lsub[0] <= 0)
                        return;

                    int_t jb = symV2PanelGid(lk);
                    if (jb >= nsupers)
                        return;

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
                        return;

                    int_t knsupc = SuperSize(jb);
                    int_t nsupr = lsub[1];

                    for (int_t lb = 0; lb < nb; ++lb)
                    {
                        int_t luptr_tmp = lloc[lb + idx_v];
                        int_t lptr_tmp = lloc[lb + idx_i];
                        int_t ik = lsub[lptr_tmp];
                        int ikcol = symV2PanelRoot(ik);
                        if (ikcol < 0 || ikcol >= Pc)
                            ABORT("SymFact V2 parallel partner-L target process column is invalid.");
                        int_t len = lsub[lptr_tmp + 1];
                        if (len <= 0)
                            continue;
                        int_t fsupc = FstBlockC(ik);
                        const int_t *row_ids = lsub + lptr_tmp + 2;

                        bool rows_already_sorted = true;
                        for (int_t i = 1; i < len; ++i)
                        {
                            if (row_ids[i - 1] > row_ids[i])
                            {
                                rows_already_sorted = false;
                                break;
                            }
                        }

                        bool rows_scatter_sorted = false;
                        if (!rows_already_sorted)
                        {
                            int_t gsupc = SuperSize(ik);
                            if (gsupc > 0)
                            {
                                row_pos_scratch.assign(
                                    static_cast<size_t>(gsupc),
                                    static_cast<int_t>(-1));
                                bool scatter_ok = true;
                                for (int_t i = 0; i < len; ++i)
                                {
                                    int_t local_row = row_ids[i] - fsupc;
                                    if (local_row < 0 || local_row >= gsupc)
                                    {
                                        scatter_ok = false;
                                        break;
                                    }
                                    if (row_pos_scratch[static_cast<size_t>(local_row)] !=
                                        static_cast<int_t>(-1))
                                    {
                                        scatter_ok = false;
                                        break;
                                    }
                                    row_pos_scratch[static_cast<size_t>(local_row)] = i;
                                }
                                if (scatter_ok)
                                {
                                    row_sorted_scratch.clear();
                                    row_sorted_scratch.reserve(static_cast<size_t>(len));
                                    for (int_t row = 0; row < gsupc; ++row)
                                    {
                                        if (row_pos_scratch[static_cast<size_t>(row)] !=
                                            static_cast<int_t>(-1))
                                            row_sorted_scratch.push_back(row);
                                    }
                                    if (row_sorted_scratch.size() !=
                                        static_cast<size_t>(len))
                                        ABORT("SymFact V2 parallel partner-L row-order scatter lost rows.");
                                    rows_scatter_sorted = true;
                                }
                            }
                        }

                        if (!rows_already_sorted && !rows_scatter_sorted)
                        {
                            row_order.clear();
                            row_order.reserve(static_cast<size_t>(len));
                            for (int_t i = 0; i < len; ++i)
                                row_order.push_back(std::make_pair(
                                    row_ids[i] - fsupc, i));
                            std::sort(row_order.begin(), row_order.end());
                        }

                        auto sorted_row_id = [&](int_t i) -> int_t
                        {
                            if (rows_already_sorted)
                                return row_ids[i] - fsupc;
                            if (rows_scatter_sorted)
                                return row_sorted_scratch[static_cast<size_t>(i)];
                            return row_order[static_cast<size_t>(i)].first;
                        };
                        auto sorted_src_row = [&](int_t i) -> int_t
                        {
                            if (rows_already_sorted)
                                return i;
                            if (rows_scatter_sorted)
                            {
                                int_t local_row =
                                    row_sorted_scratch[static_cast<size_t>(i)];
                                return row_pos_scratch[
                                    static_cast<size_t>(local_row)];
                            }
                            return row_order[static_cast<size_t>(i)].second;
                        };

                        size_t flat = static_cast<size_t>(lk) *
                                          static_cast<size_t>(Pc) +
                                      static_cast<size_t>(ikcol);
                        std::vector<int_t> &meta = symL2LSendMeta[flat];
                        size_t meta_pos = meta_write_offsets[flat];
                        if (meta_pos + static_cast<size_t>(len) + 2 >
                            meta.size())
                            ABORT("SymFact V2 parallel packed partner-L metadata overrun.");
                        meta[meta_pos++] = ik;
                        meta[meta_pos++] = len;
                        for (int_t i = 0; i < len; ++i)
                            meta[meta_pos++] = sorted_row_id(i);
                        meta_write_offsets[flat] = meta_pos;

                        size_t map_pos = map_write_offsets[flat];
                        size_t map_end = symV2PartnerLMapOffsets[flat] +
                                         map_counts[flat];
                        size_t segment_map_offset = map_pos;
                        size_t segment_map_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(knsupc),
                            "SymFact V2 parallel exact send map index segment");
                        for (int_t j = 0; j < knsupc; ++j)
                        {
                            for (int_t i = 0; i < len; ++i)
                            {
                                if (map_pos >= map_end)
                                    ABORT("SymFact V2 parallel packed partner-L send map overrun.");
                                int_t src_row = sorted_src_row(i);
                                symV2PartnerLPackedMaps[map_pos++] =
                                    luptr_tmp + src_row + j * nsupr;
                            }
                        }
                        if (map_pos != segment_map_offset + segment_map_count)
                            ABORT("SymFact V2 parallel exact send map index segment size mismatch.");
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
                };

#ifdef _OPENMP
#pragma omp parallel
                {
                    std::vector<std::pair<int_t, int_t> > row_order;
                    std::vector<int_t> row_pos_scratch;
                    std::vector<int_t> row_sorted_scratch;
#pragma omp for schedule(dynamic)
                    for (int_t lk = 0; lk < local_cols; ++lk)
                        build_partner_l_send_maps_for_lk(
                            lk, row_order, row_pos_scratch,
                            row_sorted_scratch);
                }
#else
                std::vector<std::pair<int_t, int_t> > row_order;
                std::vector<int_t> row_pos_scratch;
                std::vector<int_t> row_sorted_scratch;
                for (int_t lk = 0; lk < local_cols; ++lk)
                    build_partner_l_send_maps_for_lk(
                        lk, row_order, row_pos_scratch,
                        row_sorted_scratch);
#endif
            }
            else
            {
// SYM_V2_PC2_PARALLEL_SENDMAP_SERIAL_FALLBACK_BEGIN
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

// SYM_V2_PC2_SEGMENT_SENDMAP_BLOCK_ORDER_BEGIN
                        const int_t *row_ids = lsub + lptr_tmp + 2;
                        bool rows_already_sorted = row_l_segment_sendmap_setup;
                        if (rows_already_sorted)
                        {
                            for (int_t i = 1; i < len; ++i)
                            {
                                if (row_ids[i - 1] > row_ids[i])
                                {
                                    rows_already_sorted = false;
                                    break;
                                }
                            }
                        }
                        if (!rows_already_sorted)
                        {
                            row_order.clear();
                            row_order.reserve(static_cast<size_t>(len));
                            for (int_t i = 0; i < len; ++i)
                                row_order.push_back(std::make_pair(
                                    row_ids[i] - fsupc, i));
                            std::sort(row_order.begin(), row_order.end());
                        }
// SYM_V2_PC2_SEGMENT_SENDMAP_BLOCK_ORDER_END

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
                            meta[meta_pos++] = rows_already_sorted
                                ? row_ids[i] - fsupc
                                : row_order[i].first;
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
                                int_t src_row = rows_already_sorted
                                    ? i
                                    : row_order[i].second;
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
// SYM_V2_PC2_PARALLEL_SENDMAP_SERIAL_FALLBACK_END
            }
// SYM_V2_PC2_PARALLEL_SENDMAP_FILL_END

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

// SYM_V2_PC2_SEGMENT_SENDMAP_LEGACY_ORDER_BEGIN
                const int_t *row_ids = lsub + lptr_tmp + 2;
                bool rows_already_sorted = row_l_segment_sendmap_setup;
                if (rows_already_sorted)
                {
                    for (int_t i = 1; i < len; ++i)
                    {
                        if (row_ids[i - 1] > row_ids[i])
                        {
                            rows_already_sorted = false;
                            break;
                        }
                    }
                }
                std::vector<std::pair<int_t, int_t> > row_order;
                if (!rows_already_sorted)
                {
                    row_order.reserve(static_cast<size_t>(len));
                    for (int_t i = 0; i < len; ++i)
                        row_order.push_back(std::make_pair(row_ids[i] - fsupc, i));
                    std::sort(row_order.begin(), row_order.end());
                }
// SYM_V2_PC2_SEGMENT_SENDMAP_LEGACY_ORDER_END

                if (have_comml_send)
                {
                    std::vector<int_t> &map = host_maps[ikcol];
                    map.push_back(-(ik + 1));
                    for (int_t i = 0; i < len; ++i)
                    {
                        int_t src_row = rows_already_sorted
                            ? i
                            : row_order[i].second;
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
                    if (flat >= symL2USendMapsHost.size())
                        ABORT("SymFact GPU L2U host map slot is invalid.");
                    symL2USendMapsHost[flat].swap(host_maps[pc]);
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

            symV2PartnerLSendBufPoolCount =
                superlu_sym_v2_pc_fragment_ldl_native()
                    ? 0
                    : total_partner_send;
            symL2LSendMapPoolCount = total_partner_send;
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
            // Post-solve row-L sends intentionally use the coarse aggregated
            // row-fragment demand to avoid exact/direct maps.
            const bool exact_row_fragment_demand_setup =
                exact_fragment_demand_setup &&
                superlu_sym_v2_exact_row_fragment_demand() &&
                !superlu_sym_v2_row_l_postsolve_send() &&
// SYM_V2_PC2_PHASE4_SETUP_EXACT_ROW_GUARD_BEGIN
                !superlu_sym_v2_row_l_plan_v2_exchange();
// SYM_V2_PC2_PHASE4_SETUP_EXACT_ROW_GUARD_END
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_FLAG_BEGIN
            const bool skip_legacy_row_recv_map_setup =
                pc_fragment_schur_setup &&
                superlu_sym_v2_pc_fragment_ldl_native() &&
                superlu_sym_v2_row_l_plan_v2() &&
                superlu_sym_v2_row_l_plan_v2_exchange() &&
                !superlu_sym_v2_row_l_plan_v2_dryrun() &&
                superlu_sym_v2_row_l_skip_legacy_recv_map();
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_FLAG_END
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

// SYM_V2_PC2_PHASE3_SKIP_DENSE_ROW_DEMAND_BEGIN
            std::vector<int_t> all_row_demand_payload;
            if (superlu_sym_v2_row_l_plan_v2() &&
                !superlu_sym_v2_row_l_plan_v2_dryrun())
            {
                /* Active sparse row-down uses a row-communicator Alltoallv
                   below and bypasses the legacy dense/global row-demand
                   Allgather entirely.  Partner-L metadata allgather remains
                   because it is still used by partner fragments and by the
                   source-map/index construction. */
            }
            else
            {
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

                all_row_demand_payload.assign(
                    static_cast<size_t>(total_row_demand_count), 0);
                MPI_Allgatherv(local_row_demand_payload.empty()
                                   ? NULL
                                   : local_row_demand_payload.data(),
                               local_row_demand_count, mpi_int_t,
                               all_row_demand_payload.empty()
                                   ? NULL
                                   : all_row_demand_payload.data(),
                               row_demand_counts.data(),
                               row_demand_displs.data(), mpi_int_t, grid->comm);
            }
// SYM_V2_PC2_PHASE3_SKIP_DENSE_ROW_DEMAND_END

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
            const bool recv_map_index_setup =
                symGPU3DVersion == 2 && Pr > 1 && Pc > 1 &&
                superlu_sym_v2_recv_map_index();
            const bool recv_map_index_verify =
                recv_map_index_setup &&
                superlu_sym_v2_recv_map_index_verify();
            struct SymV2CachedPartnerBlock
            {
                int_t gid;
                size_t cols_begin;
                int_t len;
                std::vector<int_t> cols;
            };
            auto cached_block_len =
                [&](const SymV2CachedPartnerBlock &block) -> int_t
            {
                return recv_map_index_setup
                           ? block.len
                           : static_cast<int_t>(block.cols.size());
            };
            auto cached_block_col =
                [&](const SymV2CachedPartnerBlock &block,
                    size_t pos) -> int_t
            {
                return recv_map_index_setup
                           ? all_meta_payload[block.cols_begin + pos]
                           : block.cols[pos];
            };
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_partner_blocks(nsupers);
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_partner_recv_blocks(compact_count);
            size_t row_chunk_count = xlu_checked_product(
                static_cast<size_t>(nsupers), static_cast<size_t>(Pc),
                "SymFact V2 row-fragment receive table");
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_ALLOC_BEGIN
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_row_blocks;
            std::vector<std::vector<SymV2CachedPartnerBlock> >
                cached_row_recv_blocks;
            if (!skip_legacy_row_recv_map_setup)
            {
                cached_row_blocks.assign(
                    nsupers, std::vector<SymV2CachedPartnerBlock>());
                cached_row_recv_blocks.assign(
                    row_chunk_count, std::vector<SymV2CachedPartnerBlock>());
            }
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_ALLOC_END

            double tRecvCacheBuild =
                profile_setup ? SuperLU_timer_() : 0.0;
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
                            block.cols_begin = block_pos;
                            block.len = len;
                            if (!recv_map_index_setup)
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
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_CACHE_BEGIN
                    if (!skip_legacy_row_recv_map_setup &&
                        source_pr == myrow)
                    {
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_CACHE_END
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
                                block.cols_begin = row_block_pos;
                                block.len = len;
                                if (!recv_map_index_setup)
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
            if (profile_setup)
                symV2SetupProfileAdd(SYM_V2_SETUP_RECV_CACHE_BUILD,
                                     SuperLU_timer_() - tRecvCacheBuild);

            symV2PartnerLRecvSizes.assign(compact_count, 0);
            symV2PartnerLRecvIndex.assign(nsupers, std::vector<int_t>());
            symV2PartnerLRecvIndexBySrc.assign(compact_count, std::vector<int_t>());
            symV2PartnerLRecvMap.assign(compact_count, std::vector<int_t>());
            symV2RowFragRecvSizes.assign(row_chunk_count, 0);
            symV2RowFragRecvIndex.assign(nsupers, std::vector<int_t>());
            symV2RowFragRecvMap.assign(row_chunk_count, std::vector<int_t>());
            symV2PartnerLRecvMapOffsets.assign(compact_count, 0);
            symV2RowFragRecvMapOffsets.assign(row_chunk_count, 0);
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
                double tPartnerRecvIndexBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
                std::sort(blocks.begin(), blocks.end(),
                          [](const SymV2CachedPartnerBlock &a,
                             const SymV2CachedPartnerBlock &b)
                          {
                              return a.gid < b.gid;
                          });

                int_t partner_nblocks = static_cast<int_t>(blocks.size());
                int_t partner_nrows = 0;
                for (size_t ib = 0; ib < blocks.size(); ++ib)
                    partner_nrows += cached_block_len(blocks[ib]);
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
                        cached_block_len(blocks[ib]);
                    for (int_t j = 0; j < cached_block_len(blocks[ib]); ++j)
                        index[row_ptr++] =
                            cached_block_col(blocks[ib], static_cast<size_t>(j));
                }

                std::vector<std::pair<int_t, int_t> > partner_lookup;
                if (recv_map_index_setup)
                {
                    partner_lookup.reserve(static_cast<size_t>(partner_nblocks));
                    for (int_t ib = 0; ib < partner_nblocks; ++ib)
                    {
                        if (!partner_lookup.empty() &&
                            partner_lookup.back().first == blocks[ib].gid)
                            continue;
                        partner_lookup.push_back(std::make_pair(
                            blocks[ib].gid, index[px_ptr + ib]));
                    }
                }
                auto partner_linear_offset = [&](int_t gid) -> int_t
                {
                    for (int_t probe = 0; probe < partner_nblocks; ++probe)
                        if (blocks[probe].gid == gid)
                            return index[px_ptr + probe];
                    return GLOBAL_BLOCK_NOT_FOUND;
                };
                auto partner_indexed_offset = [&](int_t gid) -> int_t
                {
                    if (!recv_map_index_setup)
                        return partner_linear_offset(gid);
                    std::vector<std::pair<int_t, int_t> >::const_iterator it =
                        std::lower_bound(
                            partner_lookup.begin(), partner_lookup.end(),
                            std::make_pair(gid, static_cast<int_t>(0)),
                            [](const std::pair<int_t, int_t> &a,
                               const std::pair<int_t, int_t> &b)
                            {
                                return a.first < b.first;
                            });
                    int_t indexed =
                        (it != partner_lookup.end() && it->first == gid)
                            ? it->second
                            : GLOBAL_BLOCK_NOT_FOUND;
                    if (recv_map_index_verify &&
                        indexed != partner_linear_offset(gid))
                        ABORT("SymFact V2 indexed partner-L receive lookup mismatch.");
                    return indexed;
                };
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_PARTNER_RECV_INDEX_BUILD,
                        SuperLU_timer_() - tPartnerRecvIndexBuild);

                double tPartnerRecvLookupBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
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
                                cached_block_len(recv_blocks[rb]);
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
                                cached_block_len(
                                    recv_blocks[static_cast<size_t>(rb)]);
                            for (int_t rc = 0;
                                 rc < cached_block_len(
                                          recv_blocks[static_cast<size_t>(rb)]);
                                 ++rc)
                                src_index[src_row_ptr++] =
                                    cached_block_col(
                                        recv_blocks[static_cast<size_t>(rb)],
                                        static_cast<size_t>(rc));
                        }
                    }
                    long long expected_values = 0;
                    int_t src_offset = 0;
                    for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                    {
                        int_t recv_offset =
                            partner_indexed_offset(recv_blocks[rb].gid);
                        if (recv_offset == GLOBAL_BLOCK_NOT_FOUND)
                            ABORT("SymFact V2 partner-L receive map cannot find a block.");
                        int_t nrows = cached_block_len(recv_blocks[rb]);
                        recv_map.push_back(recv_offset);
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
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_PARTNER_RECV_LOOKUP_BUILD,
                        SuperLU_timer_() - tPartnerRecvLookupBuild);

// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_BUILD_BEGIN
                if (skip_legacy_row_recv_map_setup)
                    continue;
// SYM_V2_PC2_SKIP_LEGACY_ROW_RECV_BUILD_END
                std::vector<SymV2CachedPartnerBlock> &row_blocks =
                    cached_row_blocks[k0];
                if (row_blocks.empty())
                    continue;
                double tRowRecvIndexBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
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
                    row_nrows += cached_block_len(row_blocks[rb]);
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
                        cached_block_len(row_blocks[rb]);
                    for (int_t rc = 0; rc < cached_block_len(row_blocks[rb]); ++rc)
                        row_index[row_data_ptr++] =
                            cached_block_col(row_blocks[rb],
                                             static_cast<size_t>(rc));
                }

                std::vector<std::pair<int_t, int_t> > row_lookup;
                if (recv_map_index_setup)
                {
                    row_lookup.reserve(static_cast<size_t>(row_nblocks));
                    for (int_t rb = 0; rb < row_nblocks; ++rb)
                        row_lookup.push_back(std::make_pair(
                            row_blocks[rb].gid, row_index[row_px_ptr + rb]));
                }
                auto row_linear_offset = [&](int_t gid) -> int_t
                {
                    for (int_t probe = 0; probe < row_nblocks; ++probe)
                        if (row_blocks[probe].gid == gid)
                            return row_index[row_px_ptr + probe];
                    return GLOBAL_BLOCK_NOT_FOUND;
                };
                auto row_indexed_offset = [&](int_t gid) -> int_t
                {
                    if (!recv_map_index_setup)
                        return row_linear_offset(gid);
                    std::vector<std::pair<int_t, int_t> >::const_iterator it =
                        std::lower_bound(
                            row_lookup.begin(), row_lookup.end(),
                            std::make_pair(gid, static_cast<int_t>(0)),
                            [](const std::pair<int_t, int_t> &a,
                               const std::pair<int_t, int_t> &b)
                            {
                                return a.first < b.first;
                            });
                    int_t indexed =
                        (it != row_lookup.end() && it->first == gid)
                            ? it->second
                            : GLOBAL_BLOCK_NOT_FOUND;
                    if (recv_map_index_verify &&
                        indexed != row_linear_offset(gid))
                        ABORT("SymFact V2 indexed row-fragment receive lookup mismatch.");
                    return indexed;
                };
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_RECV_INDEX_BUILD,
                        SuperLU_timer_() - tRowRecvIndexBuild);

                double tRowRecvLookupBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
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
                        int_t recv_offset =
                            row_indexed_offset(recv_blocks[rb].gid);
                        if (recv_offset == GLOBAL_BLOCK_NOT_FOUND)
                            ABORT("SymFact V2 row-fragment receive map cannot find a block.");
                        int_t nrows = cached_block_len(recv_blocks[rb]);
                        recv_map.push_back(recv_offset);
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
                if (profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_RECV_LOOKUP_BUILD,
                        SuperLU_timer_() - tRowRecvLookupBuild);
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

                auto build_exact_send_host_maps =
                    [&](const std::vector<std::vector<int_t> > &block_gids,
                        int dest_dim,
                        std::vector<int> &sizes,
                        std::vector<size_t> &offsets,
                        std::vector<int_t> &packed_maps,
                        std::vector<std::vector<double> > &host_bufs,
                        std::vector<double *> &host_bufs_pinned,
                        const char *what)
                {
                    (void)what;
                    if (block_gids.size() != sizes.size() ||
                        block_gids.size() != host_bufs.size() ||
                        block_gids.size() != host_bufs_pinned.size())
                        ABORT("SymFact V2 exact send slot table is invalid.");
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

                    bool use_pinned_host =
                        !superlu_cuda_aware_mpi() &&
                        superlu_sym_v2_pinned_staging();
                    for (size_t slot = 0; slot < sizes.size(); ++slot)
                    {
                        if (sizes[slot] <= 0)
                            continue;
                        if (superlu_cuda_aware_mpi())
                            continue;
                        if (use_pinned_host)
                        {
                            /* Allocated after stream sizing. */
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
                {
                    build_exact_send_host_maps(
                        partner_exact_send_block_gids, Pr,
                        symV2PartnerLExactSendSizes,
                        symV2PartnerLExactSendMapOffsets,
                        symV2PartnerLExactSendMapsHost,
                        symV2PartnerLExactHostSendBufs,
                        symV2PartnerLExactHostSendBufsPinned,
                        "SymFact V2 exact partner-L send map");
                    symV2PartnerLExactSendBufPoolCount =
                        symV2PartnerLExactSendMapsHost.size();
                    symV2PartnerLExactSendMapPoolCount =
                        symV2PartnerLExactSendMapsHost.size();
                }
                if (exact_row_fragment_demand_setup &&
                    !superlu_sym_v2_pc_fragment_ldl_native())
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
                superlu_sym_v2_row_l_direct_recv() &&
                !superlu_sym_v2_row_l_postsolve_send() &&
                !superlu_sym_v2_row_l_plan_v2_exchange())
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

                auto source_value_count_for_gid =
                    [&](size_t flat, int_t gid) -> size_t
                {
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size())
                        ABORT("SymFact V2 direct row-L source count is invalid.");
                    if (symV2PartnerLSendSizes[flat] < 0)
                        ABORT("SymFact V2 direct row-L source count size is invalid.");
                    int_t lk = static_cast<int_t>(
                        flat / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 direct row-L count panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    if (ksupc <= 0)
                        ABORT("SymFact V2 direct row-L count panel width is invalid.");
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    bool found = false;
                    size_t count = 0;
                    size_t scanned = 0;
                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 direct row-L count metadata is truncated.");
                        int_t block_gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 ||
                            meta_pos + static_cast<size_t>(len) >
                                meta.size())
                            ABORT("SymFact V2 direct row-L count metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 direct row-L count segment");
                        if (block_gid == gid)
                        {
                            if (count >
                                std::numeric_limits<size_t>::max() -
                                    value_count)
                                ABORT("SymFact V2 direct row-L count overflows.");
                            count += value_count;
                            found = true;
                        }
                        if (scanned >
                            std::numeric_limits<size_t>::max() - value_count)
                            ABORT("SymFact V2 direct row-L scanned count overflows.");
                        scanned += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (scanned !=
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]))
                        ABORT("SymFact V2 direct row-L count size mismatch.");
                    if (!found)
                        ABORT("SymFact V2 direct row-L source count cannot find a requested block.");
                    return count;
                };

                double tDirectRowMapBuild =
                    profile_setup ? SuperLU_timer_() : 0.0;
                const bool ldl_native_direct_row =
                    superlu_sym_v2_pc_fragment_ldl_native();
                std::fill(symV2RowDirectSendSizes.begin(),
                          symV2RowDirectSendSizes.end(), 0);
                std::fill(symV2RowDirectSendMapOffsets.begin(),
                          symV2RowDirectSendMapOffsets.end(), 0);
                symV2RowDirectSendMapsHost.clear();
                if (ldl_native_direct_row)
                    symV2RowDirectSendBlocksHost.assign(
                        symV2RowDirectSendSizes.size(),
                        std::vector<int_t>());
                else
                    symV2RowDirectSendBlocksHost.clear();

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
                        int_t k0 = symV2PanelGid(lk);
                        if (k0 < 0 || k0 >= nsupers)
                            ABORT("SymFact V2 direct row-L panel is invalid.");
                        int_t ksupc = SuperSize(k0);
                        if (ldl_native_direct_row)
                        {
                            if (slot >= symV2RowDirectSendBlocksHost.size())
                                ABORT("SymFact V2 direct row-L block descriptor slot is invalid.");
                            std::vector<int_t> &block_desc =
                                symV2RowDirectSendBlocksHost[slot];
                            block_desc.clear();
                            block_desc.reserve(unique_blocks.size() * 2);
                            size_t map_count = 0;
                            for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                            {
                                if (unique_blocks[bi].chunk_pc < 0 ||
                                    unique_blocks[bi].chunk_pc >= Pc)
                                    ABORT("SymFact V2 direct row-L block source column is invalid.");
                                size_t flat =
                                    static_cast<size_t>(lk) *
                                        static_cast<size_t>(Pc) +
                                    static_cast<size_t>(
                                        unique_blocks[bi].chunk_pc);
                                size_t value_count = source_value_count_for_gid(
                                    flat, unique_blocks[bi].gid);
                                if (map_count >
                                    std::numeric_limits<size_t>::max() -
                                        value_count)
                                    ABORT("SymFact V2 direct row-L compact send count overflows.");
                                map_count += value_count;
                                block_desc.push_back(unique_blocks[bi].gid);
                                block_desc.push_back(static_cast<int_t>(
                                    unique_blocks[bi].chunk_pc));
                            }
                            if (map_count >
                                static_cast<size_t>(std::numeric_limits<int>::max()))
                                ABORT("SymFact V2 direct row-L compact send map is too large for MPI.");
                            symV2RowDirectSendSizes[slot] =
                                static_cast<int>(map_count);
                            continue;
                        }
                        symV2RowDirectSendMapOffsets[slot] =
                            symV2RowDirectSendMapsHost.size();
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

                if (superlu_sym_v2_pc_fragment_ldl_native())
                {
                    long long exact_max_row_stage = 0;
                    long long exact_max_row_values = 0;
                    long long exact_max_row_index = LPANEL_HEADER_SIZE;
                    long long exact_max_row_send = 0;
                    for (int_t k0 = 0; k0 < nsupers; ++k0)
                    {
                        long long stage_values = 0;
                        size_t row_recv_base =
                            static_cast<size_t>(k0) *
                            static_cast<size_t>(Pc);
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t pos = row_recv_base +
                                         static_cast<size_t>(pc);
                            if (pos < symV2RowFragRecvSizes.size())
                                stage_values +=
                                    static_cast<long long>(
                                        symV2RowFragRecvSizes[pos]);
                        }
                        exact_max_row_stage =
                            SUPERLU_MAX(exact_max_row_stage, stage_values);

                        if (static_cast<size_t>(k0) <
                                symV2RowFragRecvIndex.size() &&
                            !symV2RowFragRecvIndex[k0].empty())
                        {
                            const std::vector<int_t> &row_index =
                                symV2RowFragRecvIndex[k0];
                            if (row_index.size() < LPANEL_HEADER_SIZE)
                                ABORT("SymFact V2 direct row-L exact index is truncated.");
                            long long row_values =
                                static_cast<long long>(row_index[1]) *
                                static_cast<long long>(SuperSize(k0));
                            exact_max_row_values =
                                SUPERLU_MAX(exact_max_row_values,
                                            row_values);
                            exact_max_row_index = SUPERLU_MAX(
                                exact_max_row_index,
                                static_cast<long long>(row_index.size()));
                        }
                    }
                    for (int_t lk = 0; lk < local_cols; ++lk)
                    {
                        long long aggregate_remote_send = 0;
                        for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                        {
                            size_t slot =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc_dest);
                            long long dest_values = 0;
                            if (slot < symV2RowDirectSendSizes.size())
                                dest_values = static_cast<long long>(
                                    symV2RowDirectSendSizes[slot]);
                            exact_max_row_send =
                                SUPERLU_MAX(exact_max_row_send,
                                            dest_values);
                            if (pc_dest != mycol)
                                aggregate_remote_send += dest_values;
                        }
                        exact_max_row_send =
                            SUPERLU_MAX(exact_max_row_send,
                                        aggregate_remote_send);
                    }
                    exact_max_row_stage =
                        SUPERLU_MAX(exact_max_row_stage,
                                    exact_max_row_send);
                    if (exact_max_row_stage >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_values >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_index >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_send >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()))
                        ABORT("SymFact V2 direct row-L exact staging size exceeds int_t range.");
                    maxSymV2RowFragStageCount =
                        static_cast<int_t>(exact_max_row_stage);
                    maxSymV2RowFragValRecvCount =
                        static_cast<int_t>(exact_max_row_values);
                    maxSymV2RowFragIdxRecvCount =
                        static_cast<int_t>(exact_max_row_index);
                    maxSymV2RowFragValSendCount =
                        static_cast<int_t>(exact_max_row_send);
                }
            }


// SYM_V2_PC2_PHASE3_SPARSE_ROW_DOWN_PLAN_BEGIN
            if (pc_fragment_schur_setup && superlu_sym_v2_row_l_plan_v2())
            {
                const double row_down_setup_t = SuperLU_timer_();
                if (!superlu_sym_v2_row_l_plan_v2_block())
                    ABORT("GPU3DV2_ROW_L_PLAN_V2 currently implements block-level demand only.");
                int row_comm_size = 0;
                int row_comm_rank = -1;
                MPI_Comm_size(grid3d->rscp.comm, &row_comm_size);
                MPI_Comm_rank(grid3d->rscp.comm, &row_comm_rank);
                if (row_comm_size != Pc || row_comm_rank != mycol)
                    ABORT("SymFact V2 row-down plan assumes rscp ranks are process columns.");
                const bool row_down_dryrun =
                    superlu_sym_v2_row_l_plan_v2_dryrun();
                const bool row_down_compact =
                    superlu_sym_v2_row_l_plan_v2_compact();
                const bool row_down_direct_recv =
                    superlu_sym_v2_row_l_direct_recv() &&
                    !superlu_sym_v2_row_l_postsolve_send();
                const bool row_down_compressed_plan =
                    superlu_sym_v2_row_l_compressed_plan();
// SYM_V2_PC2_LAZY_SENDMAP_SETUP_FLAG_BEGIN
                const bool row_down_lazy_sendmap =
                    row_down_direct_recv &&
                    row_down_compressed_plan &&
                    superlu_sym_v2_pc_fragment_ldl_native() &&
                    superlu_sym_v2_row_l_lazy_sendmap();
// SYM_V2_PC2_LAZY_SENDMAP_SETUP_FLAG_END
                if (row_down_compact)
                {
                    if (row_down_dryrun)
                        ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PLAN_V2_DRYRUN=0.");
                    if (!superlu_sym_v2_row_l_plan_v2_exchange())
                        ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PLAN_V2_EXCHANGE=1.");
                    if (!superlu_sym_v2_row_l_pack_all_dest())
                        ABORT("GPU3DV2_ROW_L_PLAN_V2_COMPACT requires GPU3DV2_ROW_L_PACK_ALL_DEST=1.");
                }
                if (!row_down_dryrun &&
                    !superlu_sym_v2_row_l_plan_v2_exchange())
                    ABORT("GPU3DV2_ROW_L_PLAN_V2_DRYRUN=0 requires GPU3DV2_ROW_L_PLAN_V2_EXCHANGE=1 so rebuilt row metadata is consumed by matching row-down sends.");
                std::vector<int> saved_row_frag_recv_sizes;
                std::vector<std::vector<int_t> > saved_row_frag_recv_index;
                std::vector<std::vector<int_t> > saved_row_frag_recv_map;
                if (row_down_dryrun)
                {
                    saved_row_frag_recv_sizes = symV2RowFragRecvSizes;
                    saved_row_frag_recv_index = symV2RowFragRecvIndex;
                    saved_row_frag_recv_map = symV2RowFragRecvMap;
                }

                const size_t row_chunk_count = xlu_checked_product(
                    static_cast<size_t>(nsupers), static_cast<size_t>(Pc),
                    "SymFact V2 row-down chunk table");
                std::vector<std::vector<int_t> > requested_by_chunk(row_chunk_count);
                std::vector<std::vector<int_t> > row_down_payloads(static_cast<size_t>(Pc));

// SYM_V2_PC2_COMPRESSED_PLAN_DEMAND_ENCODE_BEGIN
                auto append_row_down_demand_record =
                    [&](std::vector<int_t> &payload, int_t panel,
                        int dest_pc, int chunk_pc,
                        const std::vector<int_t> &blocks)
                {
                    if (blocks.empty())
                        return;
                    if (blocks.size() >
                        static_cast<size_t>(std::numeric_limits<int_t>::max()))
                        ABORT("SymFact V2 row-down demand record is too large.");

                    payload.push_back(panel);
                    payload.push_back(dest_pc);
                    payload.push_back(chunk_pc);

                    if (row_down_compressed_plan)
                    {
                        std::vector<int_t> ranges;
                        ranges.reserve(blocks.size());
                        int_t start = blocks[0];
                        int_t prev = blocks[0];
                        if (start < 0)
                            ABORT("SymFact V2 row-down compressed demand has invalid block id.");
                        for (size_t bi = 1; bi < blocks.size(); ++bi)
                        {
                            int_t gid = blocks[bi];
                            if (gid <= prev)
                                ABORT("SymFact V2 row-down compressed demand is not sorted unique.");
                            if (prev < std::numeric_limits<int_t>::max() &&
                                gid == prev + 1)
                            {
                                prev = gid;
                                continue;
                            }
                            ranges.push_back(start);
                            ranges.push_back(prev - start + 1);
                            start = gid;
                            prev = gid;
                        }
                        ranges.push_back(start);
                        ranges.push_back(prev - start + 1);

                        /* Range encoding is [k0, dest_pc, chunk_pc,
                           -nranges, start0, len0, ...].  Use it only when it
                           strictly reduces the demand payload. */
                        if (ranges.size() < blocks.size())
                        {
                            size_t nranges = ranges.size() / 2;
                            if (nranges > static_cast<size_t>(
                                              std::numeric_limits<int_t>::max()))
                                ABORT("SymFact V2 row-down compressed demand has too many ranges.");
                            payload.push_back(-static_cast<int_t>(nranges));
                            payload.insert(payload.end(), ranges.begin(), ranges.end());
                            return;
                        }
                    }

                    payload.push_back(static_cast<int_t>(blocks.size()));
                    payload.insert(payload.end(), blocks.begin(), blocks.end());
                };
// SYM_V2_PC2_COMPRESSED_PLAN_DEMAND_ENCODE_END

                long long local_demand_records = 0;
                long long local_current_recv_values = 0;

                const double tRowCompactDemandBuild = SuperLU_timer_();
                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    size_t row_recv_base =
                        static_cast<size_t>(k0) * static_cast<size_t>(Pc);
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        size_t pos = row_recv_base + static_cast<size_t>(pc);
                        if (pos < symV2RowFragRecvSizes.size() &&
                            symV2RowFragRecvSizes[pos] > 0)
                            local_current_recv_values +=
                                static_cast<long long>(symV2RowFragRecvSizes[pos]);
                    }

                    std::vector<int_t> needed_copy;
                    std::vector<int_t> *needed_ptr = NULL;
                    if (row_down_compact)
                    {
                        std::vector<int_t> &needed_mut =
                            needed_row_blocks_by_panel[k0];
                        if (needed_mut.empty())
                            continue;
                        std::sort(needed_mut.begin(), needed_mut.end());
                        needed_mut.erase(
                            std::unique(needed_mut.begin(), needed_mut.end()),
                            needed_mut.end());
                        needed_ptr = &needed_mut;
                    }
                    else
                    {
                        needed_copy = needed_row_blocks_by_panel[k0];
                        if (needed_copy.empty())
                            continue;
                        std::sort(needed_copy.begin(), needed_copy.end());
                        needed_copy.erase(
                            std::unique(needed_copy.begin(),
                                        needed_copy.end()),
                            needed_copy.end());
                        needed_ptr = &needed_copy;
                    }
                    const std::vector<int_t> &needed = *needed_ptr;
                    if (needed.empty())
                        continue;
                    int source_pc = symV2PanelRoot(k0);
                    if (source_pc < 0 || source_pc >= Pc)
                        ABORT("SymFact V2 row-down plan has invalid source process column.");

                    for (size_t bi = 0; bi < needed.size(); ++bi)
                    {
                        int_t gid = needed[bi];
                        int chunk_pc = symV2PanelRoot(gid);
                        if (chunk_pc < 0 || chunk_pc >= Pc)
                            ABORT("SymFact V2 row-down plan has invalid chunk process column.");
                        requested_by_chunk[static_cast<size_t>(k0) *
                                               static_cast<size_t>(Pc) +
                                           static_cast<size_t>(chunk_pc)]
                            .push_back(gid);
                    }
                    for (int chunk_pc = 0; chunk_pc < Pc; ++chunk_pc)
                    {
                        std::vector<int_t> &blocks =
                            requested_by_chunk[static_cast<size_t>(k0) *
                                                    static_cast<size_t>(Pc) +
                                                static_cast<size_t>(chunk_pc)];
                        if (blocks.empty())
                            continue;
                        if (!row_down_compact)
                        {
                            std::sort(blocks.begin(), blocks.end());
                            blocks.erase(std::unique(blocks.begin(),
                                                     blocks.end()),
                                         blocks.end());
                        }
                        std::vector<int_t> &payload =
                            row_down_payloads[static_cast<size_t>(source_pc)];
// SYM_V2_PC2_COMPRESSED_PLAN_PAYLOAD_EMIT_BEGIN
                        /* Record encoding over grid3d->rscp.comm:
                           uncompressed: [k0, dest_pc, chunk_pc, nblocks, gids...]
                           compressed:   [k0, dest_pc, chunk_pc, -nranges,
                                          start0, len0, ...]
                           Sender rank is dest_pc; receiver rank is source_pc. */
                        append_row_down_demand_record(
                            payload, k0, mycol, chunk_pc, blocks);
// SYM_V2_PC2_COMPRESSED_PLAN_PAYLOAD_EMIT_END
                        ++local_demand_records;
                    }
                }
                if (row_down_compact && profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_COMPACT_DEMAND_BUILD,
                        SuperLU_timer_() - tRowCompactDemandBuild);

                /* Build destination-side receive metadata from the same
                   requested_by_chunk table.  This makes Phase 4 source send
                   order and destination receive/assembly order identical:
                   records are ordered by k0, chunk_pc, and then source metadata
                   order within the chunk. */
                std::vector<std::vector<SymV2CachedPartnerBlock> >
                    row_down_blocks_by_panel(nsupers);
                std::vector<std::vector<SymV2CachedPartnerBlock> >
                    row_down_recv_blocks(row_chunk_count);

                const double tRowCompactRecvMetaBuild = SuperLU_timer_();
                for (int r = 0; r < comm_size; ++r)
                {
                    size_t meta_pos = static_cast<size_t>(meta_displs[r]);
                    size_t rank_end = meta_pos + static_cast<size_t>(meta_counts[r]);
                    int source_pr = MYROW(r, grid);
                    while (meta_pos < rank_end)
                    {
                        if (meta_pos + 3 > rank_end)
                            ABORT("SymFact V2 row-down metadata payload is truncated.");
                        int_t target_pc = all_meta_payload[meta_pos++];
                        int_t k0 = all_meta_payload[meta_pos++];
                        int_t meta_len = all_meta_payload[meta_pos++];
                        if (target_pc < 0 || target_pc >= Pc || k0 < 0 ||
                            k0 >= nsupers || meta_len < 0 ||
                            meta_pos + static_cast<size_t>(meta_len) > rank_end)
                            ABORT("SymFact V2 row-down metadata payload is invalid.");
                        size_t block_pos = meta_pos;
                        size_t block_end = meta_pos + static_cast<size_t>(meta_len);
                        if (source_pr == myrow)
                        {
                            size_t req_pos = static_cast<size_t>(k0) *
                                                 static_cast<size_t>(Pc) +
                                             static_cast<size_t>(target_pc);
                            const std::vector<int_t> &requested =
                                requested_by_chunk[req_pos];
                            if (!requested.empty())
                            {
                                while (block_pos < block_end)
                                {
                                    if (block_pos + 2 > block_end)
                                        ABORT("SymFact V2 row-down metadata block is truncated.");
                                    SymV2CachedPartnerBlock block;
                                    block.gid = all_meta_payload[block_pos++];
                                    int_t len = all_meta_payload[block_pos++];
                                    if (len < 0 ||
                                        block_pos + static_cast<size_t>(len) > block_end)
                                        ABORT("SymFact V2 row-down metadata block has invalid length.");
                                    if (std::binary_search(requested.begin(),
                                                           requested.end(),
                                                           block.gid))
                                    {
                                        block.cols.assign(
                                            all_meta_payload.begin() + block_pos,
                                            all_meta_payload.begin() + block_pos + len);
                                        row_down_blocks_by_panel[k0].push_back(block);
                                        row_down_recv_blocks[req_pos].push_back(block);
                                    }
                                    block_pos += static_cast<size_t>(len);
                                }
                            }
                        }
                        meta_pos = block_end;
                    }
                }

                symV2RowFragRecvSizes.assign(row_chunk_count, 0);
                symV2RowFragRecvIndex.assign(nsupers, std::vector<int_t>());
                symV2RowFragRecvMap.assign(row_chunk_count, std::vector<int_t>());
                symV2RowDownRecvSizes.assign(row_chunk_count, 0);
                symV2RowDownSparseRecvValues = 0;
                symV2RowDownRecvMessages = 0;
                symV2RowDownSegs.clear();
                symV2RowDownSegOffsets.assign(l2u_slots + 1, 0);

                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    std::vector<SymV2CachedPartnerBlock> &blocks =
                        row_down_blocks_by_panel[k0];
                    if (blocks.empty())
                        continue;
                    std::sort(blocks.begin(), blocks.end(),
                              [](const SymV2CachedPartnerBlock &a,
                                 const SymV2CachedPartnerBlock &b)
                              { return a.gid < b.gid; });
                    std::vector<SymV2CachedPartnerBlock> unique_blocks;
                    unique_blocks.reserve(blocks.size());
                    for (size_t bi = 0; bi < blocks.size(); ++bi)
                    {
                        if (!unique_blocks.empty() &&
                            unique_blocks.back().gid == blocks[bi].gid)
                            continue;
                        unique_blocks.push_back(blocks[bi]);
                    }
                    blocks.swap(unique_blocks);

                    int_t row_nblocks = static_cast<int_t>(blocks.size());
                    int_t row_nrows = 0;
                    for (size_t bi = 0; bi < blocks.size(); ++bi)
                        row_nrows += static_cast<int_t>(blocks[bi].cols.size());
                    int_t row_index_size =
                        LPANEL_HEADER_SIZE + 2 * row_nblocks + 1 + row_nrows;
                    if (row_index_size > maxSymV2RowFragIdxRecvCount)
                        ABORT("SymFact V2 row-down cached index exceeds receive buffer.");
                    if (static_cast<int64_t>(row_nrows) *
                            static_cast<int64_t>(SuperSize(k0)) >
                        static_cast<int64_t>(maxSymV2RowFragValRecvCount))
                        ABORT("SymFact V2 row-down cached values exceed receive buffer.");

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
                    for (int_t bi = 0; bi < row_nblocks; ++bi)
                    {
                        row_index[row_gid_ptr + bi] = blocks[bi].gid;
                        row_index[row_px_ptr + bi + 1] =
                            row_index[row_px_ptr + bi] +
                            static_cast<int_t>(blocks[bi].cols.size());
                        for (size_t rc = 0; rc < blocks[bi].cols.size(); ++rc)
                            row_index[row_data_ptr++] = blocks[bi].cols[rc];
                    }

                    for (int chunk_pc = 0; chunk_pc < Pc; ++chunk_pc)
                    {
                        size_t recv_pos = static_cast<size_t>(k0) *
                                              static_cast<size_t>(Pc) +
                                          static_cast<size_t>(chunk_pc);
                        std::vector<SymV2CachedPartnerBlock> &recv_blocks =
                            row_down_recv_blocks[recv_pos];
                        std::vector<int_t> &recv_map =
                            symV2RowFragRecvMap[recv_pos];
                        long long expected_values = 0;
                        int_t src_offset = 0;
                        for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                        {
// SYM_V2_PC2_ROW_RECV_LOWER_BOUND_BEGIN
                            std::vector<SymV2CachedPartnerBlock>::const_iterator block_it =
                                std::lower_bound(
                                    blocks.begin(), blocks.end(),
                                    recv_blocks[rb].gid,
                                    [](const SymV2CachedPartnerBlock &block,
                                       int_t gid)
                                    { return block.gid < gid; });
                            if (block_it == blocks.end() ||
                                block_it->gid != recv_blocks[rb].gid)
                                ABORT("SymFact V2 row-down receive map cannot find a block.");
                            int_t ib = static_cast<int_t>(block_it - blocks.begin());
// SYM_V2_PC2_ROW_RECV_LOWER_BOUND_END
                            int_t nrows =
                                static_cast<int_t>(recv_blocks[rb].cols.size());
                            recv_map.push_back(row_index[row_px_ptr + ib]);
                            recv_map.push_back(nrows);
                            recv_map.push_back(src_offset);
                            SymV2RowDownSeg seg;
                            seg.gid = recv_blocks[rb].gid;
                            seg.chunk_pc = chunk_pc;
                            seg.nrows = nrows;
                            seg.dst_row_offset = row_index[row_px_ptr + ib];
                            seg.value_count = nrows * SuperSize(k0);
                            seg.map_offset = static_cast<size_t>(src_offset);
                            symV2RowDownSegs.push_back(seg);
                            src_offset += nrows * SuperSize(k0);
                            expected_values +=
                                static_cast<long long>(nrows) *
                                static_cast<long long>(SuperSize(k0));
                        }
                        if (expected_values >
                            static_cast<long long>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 row-down receive size is too large.");
                        symV2RowFragRecvSizes[recv_pos] =
                            static_cast<int>(expected_values);
                        symV2RowDownRecvSizes[recv_pos] =
                            static_cast<int>(expected_values);
                        symV2RowDownSparseRecvValues += expected_values;
                        if (expected_values > 0)
                            ++symV2RowDownRecvMessages;
                    }
                }
                if (row_down_compact && profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_COMPACT_RECV_META_BUILD,
                        SuperLU_timer_() - tRowCompactRecvMetaBuild);

                const double tRowCompactSendMapBuild = SuperLU_timer_();
                std::vector<int> send_counts(static_cast<size_t>(Pc), 0);
                std::vector<int> recv_counts(static_cast<size_t>(Pc), 0);
                std::vector<int> send_displs(static_cast<size_t>(Pc), 0);
                std::vector<int> recv_displs(static_cast<size_t>(Pc), 0);
                int total_send_count = 0;
                for (int pc = 0; pc < Pc; ++pc)
                {
                    if (row_down_payloads[static_cast<size_t>(pc)].size() >
                        static_cast<size_t>(std::numeric_limits<int>::max()))
                        ABORT("SymFact V2 row-down demand payload is too large for MPI.");
                    send_displs[static_cast<size_t>(pc)] = total_send_count;
                    send_counts[static_cast<size_t>(pc)] = static_cast<int>(
                        row_down_payloads[static_cast<size_t>(pc)].size());
                    total_send_count += send_counts[static_cast<size_t>(pc)];
                }
                std::vector<int_t> send_payload(static_cast<size_t>(total_send_count));
                for (int pc = 0; pc < Pc; ++pc)
                {
                    std::copy(row_down_payloads[static_cast<size_t>(pc)].begin(),
                              row_down_payloads[static_cast<size_t>(pc)].end(),
                              send_payload.begin() + send_displs[static_cast<size_t>(pc)]);
                }
                MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                             recv_counts.data(), 1, MPI_INT,
                             grid3d->rscp.comm);
                int total_recv_count = 0;
                for (int pc = 0; pc < Pc; ++pc)
                {
                    if (recv_counts[static_cast<size_t>(pc)] < 0)
                        ABORT("SymFact V2 row-down demand receive count is invalid.");
                    recv_displs[static_cast<size_t>(pc)] = total_recv_count;
                    total_recv_count += recv_counts[static_cast<size_t>(pc)];
                }
                std::vector<int_t> recv_payload(static_cast<size_t>(total_recv_count));
                MPI_Alltoallv(send_payload.empty() ? NULL : send_payload.data(),
                              send_counts.data(), send_displs.data(), mpi_int_t,
                              recv_payload.empty() ? NULL : recv_payload.data(),
                              recv_counts.data(), recv_displs.data(), mpi_int_t,
                              grid3d->rscp.comm);

                struct SymV2RowDownDirectBlock
                {
                    int_t gid;
                    int chunk_pc;
                };
// SYM_V2_PC2_RANGE_NATIVE_DIRECT_MAP_BEGIN
                struct SymV2RowDownDirectRange
                {
                    int_t start;
                    int_t len;
                    int chunk_pc;
                };
// SYM_V2_PC2_RANGE_NATIVE_DIRECT_MAP_END
                std::vector<std::vector<int_t> > slot_maps(l2u_slots);
                std::vector<std::vector<SymV2RowDownDirectBlock> >
                    slot_direct_blocks;
                std::vector<std::vector<SymV2RowDownDirectRange> >
                    slot_direct_ranges;
// SYM_V2_PC2_LAZY_SENDMAP_SLOT_STORAGE_BEGIN
                std::vector<std::vector<SymV2RowDownSendSegmentGPU> >
                    slot_send_segments;
                std::vector<int> slot_send_segment_values;
                if (row_down_lazy_sendmap)
                {
                    slot_send_segments.resize(l2u_slots);
                    slot_send_segment_values.assign(l2u_slots, 0);
                }
// SYM_V2_PC2_LAZY_SENDMAP_SLOT_STORAGE_END
                if (row_down_direct_recv)
                {
                    if (row_down_compressed_plan)
                        slot_direct_ranges.resize(l2u_slots);
                    else
                        slot_direct_blocks.resize(l2u_slots);
                }
                auto append_row_down_source_map_for_gid =
                    [&](size_t flat, int_t gid, std::vector<int_t> &out)
                {
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size() ||
                        flat >= symV2PartnerLMapOffsets.size())
                        ABORT("SymFact V2 row-down source map is invalid.");
                    int_t lk = static_cast<int_t>(
                        flat / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 row-down source panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t map_pos = symV2PartnerLMapOffsets[flat];
                    size_t map_end = map_pos +
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                    if (map_end > symV2PartnerLPackedMaps.size() ||
                        map_end < map_pos)
                        ABORT("SymFact V2 row-down source map range is invalid.");
                    bool found = false;
                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 row-down source metadata is truncated.");
                        int_t block_gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 ||
                            meta_pos + static_cast<size_t>(len) > meta.size())
                            ABORT("SymFact V2 row-down source metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 row-down source map segment");
                        if (map_pos + value_count > map_end ||
                            map_pos + value_count < map_pos)
                            ABORT("SymFact V2 row-down source map segment is invalid.");
                        if (block_gid == gid)
                        {
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() + map_pos,
                                       symV2PartnerLPackedMaps.begin() +
                                           map_pos + value_count);
                            found = true;
                        }
                        map_pos += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (map_pos != map_end)
                        ABORT("SymFact V2 row-down source map range size mismatch.");
                    if (!found)
                        ABORT("SymFact V2 row-down source map cannot find a requested block.");
                };
                auto append_row_down_map_for_record =
                    [&](int_t k0, int chunk_pc,
                        const int_t *requested_begin,
                        const int_t *requested_end,
                        std::vector<int_t> &out)
                {
                    if (requested_begin == requested_end)
                        return;
                    int_t lk = symV2PanelIndex(k0);
                    if (lk < 0)
                        return;
                    size_t flat =
                        static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                        static_cast<size_t>(chunk_pc);
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size() ||
                        flat >= symV2PartnerLMapOffsets.size())
                        ABORT("SymFact V2 row-down source metadata is invalid.");
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t map_pos = symV2PartnerLMapOffsets[flat];
                    size_t map_end = map_pos +
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                    if (map_end > symV2PartnerLPackedMaps.size() ||
                        map_end < map_pos)
                        ABORT("SymFact V2 row-down source map range is invalid.");
                    int_t ksupc = SuperSize(k0);
                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 row-down source metadata is truncated.");
                        int_t gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 || meta_pos + static_cast<size_t>(len) > meta.size())
                            ABORT("SymFact V2 row-down source metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 row-down source map segment");
                        if (map_pos + value_count > map_end ||
                            map_pos + value_count < map_pos)
                            ABORT("SymFact V2 row-down source map segment is invalid.");
                        if (std::binary_search(requested_begin,
                                               requested_end, gid))
                        {
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() + map_pos,
                                       symV2PartnerLPackedMaps.begin() + map_pos + value_count);
                        }
                        map_pos += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (map_pos != map_end)
                        ABORT("SymFact V2 row-down source map range size mismatch.");
                };

// SYM_V2_PC2_COMPRESSED_PLAN_RANGE_SOURCE_MAP_BEGIN
                auto append_row_down_map_for_ranges =
                    [&](int_t k0, int chunk_pc,
                        const int_t *ranges_begin,
                        const int_t *ranges_end,
                        std::vector<int_t> &out)
                {
                    if (ranges_begin == ranges_end)
                        return;
                    size_t range_values =
                        static_cast<size_t>(ranges_end - ranges_begin);
                    if ((range_values % 2) != 0)
                        ABORT("SymFact V2 row-down compressed demand has invalid range stride.");
                    size_t range_count = range_values / 2;
                    if (range_count == 0)
                        return;

                    int_t previous_end = 0;
                    bool have_previous = false;
                    for (size_t ri = 0; ri < range_count; ++ri)
                    {
                        int_t start = ranges_begin[2 * ri];
                        int_t len = ranges_begin[2 * ri + 1];
                        if (start < 0 || len <= 0 ||
                            start > std::numeric_limits<int_t>::max() - len)
                            ABORT("SymFact V2 row-down compressed range is invalid.");
                        int_t end_gid = start + len;
                        if (have_previous && start < previous_end)
                            ABORT("SymFact V2 row-down compressed ranges overlap or are unsorted.");
                        previous_end = end_gid;
                        have_previous = true;
                    }

                    int_t lk = symV2PanelIndex(k0);
                    if (lk < 0)
                        return;
                    size_t flat =
                        static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                        static_cast<size_t>(chunk_pc);
                    if (flat >= symL2LSendMeta.size() ||
                        flat >= symV2PartnerLSendSizes.size() ||
                        flat >= symV2PartnerLMapOffsets.size())
                        ABORT("SymFact V2 row-down compressed source metadata is invalid.");
                    const std::vector<int_t> &meta = symL2LSendMeta[flat];
                    size_t map_pos = symV2PartnerLMapOffsets[flat];
                    size_t map_end = map_pos +
                        static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                    if (map_end > symV2PartnerLPackedMaps.size() ||
                        map_end < map_pos)
                        ABORT("SymFact V2 row-down compressed source map range is invalid.");
                    int_t ksupc = SuperSize(k0);

                    auto range_contains_gid = [&](int_t gid) -> bool
                    {
                        size_t lo = 0;
                        size_t hi = range_count;
                        while (lo < hi)
                        {
                            size_t mid = lo + (hi - lo) / 2;
                            if (ranges_begin[2 * mid] <= gid)
                                lo = mid + 1;
                            else
                                hi = mid;
                        }
                        if (lo == 0)
                            return false;
                        size_t ri = lo - 1;
                        int_t start = ranges_begin[2 * ri];
                        int_t len = ranges_begin[2 * ri + 1];
                        return gid >= start && gid < start + len;
                    };

                    size_t meta_pos = 0;
                    while (meta_pos < meta.size())
                    {
                        if (meta_pos + 2 > meta.size())
                            ABORT("SymFact V2 row-down compressed source metadata is truncated.");
                        int_t gid = meta[meta_pos++];
                        int_t len = meta[meta_pos++];
                        if (len < 0 || meta_pos + static_cast<size_t>(len) > meta.size())
                            ABORT("SymFact V2 row-down compressed source metadata block is invalid.");
                        size_t value_count = xlu_checked_product(
                            static_cast<size_t>(len),
                            static_cast<size_t>(ksupc),
                            "SymFact V2 row-down compressed source map segment");
                        if (map_pos + value_count > map_end ||
                            map_pos + value_count < map_pos)
                            ABORT("SymFact V2 row-down compressed source map segment is invalid.");
                        if (range_contains_gid(gid))
                        {
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() + map_pos,
                                       symV2PartnerLPackedMaps.begin() + map_pos + value_count);
                        }
                        map_pos += value_count;
                        meta_pos += static_cast<size_t>(len);
                    }
                    if (map_pos != map_end)
                        ABORT("SymFact V2 row-down compressed source map range size mismatch.");
                };
// SYM_V2_PC2_COMPRESSED_PLAN_RANGE_SOURCE_MAP_END

                auto build_direct_row_down_slot_map =
                    [&](size_t slot,
                        std::vector<SymV2RowDownDirectBlock> &blocks,
                        std::vector<int_t> &out)
                {
                    if (blocks.empty())
                        return;
                    std::sort(blocks.begin(), blocks.end(),
                              [](const SymV2RowDownDirectBlock &a,
                                 const SymV2RowDownDirectBlock &b)
                              {
                                  if (a.gid != b.gid)
                                      return a.gid < b.gid;
                                  return a.chunk_pc < b.chunk_pc;
                              });
                    std::vector<SymV2RowDownDirectBlock> unique_blocks;
                    unique_blocks.reserve(blocks.size());
                    for (size_t bi = 0; bi < blocks.size(); ++bi)
                    {
                        if (!unique_blocks.empty() &&
                            unique_blocks.back().gid == blocks[bi].gid)
                            continue;
                        unique_blocks.push_back(blocks[bi]);
                    }
                    int_t lk = static_cast<int_t>(
                        slot / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 row-down direct panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    if (ksupc <= 0)
                        ABORT("SymFact V2 row-down direct panel width is invalid.");

                    std::vector<std::vector<int_t> > block_maps(
                        unique_blocks.size());
                    std::vector<int_t> block_nrows(
                        unique_blocks.size(), static_cast<int_t>(-1));

                    if (!row_down_compressed_plan)
                    {
                        for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                        {
                            int chunk_pc = unique_blocks[bi].chunk_pc;
                            if (chunk_pc < 0 || chunk_pc >= Pc)
                                ABORT("SymFact V2 row-down direct chunk is invalid.");
                            size_t flat =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(chunk_pc);
                            append_row_down_source_map_for_gid(
                                flat, unique_blocks[bi].gid, block_maps[bi]);
                            if (block_maps[bi].empty() ||
                                block_maps[bi].size() %
                                        static_cast<size_t>(ksupc) !=
                                    0)
                                ABORT("SymFact V2 row-down direct block map has invalid width.");
                            size_t nrows =
                                block_maps[bi].size() /
                                static_cast<size_t>(ksupc);
                            if (nrows >
                                static_cast<size_t>(
                                    std::numeric_limits<int_t>::max()))
                                ABORT("SymFact V2 row-down direct block is too large.");
                            block_nrows[bi] = static_cast<int_t>(nrows);
                        }
                    }
                    else
                    {
// SYM_V2_PC2_COMPRESSED_DIRECT_MAP_BEGIN
                        struct SymV2RowDownDirectRequest
                        {
                            int_t gid;
                            int chunk_pc;
                            size_t block_index;
                        };
                        std::vector<SymV2RowDownDirectRequest> requests;
                        requests.reserve(unique_blocks.size());
                        for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                        {
                            int chunk_pc = unique_blocks[bi].chunk_pc;
                            if (chunk_pc < 0 || chunk_pc >= Pc)
                                ABORT("SymFact V2 row-down direct chunk is invalid.");
                            SymV2RowDownDirectRequest req;
                            req.gid = unique_blocks[bi].gid;
                            req.chunk_pc = chunk_pc;
                            req.block_index = bi;
                            requests.push_back(req);
                        }
                        std::sort(requests.begin(), requests.end(),
                                  [](const SymV2RowDownDirectRequest &a,
                                     const SymV2RowDownDirectRequest &b)
                                  {
                                      if (a.chunk_pc != b.chunk_pc)
                                          return a.chunk_pc < b.chunk_pc;
                                      return a.gid < b.gid;
                                  });

                        size_t group_begin = 0;
                        while (group_begin < requests.size())
                        {
                            int chunk_pc = requests[group_begin].chunk_pc;
                            size_t group_end = group_begin + 1;
                            while (group_end < requests.size() &&
                                   requests[group_end].chunk_pc == chunk_pc)
                                ++group_end;

                            size_t flat =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(chunk_pc);
                            if (flat >= symL2LSendMeta.size() ||
                                flat >= symV2PartnerLSendSizes.size() ||
                                flat >= symV2PartnerLMapOffsets.size())
                                ABORT("SymFact V2 row-down direct source map is invalid.");
                            if (symV2PartnerLSendSizes[flat] < 0)
                                ABORT("SymFact V2 row-down direct source map size is invalid.");
                            const std::vector<int_t> &meta = symL2LSendMeta[flat];
                            size_t map_pos = symV2PartnerLMapOffsets[flat];
                            size_t map_end = map_pos +
                                static_cast<size_t>(
                                    symV2PartnerLSendSizes[flat]);
                            if (map_end > symV2PartnerLPackedMaps.size() ||
                                map_end < map_pos)
                                ABORT("SymFact V2 row-down direct source map bounds are invalid.");

                            size_t meta_pos = 0;
                            while (meta_pos < meta.size())
                            {
                                if (meta_pos + 2 > meta.size())
                                    ABORT("SymFact V2 row-down direct metadata is truncated.");
                                int_t block_gid = meta[meta_pos++];
                                int_t len = meta[meta_pos++];
                                if (len < 0 ||
                                    meta_pos + static_cast<size_t>(len) >
                                        meta.size())
                                    ABORT("SymFact V2 row-down direct metadata block is invalid.");
                                size_t value_count = xlu_checked_product(
                                    static_cast<size_t>(len),
                                    static_cast<size_t>(ksupc),
                                    "SymFact V2 row-down direct source map segment");
                                if (map_pos + value_count > map_end ||
                                    map_pos + value_count < map_pos)
                                    ABORT("SymFact V2 row-down direct source map segment is invalid.");

                                SymV2RowDownDirectRequest key;
                                key.gid = block_gid;
                                key.chunk_pc = chunk_pc;
                                key.block_index = 0;
                                std::vector<SymV2RowDownDirectRequest>::const_iterator it =
                                    std::lower_bound(
                                        requests.begin() + group_begin,
                                        requests.begin() + group_end,
                                        key,
                                        [](const SymV2RowDownDirectRequest &a,
                                           const SymV2RowDownDirectRequest &b)
                                        { return a.gid < b.gid; });
                                if (it != requests.begin() + group_end &&
                                    it->gid == block_gid)
                                {
                                    size_t bi = it->block_index;
                                    if (bi >= block_maps.size() ||
                                        block_nrows[bi] >= 0)
                                        ABORT("SymFact V2 row-down direct duplicate source block.");
                                    block_maps[bi].insert(
                                        block_maps[bi].end(),
                                        symV2PartnerLPackedMaps.begin() + map_pos,
                                        symV2PartnerLPackedMaps.begin() + map_pos + value_count);
                                    block_nrows[bi] = len;
                                }

                                map_pos += value_count;
                                meta_pos += static_cast<size_t>(len);
                            }
                            if (map_pos != map_end)
                                ABORT("SymFact V2 row-down direct source map size mismatch.");
                            group_begin = group_end;
                        }
// SYM_V2_PC2_COMPRESSED_DIRECT_MAP_END
                    }

                    for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                    {
                        if (block_nrows[bi] <= 0 || block_maps[bi].empty() ||
                            static_cast<size_t>(block_nrows[bi]) >
                                std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(ksupc) ||
                            block_maps[bi].size() !=
                                static_cast<size_t>(block_nrows[bi]) *
                                    static_cast<size_t>(ksupc))
                            ABORT("SymFact V2 row-down direct block map has invalid width.");
                    }
                    for (int_t col = 0; col < ksupc; ++col)
                    {
                        for (size_t bi = 0; bi < unique_blocks.size(); ++bi)
                        {
                            size_t src =
                                static_cast<size_t>(col) *
                                static_cast<size_t>(block_nrows[bi]);
                            out.insert(out.end(),
                                       block_maps[bi].begin() + src,
                                       block_maps[bi].begin() + src +
                                           block_nrows[bi]);
                        }
                    }
                };


// SYM_V2_PC2_RANGE_NATIVE_DIRECT_BUILDER_BEGIN
                auto build_direct_row_down_slot_range_map =
                    [&](size_t slot,
                        std::vector<SymV2RowDownDirectRange> &ranges,
                        std::vector<int_t> &out)
                {
                    if (ranges.empty())
                        return;
                    int_t lk = static_cast<int_t>(
                        slot / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 row-down range-direct panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    if (ksupc <= 0)
                        ABORT("SymFact V2 row-down range-direct panel width is invalid.");

                    std::sort(ranges.begin(), ranges.end(),
                              [](const SymV2RowDownDirectRange &a,
                                 const SymV2RowDownDirectRange &b)
                              {
                                  if (a.start != b.start)
                                      return a.start < b.start;
                                  return a.chunk_pc < b.chunk_pc;
                              });

                    std::vector<SymV2RowDownDirectRange> merged_ranges;
                    merged_ranges.reserve(ranges.size());
                    size_t requested_blocks = 0;
                    for (size_t ri = 0; ri < ranges.size(); ++ri)
                    {
                        SymV2RowDownDirectRange cur = ranges[ri];
                        if (cur.chunk_pc < 0 || cur.chunk_pc >= Pc ||
                            cur.start < 0 || cur.len <= 0 ||
                            cur.start > std::numeric_limits<int_t>::max() - cur.len)
                            ABORT("SymFact V2 row-down range-direct request is invalid.");
                        int_t cur_end = cur.start + cur.len;
                        if (!merged_ranges.empty())
                        {
                            SymV2RowDownDirectRange &prev = merged_ranges.back();
                            int_t prev_end = prev.start + prev.len;
                            if (cur.start < prev_end && cur.chunk_pc != prev.chunk_pc)
                                ABORT("SymFact V2 row-down range-direct request overlaps across chunks.");
                            if (cur.chunk_pc == prev.chunk_pc && cur.start <= prev_end)
                            {
                                if (cur_end > prev_end)
                                    prev.len = cur_end - prev.start;
                                continue;
                            }
                        }
                        merged_ranges.push_back(cur);
                    }
                    ranges.swap(merged_ranges);
                    for (size_t ri = 0; ri < ranges.size(); ++ri)
                    {
                        if (static_cast<size_t>(ranges[ri].len) >
                            std::numeric_limits<size_t>::max() - requested_blocks)
                            ABORT("SymFact V2 row-down range-direct request count overflows.");
                        requested_blocks += static_cast<size_t>(ranges[ri].len);
                    }
                    if (requested_blocks == 0)
                        return;

                    std::vector<SymV2RowDownDirectRange> scan_ranges = ranges;
                    std::sort(scan_ranges.begin(), scan_ranges.end(),
                              [](const SymV2RowDownDirectRange &a,
                                 const SymV2RowDownDirectRange &b)
                              {
                                  if (a.chunk_pc != b.chunk_pc)
                                      return a.chunk_pc < b.chunk_pc;
                                  return a.start < b.start;
                              });

                    struct SymV2RowDownDirectSegment
                    {
                        int_t gid;
                        int_t nrows;
                        size_t map_offset;
                    };
                    std::vector<SymV2RowDownDirectSegment> segments;
                    segments.reserve(requested_blocks);

                    size_t group_begin = 0;
                    while (group_begin < scan_ranges.size())
                    {
                        int chunk_pc = scan_ranges[group_begin].chunk_pc;
                        size_t group_end = group_begin + 1;
                        while (group_end < scan_ranges.size() &&
                               scan_ranges[group_end].chunk_pc == chunk_pc)
                            ++group_end;

                        size_t flat =
                            static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                            static_cast<size_t>(chunk_pc);
                        if (flat >= symL2LSendMeta.size() ||
                            flat >= symV2PartnerLSendSizes.size() ||
                            flat >= symV2PartnerLMapOffsets.size())
                            ABORT("SymFact V2 row-down range-direct source map is invalid.");
                        if (symV2PartnerLSendSizes[flat] < 0)
                            ABORT("SymFact V2 row-down range-direct source map size is invalid.");
                        const std::vector<int_t> &meta = symL2LSendMeta[flat];
                        size_t map_pos = symV2PartnerLMapOffsets[flat];
                        size_t map_end = map_pos +
                            static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                        if (map_end > symV2PartnerLPackedMaps.size() ||
                            map_end < map_pos)
                            ABORT("SymFact V2 row-down range-direct source map bounds are invalid.");

                        auto range_group_contains_gid = [&](int_t gid) -> bool
                        {
                            size_t lo = group_begin;
                            size_t hi = group_end;
                            while (lo < hi)
                            {
                                size_t mid = lo + (hi - lo) / 2;
                                if (scan_ranges[mid].start <= gid)
                                    lo = mid + 1;
                                else
                                    hi = mid;
                            }
                            if (lo == group_begin)
                                return false;
                            const SymV2RowDownDirectRange &r = scan_ranges[lo - 1];
                            return gid >= r.start && gid < r.start + r.len;
                        };

                        size_t meta_pos = 0;
                        while (meta_pos < meta.size())
                        {
                            if (meta_pos + 2 > meta.size())
                                ABORT("SymFact V2 row-down range-direct metadata is truncated.");
                            int_t block_gid = meta[meta_pos++];
                            int_t len = meta[meta_pos++];
                            if (len < 0 ||
                                meta_pos + static_cast<size_t>(len) > meta.size())
                                ABORT("SymFact V2 row-down range-direct metadata block is invalid.");
                            size_t value_count = xlu_checked_product(
                                static_cast<size_t>(len),
                                static_cast<size_t>(ksupc),
                                "SymFact V2 row-down range-direct source map segment");
                            if (map_pos + value_count > map_end ||
                                map_pos + value_count < map_pos)
                                ABORT("SymFact V2 row-down range-direct source map segment is invalid.");
                            if (range_group_contains_gid(block_gid))
                            {
                                SymV2RowDownDirectSegment seg;
                                seg.gid = block_gid;
                                seg.nrows = len;
                                seg.map_offset = map_pos;
                                segments.push_back(seg);
                            }
                            map_pos += value_count;
                            meta_pos += static_cast<size_t>(len);
                        }
                        if (map_pos != map_end)
                            ABORT("SymFact V2 row-down range-direct source map size mismatch.");
                        group_begin = group_end;
                    }

                    if (segments.size() != requested_blocks)
                        ABORT("SymFact V2 row-down range-direct did not find all requested blocks.");
                    std::sort(segments.begin(), segments.end(),
                              [](const SymV2RowDownDirectSegment &a,
                                 const SymV2RowDownDirectSegment &b)
                              { return a.gid < b.gid; });
                    for (size_t si = 1; si < segments.size(); ++si)
                    {
                        if (segments[si - 1].gid == segments[si].gid)
                            ABORT("SymFact V2 row-down range-direct duplicate source block.");
                    }

                    size_t append_count = 0;
                    for (size_t si = 0; si < segments.size(); ++si)
                    {
                        if (segments[si].nrows <= 0 ||
                            static_cast<size_t>(segments[si].nrows) >
                                std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(ksupc))
                            ABORT("SymFact V2 row-down range-direct segment has invalid width.");
                        size_t block_count = static_cast<size_t>(segments[si].nrows) *
                                             static_cast<size_t>(ksupc);
                        if (segments[si].map_offset + block_count >
                                symV2PartnerLPackedMaps.size() ||
                            segments[si].map_offset + block_count <
                                segments[si].map_offset ||
                            append_count > std::numeric_limits<size_t>::max() - block_count)
                            ABORT("SymFact V2 row-down range-direct segment map is invalid.");
                        append_count += block_count;
                    }
                    out.reserve(out.size() + append_count);
                    for (int_t col = 0; col < ksupc; ++col)
                    {
                        for (size_t si = 0; si < segments.size(); ++si)
                        {
                            size_t src = segments[si].map_offset +
                                         static_cast<size_t>(col) *
                                             static_cast<size_t>(segments[si].nrows);
                            out.insert(out.end(),
                                       symV2PartnerLPackedMaps.begin() + src,
                                       symV2PartnerLPackedMaps.begin() + src +
                                           segments[si].nrows);
                        }
                    }
                };
// SYM_V2_PC2_RANGE_NATIVE_DIRECT_BUILDER_END

// SYM_V2_PC2_LAZY_SENDMAP_SEGMENT_BUILDER_BEGIN
                auto build_direct_row_down_slot_range_segments =
                    [&](size_t slot,
                        std::vector<SymV2RowDownDirectRange> &ranges,
                        std::vector<SymV2RowDownSendSegmentGPU> &out,
                        int &value_count_out)
                {
                    value_count_out = 0;
                    if (ranges.empty())
                        return;
                    int_t lk = static_cast<int_t>(
                        slot / static_cast<size_t>(Pc));
                    int_t k0 = symV2PanelGid(lk);
                    if (k0 < 0 || k0 >= nsupers)
                        ABORT("SymFact V2 row-down lazy-sendmap panel is invalid.");
                    int_t ksupc = SuperSize(k0);
                    if (ksupc <= 0)
                        ABORT("SymFact V2 row-down lazy-sendmap panel width is invalid.");

                    std::sort(ranges.begin(), ranges.end(),
                              [](const SymV2RowDownDirectRange &a,
                                 const SymV2RowDownDirectRange &b)
                              {
                                  if (a.start != b.start)
                                      return a.start < b.start;
                                  return a.chunk_pc < b.chunk_pc;
                              });

                    std::vector<SymV2RowDownDirectRange> merged_ranges;
                    merged_ranges.reserve(ranges.size());
                    size_t requested_blocks = 0;
                    for (size_t ri = 0; ri < ranges.size(); ++ri)
                    {
                        SymV2RowDownDirectRange cur = ranges[ri];
                        if (cur.chunk_pc < 0 || cur.chunk_pc >= Pc ||
                            cur.start < 0 || cur.len <= 0 ||
                            cur.start > std::numeric_limits<int_t>::max() - cur.len)
                            ABORT("SymFact V2 row-down lazy-sendmap request is invalid.");
                        int_t cur_end = cur.start + cur.len;
                        if (!merged_ranges.empty())
                        {
                            SymV2RowDownDirectRange &prev = merged_ranges.back();
                            int_t prev_end = prev.start + prev.len;
                            if (cur.start < prev_end && cur.chunk_pc != prev.chunk_pc)
                                ABORT("SymFact V2 row-down lazy-sendmap request overlaps across chunks.");
                            if (cur.chunk_pc == prev.chunk_pc && cur.start <= prev_end)
                            {
                                if (cur_end > prev_end)
                                    prev.len = cur_end - prev.start;
                                continue;
                            }
                        }
                        merged_ranges.push_back(cur);
                    }
                    ranges.swap(merged_ranges);
                    for (size_t ri = 0; ri < ranges.size(); ++ri)
                    {
                        if (static_cast<size_t>(ranges[ri].len) >
                            std::numeric_limits<size_t>::max() - requested_blocks)
                            ABORT("SymFact V2 row-down lazy-sendmap request count overflows.");
                        requested_blocks += static_cast<size_t>(ranges[ri].len);
                    }
                    if (requested_blocks == 0)
                        return;

                    std::vector<SymV2RowDownDirectRange> scan_ranges = ranges;
                    std::sort(scan_ranges.begin(), scan_ranges.end(),
                              [](const SymV2RowDownDirectRange &a,
                                 const SymV2RowDownDirectRange &b)
                              {
                                  if (a.chunk_pc != b.chunk_pc)
                                      return a.chunk_pc < b.chunk_pc;
                                  return a.start < b.start;
                              });

                    struct SymV2RowDownDirectSegment
                    {
                        int_t gid;
                        int_t nrows;
                        size_t map_offset;
                    };
                    std::vector<SymV2RowDownDirectSegment> segments;
                    segments.reserve(requested_blocks);

                    size_t group_begin = 0;
                    while (group_begin < scan_ranges.size())
                    {
                        int chunk_pc = scan_ranges[group_begin].chunk_pc;
                        size_t group_end = group_begin + 1;
                        while (group_end < scan_ranges.size() &&
                               scan_ranges[group_end].chunk_pc == chunk_pc)
                            ++group_end;

                        size_t flat =
                            static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                            static_cast<size_t>(chunk_pc);
                        if (flat >= symL2LSendMeta.size() ||
                            flat >= symV2PartnerLSendSizes.size() ||
                            flat >= symV2PartnerLMapOffsets.size())
                            ABORT("SymFact V2 row-down lazy-sendmap source map is invalid.");
                        if (symV2PartnerLSendSizes[flat] < 0)
                            ABORT("SymFact V2 row-down lazy-sendmap source map size is invalid.");
                        const std::vector<int_t> &meta = symL2LSendMeta[flat];
                        size_t map_pos = symV2PartnerLMapOffsets[flat];
                        size_t map_end = map_pos +
                            static_cast<size_t>(symV2PartnerLSendSizes[flat]);
                        if (map_end > symV2PartnerLPackedMaps.size() ||
                            map_end < map_pos)
                            ABORT("SymFact V2 row-down lazy-sendmap source map bounds are invalid.");

                        auto range_group_contains_gid = [&](int_t gid) -> bool
                        {
                            size_t lo = group_begin;
                            size_t hi = group_end;
                            while (lo < hi)
                            {
                                size_t mid = lo + (hi - lo) / 2;
                                if (scan_ranges[mid].start <= gid)
                                    lo = mid + 1;
                                else
                                    hi = mid;
                            }
                            if (lo == group_begin)
                                return false;
                            const SymV2RowDownDirectRange &r = scan_ranges[lo - 1];
                            return gid >= r.start && gid < r.start + r.len;
                        };

                        size_t meta_pos = 0;
                        while (meta_pos < meta.size())
                        {
                            if (meta_pos + 2 > meta.size())
                                ABORT("SymFact V2 row-down lazy-sendmap metadata is truncated.");
                            int_t block_gid = meta[meta_pos++];
                            int_t len = meta[meta_pos++];
                            if (len < 0 ||
                                meta_pos + static_cast<size_t>(len) > meta.size())
                                ABORT("SymFact V2 row-down lazy-sendmap metadata block is invalid.");
                            size_t value_count = xlu_checked_product(
                                static_cast<size_t>(len),
                                static_cast<size_t>(ksupc),
                                "SymFact V2 row-down lazy-sendmap source map segment");
                            if (map_pos + value_count > map_end ||
                                map_pos + value_count < map_pos)
                                ABORT("SymFact V2 row-down lazy-sendmap source map segment is invalid.");
                            if (range_group_contains_gid(block_gid))
                            {
                                SymV2RowDownDirectSegment seg;
                                seg.gid = block_gid;
                                seg.nrows = len;
                                seg.map_offset = map_pos;
                                segments.push_back(seg);
                            }
                            map_pos += value_count;
                            meta_pos += static_cast<size_t>(len);
                        }
                        if (map_pos != map_end)
                            ABORT("SymFact V2 row-down lazy-sendmap source map size mismatch.");
                        group_begin = group_end;
                    }

                    if (segments.size() != requested_blocks)
                        ABORT("SymFact V2 row-down lazy-sendmap did not find all requested blocks.");
                    std::sort(segments.begin(), segments.end(),
                              [](const SymV2RowDownDirectSegment &a,
                                 const SymV2RowDownDirectSegment &b)
                              { return a.gid < b.gid; });
                    for (size_t si = 1; si < segments.size(); ++si)
                    {
                        if (segments[si - 1].gid == segments[si].gid)
                            ABORT("SymFact V2 row-down lazy-sendmap duplicate source block.");
                    }

                    size_t row_count = 0;
                    out.reserve(out.size() + segments.size());
                    for (size_t si = 0; si < segments.size(); ++si)
                    {
                        if (segments[si].nrows <= 0 ||
                            static_cast<size_t>(segments[si].nrows) >
                                std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(ksupc))
                            ABORT("SymFact V2 row-down lazy-sendmap segment has invalid width.");
                        size_t block_values =
                            static_cast<size_t>(segments[si].nrows) *
                            static_cast<size_t>(ksupc);
                        if (segments[si].map_offset + block_values >
                                symV2PartnerLPackedMaps.size() ||
                            segments[si].map_offset + block_values <
                                segments[si].map_offset)
                            ABORT("SymFact V2 row-down lazy-sendmap segment map is invalid.");
                        if (row_count >
                            std::numeric_limits<size_t>::max() -
                                static_cast<size_t>(segments[si].nrows))
                            ABORT("SymFact V2 row-down lazy-sendmap row count overflows.");

                        SymV2RowDownSendSegmentGPU gpu_seg;
                        gpu_seg.map_offset = segments[si].map_offset;
                        gpu_seg.nrows = segments[si].nrows;
                        if (row_count >
                            static_cast<size_t>(std::numeric_limits<int_t>::max()))
                            ABORT("SymFact V2 row-down lazy-sendmap destination offset is too large.");
                        gpu_seg.dst_row_offset = static_cast<int_t>(row_count);
                        out.push_back(gpu_seg);
                        row_count += static_cast<size_t>(segments[si].nrows);
                    }
                    size_t value_count =
                        xlu_checked_product(row_count,
                                            static_cast<size_t>(ksupc),
                                            "SymFact V2 row-down lazy-sendmap value count");
                    if (value_count >
                        static_cast<size_t>(std::numeric_limits<int>::max()))
                        ABORT("SymFact V2 row-down lazy-sendmap value count is too large.");
                    value_count_out = static_cast<int>(value_count);
                };
// SYM_V2_PC2_LAZY_SENDMAP_SEGMENT_BUILDER_END

                for (int sender_pc = 0; sender_pc < Pc; ++sender_pc)
                {
                    size_t pos = static_cast<size_t>(recv_displs[static_cast<size_t>(sender_pc)]);
                    size_t end = pos + static_cast<size_t>(recv_counts[static_cast<size_t>(sender_pc)]);
                    while (pos < end)
                    {
                        if (pos + 4 > end)
                            ABORT("SymFact V2 row-down demand record is truncated.");
                        int_t k0 = recv_payload[pos++];
                        int dest_pc = static_cast<int>(recv_payload[pos++]);
                        int chunk_pc = static_cast<int>(recv_payload[pos++]);
// SYM_V2_PC2_COMPRESSED_PLAN_DEMAND_DECODE_BEGIN
                        int_t encoded_count = recv_payload[pos++];
                        if (encoded_count == 0 ||
                            encoded_count == std::numeric_limits<int_t>::min())
                            ABORT("SymFact V2 row-down demand record is invalid.");
                        const bool range_encoded = encoded_count < 0;
                        int_t encoded_items = range_encoded
                            ? -encoded_count
                            : encoded_count;
                        if (encoded_items <= 0)
                            ABORT("SymFact V2 row-down demand record is invalid.");
                        size_t payload_items =
                            static_cast<size_t>(encoded_items);
                        if (range_encoded)
                        {
                            if (payload_items >
                                std::numeric_limits<size_t>::max() / 2)
                                ABORT("SymFact V2 row-down compressed demand overflows.");
                            payload_items *= 2;
                        }
                        if (k0 < 0 || k0 >= nsupers || dest_pc < 0 || dest_pc >= Pc ||
                            chunk_pc < 0 || chunk_pc >= Pc || payload_items > end - pos)
                            ABORT("SymFact V2 row-down demand record is invalid.");
                        if (dest_pc != sender_pc || symV2PanelRoot(k0) != mycol)
                            ABORT("SymFact V2 row-down demand arrived at wrong source column.");
                        const size_t request_pos = pos;
                        const size_t request_end = pos + payload_items;
                        if (range_encoded)
                        {
                            int_t previous_end = 0;
                            bool have_previous = false;
                            for (size_t rp = request_pos; rp < request_end; rp += 2)
                            {
                                int_t start = recv_payload[rp];
                                int_t len = recv_payload[rp + 1];
                                if (start < 0 || len <= 0 ||
                                    start > std::numeric_limits<int_t>::max() - len)
                                    ABORT("SymFact V2 row-down compressed demand range is invalid.");
                                int_t end_gid = start + len;
                                if (have_previous && start < previous_end)
                                    ABORT("SymFact V2 row-down compressed demand ranges overlap or are unsorted.");
                                previous_end = end_gid;
                                have_previous = true;
                            }
                        }
                        pos = request_end;
                        int_t lk = symV2PanelIndex(k0);
                        if (lk < 0)
                            continue;
                        size_t slot =
                            static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                            static_cast<size_t>(dest_pc);
                        if (slot >= slot_maps.size())
                            ABORT("SymFact V2 row-down demand slot is invalid.");
                        if (row_down_direct_recv)
                        {
                            if (row_down_compressed_plan)
                            {
                                if (slot >= slot_direct_ranges.size())
                                    ABORT("SymFact V2 row-down range-direct slot is invalid.");
                                std::vector<SymV2RowDownDirectRange> &ranges =
                                    slot_direct_ranges[slot];
                                auto append_direct_range =
                                    [&](int_t start, int_t len)
                                {
                                    if (start < 0 || len <= 0 ||
                                        start > std::numeric_limits<int_t>::max() - len)
                                        ABORT("SymFact V2 row-down range-direct demand is invalid.");
                                    SymV2RowDownDirectRange range;
                                    range.start = start;
                                    range.len = len;
                                    range.chunk_pc = chunk_pc;
                                    ranges.push_back(range);
                                };
                                if (range_encoded)
                                {
                                    for (size_t rp = request_pos; rp < request_end; rp += 2)
                                        append_direct_range(recv_payload[rp],
                                                            recv_payload[rp + 1]);
                                }
                                else
                                {
                                    int_t run_start = recv_payload[request_pos];
                                    int_t prev = run_start;
                                    if (run_start < 0)
                                        ABORT("SymFact V2 row-down range-direct demand has invalid gid.");
                                    for (size_t ri = request_pos + 1; ri < request_end; ++ri)
                                    {
                                        int_t gid = recv_payload[ri];
                                        if (gid <= prev)
                                            ABORT("SymFact V2 row-down range-direct explicit demand is not sorted unique.");
                                        if (prev < std::numeric_limits<int_t>::max() &&
                                            gid == prev + 1)
                                        {
                                            prev = gid;
                                            continue;
                                        }
                                        append_direct_range(run_start,
                                                            prev - run_start + 1);
                                        run_start = gid;
                                        prev = gid;
                                    }
                                    append_direct_range(run_start,
                                                        prev - run_start + 1);
                                }
                            }
                            else
                            {
                                if (slot >= slot_direct_blocks.size())
                                    ABORT("SymFact V2 row-down direct slot is invalid.");
                                std::vector<SymV2RowDownDirectBlock> &blocks =
                                    slot_direct_blocks[slot];
                                if (range_encoded)
                                {
                                    for (size_t rp = request_pos; rp < request_end; rp += 2)
                                    {
                                        int_t start = recv_payload[rp];
                                        int_t len = recv_payload[rp + 1];
                                        for (int_t off = 0; off < len; ++off)
                                        {
                                            SymV2RowDownDirectBlock block;
                                            block.gid = start + off;
                                            block.chunk_pc = chunk_pc;
                                            blocks.push_back(block);
                                        }
                                    }
                                }
                                else
                                {
                                    for (size_t ri = request_pos; ri < request_end; ++ri)
                                    {
                                        SymV2RowDownDirectBlock block;
                                        block.gid = recv_payload[ri];
                                        block.chunk_pc = chunk_pc;
                                        blocks.push_back(block);
                                    }
                                }
                            }
                        }
                        else if (row_down_compact)
                        {
                            if (range_encoded)
                                append_row_down_map_for_ranges(
                                    k0, chunk_pc,
                                    recv_payload.data() + request_pos,
                                    recv_payload.data() + request_end,
                                    slot_maps[slot]);
                            else
                                append_row_down_map_for_record(
                                    k0, chunk_pc, recv_payload.data() + request_pos,
                                    recv_payload.data() + request_end, slot_maps[slot]);
                        }
                        else
                        {
                            if (range_encoded)
                            {
                                append_row_down_map_for_ranges(
                                    k0, chunk_pc,
                                    recv_payload.data() + request_pos,
                                    recv_payload.data() + request_end,
                                    slot_maps[slot]);
                            }
                            else
                            {
                                std::vector<int_t> requested(
                                    recv_payload.begin() + request_pos,
                                    recv_payload.begin() + request_end);
                                std::sort(requested.begin(), requested.end());
                                requested.erase(
                                    std::unique(requested.begin(),
                                                requested.end()),
                                    requested.end());
                                append_row_down_map_for_record(
                                    k0, chunk_pc, requested.data(),
                                    requested.data() + requested.size(),
                                    slot_maps[slot]);
                            }
                        }
// SYM_V2_PC2_COMPRESSED_PLAN_DEMAND_DECODE_END
                    }
                }
                if (row_down_direct_recv)
                {
                    if (row_down_compressed_plan)
                    {
// SYM_V2_PC2_LAZY_SENDMAP_BUILD_BRANCH_BEGIN
                        if (row_down_lazy_sendmap)
                        {
                            for (size_t slot = 0;
                                 slot < slot_direct_ranges.size(); ++slot)
                            {
                                if (slot >= slot_send_segments.size() ||
                                    slot >= slot_send_segment_values.size())
                                    ABORT("SymFact V2 row-down lazy-sendmap slot table is invalid.");
                                build_direct_row_down_slot_range_segments(
                                    slot, slot_direct_ranges[slot],
                                    slot_send_segments[slot],
                                    slot_send_segment_values[slot]);
                            }
                        }
                        else
                        {
// SYM_V2_PC2_LAZY_SENDMAP_BUILD_BRANCH_END
                            for (size_t slot = 0; slot < slot_direct_ranges.size();
                                 ++slot)
                                build_direct_row_down_slot_range_map(
                                    slot, slot_direct_ranges[slot], slot_maps[slot]);
                        }
                    }
                    else
                    {
                        for (size_t slot = 0; slot < slot_direct_blocks.size();
                             ++slot)
                            build_direct_row_down_slot_map(
                                slot, slot_direct_blocks[slot], slot_maps[slot]);
                    }
                }

                symV2RowDownSendMapsHost.clear();
// SYM_V2_PC2_LAZY_SENDMAP_STORE_BEGIN
                symV2RowDownSendSegsHost.clear();
                symV2RowDownSparseSendValues = 0;
                symV2RowDownSendMessages = 0;
                std::vector<std::vector<int_t> > size_reply_payloads(static_cast<size_t>(Pc));
                for (size_t slot = 0; slot < slot_maps.size(); ++slot)
                {
                    int slot_send_size = 0;
                    if (row_down_lazy_sendmap)
                    {
                        if (slot >= slot_send_segment_values.size() ||
                            slot >= slot_send_segments.size() ||
                            slot >= symV2RowDownSendSegOffsets.size() ||
                            slot >= symV2RowDownSendSegCounts.size())
                            ABORT("SymFact V2 row-down lazy-sendmap slot metadata is invalid.");
                        symV2RowDownSendMapOffsets[slot] = 0;
                        symV2RowDownSendSegOffsets[slot] =
                            symV2RowDownSendSegsHost.size();
                        if (slot_send_segments[slot].size() >
                            static_cast<size_t>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 row-down lazy-sendmap has too many segments.");
                        symV2RowDownSendSegCounts[slot] =
                            static_cast<int>(slot_send_segments[slot].size());
                        slot_send_size = slot_send_segment_values[slot];
                        symV2RowDownSendSegsHost.insert(
                            symV2RowDownSendSegsHost.end(),
                            slot_send_segments[slot].begin(),
                            slot_send_segments[slot].end());
                    }
                    else
                    {
                        symV2RowDownSendMapOffsets[slot] =
                            symV2RowDownSendMapsHost.size();
                        if (slot_maps[slot].size() >
                            static_cast<size_t>(std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 row-down send map is too large for MPI.");
                        slot_send_size =
                            static_cast<int>(slot_maps[slot].size());
                        symV2RowDownSendMapsHost.insert(
                            symV2RowDownSendMapsHost.end(),
                            slot_maps[slot].begin(), slot_maps[slot].end());
                    }
                    if (slot_send_size < 0)
                        ABORT("SymFact V2 row-down send size is invalid.");
                    symV2RowDownSendSizes[slot] = slot_send_size;
                    symV2RowDownSparseSendValues +=
                        static_cast<long long>(slot_send_size);
                    if (slot_send_size > 0)
                    {
                        ++symV2RowDownSendMessages;
                        int_t lk = static_cast<int_t>(slot / static_cast<size_t>(Pc));
                        int dest_pc = static_cast<int>(slot % static_cast<size_t>(Pc));
                        int_t k0 = symV2PanelGid(lk);
                        size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(k0);
                        size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(
                            static_cast<int_t>(slot_send_size));
                    }
                }
// SYM_V2_PC2_LAZY_SENDMAP_STORE_END
                if (row_down_compact && profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_COMPACT_SEND_MAP_BUILD,
                        SuperLU_timer_() - tRowCompactSendMapBuild);

                const double tRowCompactSizeCheck = SuperLU_timer_();
                std::vector<int> size_send_counts(static_cast<size_t>(Pc), 0);
                std::vector<int> size_recv_counts(static_cast<size_t>(Pc), 0);
                std::vector<int> size_send_displs(static_cast<size_t>(Pc), 0);
                std::vector<int> size_recv_displs(static_cast<size_t>(Pc), 0);
                int total_size_send = 0;
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_send_displs[static_cast<size_t>(pc)] = total_size_send;
                    size_send_counts[static_cast<size_t>(pc)] = static_cast<int>(
                        size_reply_payloads[static_cast<size_t>(pc)].size());
                    total_size_send += size_send_counts[static_cast<size_t>(pc)];
                }
                std::vector<int_t> size_send_payload(static_cast<size_t>(total_size_send));
                for (int pc = 0; pc < Pc; ++pc)
                    std::copy(size_reply_payloads[static_cast<size_t>(pc)].begin(),
                              size_reply_payloads[static_cast<size_t>(pc)].end(),
                              size_send_payload.begin() + size_send_displs[static_cast<size_t>(pc)]);
                MPI_Alltoall(size_send_counts.data(), 1, MPI_INT,
                             size_recv_counts.data(), 1, MPI_INT,
                             grid3d->rscp.comm);
                int total_size_recv = 0;
                for (int pc = 0; pc < Pc; ++pc)
                {
                    size_recv_displs[static_cast<size_t>(pc)] = total_size_recv;
                    total_size_recv += size_recv_counts[static_cast<size_t>(pc)];
                }
                std::vector<int_t> size_recv_payload(static_cast<size_t>(total_size_recv));
                MPI_Alltoallv(size_send_payload.empty() ? NULL : size_send_payload.data(),
                              size_send_counts.data(), size_send_displs.data(), mpi_int_t,
                              size_recv_payload.empty() ? NULL : size_recv_payload.data(),
                              size_recv_counts.data(), size_recv_displs.data(), mpi_int_t,
                              grid3d->rscp.comm);
                for (int source_pc = 0; source_pc < Pc; ++source_pc)
                {
                    size_t pos = static_cast<size_t>(size_recv_displs[static_cast<size_t>(source_pc)]);
                    size_t end = pos + static_cast<size_t>(size_recv_counts[static_cast<size_t>(source_pc)]);
                    while (pos < end)
                    {
                        if (pos + 2 > end)
                            ABORT("SymFact V2 row-down size reply is truncated.");
                        int_t k0 = size_recv_payload[pos++];
                        int_t send_size = size_recv_payload[pos++];
                        if (k0 < 0 || k0 >= nsupers || send_size < 0)
                            ABORT("SymFact V2 row-down size reply is invalid.");
                        long long expected = 0;
                        for (int chunk_pc = 0; chunk_pc < Pc; ++chunk_pc)
                            expected += symV2RowFragRecvSizes[
                                static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                                static_cast<size_t>(chunk_pc)];
                        if (source_pc == symV2PanelRoot(k0) &&
                            expected != static_cast<long long>(send_size))
                            ABORT("SymFact V2 row-down send/receive size mismatch.");
                    }
                }
                if (row_down_compact && profile_setup)
                    symV2SetupProfileAdd(
                        SYM_V2_SETUP_ROW_COMPACT_SIZE_CHECK,
                        SuperLU_timer_() - tRowCompactSizeCheck);

                symV2RowDownCurrentRecvValues = local_current_recv_values;
                symV2RowDownDemandRecords = local_demand_records;
                for (int_t k0 = 0; k0 < nsupers; ++k0)
                    if (!needed_row_blocks_by_panel[k0].empty())
                        symV2RowDownPlanReady[static_cast<size_t>(k0)] = 1;
                symV2RowDownSetupSeconds = SuperLU_timer_() - row_down_setup_t;
                if (!row_down_dryrun &&
                    superlu_sym_v2_row_l_plan_v2_exchange())
                {
                    long long exact_max_row_stage = 0;
                    long long exact_max_row_values = 0;
                    long long exact_max_row_index = LPANEL_HEADER_SIZE;
                    long long exact_max_row_send = 0;
                    for (int_t k0 = 0; k0 < nsupers; ++k0)
                    {
                        long long stage_values = 0;
                        size_t row_recv_base =
                            static_cast<size_t>(k0) *
                            static_cast<size_t>(Pc);
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t pos = row_recv_base +
                                         static_cast<size_t>(pc);
                            if (pos < symV2RowFragRecvSizes.size())
                                stage_values +=
                                    static_cast<long long>(
                                        symV2RowFragRecvSizes[pos]);
                        }
                        exact_max_row_stage =
                            SUPERLU_MAX(exact_max_row_stage, stage_values);
                        if (static_cast<size_t>(k0) <
                                symV2RowFragRecvIndex.size() &&
                            !symV2RowFragRecvIndex[k0].empty())
                        {
                            const std::vector<int_t> &row_index =
                                symV2RowFragRecvIndex[k0];
                            long long row_values =
                                static_cast<long long>(row_index[1]) *
                                static_cast<long long>(SuperSize(k0));
                            exact_max_row_values =
                                SUPERLU_MAX(exact_max_row_values, row_values);
                            exact_max_row_index = SUPERLU_MAX(
                                exact_max_row_index,
                                static_cast<long long>(row_index.size()));
                        }
                    }
                    for (int_t lk = 0; lk < local_cols; ++lk)
                    {
                        long long send_values = 0;
                        for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
                        {
                            size_t slot =
                                static_cast<size_t>(lk) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc_dest);
                            long long dest_values = 0;
                            if (slot < symV2RowDownSendSizes.size())
                                dest_values = static_cast<long long>(
                                    symV2RowDownSendSizes[slot]);
                            exact_max_row_send =
                                SUPERLU_MAX(exact_max_row_send,
                                            dest_values);
                            if (pc_dest != mycol)
                                send_values += dest_values;
                        }
                        exact_max_row_send =
                            SUPERLU_MAX(exact_max_row_send, send_values);
                    }
                    if (exact_max_row_stage >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_values >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_index >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()) ||
                        exact_max_row_send >
                            static_cast<long long>(
                                std::numeric_limits<int_t>::max()))
                        ABORT("SymFact V2 exact row-down staging size exceeds int_t range.");
                    maxSymV2RowFragStageCount =
                        static_cast<int_t>(exact_max_row_stage);
                    maxSymV2RowFragValRecvCount =
                        static_cast<int_t>(exact_max_row_values);
                    maxSymV2RowFragIdxRecvCount =
                        static_cast<int_t>(exact_max_row_index);
                    maxSymV2RowFragValSendCount =
                        static_cast<int_t>(exact_max_row_send);
                }
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_CURRENT_RECV_VALUES,
                    symV2RowDownCurrentRecvValues);
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_SPARSE_SEND_VALUES,
                    symV2RowDownSparseSendValues);
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_SPARSE_RECV_VALUES,
                    symV2RowDownSparseRecvValues);
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_SAVED_RECV_VALUES,
                    SUPERLU_MAX((long long)0,
                                symV2RowDownCurrentRecvValues -
                                    symV2RowDownSparseRecvValues));
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_DEMAND_RECORDS,
                    symV2RowDownDemandRecords);
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_SEND_MESSAGES,
                    symV2RowDownSendMessages);
                symV2ProfileScalarSet(
                    SYM_V2_PROFILE_ROW_RECV_MESSAGES,
                    symV2RowDownRecvMessages);

                if (superlu_sym_v2_row_l_plan_v2_verify())
                {
                    long long local_stats[5] = {
                        symV2RowDownSparseSendValues,
                        symV2RowDownSparseRecvValues,
                        symV2RowDownCurrentRecvValues,
                        symV2RowDownDemandRecords,
                        symV2RowDownSendMessages + symV2RowDownRecvMessages
                    };
                    long long global_stats[5] = {0, 0, 0, 0, 0};
                    double global_setup = 0.0;
                    MPI_Reduce(local_stats, global_stats, 5, MPI_LONG_LONG,
                               MPI_SUM, 0, grid->comm);
                    MPI_Reduce(&symV2RowDownSetupSeconds, &global_setup, 1,
                               MPI_DOUBLE, MPI_MAX, 0, grid->comm);
                    if (grid3d->iam == 0)
                    {
                        std::printf("SymFact V2 row-down plan v2: sparse_send_values=%lld sparse_recv_values=%lld current_recv_values=%lld demand_records=%lld row_messages=%lld setup_max=%g communicator=rscp ordering=k/chunk/source-meta dryrun=%d\n",
                                    global_stats[0], global_stats[1], global_stats[2],
                                    global_stats[3], global_stats[4], global_setup,
                                    superlu_sym_v2_row_l_plan_v2_dryrun() ? 1 : 0);
                        std::fflush(stdout);
                    }
                }
                if (row_down_dryrun)
                {
                    symV2RowFragRecvSizes.swap(saved_row_frag_recv_sizes);
                    symV2RowFragRecvIndex.swap(saved_row_frag_recv_index);
                    symV2RowFragRecvMap.swap(saved_row_frag_recv_map);
                }
            }
// SYM_V2_PC2_PHASE3_SPARSE_ROW_DOWN_PLAN_END
// SYM_V2_PC2_PHASE4_ALLOC_ROW_DOWN_GPU_MAPS_BEGIN
                if (superlu_sym_v2_row_l_plan_v2_exchange())
                {
                    if (superlu_sym_v2_row_l_plan_v2_dryrun())
                        ABORT("GPU3DV2_ROW_L_PLAN_V2_EXCHANGE requires GPU3DV2_ROW_L_PLAN_V2_DRYRUN=0.");
// SYM_V2_PC2_LAZY_SENDMAP_ALLOC_BEGIN
                    const bool row_down_lazy_sendmap_alloc =
                        superlu_sym_v2_pc_fragment_ldl_native() &&
                        superlu_sym_v2_row_l_direct_recv() &&
                        superlu_sym_v2_row_l_compressed_plan() &&
                        superlu_sym_v2_row_l_lazy_sendmap();
                    symV2RowDownSendMapPoolCount =
                        row_down_lazy_sendmap_alloc
                            ? 0
                            : symV2RowDownSendMapsHost.size();
                    symV2RowDownSendSegPoolCount =
                        row_down_lazy_sendmap_alloc
                            ? symV2RowDownSendSegsHost.size()
                            : 0;
                    symV2RowDownSendMapsGPU.assign(l2u_slots, NULL);
                    symV2RowDownSendSegsGPU.assign(l2u_slots, NULL);
                    for (size_t slot = 0; slot < symV2RowDownSendSizes.size(); ++slot)
                    {
                        if (symV2RowDownSendSizes[slot] <= 0)
                            continue;
                        if (row_down_lazy_sendmap_alloc)
                        {
                            if (slot >= symV2RowDownSendSegOffsets.size() ||
                                slot >= symV2RowDownSendSegCounts.size())
                                ABORT("SymFact V2 row-down lazy-sendmap offset table is invalid.");
                            size_t off = symV2RowDownSendSegOffsets[slot];
                            int count = symV2RowDownSendSegCounts[slot];
                            if (count <= 0)
                                ABORT("SymFact V2 row-down lazy-sendmap slot has no segments.");
                            if (off + static_cast<size_t>(count) >
                                    symV2RowDownSendSegPoolCount ||
                                off + static_cast<size_t>(count) < off)
                                ABORT("SymFact V2 row-down lazy-sendmap segment offset is invalid.");
                        }
                        else
                        {
                            size_t off = symV2RowDownSendMapOffsets[slot];
                            if (off + static_cast<size_t>(symV2RowDownSendSizes[slot]) >
                                    symV2RowDownSendMapPoolCount ||
                                off + static_cast<size_t>(symV2RowDownSendSizes[slot]) < off)
                                ABORT("SymFact V2 row-down send map offset is invalid.");
                        }
                    }
// SYM_V2_PC2_LAZY_SENDMAP_ALLOC_END
                }
// SYM_V2_PC2_PHASE4_ALLOC_ROW_DOWN_GPU_MAPS_END

            if (pc_fragment_schur_setup &&
                superlu_sym_v2_row_hybrid_cost() &&
                !superlu_sym_v2_row_l_postsolve_send())
            {
// SYM_V2_PC2_PHASE5_COST_HYBRID_SELECTOR_BEGIN
                const double margin = superlu_sym_v2_row_hybrid_margin();
                const double lat_s = superlu_sym_v2_cost_lat_us() * 1.0e-6;
                const double net_Bps = superlu_sym_v2_cost_net_gbps() * 1.0e9;
                const double pcie_Bps = superlu_sym_v2_cost_pcie_gbps() * 1.0e9;
                const double pack_Bps = superlu_sym_v2_cost_pack_gbps() * 1.0e9;
                const double asm_Bps = superlu_sym_v2_cost_asm_gbps() * 1.0e9;
                const double kernel_s = superlu_sym_v2_cost_kernel_us() * 1.0e-6;
                const double setup_weight = superlu_sym_v2_cost_setup_weight();

                std::vector<long long> local_frag_values(static_cast<size_t>(nsupers), 0);
                std::vector<long long> local_frag_messages(static_cast<size_t>(nsupers), 0);
                std::vector<long long> global_frag_values(static_cast<size_t>(nsupers), 0);
                std::vector<long long> global_frag_messages(static_cast<size_t>(nsupers), 0);
                const std::vector<int> &frag_sizes =
                    (superlu_sym_v2_row_l_plan_v2() && !symV2RowDownRecvSizes.empty())
                        ? symV2RowDownRecvSizes
                        : symV2RowFragRecvSizes;
                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    size_t row_recv_base = static_cast<size_t>(k0) * static_cast<size_t>(Pc);
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        size_t pos = row_recv_base + static_cast<size_t>(pc);
                        if (pos >= frag_sizes.size())
                            ABORT("SymFact V2 cost hybrid row-fragment receive size is missing.");
                        int count = frag_sizes[pos];
                        if (count > 0)
                        {
                            local_frag_values[static_cast<size_t>(k0)] +=
                                static_cast<long long>(count);
                            local_frag_messages[static_cast<size_t>(k0)] += 1;
                        }
                    }
                }
                MPI_Allreduce(local_frag_values.data(), global_frag_values.data(),
                              static_cast<int>(nsupers), MPI_LONG_LONG,
                              MPI_SUM, grid->comm);
                MPI_Allreduce(local_frag_messages.data(), global_frag_messages.data(),
                              static_cast<int>(nsupers), MPI_LONG_LONG,
                              MPI_SUM, grid->comm);
                double global_row_plan_setup = 0.0;
                long long global_row_plan_records = 0;
                const long long local_row_plan_records = symV2RowDownDemandRecords;
                MPI_Allreduce(&symV2RowDownSetupSeconds, &global_row_plan_setup,
                              1, MPI_DOUBLE, MPI_MAX, grid->comm);
                MPI_Allreduce(&local_row_plan_records, &global_row_plan_records,
                              1, MPI_LONG_LONG, MPI_SUM, grid->comm);

                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    long long full_row_values = 0;
                    long long full_row_messages = 0;
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t pos =
                                (static_cast<size_t>(k0) * static_cast<size_t>(Pc) +
                                 static_cast<size_t>(pc)) * static_cast<size_t>(Pr) +
                                static_cast<size_t>(pr);
                            if (pos >= global_recv_sizes.size())
                                ABORT("SymFact V2 cost hybrid full-row source size is missing.");
                            if (global_recv_sizes[pos] > 0)
                            {
                                full_row_values += static_cast<long long>(global_recv_sizes[pos]);
                                full_row_messages += 1;
                            }
                        }
                    }
                    full_row_values *= static_cast<long long>(SUPERLU_MAX(Pc - 1, 1));
                    full_row_messages *= static_cast<long long>(SUPERLU_MAX(Pc - 1, 1));

                    const long long frag_values = global_frag_values[static_cast<size_t>(k0)];
                    const long long frag_messages = global_frag_messages[static_cast<size_t>(k0)];
                    const double frag_bytes = static_cast<double>(frag_values) * sizeof(double);
                    const double full_bytes = static_cast<double>(full_row_values) * sizeof(double);
                    const double setup_share =
                        (global_row_plan_records > 0)
                            ? global_row_plan_setup / static_cast<double>(SUPERLU_MAX((long long)1, global_row_plan_records))
                            : 0.0;
                    const double frag_cost =
                        lat_s * static_cast<double>(frag_messages) +
                        frag_bytes / net_Bps +
                        (2.0 * frag_bytes) / pcie_Bps +
                        frag_bytes / pack_Bps +
                        frag_bytes / asm_Bps +
                        kernel_s * static_cast<double>(2 * frag_messages) +
                        setup_weight * setup_share;
                    const double full_cost =
                        lat_s * static_cast<double>(full_row_messages) +
                        full_bytes / net_Bps +
                        full_bytes / pcie_Bps +
                        kernel_s * static_cast<double>(full_row_messages);
                    const bool use_fragments =
                        frag_values > 0 && full_row_values > 0 &&
                        frag_cost < margin * full_cost;
                    symV2UsePcFragmentSchur[static_cast<size_t>(k0)] =
                        use_fragments ? 1 : 0;
                    if (superlu_sym_v2_row_hybrid_trace() && grid3d->iam == 0)
                    {
                        std::printf("SymFact V2 row hybrid cost k=%lld use_frag=%d frag_cost=%e full_cost=%e frag_values=%lld full_values=%lld frag_msgs=%lld full_msgs=%lld setup_share=%e margin=%g source=%s\n",
                                    static_cast<long long>(k0), use_fragments ? 1 : 0,
                                    frag_cost, full_cost, frag_values, full_row_values,
                                    frag_messages, full_row_messages, setup_share, margin,
                                    (superlu_sym_v2_row_l_plan_v2() && !symV2RowDownRecvSizes.empty()) ? "row_down_plan" : "legacy_rowfrag");
                    }
                }
                if (superlu_sym_v2_row_hybrid_trace() && grid3d->iam == 0)
                    std::fflush(stdout);
// SYM_V2_PC2_PHASE5_COST_HYBRID_SELECTOR_END
            }
            else if (pc_fragment_schur_setup &&
                superlu_sym_v2_hybrid_row_bcast() &&
                !superlu_sym_v2_row_l_postsolve_send())
            {
                double ratio =
                    xlu_env_double("GPU3DV2_HYBRID_ROW_BCAST_RATIO", 0.90);
                if (ratio <= 0.0)
                    ABORT("GPU3DV2_HYBRID_ROW_BCAST_RATIO must be positive.");
                std::vector<long long> local_row_fragment_values(
                    static_cast<size_t>(nsupers), 0);
                std::vector<long long> global_row_fragment_values(
                    static_cast<size_t>(nsupers), 0);
                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    size_t row_recv_base =
                        static_cast<size_t>(k0) *
                        static_cast<size_t>(Pc);
                    for (int pc = 0; pc < Pc; ++pc)
                    {
                        size_t pos = row_recv_base + static_cast<size_t>(pc);
                        if (pos >= symV2RowFragRecvSizes.size())
                            ABORT("SymFact V2 hybrid row-fragment receive size is missing.");
                        int count = symV2RowFragRecvSizes[pos];
                        if (count > 0)
                            local_row_fragment_values[static_cast<size_t>(k0)] +=
                                static_cast<long long>(count);
                    }
                }
                MPI_Allreduce(local_row_fragment_values.data(),
                              global_row_fragment_values.data(),
                              static_cast<int>(nsupers), MPI_LONG_LONG,
                              MPI_SUM, grid->comm);
                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    long long full_row_values = 0;
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t pos =
                                (static_cast<size_t>(k0) *
                                     static_cast<size_t>(Pc) +
                                 static_cast<size_t>(pc)) *
                                    static_cast<size_t>(Pr) +
                                static_cast<size_t>(pr);
                            if (pos >= global_recv_sizes.size())
                                ABORT("SymFact V2 hybrid row source size is missing.");
                            if (global_recv_sizes[pos] > 0)
                                full_row_values +=
                                    static_cast<long long>(
                                        global_recv_sizes[pos]);
                        }
                    }
                    full_row_values *= static_cast<long long>(Pc - 1);
                    long long fragment_values =
                        global_row_fragment_values[static_cast<size_t>(k0)];
                    bool use_fragments =
                        fragment_values > 0 && full_row_values > 0 &&
                        static_cast<double>(fragment_values) <
                            ratio * static_cast<double>(full_row_values);
                    symV2UsePcFragmentSchur[static_cast<size_t>(k0)] =
                        use_fragments ? 1 : 0;
                }
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

            if (superlu_sym_v2_pcfrag_taskflow() &&
                superlu_sym_v2_pc_fragment_ldl_native() &&
                superlu_sym_v2_row_l_plan_v2_exchange() &&
                superlu_sym_v2_row_l_direct_recv() &&
                superlu_sym_v2_row_l_compressed_plan() &&
                superlu_sym_v2_row_l_lazy_sendmap() &&
                !superlu_cuda_aware_mpi())
            {
                if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
                    (!symV2PcFragTaskflowGlobalOutputLocks.empty() ||
                     symV2PcFragTaskflowGlobalOutputLocksLive != 0))
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE setup found stale global output locks.");
                if (superlu_sym_v2_pcfrag_taskflow_piece_max_rows() > 0 &&
                    !(superlu_sym_v2_pcfrag_taskflow_async_core() &&
                      superlu_sym_v2_pcfrag_taskflow_coalesce_col()))
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW_PIECE_MAX_ROWS>0 requires async-core column coalescing.");
                const char *taskflow_setup_diag =
                    std::getenv("GPU3DV2_PCFRAG_TASKFLOW_SETUP_DIAG");
                auto taskflow_setup_mark = [&](const char *where) {
                    if (taskflow_setup_diag != NULL &&
                        taskflow_setup_diag[0] != '\0')
                    {
                        fprintf(stderr,
                                "[rank %d] taskflow setup %s\n",
                                iam, where);
                        fflush(stderr);
                    }
                };
                taskflow_setup_mark("enter");
                std::vector<size_t> taskflow_index_counts;
                std::vector<size_t> taskflow_value_counts;
                std::vector<size_t> taskflow_group_index_counts;
                std::vector<size_t> taskflow_group_value_counts;
                std::vector<size_t> taskflow_pinned_counts;
                std::vector<size_t> taskflow_event_counts;
                auto add_fragment_taskflow_counts =
                    [&](const std::vector<int_t> &frag, int_t k0,
                        size_t &index_total, size_t &value_total) {
                        if (frag.empty())
                            return;
                        if (frag.size() < LPANEL_HEADER_SIZE)
                            ABORT("SymFact V2 taskflow prewarm saw truncated metadata.");
                        int_t nblocks = frag[0];
                        if (nblocks < 0)
                            ABORT("SymFact V2 taskflow prewarm saw invalid block count.");
                        size_t nb = static_cast<size_t>(nblocks);
                        size_t row_offset_base =
                            xlu_checked_sum_size(LPANEL_HEADER_SIZE, nb,
                                                 "taskflow prewarm row offsets");
                        size_t row_data_base =
                            xlu_checked_sum_size(row_offset_base, nb,
                                                 "taskflow prewarm row data");
                        row_data_base =
                            xlu_checked_sum_size(row_data_base, 1,
                                                 "taskflow prewarm row data");
                        if (row_data_base > frag.size())
                            ABORT("SymFact V2 taskflow prewarm saw invalid metadata size.");
                        int_t ksupc = frag.size() >= 4 ? frag[3] : SuperSize(k0);
                        if (ksupc < 0)
                            ABORT("SymFact V2 taskflow prewarm saw invalid ksupc.");
                        for (int_t b = 0; b < nblocks; ++b)
                        {
                            size_t row_pos =
                                row_offset_base + static_cast<size_t>(b);
                            int_t row_begin = frag[row_pos];
                            int_t row_end = frag[row_pos + 1];
                            if (row_begin < 0 || row_end < row_begin)
                                ABORT("SymFact V2 taskflow prewarm saw invalid row range.");
                            size_t nrows =
                                static_cast<size_t>(row_end - row_begin);
                            size_t src_begin =
                                xlu_checked_sum_size(row_data_base,
                                                     static_cast<size_t>(row_begin),
                                                     "taskflow prewarm row list");
                            size_t src_end =
                                xlu_checked_sum_size(src_begin, nrows,
                                                     "taskflow prewarm row list");
                            if (src_end > frag.size())
                                ABORT("SymFact V2 taskflow prewarm saw truncated row list.");
                            size_t piece_index_count =
                                xlu_checked_sum_size(
                                    static_cast<size_t>(LPANEL_HEADER_SIZE + 3),
                                    nrows, "taskflow prewarm index count");
                            index_total =
                                xlu_checked_sum_size(index_total,
                                                     piece_index_count,
                                                     "taskflow prewarm index total");
                            size_t piece_value_count =
                                xlu_checked_product(
                                    nrows, static_cast<size_t>(ksupc),
                                    "taskflow prewarm value count");
	                            value_total =
	                                xlu_checked_sum_size(value_total,
	                                                     piece_value_count,
	                                                     "taskflow prewarm value total");
	                        }
	                    };
                auto taskflow_frag_nblocks =
                    [](const std::vector<int_t> &frag) -> int_t {
                    return frag.empty() ? 0 : frag[0];
                };
                auto taskflow_frag_gid =
                    [](const std::vector<int_t> &frag, int_t b) -> int_t {
                    return frag[LPANEL_HEADER_SIZE + b];
                };
                struct TaskflowPrewarmGidPiece
                {
                    int_t gid;
                    int piece;
                    bool operator<(const TaskflowPrewarmGidPiece &other) const
                    {
                        return gid < other.gid;
                    }
                };
                auto taskflow_actual_task_count =
                    [&](int_t k0, const std::vector<int_t> &row_frag,
                        const std::vector<int_t> &partner_frag) -> size_t {
                    int_t nr = taskflow_frag_nblocks(row_frag);
                    int_t nc = taskflow_frag_nblocks(partner_frag);
                    if (nr <= 0 || nc <= 0)
                        return 0;
                    std::vector<TaskflowPrewarmGidPiece> row_gid_to_piece;
                    row_gid_to_piece.reserve(static_cast<size_t>(nr));
                    for (int_t rb = 0; rb < nr; ++rb)
                    {
                        int_t gid = taskflow_frag_gid(row_frag, rb);
                        row_gid_to_piece.push_back(
                            TaskflowPrewarmGidPiece{
                                gid, static_cast<int>(rb)});
                    }
                    std::sort(row_gid_to_piece.begin(),
                              row_gid_to_piece.end());
                    auto row_piece_for_gid = [&](int_t gid) -> int {
                        TaskflowPrewarmGidPiece key{gid, -1};
                        std::vector<TaskflowPrewarmGidPiece>::const_iterator it =
                            std::lower_bound(row_gid_to_piece.begin(),
                                             row_gid_to_piece.end(), key);
                        if (it == row_gid_to_piece.end() || it->gid != gid)
                            return -1;
                        return it->piece;
                    };

                    size_t actual_tasks = 0;
                    for (int_t cb = 0; cb < nc; ++cb)
                    {
                        int_t gj = taskflow_frag_gid(partner_frag, cb);
                        if (gj == k0)
                            continue;
                        int_t local_panel_j = symV2PanelIndex(gj);
                        if (local_panel_j == GLOBAL_BLOCK_NOT_FOUND ||
                            local_panel_j < 0 ||
                            local_panel_j >= symV2PanelCount())
                            continue;
                        xlpanel_t<double> &dst_panel =
                            lPanelVec[static_cast<size_t>(local_panel_j)];
                        if (dst_panel.index == NULL)
                            continue;
                        for (int_t li = 0; li < dst_panel.nblocks(); ++li)
                        {
                            int_t gi = dst_panel.gid(li);
                            if (gi == k0 || gi < gj)
                                continue;
                            if (row_piece_for_gid(gi) < 0)
                                continue;
                            actual_tasks = xlu_checked_sum_size(
                                actual_tasks, static_cast<size_t>(1),
                                "taskflow prewarm sparse task count");
                        }
                    }
                    return actual_tasks;
                };

                for (int_t k0 = 0; k0 < nsupers; ++k0)
                {
                    if (!symV2UsePcFragmentSchurPanel(k0))
                        continue;
                    if (static_cast<size_t>(k0) >= symV2RowFragRecvIndex.size() ||
                        static_cast<size_t>(k0) >= symV2PartnerLRecvIndex.size())
                        ABORT("SymFact V2 taskflow prewarm metadata is missing.");
                    size_t index_total = 0;
                    size_t value_total = 0;
                    add_fragment_taskflow_counts(symV2RowFragRecvIndex[k0],
                                                 k0, index_total, value_total);
	                    add_fragment_taskflow_counts(symV2PartnerLRecvIndex[k0],
	                                                 k0, index_total, value_total);
	                    if (index_total > 0)
	                        taskflow_index_counts.push_back(index_total);
	                    if (value_total > 0)
	                        taskflow_value_counts.push_back(value_total);
	                    const std::vector<int_t> &row_frag =
	                        symV2RowFragRecvIndex[k0];
	                    const std::vector<int_t> &partner_frag =
	                        symV2PartnerLRecvIndex[k0];
	                    int_t nr = taskflow_frag_nblocks(row_frag);
	                    int_t nc = taskflow_frag_nblocks(partner_frag);
		                    size_t event_total = xlu_checked_sum_size(
		                        static_cast<size_t>(nr),
		                        static_cast<size_t>(nc),
		                        "taskflow prewarm piece events");
                    size_t sparse_task_count =
                        taskflow_actual_task_count(k0, row_frag,
                                                   partner_frag);
                    if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
                        superlu_sym_v2_pcfrag_taskflow_coalesce_col() &&
                        sparse_task_count > 0)
                    {
                        size_t pair_index_total = xlu_checked_product(
                            static_cast<size_t>(2), sparse_task_count,
                            "taskflow coalesced group pair index total");
                        size_t group_index_total = xlu_checked_sum_size(
                            index_total, pair_index_total,
                            "taskflow coalesced group index total");
                        taskflow_group_index_counts.push_back(
                            group_index_total);
                        if (value_total > 0)
                            taskflow_group_value_counts.push_back(
                                value_total);
                    }
                    size_t task_event_count = sparse_task_count;
                    if (superlu_sym_v2_pcfrag_taskflow_async_core())
                    {
                        task_event_count = SUPERLU_MIN(
                            sparse_task_count,
                            static_cast<size_t>(
                                superlu_sym_v2_pcfrag_taskflow_progress_budget()));
                    }
                    else
                    {
                        task_event_count = 0;
                    }
		                    event_total = xlu_checked_sum_size(
		                        event_total, task_event_count,
		                        "taskflow prewarm sparse task events");
		                    if (event_total > 0)
		                        taskflow_event_counts.push_back(event_total);
	                    size_t partner_recv_base =
	                        static_cast<size_t>(k0) * static_cast<size_t>(Pr);
                    if (partner_recv_base + static_cast<size_t>(Pr) >
                        symV2PartnerLRecvSizes.size())
                        ABORT("SymFact V2 taskflow prewarm partner sizes are missing.");
                    size_t partner_recv_total = 0;
                    int_t kcol0 = symV2PanelRoot(k0);
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        size_t pos = partner_recv_base +
                                     static_cast<size_t>(pr);
                        int count = symV2PartnerLRecvSizes[pos];
                        int src = PNUM(pr, kcol0, grid);
                        if (count > 0 && src != iam)
                            partner_recv_total = xlu_checked_sum_size(
                                partner_recv_total,
                                static_cast<size_t>(count),
                                "taskflow prewarm partner receive total");
                    }
                    if (partner_recv_total > 0)
                        taskflow_pinned_counts.push_back(partner_recv_total);

                    size_t row_recv_base =
                        static_cast<size_t>(k0) * static_cast<size_t>(Pc);
                    if (row_recv_base + static_cast<size_t>(Pc) >
                        symV2RowFragRecvSizes.size())
                        ABORT("SymFact V2 taskflow prewarm row sizes are missing.");
                    if (kcol0 != mycol)
                    {
                        size_t row_recv_total = 0;
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            int count = symV2RowFragRecvSizes[
                                row_recv_base + static_cast<size_t>(pc)];
                            if (count > 0)
                                row_recv_total = xlu_checked_sum_size(
                                    row_recv_total,
                                    static_cast<size_t>(count),
                                    "taskflow prewarm row receive total");
                        }
                        if (row_recv_total > 0)
                            taskflow_pinned_counts.push_back(row_recv_total);
                    }
                    else
                    {
                        int_t lk0 = symV2PanelIndex(k0);
                        if (lk0 < 0)
                            continue;

                        size_t partner_send_total = 0;
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            size_t flat =
                                static_cast<size_t>(lk0) *
                                    static_cast<size_t>(Pc) +
                                static_cast<size_t>(pc);
                            if (flat >= symV2PartnerLSendSizes.size())
                                ABORT("SymFact V2 taskflow prewarm partner send size is missing.");
                            int count = symV2PartnerLSendSizes[flat];
                            if (count <= 0)
                                continue;
                            bool active_remote_dest = false;
                            for (int pr = 0; pr < Pr; ++pr)
                            {
                                size_t active_pos =
                                    flat * static_cast<size_t>(Pr) +
                                    static_cast<size_t>(pr);
                                if (active_pos >= symV2PartnerLSendRowActive.size())
                                    ABORT("SymFact V2 taskflow prewarm partner send mask is missing.");
                                if (symV2PartnerLSendRowActive[active_pos] &&
                                    PNUM(pr, pc, grid) != iam)
                                {
                                    active_remote_dest = true;
                                    break;
                                }
                            }
                            if (active_remote_dest)
                                partner_send_total = xlu_checked_sum_size(
                                    partner_send_total,
                                    static_cast<size_t>(count),
                                    "taskflow prewarm partner send total");
                        }
                        if (partner_send_total > 0)
                            taskflow_pinned_counts.push_back(partner_send_total);

                        size_t row_send_total = 0;
                        size_t row_send_base =
                            static_cast<size_t>(lk0) *
                            static_cast<size_t>(Pc);
                        if (row_send_base + static_cast<size_t>(Pc) >
                            symV2RowDownSendSizes.size())
                            ABORT("SymFact V2 taskflow prewarm row send sizes are missing.");
                        for (int pc = 0; pc < Pc; ++pc)
                        {
                            if (pc == mycol)
                                continue;
                            int count =
                                symV2RowDownSendSizes[
                                    row_send_base + static_cast<size_t>(pc)];
                            if (count > 0)
                                row_send_total = xlu_checked_sum_size(
                                    row_send_total,
                                    static_cast<size_t>(count),
                                    "taskflow prewarm row send total");
                        }
                        if (row_send_total > 0)
                            taskflow_pinned_counts.push_back(row_send_total);
                    }
                }
                taskflow_setup_mark("after_count_scan");

                int_t num_lookahead = getNumLookAhead(options);
                if (num_lookahead <= 0)
                    num_lookahead = 1;
                size_t active_slots =
                    static_cast<size_t>(SUPERLU_MAX(static_cast<int_t>(1),
                                                    num_lookahead));
                if (A_gpu.numCudaStreams > 0 &&
                    A_gpu.numCudaStreams <= MAX_CUDA_STREAMS)
                {
                    size_t stream_slots =
                        static_cast<size_t>(A_gpu.numCudaStreams);
                    active_slots = SUPERLU_MAX(active_slots, stream_slots);
                }
                if (superlu_sym_v2_pcfrag_taskflow_async_core())
                {
                    taskflow_setup_mark("before_state_resize");
                    if (symV2PcFragTaskStates.size() <
                        static_cast<size_t>(nsupers))
                        symV2PcFragTaskStates.resize(
                            static_cast<size_t>(nsupers));
                    taskflow_setup_mark("after_state_resize");
                    size_t progress_scratch_count = xlu_checked_sum_size(
                        xlu_checked_product(static_cast<size_t>(Pr),
                                            static_cast<size_t>(Pc),
                                            "taskflow progress scratch"),
                        static_cast<size_t>(Pc),
                        "taskflow progress scratch");
                    progress_scratch_count = SUPERLU_MAX(
                        progress_scratch_count, static_cast<size_t>(Pr));
                    for (int_t k0 = 0; k0 < nsupers; ++k0)
                    {
                        if (!symV2UsePcFragmentSchurPanel(k0))
                            continue;
                        SymV2PcFragPanelTaskState &state =
                            symV2PcFragTaskStates[static_cast<size_t>(k0)];
                        if (state.producer_progress_indices.capacity() <
                            progress_scratch_count)
                            state.producer_progress_indices.reserve(
                                progress_scratch_count);
                        if (state.producer_progress_statuses.capacity() <
                            progress_scratch_count)
                            state.producer_progress_statuses.reserve(
                                progress_scratch_count);
                        size_t partner_request_count =
                            static_cast<size_t>(Pr);
                        size_t row_request_count = SUPERLU_MAX(
                            static_cast<size_t>(1), static_cast<size_t>(Pc));
                        if (state.producer_partner_recv_reqs.capacity() <
                            partner_request_count)
                            state.producer_partner_recv_reqs.reserve(
                                partner_request_count);
                        if (state.producer_partner_recv_prs.capacity() <
                            partner_request_count)
                            state.producer_partner_recv_prs.reserve(
                                partner_request_count);
                        if (state.producer_partner_recv_sizes.capacity() <
                            partner_request_count)
                            state.producer_partner_recv_sizes.reserve(
                                partner_request_count);
                        if (state.producer_partner_recv_offsets.capacity() <
                            partner_request_count)
                            state.producer_partner_recv_offsets.reserve(
                                partner_request_count);
                        if (state.producer_partner_recv_done.capacity() <
                            partner_request_count)
                            state.producer_partner_recv_done.reserve(
                                partner_request_count);
                        if (state.producer_row_recv_reqs.capacity() <
                            row_request_count)
                            state.producer_row_recv_reqs.reserve(
                                row_request_count);
                        if (state.producer_row_recv_pcs.capacity() <
                            row_request_count)
                            state.producer_row_recv_pcs.reserve(
                                row_request_count);
                        if (state.producer_row_recv_sizes.capacity() <
                            row_request_count)
                            state.producer_row_recv_sizes.reserve(
                                row_request_count);
                        if (state.producer_row_recv_offsets.capacity() <
                            row_request_count)
                            state.producer_row_recv_offsets.reserve(
                                row_request_count);
                        if (state.producer_row_recv_done.capacity() <
                            row_request_count)
                            state.producer_row_recv_done.reserve(
                                row_request_count);
                        if (state.producer_send_reqs.capacity() <
                            progress_scratch_count)
                            state.producer_send_reqs.reserve(
                                progress_scratch_count);
                        if (state.producer_partner_progressive_assembled.capacity() <
                            static_cast<size_t>(Pr))
                            state.producer_partner_progressive_assembled.reserve(
                                static_cast<size_t>(Pr));
                    }
                    taskflow_setup_mark("after_state_reserves");
                }
                std::vector<size_t> taskflow_pinned_counts_by_panel =
                    taskflow_pinned_counts;
                std::sort(taskflow_index_counts.begin(),
                          taskflow_index_counts.end(),
                          [](size_t a, size_t b) { return a > b; });
                std::sort(taskflow_value_counts.begin(),
                          taskflow_value_counts.end(),
                          [](size_t a, size_t b) { return a > b; });
                if (superlu_sym_v2_pcfrag_taskflow_async_core())
                {
                    if (taskflow_group_index_counts.empty() &&
                        taskflow_group_value_counts.empty())
                    {
                        taskflow_group_index_counts = taskflow_index_counts;
                        taskflow_group_value_counts = taskflow_value_counts;
                    }
                    std::sort(taskflow_group_index_counts.begin(),
                              taskflow_group_index_counts.end(),
                              [](size_t a, size_t b) { return a > b; });
                    std::sort(taskflow_group_value_counts.begin(),
                              taskflow_group_value_counts.end(),
                              [](size_t a, size_t b) { return a > b; });
                }
	                std::sort(taskflow_pinned_counts.begin(),
	                          taskflow_pinned_counts.end(),
	                          [](size_t a, size_t b) { return a > b; });
	                std::sort(taskflow_event_counts.begin(),
	                          taskflow_event_counts.end(),
	                          [](size_t a, size_t b) { return a > b; });
                size_t group_slots = active_slots;
                if (superlu_sym_v2_pcfrag_taskflow_async_core())
                {
                    size_t progress_slots = static_cast<size_t>(
                        SUPERLU_MAX(1,
                                    superlu_sym_v2_pcfrag_taskflow_effective_progress_budget()));
                    group_slots = SUPERLU_MAX(group_slots, progress_slots);
                }
                if (taskflow_group_index_counts.size() > group_slots)
                    taskflow_group_index_counts.resize(group_slots);
                if (taskflow_group_value_counts.size() > group_slots)
                    taskflow_group_value_counts.resize(group_slots);
                if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
                    superlu_sym_v2_pcfrag_taskflow_coalesce_col())
                {
                    if (!taskflow_group_index_counts.empty())
                    {
                        size_t fill = taskflow_group_index_counts[0];
                        while (taskflow_group_index_counts.size() <
                               group_slots)
                            taskflow_group_index_counts.push_back(fill);
                    }
                    if (!taskflow_group_value_counts.empty())
                    {
                        size_t fill = taskflow_group_value_counts[0];
                        while (taskflow_group_value_counts.size() <
                               group_slots)
                            taskflow_group_value_counts.push_back(fill);
                    }
                }
		                if (taskflow_index_counts.size() > active_slots)
		                    taskflow_index_counts.resize(active_slots);
		                if (taskflow_value_counts.size() > active_slots)
		                    taskflow_value_counts.resize(active_slots);
		                if (taskflow_event_counts.size() > active_slots)
		                    taskflow_event_counts.resize(active_slots);
	                size_t pinned_slots =
	                    xlu_checked_product(active_slots, 4,
	                                        "taskflow prewarm pinned slots");
                if (taskflow_pinned_counts.size() > pinned_slots)
                    taskflow_pinned_counts.resize(pinned_slots);
	                size_t temporal_pinned_slots =
	                    superlu_sym_v2_pcfrag_taskflow_async_core()
	                        ? static_cast<size_t>(0)
	                        : SUPERLU_MIN(
	                              pinned_slots,
	                              taskflow_pinned_counts_by_panel.size());
                for (size_t i = 0; i < temporal_pinned_slots; ++i)
                    taskflow_pinned_counts.push_back(
                        taskflow_pinned_counts_by_panel[i]);
                taskflow_setup_mark("after_pool_sizing");

                const char *taskflow_prewarm_diag =
                    std::getenv("GPU3DV2_PCFRAG_TASKFLOW_PREWARM_DIAG");
                if (taskflow_prewarm_diag != NULL &&
                    taskflow_prewarm_diag[0] != '\0')
                {
                    auto print_prewarm_counts =
                        [&](const char *name,
                            const std::vector<size_t> &counts,
                            size_t element_size) {
                        size_t sum = 0;
                        size_t max_count = 0;
                        for (size_t i = 0; i < counts.size(); ++i)
                        {
                            sum = xlu_checked_sum_size(
                                sum, counts[i],
                                "taskflow prewarm diagnostic sum");
                            max_count = SUPERLU_MAX(max_count, counts[i]);
                        }
                        size_t bytes = xlu_checked_product(
                            sum, element_size,
                            "taskflow prewarm diagnostic bytes");
                        fprintf(stderr,
                                "[rank %d] taskflow prewarm %s: "
                                "blocks=%zu sum=%zu max=%zu bytes=%zu\n",
                                iam, name, counts.size(), sum,
                                max_count, bytes);
                        size_t top = SUPERLU_MIN(counts.size(),
                                                 static_cast<size_t>(8));
                        for (size_t i = 0; i < top; ++i)
                            fprintf(stderr,
                                    "[rank %d] taskflow prewarm %s[%zu]=%zu\n",
                                    iam, name, i, counts[i]);
                    };
                    print_prewarm_counts("index", taskflow_index_counts,
                                         sizeof(int_t));
                    print_prewarm_counts("value", taskflow_value_counts,
                                         sizeof(double));
                    print_prewarm_counts("group_index",
                                         taskflow_group_index_counts,
                                         sizeof(int_t));
                    print_prewarm_counts("group_value",
                                         taskflow_group_value_counts,
                                         sizeof(double));
                    print_prewarm_counts("pinned", taskflow_pinned_counts,
                                         sizeof(double));
                    print_prewarm_counts("events", taskflow_event_counts,
                                         sizeof(cudaEvent_t));
                    fflush(stderr);
                    const char *taskflow_prewarm_diag_abort =
                        std::getenv(
                            "GPU3DV2_PCFRAG_TASKFLOW_PREWARM_DIAG_ABORT");
                    if (taskflow_prewarm_diag_abort != NULL &&
                        taskflow_prewarm_diag_abort[0] != '\0')
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW prewarm diagnostic abort.");
                }

                std::vector<unsigned char> matched_index_blocks(
                    symV2PcFragTaskflowIndexBlockPool.size(), 0);
                auto index_pool_has_block = [&](size_t capacity) -> bool {
                    for (size_t i = 0;
                         i < symV2PcFragTaskflowIndexBlockPool.size(); ++i)
                    {
                        if (matched_index_blocks[i] ||
                            symV2PcFragTaskflowIndexBlockPool[i].ptr == NULL ||
                            symV2PcFragTaskflowIndexBlockPool[i].capacity <
                                capacity)
                            continue;
                        matched_index_blocks[i] = 1;
                        return true;
                    }
                    return false;
                };
                int taskflow_device_pool_copies =
                    superlu_sym_v2_pcfrag_taskflow_async_core() ? 1 : 2;
                if (superlu_sym_v2_pcfrag_taskflow_async_core() &&
                    superlu_sym_v2_pcfrag_taskflow_coalesce_col())
                    taskflow_device_pool_copies = 2;
	                for (size_t i = 0; i < taskflow_index_counts.size(); ++i)
	                {
                    if (i == 0)
                        taskflow_setup_mark("before_index_prewarm");
	                    for (int copy = 0; copy < taskflow_device_pool_copies; ++copy)
	                    {
                        size_t capacity = taskflow_index_counts[i];
                        if (capacity == 0 || index_pool_has_block(capacity))
                            continue;
                        int_t *ptr = NULL;
                        gpuErrchk(cudaMalloc(
                            reinterpret_cast<void **>(&ptr),
                            sizeof(int_t) * capacity));
                        symV2PcFragTaskflowIndexBlockPool.push_back(
                            SymV2PcFragGpuIndexBlock(ptr, capacity));
                        matched_index_blocks.push_back(1);
	                        ++symV2PcFragTaskflowStats
	                              .arena_index_prewarm_blocks;
	                    }
	                }
                taskflow_setup_mark("after_index_prewarm");

                std::vector<unsigned char> matched_group_index_blocks(
                    symV2PcFragTaskflowGroupIndexBlockPool.size(), 0);
                auto group_index_pool_has_block =
                    [&](size_t capacity) -> bool {
                    for (size_t i = 0;
                         i < symV2PcFragTaskflowGroupIndexBlockPool.size();
                         ++i)
                    {
                        if (matched_group_index_blocks[i] ||
                            symV2PcFragTaskflowGroupIndexBlockPool[i].ptr ==
                                NULL ||
                            symV2PcFragTaskflowGroupIndexBlockPool[i]
                                    .capacity < capacity)
                            continue;
                        matched_group_index_blocks[i] = 1;
                        return true;
                    }
                    return false;
                };
                for (size_t i = 0; i < taskflow_group_index_counts.size();
                     ++i)
                {
                    if (i == 0)
                        taskflow_setup_mark("before_group_index_prewarm");
                    size_t capacity = taskflow_group_index_counts[i];
                    if (capacity == 0 ||
                        group_index_pool_has_block(capacity))
                        continue;
                    int_t *ptr = NULL;
                    gpuErrchk(cudaMalloc(
                        reinterpret_cast<void **>(&ptr),
                        sizeof(int_t) * capacity));
                    symV2PcFragTaskflowGroupIndexBlockPool.push_back(
                        SymV2PcFragGpuIndexBlock(ptr, capacity));
                    matched_group_index_blocks.push_back(1);
                    ++symV2PcFragTaskflowStats.arena_index_prewarm_blocks;
                }
                taskflow_setup_mark("after_group_index_prewarm");

                std::vector<unsigned char> matched_value_blocks(
                    symV2PcFragTaskflowValueBlockPool.size(), 0);
                auto value_pool_has_block = [&](size_t capacity) -> bool {
                    for (size_t i = 0;
                         i < symV2PcFragTaskflowValueBlockPool.size(); ++i)
                    {
                        if (matched_value_blocks[i] ||
                            symV2PcFragTaskflowValueBlockPool[i].ptr == NULL ||
                            symV2PcFragTaskflowValueBlockPool[i].capacity <
                                capacity)
                            continue;
                        matched_value_blocks[i] = 1;
                        return true;
                    }
                    return false;
                };
	                for (size_t i = 0; i < taskflow_value_counts.size(); ++i)
	                {
                    if (i == 0)
                        taskflow_setup_mark("before_value_prewarm");
	                    for (int copy = 0; copy < taskflow_device_pool_copies; ++copy)
	                    {
                        size_t capacity = taskflow_value_counts[i];
                        if (capacity == 0 || value_pool_has_block(capacity))
                            continue;
                        double *ptr = NULL;
                        gpuErrchk(cudaMalloc(
                            reinterpret_cast<void **>(&ptr),
                            sizeof(double) * capacity));
                        symV2PcFragTaskflowValueBlockPool.push_back(
                            SymV2PcFragGpuValueBlock(ptr, capacity));
                        matched_value_blocks.push_back(1);
	                        ++symV2PcFragTaskflowStats
	                              .arena_value_prewarm_blocks;
	                    }
	                }
                taskflow_setup_mark("after_value_prewarm");

                std::vector<unsigned char> matched_group_value_blocks(
                    symV2PcFragTaskflowGroupValueBlockPool.size(), 0);
                auto group_value_pool_has_block =
                    [&](size_t capacity) -> bool {
                    for (size_t i = 0;
                         i < symV2PcFragTaskflowGroupValueBlockPool.size();
                         ++i)
                    {
                        if (matched_group_value_blocks[i] ||
                            symV2PcFragTaskflowGroupValueBlockPool[i].ptr ==
                                NULL ||
                            symV2PcFragTaskflowGroupValueBlockPool[i]
                                    .capacity < capacity)
                            continue;
                        matched_group_value_blocks[i] = 1;
                        return true;
                    }
                    return false;
                };
                for (size_t i = 0; i < taskflow_group_value_counts.size();
                     ++i)
                {
                    if (i == 0)
                        taskflow_setup_mark("before_group_value_prewarm");
                    size_t capacity = taskflow_group_value_counts[i];
                    if (capacity == 0 ||
                        group_value_pool_has_block(capacity))
                        continue;
                    double *ptr = NULL;
                    gpuErrchk(cudaMalloc(
                        reinterpret_cast<void **>(&ptr),
                        sizeof(double) * capacity));
                    symV2PcFragTaskflowGroupValueBlockPool.push_back(
                        SymV2PcFragGpuValueBlock(ptr, capacity));
                    matched_group_value_blocks.push_back(1);
                    ++symV2PcFragTaskflowStats.arena_value_prewarm_blocks;
                }
                taskflow_setup_mark("after_group_value_prewarm");

                std::vector<unsigned char> matched_pinned_blocks(
                    symV2PcFragTaskflowPinnedBlockPool.size(), 0);
                auto pinned_pool_has_block = [&](size_t capacity) -> bool {
                    for (size_t i = 0;
                         i < symV2PcFragTaskflowPinnedBlockPool.size(); ++i)
                    {
                        if (matched_pinned_blocks[i] ||
                            symV2PcFragTaskflowPinnedBlockPool[i].ptr == NULL ||
                            symV2PcFragTaskflowPinnedBlockPool[i].capacity <
                                capacity)
                            continue;
                        matched_pinned_blocks[i] = 1;
                        return true;
                    }
                    return false;
                };
	                for (size_t i = 0; i < taskflow_pinned_counts.size(); ++i)
	                {
                    if (i == 0)
                        taskflow_setup_mark("before_pinned_prewarm");
	                    size_t capacity = taskflow_pinned_counts[i];
                    if (capacity == 0 || pinned_pool_has_block(capacity))
                        continue;
                    double *ptr = NULL;
                    gpuErrchk(cudaMallocHost(
                        reinterpret_cast<void **>(&ptr),
                        sizeof(double) * capacity));
	                    symV2PcFragTaskflowPinnedBlockPool.push_back(
	                        SymV2PcFragHostValueBlock(ptr, capacity));
	                    matched_pinned_blocks.push_back(1);
		                    ++symV2PcFragTaskflowStats
		                          .arena_pinned_prewarm_blocks;
		                }
                taskflow_setup_mark("after_pinned_prewarm");

		                size_t event_target = 0;
	                for (size_t i = 0; i < taskflow_event_counts.size(); ++i)
	                    event_target = xlu_checked_sum_size(
	                        event_target, taskflow_event_counts[i],
	                        "taskflow prewarm event total");
                taskflow_setup_mark("before_event_prewarm");
		                while (symV2PcFragTaskflowEventPool.size() < event_target)
		                {
	                    cudaEvent_t event = NULL;
	                    gpuErrchk(cudaEventCreateWithFlags(
	                        &event, cudaEventDisableTiming));
	                    symV2PcFragTaskflowEventPool.push_back(event);
		                    ++symV2PcFragTaskflowStats
		                          .arena_event_prewarm_blocks;
		                }
                taskflow_setup_mark("after_event_prewarm");
		                if (superlu_sym_v2_pcfrag_taskflow_async_core())
		                {
	                    taskflow_setup_mark("before_gemm_event_prewarm");
		                    size_t taskflow_gemm_slots = active_slots;
		                    taskflow_gemm_slots = SUPERLU_MAX(
		                        taskflow_gemm_slots,
		                        static_cast<size_t>(MAX_CUDA_STREAMS));
		                    int taskflow_gemm_raw_streams = A_gpu.numCudaStreams;
		                    if (taskflow_gemm_raw_streams > 0 &&
		                        taskflow_gemm_raw_streams <= 1024)
		                        taskflow_gemm_slots = SUPERLU_MAX(
		                            taskflow_gemm_slots,
		                            static_cast<size_t>(
		                                taskflow_gemm_raw_streams));
		                    int taskflow_gemm_resource_count =
		                        SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT;
		                    if (std::getenv("GPU3DV2_PCFRAG_TASKFLOW_SETUP_DIAG") != NULL)
		                    {
		                        std::printf("[rank %d] taskflow GEMM resource prewarm slots=%zu raw_streams=%d resource_count=%d current=%zu\n",
		                                    (grid3d != NULL) ? grid3d->iam : -1,
		                                    taskflow_gemm_slots,
		                                    taskflow_gemm_raw_streams,
		                                    taskflow_gemm_resource_count,
		                                    symV2PcFragTaskflowGemmResources.size());
		                        std::fflush(stdout);
		                    }
		                    if (taskflow_gemm_slots == 0 ||
		                        taskflow_gemm_slots > 1024 ||
		                        taskflow_gemm_resource_count <= 0 ||
		                        taskflow_gemm_resource_count > 64)
		                        ABORT("GPU3DV2_PCFRAG_TASKFLOW invalid GEMM resource prewarm dimensions.");
		                    size_t nres = xlu_checked_product(
		                        taskflow_gemm_slots,
		                        static_cast<size_t>(taskflow_gemm_resource_count),
		                        "taskflow GEMM resource count");
	                    symV2PcFragTaskflowGemmResources.resize(nres);
	                    for (size_t i = 0; i < nres; ++i)
	                    {
	                        SymV2PcFragGemmResourceState &res =
	                            symV2PcFragTaskflowGemmResources[i];
	                        if (res.tail_event == NULL)
	                        {
	                            gpuErrchk(cudaEventCreateWithFlags(
	                                &res.tail_event,
	                                cudaEventDisableTiming));
	                            ++symV2PcFragTaskflowStats
	                                  .arena_event_prewarm_blocks;
	                        }
	                        res.recorded = 0;
	                        res.owner_stream_id = static_cast<int>(
	                            i / static_cast<size_t>(
	                                    SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT));
	                        res.resource_kind = static_cast<int>(
	                            i % static_cast<size_t>(
	                                    SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT));
	                        res.active_task_id = -1;
	                        res.waits = 0;
	                        res.updates = 0;
	                    }
                    taskflow_setup_mark("after_gemm_event_prewarm");
	                }
                taskflow_setup_mark("after_pool_prewarm");
	            }
	        }
        xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after send GPU L2U map setup");
    }
#endif

    xlu_sym_gpu3d_trace(grid3d, "exit initSymFactWorkspace");
    return 0;
}

template <typename Ftype>
bool xLUstruct_t<Ftype>::symV2UsePcFragmentSchurPanel(int_t k) const
{
#ifdef HAVE_CUDA
    if (!(superlu_sym_v2_pc_fragment_schur() && Pr > 1 && Pc > 1))
        return false;
    if (k < 0 || k >= nsupers)
        return false;
    if (static_cast<size_t>(k) >= symV2UsePcFragmentSchur.size())
        return true;
    return symV2UsePcFragmentSchur[static_cast<size_t>(k)] != 0;
#else
    return false;
#endif
}

template <typename Ftype>
bool xLUstruct_t<Ftype>::symV2UsePcFragmentTaskflowPanel(int_t k) const
{
#ifdef HAVE_CUDA
    return superlu_sym_v2_pcfrag_taskflow() &&
           symV2UsePcFragmentSchurPanel(k);
#else
    return false;
#endif
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
// SYM_V2_PC2_PHASE4_FREE_ROW_DOWN_GPU_MAPS_BEGIN
    if (symV2RowDownSendMapPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2RowDownSendMapPoolGPU));
        symV2RowDownSendMapPoolGPU = NULL;
    }
// SYM_V2_PC2_LAZY_SENDMAP_FREE_BEGIN
    if (symV2RowDownSendSegPoolGPU != NULL)
    {
        gpuErrchk(cudaFree(symV2RowDownSendSegPoolGPU));
        symV2RowDownSendSegPoolGPU = NULL;
    }
    symV2RowDownSendSegPoolCount = 0;
// SYM_V2_PC2_LAZY_SENDMAP_FREE_END
// SYM_V2_PC2_PHASE4_FREE_ROW_DOWN_GPU_MAPS_END

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
// SYM_V2_PC2_PHASE1_FREE_SEND_STAGING_BEGIN
    if (symV2RowFragHostSendPoolPinned != NULL)
    {
        gpuErrchk(cudaFreeHost(symV2RowFragHostSendPoolPinned));
        symV2RowFragHostSendPoolPinned = NULL;
    }
    else
    {
        for (size_t i = 0; i < symV2RowFragHostSendBufs.size(); ++i)
        {
            if (symV2RowFragHostSendBufs[i] == NULL)
                continue;
            if (symV2RowFragHostSendPinned)
            {
                gpuErrchk(cudaFreeHost(symV2RowFragHostSendBufs[i]));
            }
            else
            {
                SUPERLU_FREE(symV2RowFragHostSendBufs[i]);
            }
            symV2RowFragHostSendBufs[i] = NULL;
        }
    }
    symV2RowFragHostSendBufs.clear();
    symV2RowFragHostSendPinned = 0;
    symV2RowFragHostSendPoolPinnedCount = 0;
    symV2RowFragSendCountsScratch.clear();
    symV2RowFragSendOffsetsScratch.clear();
    symV2RowFragSendReqsScratch.clear();
// SYM_V2_PC2_PHASE1_FREE_SEND_STAGING_END

    symV2PartnerLHostSendScratchOffsets.clear();
    symV2ExchangeSendSizesScratch.clear();
    symV2ExchangeRecvSizesScratch.clear();
    symV2ExchangeRecvOffsetsScratch.clear();
    symV2ExchangeRecvReqsScratch.clear();
    symV2ExchangeSendReqsScratch.clear();
    symV2PartnerLHostSendBufs.clear();
    symV2PartnerLExactHostSendBufs.clear();
    symV2RowFragExactHostSendBufs.clear();
    symL2USendMapsHost.clear();
    symL2ULocalMapsHost.clear();
    symV2PartnerLMapOffsets.clear();
    symV2PartnerLPackedMaps.clear();
    symV2PartnerLExactSendMapsHost.clear();
    symV2PartnerLExactSendMapOffsets.clear();
    symV2RowFragExactSendMapsHost.clear();
    symV2RowFragExactSendMapOffsets.clear();
    symV2RowDirectSendSizes.clear();
    symV2RowDirectSendMapOffsets.clear();
    symV2RowDirectSendMapsHost.clear();
    symV2RowDirectSendBlocksHost.clear();
    symV2RowDirectSendMapScratchHost.clear();
    symV2RowDownSendSizes.clear();
    symV2RowDownSendMapOffsets.clear();
    symV2RowDownSendMapsHost.clear();
// SYM_V2_PC2_LAZY_SENDMAP_CLEAR_BEGIN
    symV2RowDownSendSegsHost.clear();
    symV2RowDownSendSegOffsets.clear();
    symV2RowDownSendSegCounts.clear();
    symV2RowDownSendSegsGPU.clear();
    symV2RowDownSendSegPoolCount = 0;
// SYM_V2_PC2_LAZY_SENDMAP_CLEAR_END
    symV2RowDownSegOffsets.clear();
    symV2RowDownSegs.clear();
    symV2RowDownRecvSizes.clear();
    symV2RowDownPlanReady.clear();
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
    symV2PartnerLRecvMapOffsets.clear();
    symV2PartnerLRecvMapsGPU.clear();
    symV2RowFragRecvSizes.clear();
    symV2RowFragRecvIndex.clear();
    symV2RowFragRecvMap.clear();
    symV2RowFragRecvMapOffsets.clear();
    symV2RowFragRecvMapsGPU.clear();
    symL2ULocalMapsGPU.clear();
    symPanelReadyEventIds.clear();
    symV2UsePcFragmentSchur.clear();
    symV2RawPanelNodes.clear();
    for (size_t sx = 0; sx < symV2PcFragTaskStates.size(); ++sx)
    {
        SymV2PcFragPanelTaskState &state = symV2PcFragTaskStates[sx];
        for (size_t p = 0; p < state.row_pieces.size(); ++p)
        {
            if (state.row_pieces[p].ready_event != NULL)
                gpuErrchk(cudaEventDestroy(state.row_pieces[p].ready_event));
            if (state.row_pieces[p].done_event != NULL)
                gpuErrchk(cudaEventDestroy(state.row_pieces[p].done_event));
        }
        for (size_t p = 0; p < state.partner_pieces.size(); ++p)
        {
            if (state.partner_pieces[p].ready_event != NULL)
                gpuErrchk(cudaEventDestroy(state.partner_pieces[p].ready_event));
            if (state.partner_pieces[p].done_event != NULL)
                gpuErrchk(cudaEventDestroy(state.partner_pieces[p].done_event));
        }
        for (size_t t = 0; t < state.tasks.size(); ++t)
            if (state.tasks[t].done_event != NULL)
                gpuErrchk(cudaEventDestroy(state.tasks[t].done_event));
        if (state.producer_partner_recv_host_values != NULL)
            gpuErrchk(cudaFreeHost(state.producer_partner_recv_host_values));
        if (state.producer_row_recv_host_values != NULL)
            gpuErrchk(cudaFreeHost(state.producer_row_recv_host_values));
        if (state.producer_partner_send_host_values != NULL)
            gpuErrchk(cudaFreeHost(state.producer_partner_send_host_values));
        if (state.producer_row_send_host_values != NULL)
            gpuErrchk(cudaFreeHost(state.producer_row_send_host_values));
        if (state.d_index_pool != NULL)
            gpuErrchk(cudaFree(state.d_index_pool));
        if (state.d_value_pool != NULL)
            gpuErrchk(cudaFree(state.d_value_pool));
        if (state.d_group_index_pool != NULL)
            gpuErrchk(cudaFree(state.d_group_index_pool));
        if (state.d_group_value_pool != NULL)
            gpuErrchk(cudaFree(state.d_group_value_pool));
        state.reset();
    }
    symV2PcFragTaskStates.clear();
    for (size_t r = 0; r < symV2PcFragTaskflowGemmResources.size(); ++r)
        if (symV2PcFragTaskflowGemmResources[r].tail_event != NULL)
            gpuErrchk(cudaEventDestroy(
                symV2PcFragTaskflowGemmResources[r].tail_event));
    symV2PcFragTaskflowGemmResources.clear();
    symV2PcFragTaskflowGlobalOutputLocks.clear();
    symV2PcFragTaskflowOutputPanelOffsets.clear();
    symV2PcFragTaskflowGlobalOutputLockState.clear();
    symV2PcFragTaskflowGlobalOutputLocksLive = 0;
    for (size_t ev = 0; ev < symV2PcFragTaskflowEventPool.size(); ++ev)
        if (symV2PcFragTaskflowEventPool[ev] != NULL)
            gpuErrchk(cudaEventDestroy(symV2PcFragTaskflowEventPool[ev]));
    symV2PcFragTaskflowEventPool.clear();
    for (size_t b = 0; b < symV2PcFragTaskflowIndexBlockPool.size(); ++b)
        if (symV2PcFragTaskflowIndexBlockPool[b].ptr != NULL)
            gpuErrchk(cudaFree(symV2PcFragTaskflowIndexBlockPool[b].ptr));
    symV2PcFragTaskflowIndexBlockPool.clear();
    for (size_t b = 0; b < symV2PcFragTaskflowValueBlockPool.size(); ++b)
        if (symV2PcFragTaskflowValueBlockPool[b].ptr != NULL)
            gpuErrchk(cudaFree(symV2PcFragTaskflowValueBlockPool[b].ptr));
    symV2PcFragTaskflowValueBlockPool.clear();
    for (size_t b = 0;
         b < symV2PcFragTaskflowGroupIndexBlockPool.size(); ++b)
        if (symV2PcFragTaskflowGroupIndexBlockPool[b].ptr != NULL)
            gpuErrchk(cudaFree(
                symV2PcFragTaskflowGroupIndexBlockPool[b].ptr));
    symV2PcFragTaskflowGroupIndexBlockPool.clear();
    for (size_t b = 0;
         b < symV2PcFragTaskflowGroupValueBlockPool.size(); ++b)
        if (symV2PcFragTaskflowGroupValueBlockPool[b].ptr != NULL)
            gpuErrchk(cudaFree(
                symV2PcFragTaskflowGroupValueBlockPool[b].ptr));
    symV2PcFragTaskflowGroupValueBlockPool.clear();
    for (size_t b = 0; b < symV2PcFragTaskflowPinnedBlockPool.size(); ++b)
        if (symV2PcFragTaskflowPinnedBlockPool[b].ptr != NULL)
            gpuErrchk(cudaFreeHost(
                symV2PcFragTaskflowPinnedBlockPool[b].ptr));
    symV2PcFragTaskflowPinnedBlockPool.clear();
// SYM_V2_PC2_PHASE6_FREE_EXCHANGE_STATES_BEGIN
    for (size_t sx = 0; sx < symV2RowExchangeStates.size(); ++sx)
    {
        if (symV2RowExchangeStates[sx].active)
            ABORT("SymFact V2 row exchange state is still active during free.");
#ifdef HAVE_CUDA
        if (symV2RowExchangeStates[sx].pack_done != NULL)
            gpuErrchk(cudaEventDestroy(symV2RowExchangeStates[sx].pack_done));
        if (symV2RowExchangeStates[sx].d2h_done != NULL)
            gpuErrchk(cudaEventDestroy(symV2RowExchangeStates[sx].d2h_done));
        if (symV2RowExchangeStates[sx].h2d_done != NULL)
            gpuErrchk(cudaEventDestroy(symV2RowExchangeStates[sx].h2d_done));
#endif
    }
    symV2RowExchangeStates.clear();
// SYM_V2_PC2_PHASE6_FREE_EXCHANGE_STATES_END

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

#ifdef HAVE_CUDA
template <typename Ftype>
size_t xLUstruct_t<Ftype>::symV2DelayedGpuMetadataBytes() const
{
    return 0;
}

template <>
inline size_t xLUstruct_t<double>::symV2DelayedGpuMetadataBytes() const
{
    if (options->SymFact != YES || !superlu_acc_offload)
        return 0;

    size_t total = 0;
    auto add_bytes = [&](size_t count, size_t elem_size, const char *what)
    {
        if (count == 0)
            return;
        size_t bytes = xlu_checked_product(count, elem_size, what);
        if (total > std::numeric_limits<size_t>::max() - bytes)
            ABORT("SymFact V2 delayed GPU metadata byte count overflows.");
        total += bytes;
    };
    auto add_host_maps = [&](const std::vector<std::vector<int_t> > &maps,
                             size_t elem_size,
                             const char *what)
    {
        for (size_t i = 0; i < maps.size(); ++i)
            add_bytes(maps[i].size(), elem_size, what);
    };

    add_host_maps(symL2ULocalMapsHost, sizeof(int_t),
                  "SymFact local GPU L2U map");
    add_host_maps(symL2USendMapsHost, sizeof(int_t),
                  "SymFact GPU L2U send map");
    add_host_maps(symL2USendMapsHost, sizeof(double),
                  "SymFact GPU L2U send buffer");

    if (!superlu_sym_v2_pc_fragment_ldl_native())
        add_bytes(symV2PartnerLSendBufPoolCount, sizeof(double),
                  "SymFact V2 partner-L send buffer pool");
    add_bytes(symL2LSendMapPoolCount, sizeof(int_t),
              "SymFact GPU L2L send map pool");
    add_bytes(symV2PartnerLExactSendBufPoolCount, sizeof(double),
              "SymFact V2 exact partner-L send buffer pool");
    add_bytes(symV2PartnerLExactSendMapPoolCount, sizeof(int_t),
              "SymFact V2 exact partner-L send map pool");
    if (!superlu_sym_v2_rowfrag_destination_path())
    {
        add_bytes(symV2RowFragExactSendBufPoolCount, sizeof(double),
                  "SymFact V2 exact row-fragment send buffer pool");
        add_bytes(symV2RowFragExactSendMapPoolCount, sizeof(int_t),
                  "SymFact V2 exact row-fragment send map pool");
    }
    add_bytes(symV2PartnerLRecvMapPoolCount, sizeof(int_t),
              "SymFact V2 partner-L receive map pool");
    add_bytes(symV2RowFragRecvMapPoolCount, sizeof(int_t),
              "SymFact V2 row-fragment receive map pool");
    add_bytes(symV2RowDownSendMapPoolCount, sizeof(int_t),
              "SymFact V2 row-down send map pool");
// SYM_V2_PC2_LAZY_SENDMAP_BYTES_BEGIN
    add_bytes(symV2RowDownSendSegPoolCount,
              sizeof(SymV2RowDownSendSegmentGPU),
              "SymFact V2 row-down send segment pool");
// SYM_V2_PC2_LAZY_SENDMAP_BYTES_END

    return total;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::materializeSymFactGpuMetadata()
{
    return 0;
}

template <>
inline int xLUstruct_t<double>::materializeSymFactGpuMetadata()
{
    if (options->SymFact != YES || !superlu_acc_offload)
        return 0;

    int profile_setup = symV2SetupProfileActive() ? 1 : 0;
    double tPartnerSendGPU = profile_setup ? SuperLU_timer_() : 0.0;

    auto copy_int_pool = [&](int_t **dst,
                             const std::vector<int_t> &host,
                             size_t count,
                             const char *what)
    {
        if (count == 0)
            return;
        if (*dst != NULL)
            ABORT("SymFact V2 delayed GPU metadata pool was already allocated.");
        if (host.size() != count)
            ABORT("SymFact V2 delayed GPU metadata host map size mismatch.");
        gpuErrchk(cudaMalloc(
            (void **)dst,
            xlu_checked_product(count, sizeof(int_t), what)));
        gpuErrchk(cudaMemcpy(*dst, host.data(), sizeof(int_t) * count,
                             cudaMemcpyHostToDevice));
    };

    if (!symL2ULocalMapsHost.empty())
    {
        if (symL2ULocalMapsGPU.size() != symL2ULocalMapsHost.size())
            ABORT("SymFact local GPU L2U map table size mismatch.");
        for (size_t lk = 0; lk < symL2ULocalMapsHost.size(); ++lk)
        {
            const std::vector<int_t> &map = symL2ULocalMapsHost[lk];
            if (map.empty())
                continue;
            if (symL2ULocalMapsGPU[lk] != NULL)
                ABORT("SymFact local GPU L2U map was already allocated.");
            gpuErrchk(cudaMalloc(
                (void **)&symL2ULocalMapsGPU[lk],
                xlu_checked_product(map.size(), sizeof(int_t),
                                    "SymFact local GPU L2U map")));
            gpuErrchk(cudaMemcpy(symL2ULocalMapsGPU[lk], map.data(),
                                 sizeof(int_t) * map.size(),
                                 cudaMemcpyHostToDevice));
        }
        symL2ULocalMapsHost.clear();
    }

    if (!symL2USendMapsHost.empty())
    {
        if (symL2USendBufsGPU.size() != symL2USendMapsHost.size() ||
            symL2USendMapsGPU.size() != symL2USendMapsHost.size())
            ABORT("SymFact GPU L2U send map table size mismatch.");
        for (size_t flat = 0; flat < symL2USendMapsHost.size(); ++flat)
        {
            const std::vector<int_t> &map = symL2USendMapsHost[flat];
            if (map.empty())
                continue;
            if (symL2USendBufsGPU[flat] != NULL ||
                symL2USendMapsGPU[flat] != NULL)
                ABORT("SymFact GPU L2U send map was already allocated.");
            gpuErrchk(cudaMalloc(
                (void **)&symL2USendBufsGPU[flat],
                xlu_checked_product(map.size(), sizeof(double),
                                    "SymFact GPU L2U send buffer")));
            gpuErrchk(cudaMalloc(
                (void **)&symL2USendMapsGPU[flat],
                xlu_checked_product(map.size(), sizeof(int_t),
                                    "SymFact GPU L2U send map")));
            gpuErrchk(cudaMemcpy(symL2USendMapsGPU[flat], map.data(),
                                 sizeof(int_t) * map.size(),
                                 cudaMemcpyHostToDevice));
        }
        symL2USendMapsHost.clear();
    }

    if (symL2LSendMapPoolCount > 0 &&
        superlu_sym_v2_pc_fragment_ldl_native())
    {
        if (symV2PartnerLPackedMaps.size() != symL2LSendMapPoolCount)
            ABORT("SymFact V2 partner-L send map pool size mismatch.");
        if (symV2PartnerLSendBufPoolGPU != NULL ||
            symL2LSendMapPoolGPU != NULL)
            ABORT("SymFact V2 partner-L send map pool was already allocated.");
        copy_int_pool(&symL2LSendMapPoolGPU, symV2PartnerLPackedMaps,
                      symL2LSendMapPoolCount,
                      "SymFact GPU L2L send map pool");
        for (size_t flat = 0; flat < symV2PartnerLSendSizes.size(); ++flat)
        {
            int size = symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            if (flat >= symV2PartnerLMapOffsets.size())
                ABORT("SymFact V2 partner-L send offset is missing.");
            size_t offset = symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    symL2LSendMapPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner-L send offset is invalid.");
            symL2LSendMapsGPU[flat] = symL2LSendMapPoolGPU + offset;
        }
    }
    else if (symV2PartnerLSendBufPoolCount > 0)
    {
        if (symV2PartnerLPackedMaps.size() != symL2LSendMapPoolCount ||
            symV2PartnerLSendBufPoolCount != symL2LSendMapPoolCount)
            ABORT("SymFact V2 partner-L send pool size mismatch.");
        if (symV2PartnerLSendBufPoolGPU != NULL ||
            symL2LSendMapPoolGPU != NULL)
            ABORT("SymFact V2 partner-L send pool was already allocated.");
        gpuErrchk(cudaMalloc(
            (void **)&symV2PartnerLSendBufPoolGPU,
            xlu_checked_product(symV2PartnerLSendBufPoolCount,
                                sizeof(double),
                                "SymFact V2 partner-L send buffer pool")));
        copy_int_pool(&symL2LSendMapPoolGPU, symV2PartnerLPackedMaps,
                      symL2LSendMapPoolCount,
                      "SymFact GPU L2L send map pool");
        for (size_t flat = 0; flat < symV2PartnerLSendSizes.size(); ++flat)
        {
            int size = symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            if (flat >= symV2PartnerLMapOffsets.size())
                ABORT("SymFact V2 partner-L send offset is missing.");
            size_t offset = symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    symV2PartnerLSendBufPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner-L send offset is invalid.");
            symV2PartnerLSendBufsGPU[flat] =
                symV2PartnerLSendBufPoolGPU + offset;
            symL2LSendMapsGPU[flat] = symL2LSendMapPoolGPU + offset;
        }
    }

    if (symV2PartnerLExactSendBufPoolCount > 0)
    {
        if (symV2PartnerLExactSendMapsHost.size() !=
                symV2PartnerLExactSendMapPoolCount ||
            symV2PartnerLExactSendBufPoolCount !=
                symV2PartnerLExactSendMapPoolCount)
            ABORT("SymFact V2 exact partner-L send pool size mismatch.");
        if (symV2PartnerLExactSendBufPoolGPU != NULL ||
            symV2PartnerLExactSendMapPoolGPU != NULL)
            ABORT("SymFact V2 exact partner-L send pool was already allocated.");
        gpuErrchk(cudaMalloc(
            (void **)&symV2PartnerLExactSendBufPoolGPU,
            xlu_checked_product(symV2PartnerLExactSendBufPoolCount,
                                sizeof(double),
                                "SymFact V2 exact partner-L send buffer pool")));
        copy_int_pool(&symV2PartnerLExactSendMapPoolGPU,
                      symV2PartnerLExactSendMapsHost,
                      symV2PartnerLExactSendMapPoolCount,
                      "SymFact V2 exact partner-L send map pool");
        for (size_t slot = 0; slot < symV2PartnerLExactSendSizes.size();
             ++slot)
        {
            int size = symV2PartnerLExactSendSizes[slot];
            if (size <= 0)
                continue;
            if (slot >= symV2PartnerLExactSendMapOffsets.size())
                ABORT("SymFact V2 exact partner-L send offset is missing.");
            size_t offset = symV2PartnerLExactSendMapOffsets[slot];
            if (offset + static_cast<size_t>(size) >
                    symV2PartnerLExactSendBufPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 exact partner-L send offset is invalid.");
            symV2PartnerLExactSendBufsGPU[slot] =
                symV2PartnerLExactSendBufPoolGPU + offset;
            symV2PartnerLExactSendMapsGPU[slot] =
                symV2PartnerLExactSendMapPoolGPU + offset;
        }
    }

    if (!superlu_sym_v2_rowfrag_destination_path() &&
        symV2RowFragExactSendMapPoolCount > 0)
    {
        if (symV2RowFragExactSendMapsHost.size() !=
                symV2RowFragExactSendMapPoolCount ||
            symV2RowFragExactSendBufPoolCount !=
                symV2RowFragExactSendMapPoolCount)
            ABORT("SymFact V2 exact row-fragment send pool size mismatch.");
        if (symV2RowFragExactSendBufPoolGPU != NULL ||
            symV2RowFragExactSendMapPoolGPU != NULL)
            ABORT("SymFact V2 exact row-fragment send pool was already allocated.");
        gpuErrchk(cudaMalloc(
            (void **)&symV2RowFragExactSendBufPoolGPU,
            xlu_checked_product(symV2RowFragExactSendBufPoolCount,
                                sizeof(double),
                                "SymFact V2 exact row-fragment send buffer pool")));
        copy_int_pool(&symV2RowFragExactSendMapPoolGPU,
                      symV2RowFragExactSendMapsHost,
                      symV2RowFragExactSendMapPoolCount,
                      "SymFact V2 exact row-fragment send map pool");
        for (size_t slot = 0; slot < symV2RowFragExactSendSizes.size();
             ++slot)
        {
            int size = symV2RowFragExactSendSizes[slot];
            if (size <= 0)
                continue;
            if (slot >= symV2RowFragExactSendMapOffsets.size())
                ABORT("SymFact V2 exact row-fragment send offset is missing.");
            size_t offset = symV2RowFragExactSendMapOffsets[slot];
            if (offset + static_cast<size_t>(size) >
                    symV2RowFragExactSendBufPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 exact row-fragment send offset is invalid.");
            symV2RowFragExactSendBufsGPU[slot] =
                symV2RowFragExactSendBufPoolGPU + offset;
            symV2RowFragExactSendMapsGPU[slot] =
                symV2RowFragExactSendMapPoolGPU + offset;
        }
    }

    if (symV2RowDownSendMapPoolCount > 0)
    {
        copy_int_pool(&symV2RowDownSendMapPoolGPU,
                      symV2RowDownSendMapsHost,
                      symV2RowDownSendMapPoolCount,
                      "SymFact V2 row-down send map pool");
        if (symV2RowDownSendMapsGPU.size() !=
            symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send map table size mismatch.");
        for (size_t slot = 0; slot < symV2RowDownSendSizes.size(); ++slot)
        {
            int size = symV2RowDownSendSizes[slot];
            if (size <= 0)
                continue;
            size_t offset = symV2RowDownSendMapOffsets[slot];
            if (offset + static_cast<size_t>(size) >
                    symV2RowDownSendMapPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 row-down send map offset is invalid.");
            symV2RowDownSendMapsGPU[slot] =
                symV2RowDownSendMapPoolGPU + offset;
        }
    }
// SYM_V2_PC2_LAZY_SENDMAP_MATERIALIZE_BEGIN
    if (symV2RowDownSendSegPoolCount > 0)
    {
        if (symV2RowDownSendSegPoolGPU != NULL)
            ABORT("SymFact V2 row-down send segment pool was already allocated.");
        if (symV2RowDownSendSegsHost.size() !=
            symV2RowDownSendSegPoolCount)
            ABORT("SymFact V2 row-down send segment pool size mismatch.");
        gpuErrchk(cudaMalloc(
            (void **)&symV2RowDownSendSegPoolGPU,
            xlu_checked_product(symV2RowDownSendSegPoolCount,
                                sizeof(SymV2RowDownSendSegmentGPU),
                                "SymFact V2 row-down send segment pool")));
        gpuErrchk(cudaMemcpy(symV2RowDownSendSegPoolGPU,
                             symV2RowDownSendSegsHost.data(),
                             sizeof(SymV2RowDownSendSegmentGPU) *
                                 symV2RowDownSendSegPoolCount,
                             cudaMemcpyHostToDevice));
        if (symV2RowDownSendSegsGPU.size() !=
                symV2RowDownSendSizes.size() ||
            symV2RowDownSendSegOffsets.size() !=
                symV2RowDownSendSizes.size() ||
            symV2RowDownSendSegCounts.size() !=
                symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send segment table size mismatch.");
        for (size_t slot = 0; slot < symV2RowDownSendSizes.size(); ++slot)
        {
            if (symV2RowDownSendSizes[slot] <= 0)
                continue;
            int count = symV2RowDownSendSegCounts[slot];
            if (count <= 0)
                ABORT("SymFact V2 row-down send segment slot has no descriptors.");
            size_t offset = symV2RowDownSendSegOffsets[slot];
            if (offset + static_cast<size_t>(count) >
                    symV2RowDownSendSegPoolCount ||
                offset + static_cast<size_t>(count) < offset)
                ABORT("SymFact V2 row-down send segment offset is invalid.");
            symV2RowDownSendSegsGPU[slot] =
                symV2RowDownSendSegPoolGPU + offset;
        }
    }
// SYM_V2_PC2_LAZY_SENDMAP_MATERIALIZE_END

    auto materialize_recv_maps =
        [&](const std::vector<std::vector<int_t> > &maps,
            const std::vector<size_t> &offsets,
            std::vector<int_t *> &gpu_maps,
            int_t **pool_gpu,
            size_t pool_count,
            const char *what)
    {
        if (pool_count == 0)
            return;
        if (maps.size() != offsets.size() || maps.size() != gpu_maps.size())
            ABORT("SymFact V2 receive map table size mismatch.");
        std::vector<int_t> packed(pool_count, 0);
        for (size_t pos = 0; pos < maps.size(); ++pos)
        {
            if (maps[pos].empty())
                continue;
            size_t offset = offsets[pos];
            if (offset + maps[pos].size() > pool_count ||
                offset + maps[pos].size() < offset)
                ABORT("SymFact V2 receive map offset is invalid.");
            std::copy(maps[pos].begin(), maps[pos].end(),
                      packed.begin() + offset);
        }
        copy_int_pool(pool_gpu, packed, pool_count, what);
        for (size_t pos = 0; pos < maps.size(); ++pos)
        {
            if (!maps[pos].empty())
                gpu_maps[pos] = *pool_gpu + offsets[pos];
        }
    };

    materialize_recv_maps(symV2PartnerLRecvMap,
                          symV2PartnerLRecvMapOffsets,
                          symV2PartnerLRecvMapsGPU,
                          &symV2PartnerLRecvMapPoolGPU,
                          symV2PartnerLRecvMapPoolCount,
                          "SymFact V2 partner-L receive map pool");
    materialize_recv_maps(symV2RowFragRecvMap,
                          symV2RowFragRecvMapOffsets,
                          symV2RowFragRecvMapsGPU,
                          &symV2RowFragRecvMapPoolGPU,
                          symV2RowFragRecvMapPoolCount,
                          "SymFact V2 row-fragment receive map pool");

    if (!superlu_cuda_aware_mpi())
    {
        if (superlu_sym_v2_pinned_staging())
        {
            if (superlu_sym_v2_pinned_staging_pool() &&
                symV2PartnerLHostSendPoolPinnedCount > 0)
            {
                if (symV2PartnerLHostSendPoolPinned != NULL)
                    ABORT("SymFact V2 pooled pinned send staging was already allocated.");
                gpuErrchk(cudaMallocHost(
                    (void **)&symV2PartnerLHostSendPoolPinned,
                    xlu_checked_product(symV2PartnerLHostSendPoolPinnedCount,
                                        sizeof(double),
                                        "SymFact V2 pooled pinned send staging")));
                for (size_t flat = 0; flat < symV2PartnerLSendSizes.size();
                     ++flat)
                {
                    int size = symV2PartnerLSendSizes[flat];
                    if (size <= 0)
                        continue;
                    size_t offset =
                        symV2PartnerLHostSendScratchOffsets[flat];
                    if (offset + static_cast<size_t>(size) >
                            symV2PartnerLHostSendPoolPinnedCount ||
                        offset + static_cast<size_t>(size) < offset)
                        ABORT("SymFact V2 pooled send staging map is invalid.");
                    symV2PartnerLHostSendBufsPinned[flat] =
                        symV2PartnerLHostSendPoolPinned + offset;
                }
            }
            else if (!superlu_sym_v2_pinned_staging_pool())
            {
                for (size_t flat = 0; flat < symV2PartnerLSendSizes.size();
                     ++flat)
                {
                    if (symV2PartnerLSendSizes[flat] <= 0)
                        continue;
                    if (symV2PartnerLHostSendBufsPinned[flat] != NULL)
                        ABORT("SymFact V2 pinned send staging was already allocated.");
                    gpuErrchk(cudaMallocHost(
                        (void **)&symV2PartnerLHostSendBufsPinned[flat],
                        xlu_checked_product(
                            static_cast<size_t>(symV2PartnerLSendSizes[flat]),
                            sizeof(double),
                            "SymFact V2 pinned send staging")));
                }
            }

            for (size_t slot = 0;
                 slot < symV2PartnerLExactSendSizes.size(); ++slot)
            {
                if (symV2PartnerLExactSendSizes[slot] <= 0)
                    continue;
                if (symV2PartnerLExactHostSendBufsPinned[slot] != NULL)
                    ABORT("SymFact V2 exact pinned send staging was already allocated.");
                gpuErrchk(cudaMallocHost(
                    (void **)&symV2PartnerLExactHostSendBufsPinned[slot],
                    xlu_checked_product(
                        static_cast<size_t>(
                            symV2PartnerLExactSendSizes[slot]),
                        sizeof(double),
                        "SymFact V2 exact partner-L pinned send staging")));
            }
            for (size_t slot = 0;
                 slot < symV2RowFragExactSendSizes.size(); ++slot)
            {
                if (symV2RowFragExactSendSizes[slot] <= 0)
                    continue;
                if (symV2RowFragExactHostSendBufsPinned[slot] != NULL)
                    ABORT("SymFact V2 exact row-fragment pinned send staging was already allocated.");
                gpuErrchk(cudaMallocHost(
                    (void **)&symV2RowFragExactHostSendBufsPinned[slot],
                    xlu_checked_product(
                        static_cast<size_t>(
                            symV2RowFragExactSendSizes[slot]),
                        sizeof(double),
                        "SymFact V2 exact row-fragment pinned send staging")));
            }
        }
        else
        {
            for (size_t slot = 0;
                 slot < symV2PartnerLExactSendSizes.size(); ++slot)
            {
                if (symV2PartnerLExactSendSizes[slot] > 0 &&
                    symV2PartnerLExactHostSendBufs[slot].empty())
                    symV2PartnerLExactHostSendBufs[slot].resize(
                        static_cast<size_t>(
                            symV2PartnerLExactSendSizes[slot]));
            }
            for (size_t slot = 0;
                 slot < symV2RowFragExactSendSizes.size(); ++slot)
            {
                if (symV2RowFragExactSendSizes[slot] > 0 &&
                    symV2RowFragExactHostSendBufs[slot].empty())
                    symV2RowFragExactHostSendBufs[slot].resize(
                        static_cast<size_t>(
                            symV2RowFragExactSendSizes[slot]));
            }
        }
    }

    if (profile_setup)
        symV2SetupProfileAdd(SYM_V2_SETUP_PARTNER_SEND_GPU_ALLOC_COPY,
                             SuperLU_timer_() - tPartnerSendGPU);

    if (!superlu_sym_v2_pc_fragment_ldl_native())
        std::vector<int_t>().swap(symV2PartnerLPackedMaps);
    std::vector<int_t>().swap(symV2PartnerLExactSendMapsHost);
    if (symV2RowDownSendMapPoolGPU != NULL)
        std::vector<int_t>().swap(symV2RowDownSendMapsHost);
// SYM_V2_PC2_LAZY_SENDMAP_HOST_RELEASE_BEGIN
    if (symV2RowDownSendSegPoolGPU != NULL)
        std::vector<SymV2RowDownSendSegmentGPU>().swap(
            symV2RowDownSendSegsHost);
// SYM_V2_PC2_LAZY_SENDMAP_HOST_RELEASE_END

    return 0;
}
#endif

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
