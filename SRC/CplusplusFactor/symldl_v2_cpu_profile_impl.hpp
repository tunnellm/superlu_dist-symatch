#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "xlupanels.hpp"

struct SymLDLV2CpuCapacitySnapshot
{
    uint64_t hash;
    uint64_t total;
};

static inline void symldl_v2_cpu_capacity_mix(
    SymLDLV2CpuCapacitySnapshot *snapshot, size_t capacity)
{
    snapshot->hash ^= static_cast<uint64_t>(capacity) +
                      UINT64_C(0x9e3779b97f4a7c15) +
                      (snapshot->hash << 6) + (snapshot->hash >> 2);
    snapshot->total += static_cast<uint64_t>(capacity);
}

template <typename T>
static inline void symldl_v2_cpu_capacity_mix_vector(
    SymLDLV2CpuCapacitySnapshot *snapshot, const std::vector<T> &values)
{
    symldl_v2_cpu_capacity_mix(snapshot, values.capacity());
}

template <typename Ftype>
static SymLDLV2CpuCapacitySnapshot symldl_v2_cpu_capacity_snapshot(
    const xLUstruct_t<Ftype> *lu)
{
    SymLDLV2CpuCapacitySnapshot snapshot = {
        UINT64_C(0xcbf29ce484222325), 0
    };
#define SYM_LDL_V2_CPU_MIX_CAPACITY(name) \
    symldl_v2_cpu_capacity_mix_vector(&snapshot, lu->name)
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRawPanelBufs);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerSendBufs);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerRecvBufs);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowSendBufs);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowRecvBufs);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerSegOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerSendOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerSendSizes);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerRecvSizes);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerSegments);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerRowPermutations);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowSegOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowSendOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowSendSizes);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowSegments);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRowPermutations);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRequests);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerRecvOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuRequestPeers);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuWaitIndices);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuWaitStatuses);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerRecvChunksRemaining);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuPartnerUpdateSubmitted);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuSlotRequestCounts);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuSlotSendBegins);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuReductionPanelSlots);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuReductionChunksRemaining);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuThreadProfiles);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2CpuOutputLockOffsets);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2PartnerLRecvIndexBySrc);
    for (size_t i = 0; i < lu->symV2PartnerLRecvIndexBySrc.size(); ++i)
        symldl_v2_cpu_capacity_mix_vector(
            &snapshot, lu->symV2PartnerLRecvIndexBySrc[i]);
    SYM_LDL_V2_CPU_MIX_CAPACITY(symV2RowFragRecvIndex);
    for (size_t i = 0; i < lu->symV2RowFragRecvIndex.size(); ++i)
        symldl_v2_cpu_capacity_mix_vector(
            &snapshot, lu->symV2RowFragRecvIndex[i]);
#undef SYM_LDL_V2_CPU_MIX_CAPACITY
    return snapshot;
}

template <typename Ftype>
static uint64_t symldl_v2_cpu_count_outstanding_requests(
    const xLUstruct_t<Ftype> *lu)
{
    uint64_t count = 0;
    for (size_t i = 0; i < lu->symV2CpuRequests.size(); ++i)
        if (lu->symV2CpuRequests[i] != MPI_REQUEST_NULL)
            ++count;
    return count;
}

template <typename Ftype>
static void symldl_v2_cpu_profile_print(xLUstruct_t<Ftype> *lu)
{
    enum { timer_count = 28, counter_count = 27 };
    double local_timers[timer_count] = {
        lu->symV2CpuPlanBuildTime,
        lu->symV2CpuWorkspaceInitTime,
        lu->symV2CpuSchedulerTime,
        lu->symV2CpuPanelIssueTime,
        lu->symV2CpuPanelExchangeTime,
        lu->symV2CpuFragmentExchangeTime,
        lu->symV2CpuPartnerPackTime,
        lu->symV2CpuRowPackTime,
        lu->symV2CpuRecvPostTime,
        lu->symV2CpuRecvProgressTime,
        lu->symV2CpuRecvWaitTime,
        lu->symV2CpuSendPostTime,
        lu->symV2CpuSendDrainTime,
        lu->symV2CpuFragmentAssemblyTime,
        lu->symV2CpuDiagFactorTime,
        lu->symV2CpuInvDiagCommTime,
        lu->symV2CpuWTransformTime,
        lu->symV2CpuSchurTime,
        lu->symV2CpuGemmTime,
        lu->symV2CpuLookaheadGemmTime,
        lu->symV2CpuExcludeGemmTime,
        lu->symV2CpuDirectScatterTime,
        lu->symV2CpuMappedScatterTime,
        lu->symV2CpuScatterLockWaitTime,
        lu->symV2CpuSchedulerIdleTime,
        lu->symV2CpuSlotBackpressureTime,
        lu->symV2CpuReductionTime,
        lu->SCT->pdgstrfTimer
    };
    double max_timers[timer_count] = {0.0};
    MPI_Reduce(local_timers, max_timers, timer_count, MPI_DOUBLE, MPI_MAX, 0,
               lu->grid3d->comm);

    unsigned long long local_counters[counter_count] = {
        lu->symV2CpuPanelsIssued,
        lu->symV2CpuPanelsCompleted,
        lu->symV2CpuSchurTasks,
        lu->symV2CpuGroupedGemms,
        lu->symV2CpuGemmFlops,
        lu->symV2CpuDirectScatters,
        lu->symV2CpuMappedScatters,
        lu->symV2CpuLookaheadTasks,
        lu->symV2CpuExcludeTasks,
        lu->symV2CpuOutputLockAttempts,
        lu->symV2CpuOutputLockConflicts,
        lu->symV2CpuPartnerBytes,
        lu->symV2CpuRowBytes,
        lu->symV2CpuInvDiagBytes,
        lu->symV2CpuReductionBytes,
        lu->symV2CpuSlotBackpressureEvents,
        lu->symV2CpuMpiTestsomeCalls,
        lu->symV2CpuMpiWaitsomeCalls,
        lu->symV2CpuMpiCompletions,
        lu->symV2CpuSendDrainCalls,
        lu->symV2CpuOversizedMpiChunks,
        lu->symV2CpuRuntimeAllocations,
        lu->symV2CpuRuntimeVectorGrowths,
        lu->symV2CpuOutstandingRequests,
        lu->symV2CpuTaskBatches,
        lu->symV2CpuDeferredTaskBatches,
        lu->symV2CpuInlineTaskBatches
    };
    unsigned long long sum_counters[counter_count] = {0};
    MPI_Reduce(local_counters, sum_counters, counter_count,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, lu->grid3d->comm);
    unsigned long long local_high_water = lu->symV2CpuActiveSlotHighWater;
    unsigned long long max_high_water = 0;
    MPI_Reduce(&local_high_water, &max_high_water, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, lu->grid3d->comm);

    unsigned long long local_shapes[17] = {0};
    unsigned long long local_shape_max[3] = {0};
    for (size_t thread = 0; thread < lu->symV2CpuThreadProfiles.size();
         ++thread)
    {
        const SymLDLV2CpuThreadProfile &profile =
            lu->symV2CpuThreadProfiles[thread];
        local_shapes[0] += profile.gemms;
        local_shapes[1] += profile.small_gemms;
        local_shapes[2] += profile.medium_gemms;
        local_shapes[3] += profile.large_gemms;
        local_shapes[4] += profile.m_sum;
        local_shapes[5] += profile.n_sum;
        local_shapes[6] += profile.k_sum;
        local_shapes[7] += profile.mapped_scatter_values;
        local_shapes[8] += profile.mapped_row_exact;
        local_shapes[9] += profile.mapped_row_contiguous;
        local_shapes[10] += profile.mapped_column_full;
        local_shapes[11] += profile.mapped_column_contiguous;
        local_shapes[12] += profile.mapped_rectangular;
        local_shapes[13] += profile.mapped_sorted_rows;
        local_shapes[14] += profile.mapped_destination_full;
        local_shapes[15] += profile.mapped_row_contiguous_values;
        local_shapes[16] += profile.mapped_rectangular_values;
        local_shape_max[0] = SUPERLU_MAX(
            local_shape_max[0],
            static_cast<unsigned long long>(profile.max_m));
        local_shape_max[1] = SUPERLU_MAX(
            local_shape_max[1],
            static_cast<unsigned long long>(profile.max_n));
        local_shape_max[2] = SUPERLU_MAX(
            local_shape_max[2],
            static_cast<unsigned long long>(profile.max_k));
    }
    unsigned long long sum_shapes[17] = {0};
    unsigned long long max_shapes[3] = {0};
    MPI_Reduce(local_shapes, sum_shapes, 17, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, lu->grid3d->comm);
    MPI_Reduce(local_shape_max, max_shapes, 3, MPI_UNSIGNED_LONG_LONG,
               MPI_MAX, 0, lu->grid3d->comm);
    int max_workers = 1;
    MPI_Reduce(&lu->symV2CpuWorkerCount, &max_workers, 1, MPI_INT,
               MPI_MAX, 0, lu->grid3d->comm);

    if (lu->grid3d->iam != 0)
        return;
    std::printf(
        "SymFact V2 CPU setup profile (max-rank): plan_build=%.6f workspace_init=%.6f factor_total=%.6f\n",
        max_timers[0], max_timers[1], max_timers[27]);
    std::printf(
        "SymFact V2 CPU scheduler profile (max-rank): scheduler=%.6f panel_issue=%.6f idle=%.6f slot_backpressure=%.6f active_slots=%llu\n",
        max_timers[2], max_timers[3], max_timers[24], max_timers[25],
        max_high_water);
    std::printf(
        "SymFact V2 CPU communication profile (max-rank): panel_exchange=%.6f fragment_exchange=%.6f partner_pack=%.6f row_pack=%.6f recv_post=%.6f recv_progress=%.6f recv_wait=%.6f send_post=%.6f send_drain=%.6f assembly=%.6f reduction=%.6f\n",
        max_timers[4], max_timers[5], max_timers[6], max_timers[7],
        max_timers[8], max_timers[9], max_timers[10], max_timers[11],
        max_timers[12], max_timers[13], max_timers[26]);
    std::printf(
        "SymFact V2 CPU compute profile (max-rank): diag=%.6f invdiag_comm=%.6f w_transform=%.6f schur=%.6f gemm=%.6f lookahead_gemm=%.6f exclude_gemm=%.6f direct=%.6f mapped=%.6f lock_wait=%.6f\n",
        max_timers[14], max_timers[15], max_timers[16], max_timers[17],
        max_timers[18], max_timers[19], max_timers[20], max_timers[21],
        max_timers[22], max_timers[23]);
    std::printf(
        "SymFact V2 CPU work counters (sum): panels=%llu/%llu tasks=%llu task_batches=%llu deferred_batches=%llu inline_batches=%llu gemms=%llu gemm_flops=%llu direct=%llu mapped=%llu lookahead=%llu exclude=%llu lock_attempts=%llu lock_conflicts=%llu\n",
        sum_counters[0], sum_counters[1], sum_counters[2], sum_counters[24],
        sum_counters[25], sum_counters[26],
        sum_counters[3], sum_counters[4], sum_counters[5], sum_counters[6],
        sum_counters[7], sum_counters[8], sum_counters[9], sum_counters[10]);
    double gemm_count = static_cast<double>(sum_shapes[0]);
    std::printf(
        "SymFact V2 CPU GEMM profile (sum/max): count=%llu small=%llu medium=%llu large=%llu avg_m=%.2f avg_n=%.2f avg_k=%.2f max_m=%llu max_n=%llu max_k=%llu\n",
        sum_shapes[0], sum_shapes[1], sum_shapes[2], sum_shapes[3],
        gemm_count > 0.0 ? sum_shapes[4] / gemm_count : 0.0,
        gemm_count > 0.0 ? sum_shapes[5] / gemm_count : 0.0,
        gemm_count > 0.0 ? sum_shapes[6] / gemm_count : 0.0,
        max_shapes[0], max_shapes[1], max_shapes[2]);
    std::printf(
        "SymFact V2 CPU mapped-scatter profile (sum): values=%llu row_exact=%llu row_contiguous=%llu column_full=%llu column_contiguous=%llu rectangular=%llu sorted_rows=%llu destination_full=%llu row_contiguous_values=%llu rectangular_values=%llu\n",
        sum_shapes[7], sum_shapes[8], sum_shapes[9], sum_shapes[10],
        sum_shapes[11], sum_shapes[12], sum_shapes[13], sum_shapes[14],
        sum_shapes[15], sum_shapes[16]);
    const char *blas_threads = std::getenv("OPENBLAS_NUM_THREADS");
    const char *blas_source = "OPENBLAS_NUM_THREADS";
    if (blas_threads == NULL || blas_threads[0] == '\0')
    {
        blas_threads = std::getenv("MKL_NUM_THREADS");
        blas_source = "MKL_NUM_THREADS";
    }
    if (blas_threads == NULL || blas_threads[0] == '\0')
    {
        blas_threads = "caller-controlled";
        blas_source = "generic-BLAS";
    }
    std::printf(
        "SymFact V2 CPU execution policy: outer_task_workers=%d min_deferred_work=%" PRIu64 " blas_threads=%s source=%s\n",
        max_workers, symldl_v2_cpu_min_deferred_work(), blas_threads,
        blas_source);
    std::printf(
        "SymFact V2 CPU communication counters (sum): partner_bytes=%llu row_bytes=%llu invdiag_bytes=%llu reduction_bytes=%llu backpressure=%llu testsome=%llu waitsome=%llu mpi_completions=%llu send_drains=%llu oversized_chunks=%llu\n",
        sum_counters[11], sum_counters[12], sum_counters[13],
        sum_counters[14], sum_counters[15], sum_counters[16],
        sum_counters[17], sum_counters[18], sum_counters[19],
        sum_counters[20]);
    std::printf(
        "SymFact V2 CPU runtime invariants (sum): allocations=%llu vector_growths=%llu outstanding_requests=%llu\n",
        sum_counters[21], sum_counters[22], sum_counters[23]);
}
