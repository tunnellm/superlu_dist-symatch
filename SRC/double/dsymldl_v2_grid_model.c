#include "dsymldl_v2_grid_model.h"
#include "dsymldl_v2_runtime_model.h"
#include "dsymldl_v2_workspace_size.h"

#ifdef GPU_ACC
#include "gpu_wrapper.h"
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

static void dSymLDLV2SetModelError(char *error, size_t error_size,
                                   const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static uint64_t dSymLDLV2AddBytes(uint64_t left, uint64_t right)
{
    return UINT64_MAX - left < right ? UINT64_MAX : left + right;
}

static uint64_t dSymLDLV2MultiplyBytes(uint64_t left, uint64_t right)
{
    return left != 0 && right > UINT64_MAX / left
               ? UINT64_MAX
               : left * right;
}

static uint64_t dSymLDLV2RuntimeGPUOverhead(uint64_t tracked_bytes)
{
    return SUPERLU_MAX(
        dSymLDLV2MultiplyBytes(256, (uint64_t) 1024 * 1024),
        tracked_bytes / 100);
}

static uint64_t dSymLDLV2MaximumBytes(const uint64_t *values, size_t count)
{
    uint64_t maximum = 0;
    for (size_t i = 0; i < count; ++i)
        if (values[i] > maximum)
            maximum = values[i];
    return maximum;
}

static int dSymLDLV2AllocationFits(size_t count, size_t element_size)
{
    return element_size == 0 || count <= SIZE_MAX / element_size;
}

static int dSymLDLV2CompareIntT(const void *left, const void *right)
{
    int_t lhs = *(const int_t *) left;
    int_t rhs = *(const int_t *) right;
    return lhs < rhs ? -1 : lhs > rhs;
}

static uint64_t dSymLDLV2ReadHostAvailable(void)
{
#ifdef _WIN32
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ?
               (uint64_t) status.ullAvailPhys : 0;
#else
    FILE *file = fopen("/proc/meminfo", "r");
    if (file != NULL)
    {
        char line[256];
        while (fgets(line, sizeof(line), file) != NULL)
        {
            unsigned long long kibibytes;
            if (sscanf(line, "MemAvailable: %llu kB", &kibibytes) == 1)
            {
                fclose(file);
                return dSymLDLV2MultiplyBytes((uint64_t) kibibytes, 1024);
            }
        }
        fclose(file);
    }

    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return 0;
    return dSymLDLV2MultiplyBytes((uint64_t) pages, (uint64_t) page_size);
#endif
}

int dSymLDLV2BuildStructuralSummary(
    int_t nsupers, const Glu_persist_t *persist,
    const Glu_freeable_t *symbolic, dSymLDLV2StructuralSummary *summary,
    char *error, size_t error_size)
{
    const char *invalid_reason = "structure";
    int_t invalid_k = -1;
    int_t invalid_position = -1;
    int_t invalid_row = -1;
    int_t invalid_supernode = -1;
    int_t invalid_previous = -1;
    int_t invalid_begin = -1;
    int_t invalid_end = -1;
    int_t invalid_columns = -1;
    int_t *row_counts = NULL;
    int_t *touched_supernodes = NULL;

    if (nsupers < 1 || persist == NULL || persist->xsup == NULL ||
        persist->supno == NULL || symbolic == NULL ||
        symbolic->xlsub == NULL || symbolic->lsub == NULL || summary == NULL)
    {
        dSymLDLV2SetModelError(error, error_size,
                               "SymLDL structural summary input is incomplete.");
        return 0;
    }

    memset(summary, 0, sizeof(*summary));
    summary->nsupers = nsupers;
    summary->n = persist->xsup[nsupers];
    if (summary->n < 1 || persist->xsup[0] != 0 ||
        (uint64_t) nsupers + 1 > SIZE_MAX / sizeof(int_t))
    {
        invalid_reason = "supernode bounds";
        goto invalid_structure;
    }
    for (int_t k = 0; k < nsupers; ++k)
        if (persist->xsup[k] < 0 ||
            persist->xsup[k + 1] <= persist->xsup[k] ||
            persist->xsup[k + 1] > summary->n)
        {
            invalid_reason = "supernode partition";
            invalid_k = k;
            goto invalid_structure;
        }
    summary->panel_block_offsets = INT_T_ALLOC(nsupers + 1);
    summary->panel_rows = INT_T_ALLOC(nsupers);
    row_counts = intCalloc_dist(nsupers);
    touched_supernodes = INT_T_ALLOC(nsupers);
    if (summary->panel_block_offsets == NULL || summary->panel_rows == NULL ||
        row_counts == NULL || touched_supernodes == NULL)
        goto allocation_failed;

    int_t block_count = 0;
    int_t maximum_int_t = sizeof(int_t) == sizeof(int64_t)
                              ? (int_t) INT64_MAX
                              : (int_t) INT32_MAX;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t first_column = persist->xsup[k];
        int_t begin = symbolic->xlsub[first_column];
        int_t end = symbolic->xlsub[first_column + 1];
        int_t columns = persist->xsup[k + 1] - persist->xsup[k];
        int_t touched_count = 0;
        invalid_k = k;
        invalid_begin = begin;
        invalid_end = end;
        invalid_columns = columns;
        if (begin < 0 || end < begin || end - begin < columns)
        {
            invalid_reason = "panel bounds";
            goto invalid_structure;
        }
        summary->panel_block_offsets[k] = block_count;
        summary->panel_rows[k] = SUPERLU_MAX((int_t) 0, end - begin);
        summary->symbolic_row_entries = dSymLDLV2AddBytes(
            summary->symbolic_row_entries,
            (uint64_t) summary->panel_rows[k]);
        summary->factor_value_entries = dSymLDLV2AddBytes(
            summary->factor_value_entries,
            dSymLDLV2MultiplyBytes(
                (uint64_t) summary->panel_rows[k],
                (uint64_t) (persist->xsup[k + 1] - persist->xsup[k])));
        if (summary->symbolic_row_entries == UINT64_MAX ||
            summary->factor_value_entries == UINT64_MAX)
            goto summary_overflow;
        for (int_t position = begin; position < end; ++position)
        {
            int_t row = symbolic->lsub[position];
            if (row < 0 || row >= summary->n)
            {
                invalid_reason = "row bounds";
                invalid_k = k;
                invalid_position = position;
                invalid_row = row;
                goto invalid_structure;
            }
            int_t row_supernode = persist->supno[row];
            if (row_supernode < k || row_supernode >= nsupers)
            {
                invalid_reason = "row supernode";
                invalid_k = k;
                invalid_position = position;
                invalid_row = row;
                invalid_supernode = row_supernode;
                goto invalid_structure;
            }
            if (row_counts[row_supernode] == 0)
            {
                touched_supernodes[touched_count++] = row_supernode;
            }
            if (row_counts[row_supernode] == maximum_int_t)
            {
                invalid_reason = "row count";
                invalid_position = position;
                invalid_row = row;
                invalid_supernode = row_supernode;
                goto invalid_structure;
            }
            ++row_counts[row_supernode];
        }
        if (row_counts[k] < columns)
        {
            invalid_reason = "diagonal rows";
            invalid_supernode = k;
            goto invalid_structure;
        }
        if (touched_count > maximum_int_t - block_count)
        {
            invalid_reason = "block count";
            goto invalid_structure;
        }
        block_count += touched_count;
        for (int_t i = 0; i < touched_count; ++i)
            row_counts[touched_supernodes[i]] = 0;
    }
    summary->panel_block_offsets[nsupers] = block_count;
    summary->block_count = block_count;
    if ((uint64_t) block_count > SIZE_MAX / sizeof(int_t))
        goto summary_overflow;
    summary->block_row_supernode = INT_T_ALLOC(block_count);
    summary->block_row_count = INT_T_ALLOC(block_count);
    if (block_count > 0 && (summary->block_row_supernode == NULL ||
                            summary->block_row_count == NULL))
        goto allocation_failed;

    int_t block = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t first_column = persist->xsup[k];
        int_t begin = symbolic->xlsub[first_column];
        int_t end = symbolic->xlsub[first_column + 1];
        int_t touched_count = 0;
        for (int_t position = begin; position < end; ++position)
        {
            int_t row_supernode = persist->supno[symbolic->lsub[position]];
            if (row_counts[row_supernode] == 0)
                touched_supernodes[touched_count++] = row_supernode;
            ++row_counts[row_supernode];
        }
        qsort(touched_supernodes, (size_t) touched_count, sizeof(int_t),
              dSymLDLV2CompareIntT);
        for (int_t i = 0; i < touched_count; ++i)
        {
            int_t row_supernode = touched_supernodes[i];
            summary->block_row_supernode[block] = row_supernode;
            summary->block_row_count[block] = row_counts[row_supernode];
            row_counts[row_supernode] = 0;
            ++block;
        }
    }
    if (block != block_count)
    {
        invalid_reason = "block reconstruction";
        goto invalid_structure;
    }
    SUPERLU_FREE(row_counts);
    SUPERLU_FREE(touched_supernodes);
    return 1;

summary_overflow:
    SUPERLU_FREE(row_counts);
    SUPERLU_FREE(touched_supernodes);
    dSymLDLV2StructuralSummaryDestroy(summary);
    dSymLDLV2SetModelError(error, error_size,
                           "SymLDL structural summary size overflows.");
    return 0;
invalid_structure:
    SUPERLU_FREE(row_counts);
    SUPERLU_FREE(touched_supernodes);
    dSymLDLV2StructuralSummaryDestroy(summary);
    if (error != NULL && error_size > 0)
        snprintf(error, error_size,
                 "SymLDL structural summary is inconsistent (%s, panel=%lld, position=%lld, row=%lld, supernode=%lld, previous=%lld, range=[%lld,%lld), columns=%lld).",
                 invalid_reason, (long long) invalid_k,
                 (long long) invalid_position, (long long) invalid_row,
                 (long long) invalid_supernode, (long long) invalid_previous,
                 (long long) invalid_begin, (long long) invalid_end,
                 (long long) invalid_columns);
    return 0;
allocation_failed:
    SUPERLU_FREE(row_counts);
    SUPERLU_FREE(touched_supernodes);
    dSymLDLV2StructuralSummaryDestroy(summary);
    dSymLDLV2SetModelError(error, error_size,
                           "SymLDL structural summary allocation failed.");
    return 0;
}

void dSymLDLV2StructuralSummaryDestroy(
    dSymLDLV2StructuralSummary *summary)
{
    if (summary == NULL)
        return;
    SUPERLU_FREE(summary->panel_block_offsets);
    SUPERLU_FREE(summary->block_row_supernode);
    SUPERLU_FREE(summary->block_row_count);
    SUPERLU_FREE(summary->panel_rows);
    memset(summary, 0, sizeof(*summary));
}

int dSymLDLV2DetectAvailableMemory(MPI_Comm communicator,
                                    dSymLDLV2AvailableMemory *available,
                                    char *error, size_t error_size)
{
    MPI_Comm node_communicator = MPI_COMM_NULL;
    int local_ranks = 1;
    int maximum_local_ranks = 1;
    int local_ranks_per_gpu = 1;
    int maximum_ranks_per_gpu = 1;
    uint64_t host_available = dSymLDLV2ReadHostAvailable();
    uint64_t minimum_host_available = 0;
    uint64_t gpu_available = 0;
    uint64_t minimum_gpu_available = 0;
    int gpu_known = 0;
    int all_gpu_known = 0;

    if (available == NULL)
    {
        dSymLDLV2SetModelError(error, error_size,
                               "Available-memory result is missing.");
        return 0;
    }
    memset(available, 0, sizeof(*available));
    if (MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &node_communicator) != MPI_SUCCESS)
    {
        dSymLDLV2SetModelError(error, error_size,
                               "Node-local communicator creation failed.");
        return 0;
    }
    MPI_Comm_size(node_communicator, &local_ranks);
    MPI_Allreduce(&local_ranks, &maximum_local_ranks, 1, MPI_INT, MPI_MAX,
                  communicator);
    MPI_Allreduce(&host_available, &minimum_host_available, 1,
                  MPI_UINT64_T, MPI_MIN, communicator);

#ifdef GPU_ACC
    {
#define DSYMLDL_V2_DEVICE_IDENTITY_SIZE 64
        int device_count = 0;
        int device_id = -1;
        int ranks_on_device = 1;
        int device_ready = 0;
        int allocation_ready = 0;
        int all_allocations_ready = 0;
        char device_identity[DSYMLDL_V2_DEVICE_IDENTITY_SIZE] = {0};
        char *node_device_identities = NULL;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (gpuGetDeviceCount(&device_count) == gpuSuccess &&
            device_count > 0 && gpuGetDevice(&device_id) == gpuSuccess &&
            device_id >= 0 && device_id < device_count &&
            gpuDeviceGetPCIBusId(device_identity,
                                 DSYMLDL_V2_DEVICE_IDENTITY_SIZE,
                                 device_id) == gpuSuccess)
            device_ready = 1;

        if ((size_t) local_ranks <=
            SIZE_MAX / DSYMLDL_V2_DEVICE_IDENTITY_SIZE)
            node_device_identities = (char *) malloc(
                (size_t) local_ranks * DSYMLDL_V2_DEVICE_IDENTITY_SIZE);
        allocation_ready = node_device_identities != NULL;
        MPI_Allreduce(&allocation_ready, &all_allocations_ready, 1, MPI_INT,
                      MPI_MIN, node_communicator);
        if (all_allocations_ready)
        {
            MPI_Allgather(device_identity,
                          DSYMLDL_V2_DEVICE_IDENTITY_SIZE, MPI_CHAR,
                          node_device_identities,
                          DSYMLDL_V2_DEVICE_IDENTITY_SIZE, MPI_CHAR,
                          node_communicator);
            if (device_ready)
            {
                ranks_on_device = 0;
                for (int rank = 0; rank < local_ranks; ++rank)
                {
                    const char *identity = node_device_identities +
                        (size_t) rank * DSYMLDL_V2_DEVICE_IDENTITY_SIZE;
                    if (memcmp(device_identity, identity,
                               DSYMLDL_V2_DEVICE_IDENTITY_SIZE) == 0)
                        ++ranks_on_device;
                }
                if (ranks_on_device > 0 &&
                    gpuMemGetInfo(&free_bytes, &total_bytes) == gpuSuccess)
                {
                    gpu_known = 1;
                    local_ranks_per_gpu = ranks_on_device;
                }
            }
        }
        free(node_device_identities);
        if (gpu_known)
        {
            gpu_available = (uint64_t) free_bytes /
                            (uint64_t) SUPERLU_MAX(1, ranks_on_device);
        }
#undef DSYMLDL_V2_DEVICE_IDENTITY_SIZE
    }
#endif
    MPI_Allreduce(&local_ranks_per_gpu, &maximum_ranks_per_gpu, 1,
                  MPI_INT, MPI_MAX, communicator);
    MPI_Allreduce(&gpu_known, &all_gpu_known, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (all_gpu_known)
        MPI_Allreduce(&gpu_available, &minimum_gpu_available, 1,
                      MPI_UINT64_T, MPI_MIN, communicator);

    MPI_Comm_free(&node_communicator);
    available->available_host_bytes_per_node = minimum_host_available;
    available->available_gpu_bytes = minimum_gpu_available;
    available->ranks_per_node = maximum_local_ranks;
    available->ranks_per_gpu = maximum_ranks_per_gpu;
    available->host_memory_known = minimum_host_available > 0;
    available->gpu_memory_known = all_gpu_known;
    return 1;
}

static int dSymLDLV2ForestState(const dSymLDLV2PartitionPlan *plan,
                                int z, int_t target_tree)
{
    int_t tree = plan->pz - 1 + z;
    for (int_t level = 0; level < plan->max_z_levels; ++level)
    {
        if (tree == target_tree)
            return z % ((int) 1 << level) == 0 ? 1 : 2;
        if (level + 1 < plan->max_z_levels)
            tree = (tree - 1) / 2;
    }
    return 0;
}

static uint64_t dSymLDLV2UsableBytes(uint64_t available, uint64_t reserve)
{
    return available > reserve ? available - reserve : 0;
}

int dSymLDLV2EvaluateGridCandidate(
    const dSymLDLV2GridModelInput *input,
    const dSymLDLV2PartitionPlan *plan,
    dSymLDLV2GridCandidate *candidate, char *error, size_t error_size)
{
    if (input == NULL || input->partition_input == NULL ||
        input->structure == NULL || plan == NULL || candidate == NULL ||
        plan->pr != candidate->pr || plan->pc != candidate->pc ||
        plan->pz != candidate->pz)
    {
        dSymLDLV2SetModelError(error, error_size,
                               "SymLDL grid model input is inconsistent.");
        return 0;
    }

    const dSymLDLV2StructuralSummary *structure = input->structure;
    const dSymLDLV2PartitionPlanInput *partition = input->partition_input;
    if (plan->pr < 1 || plan->pc < 1 || plan->pz < 1 ||
        structure->nsupers != plan->nsupers ||
        partition->nsupers != structure->nsupers ||
        partition->xsup == NULL ||
        structure->n < 1 || structure->panel_block_offsets == NULL ||
        structure->panel_rows == NULL || plan->diag_root == NULL ||
        plan->panel_root == NULL || plan->forest_of_supernode == NULL ||
        plan->factor_level_count < 1 || plan->factor_level_ptr == NULL ||
        plan->factor_nodes == NULL ||
        structure->block_count < 0 ||
        (structure->block_count > 0 &&
         (structure->block_row_supernode == NULL ||
          structure->block_row_count == NULL)))
    {
        dSymLDLV2SetModelError(
            error, error_size,
            "SymLDL grid model structure is inconsistent.");
        return 0;
    }
    if (structure->panel_block_offsets[0] != 0 ||
        structure->panel_block_offsets[structure->nsupers] !=
            structure->block_count)
    {
        dSymLDLV2SetModelError(
            error, error_size,
            "SymLDL grid model block offsets are inconsistent.");
        return 0;
    }
    if (plan->pr > 1 && plan->pc > 1 &&
        (!input->runtime.pc_fragment_schur ||
         !input->runtime.pc_fragment_ldl_native ||
         input->runtime.cuda_aware_mpi))
    {
        memset(&candidate->memory, 0, sizeof(candidate->memory));
        memset(&candidate->performance, 0, sizeof(candidate->performance));
        candidate->status = DSYMLDL_V2_GRID_REJECT_IMPLEMENTATION;
        return 1;
    }
    for (int_t k = 0; k < structure->nsupers; ++k)
    {
        if (structure->panel_block_offsets[k] < 0 ||
            structure->panel_block_offsets[k + 1] <
                structure->panel_block_offsets[k] ||
            plan->diag_root[k] < 0 || plan->diag_root[k] >= plan->pr ||
            plan->panel_root[k] < 0 || plan->panel_root[k] >= plan->pc ||
            plan->forest_of_supernode[k] < 0 ||
            plan->forest_of_supernode[k] >= plan->forest_count)
        {
            dSymLDLV2SetModelError(
                error, error_size,
                "SymLDL grid model plan is inconsistent.");
            return 0;
        }
    }
    if ((size_t) plan->pr > SIZE_MAX / (size_t) plan->pc)
        goto overflow;
    size_t ranks_2d = (size_t) plan->pr * (size_t) plan->pc;
    if (ranks_2d > SIZE_MAX / (size_t) plan->pz)
        goto overflow;
    size_t ranks = ranks_2d * (size_t) plan->pz;
    if (!dSymLDLV2AllocationFits(ranks, sizeof(uint64_t)) ||
        !dSymLDLV2AllocationFits(ranks_2d, sizeof(uint64_t)) ||
        !dSymLDLV2AllocationFits((size_t) plan->pc, sizeof(uint64_t)) ||
        !dSymLDLV2AllocationFits((size_t) plan->pr, sizeof(uint64_t)))
        goto overflow;
    uint64_t *factor = (uint64_t *) calloc(ranks, sizeof(uint64_t));
    uint64_t *metadata = (uint64_t *) calloc(ranks, sizeof(uint64_t));
    uint64_t *panel_factor =
        (uint64_t *) calloc(ranks_2d, sizeof(uint64_t));
    uint64_t *panel_metadata =
        (uint64_t *) calloc(ranks_2d, sizeof(uint64_t));
    uint64_t *panel_row_counts =
        (uint64_t *) calloc(ranks_2d, sizeof(uint64_t));
    uint64_t *panel_block_counts =
        (uint64_t *) calloc(ranks_2d, sizeof(uint64_t));
    uint64_t *partner_value_counts =
        (uint64_t *) calloc((size_t) plan->pc, sizeof(uint64_t));
    uint64_t *partner_row_counts =
        (uint64_t *) calloc((size_t) plan->pc, sizeof(uint64_t));
    uint64_t *partner_block_counts =
        (uint64_t *) calloc((size_t) plan->pc, sizeof(uint64_t));
    uint64_t *row_value_counts =
        (uint64_t *) calloc((size_t) plan->pr, sizeof(uint64_t));
    uint64_t *row_row_counts =
        (uint64_t *) calloc((size_t) plan->pr, sizeof(uint64_t));
    uint64_t *row_block_counts =
        (uint64_t *) calloc((size_t) plan->pr, sizeof(uint64_t));
    if (factor == NULL || metadata == NULL ||
        panel_factor == NULL || panel_metadata == NULL ||
        panel_row_counts == NULL || panel_block_counts == NULL ||
        partner_value_counts == NULL || partner_row_counts == NULL ||
        partner_block_counts == NULL || row_value_counts == NULL ||
        row_row_counts == NULL || row_block_counts == NULL)
        goto allocation_failed;

    uint64_t symbolic_row_entries = structure->symbolic_row_entries;
    uint64_t maximum_l_panel_value_count = 0;
    uint64_t maximum_l_panel_index_count = 0;
    uint64_t maximum_partner_value_count = 0;
    uint64_t maximum_partner_index_count = 0;
    uint64_t maximum_row_value_count = 0;
    uint64_t maximum_row_index_count = 0;

    for (int_t k = 0; k < structure->nsupers; ++k)
    {
        memset(panel_factor, 0, ranks_2d * sizeof(uint64_t));
        memset(panel_metadata, 0, ranks_2d * sizeof(uint64_t));
        memset(panel_row_counts, 0, ranks_2d * sizeof(uint64_t));
        memset(panel_block_counts, 0, ranks_2d * sizeof(uint64_t));
        memset(partner_value_counts, 0,
               (size_t) plan->pc * sizeof(uint64_t));
        memset(partner_row_counts, 0,
               (size_t) plan->pc * sizeof(uint64_t));
        memset(partner_block_counts, 0,
               (size_t) plan->pc * sizeof(uint64_t));
        memset(row_value_counts, 0,
               (size_t) plan->pr * sizeof(uint64_t));
        memset(row_row_counts, 0,
               (size_t) plan->pr * sizeof(uint64_t));
        memset(row_block_counts, 0,
               (size_t) plan->pr * sizeof(uint64_t));
        int_t columns = partition->xsup[k + 1] - partition->xsup[k];
        for (int_t block = structure->panel_block_offsets[k];
             block < structure->panel_block_offsets[k + 1]; ++block)
        {
            int_t row_supernode = structure->block_row_supernode[block];
            int_t rows = structure->block_row_count[block];
            if (row_supernode < k || row_supernode >= structure->nsupers ||
                rows < 1)
                goto invalid_allocated;
            int rank_2d = plan->diag_root[row_supernode] * plan->pc +
                          plan->panel_root[k];
            uint64_t values = dSymLDLV2MultiplyBytes(
                dSymLDLV2MultiplyBytes((uint64_t) rows, (uint64_t) columns),
                sizeof(double));
            panel_factor[rank_2d] =
                dSymLDLV2AddBytes(panel_factor[rank_2d], values);
            panel_row_counts[rank_2d] = dSymLDLV2AddBytes(
                panel_row_counts[rank_2d], (uint64_t) rows);
            panel_block_counts[rank_2d] = dSymLDLV2AddBytes(
                panel_block_counts[rank_2d], 1);
            if (row_supernode != k)
            {
                int partner = plan->panel_root[row_supernode];
                int source_row = plan->diag_root[row_supernode];
                uint64_t value_count = dSymLDLV2MultiplyBytes(
                    (uint64_t) rows, (uint64_t) columns);
                partner_value_counts[partner] = dSymLDLV2AddBytes(
                    partner_value_counts[partner], value_count);
                partner_row_counts[partner] = dSymLDLV2AddBytes(
                    partner_row_counts[partner], (uint64_t) rows);
                partner_block_counts[partner] = dSymLDLV2AddBytes(
                    partner_block_counts[partner], 1);
                row_value_counts[source_row] = dSymLDLV2AddBytes(
                    row_value_counts[source_row], value_count);
                row_row_counts[source_row] = dSymLDLV2AddBytes(
                    row_row_counts[source_row], (uint64_t) rows);
                row_block_counts[source_row] = dSymLDLV2AddBytes(
                    row_block_counts[source_row], 1);
            }
        }

        for (size_t rank_2d = 0; rank_2d < ranks_2d; ++rank_2d)
        {
            if (panel_block_counts[rank_2d] > 0)
            {
                uint64_t index_count = dSymLDLV2AddBytes(
                    (uint64_t) DSYMLDL_V2_PANEL_HEADER_ENTRIES + 1,
                    dSymLDLV2AddBytes(
                        panel_row_counts[rank_2d],
                        dSymLDLV2MultiplyBytes(
                            2, panel_block_counts[rank_2d])));
                panel_metadata[rank_2d] = dSymLDLV2MultiplyBytes(
                    index_count, sizeof(int_t));
                maximum_l_panel_index_count = SUPERLU_MAX(
                    maximum_l_panel_index_count, index_count);
            }
            maximum_l_panel_value_count = SUPERLU_MAX(
                maximum_l_panel_value_count,
                panel_factor[rank_2d] / sizeof(double));
        }
        for (int pc = 0; pc < plan->pc; ++pc)
        {
            uint64_t index_count = partner_block_counts[pc] > 0
                ? dSymLDLV2AddBytes(
                      (uint64_t) DSYMLDL_V2_PANEL_HEADER_ENTRIES + 1,
                      dSymLDLV2AddBytes(
                          partner_row_counts[pc],
                          dSymLDLV2MultiplyBytes(
                              2, partner_block_counts[pc])))
                : 0;
            maximum_partner_value_count = SUPERLU_MAX(
                maximum_partner_value_count, partner_value_counts[pc]);
            maximum_partner_index_count = SUPERLU_MAX(
                maximum_partner_index_count, index_count);
        }
        for (int pr = 0; pr < plan->pr; ++pr)
        {
            uint64_t index_count = row_block_counts[pr] > 0
                ? dSymLDLV2AddBytes(
                      (uint64_t) DSYMLDL_V2_PANEL_HEADER_ENTRIES + 1,
                      dSymLDLV2AddBytes(
                          row_row_counts[pr],
                          dSymLDLV2MultiplyBytes(
                              2, row_block_counts[pr])))
                : 0;
            maximum_row_value_count = SUPERLU_MAX(
                maximum_row_value_count, row_value_counts[pr]);
            maximum_row_index_count = SUPERLU_MAX(
                maximum_row_index_count, index_count);
        }

        int diagonal_rank =
            plan->diag_root[k] * plan->pc + plan->panel_root[k];
        uint64_t diagonal_bytes = dSymLDLV2MultiplyBytes(
            dSymLDLV2MultiplyBytes((uint64_t) columns, (uint64_t) columns),
            sizeof(double));
        for (int pr = 0; pr < plan->pr; ++pr)
        {
            int rank_2d = pr * plan->pc + plan->panel_root[k];
            panel_factor[rank_2d] = dSymLDLV2AddBytes(
                panel_factor[rank_2d], diagonal_bytes);
        }
        if (plan->pr == 1 && plan->pc > 1)
        {
            /* The Pr=1 path forwards the persistent diagonal across columns. */
            for (int pc = 0; pc < plan->pc; ++pc)
            {
                if (pc == plan->panel_root[k])
                    continue;
                panel_factor[pc] = dSymLDLV2AddBytes(
                    panel_factor[pc], diagonal_bytes);
            }
        }
        panel_metadata[diagonal_rank] = dSymLDLV2AddBytes(
            panel_metadata[diagonal_rank],
            dSymLDLV2MultiplyBytes((uint64_t) columns, sizeof(int)));
        int_t tree = plan->forest_of_supernode[k];
        for (int z = 0; z < plan->pz; ++z)
        {
            int state = dSymLDLV2ForestState(plan, z, tree);
            if (state == 0)
                continue;
            for (size_t rank_2d = 0; rank_2d < ranks_2d; ++rank_2d)
            {
                size_t rank = (size_t) z * ranks_2d + rank_2d;
                factor[rank] =
                    dSymLDLV2AddBytes(factor[rank], panel_factor[rank_2d]);
                metadata[rank] = dSymLDLV2AddBytes(
                    metadata[rank], panel_metadata[rank_2d]);
            }
        }
    }

    uint64_t factor_maximum = dSymLDLV2MaximumBytes(factor, ranks);
    uint64_t panel_metadata_maximum =
        dSymLDLV2MaximumBytes(metadata, ranks);
    if (factor_maximum == UINT64_MAX ||
        panel_metadata_maximum == UINT64_MAX ||
        maximum_l_panel_value_count == UINT64_MAX ||
        maximum_l_panel_index_count == UINT64_MAX ||
        maximum_partner_value_count == UINT64_MAX ||
        maximum_partner_index_count == UINT64_MAX ||
        maximum_row_value_count == UINT64_MAX ||
        maximum_row_index_count == UINT64_MAX)
        goto overflow_allocated;
    uint64_t host_plan_metadata = dSymLDLV2MultiplyBytes(
        (uint64_t) structure->nsupers,
        8 * sizeof(int) + 14 * sizeof(int_t));
    uint64_t gpu_plan_metadata = dSymLDLV2MultiplyBytes(
        (uint64_t) structure->nsupers,
        4 * sizeof(int_t) + 4 * sizeof(void *));
    uint64_t host_metadata_maximum = dSymLDLV2AddBytes(
        panel_metadata_maximum, host_plan_metadata);
    uint64_t gpu_metadata_maximum = dSymLDLV2AddBytes(
        panel_metadata_maximum, gpu_plan_metadata);
    if (plan->pr > 1)
    {
        /* Upper-bound the native host gather map by one index per factor value. */
        uint64_t packed_map_entries = factor_maximum / sizeof(double);
        uint64_t packed_map_bytes = dSymLDLV2MultiplyBytes(
            packed_map_entries, sizeof(int_t));
        host_metadata_maximum = dSymLDLV2AddBytes(
            host_metadata_maximum, packed_map_bytes);
        /* Send metadata and receive layouts are index-sized and sparse. */
        host_metadata_maximum = dSymLDLV2AddBytes(
            host_metadata_maximum, panel_metadata_maximum);
        gpu_metadata_maximum = dSymLDLV2AddBytes(
            gpu_metadata_maximum, panel_metadata_maximum);
    }
    if (host_plan_metadata == UINT64_MAX ||
        gpu_plan_metadata == UINT64_MAX ||
        host_metadata_maximum == UINT64_MAX ||
        gpu_metadata_maximum == UINT64_MAX)
        goto overflow_allocated;

    int requested_streams = SUPERLU_MAX(1, input->runtime.gpu_streams);
    uint64_t panel_value_count_64 = maximum_l_panel_value_count;
    uint64_t panel_index_count_64 = maximum_l_panel_index_count;
    if (panel_value_count_64 > SIZE_MAX || panel_index_count_64 > SIZE_MAX ||
        maximum_partner_value_count > SIZE_MAX ||
        maximum_partner_index_count > SIZE_MAX ||
        maximum_row_value_count > SIZE_MAX ||
        maximum_row_index_count > SIZE_MAX)
        goto overflow_allocated;
    size_t panel_value_count = (size_t) panel_value_count_64;
    size_t panel_index_count = (size_t) panel_index_count_64;
    int pc_fragment_schur = plan->pr > 1 && plan->pc > 1;
    dSymLDLV2StreamWorkspaceCounts stream_counts;
    memset(&stream_counts, 0, sizeof(stream_counts));
    stream_counts.l_value_count = panel_value_count;
    stream_counts.partner_value_count =
        plan->pr <= 1 ? panel_value_count
                      : (size_t) maximum_partner_value_count;
    stream_counts.partner_stage_count =
        plan->pr <= 1 ? panel_value_count
                      : (size_t) maximum_partner_value_count;
    stream_counts.partner_send_stage_count =
        plan->pr > 1 ? panel_value_count : 0;
    uint64_t row_stage_count =
        pc_fragment_schur
            ? dSymLDLV2MultiplyBytes(
                  maximum_row_value_count,
                  (uint64_t) SUPERLU_MAX(1, plan->pc - 1))
            : 0;
    if (row_stage_count > SIZE_MAX)
        goto overflow_allocated;
    stream_counts.row_stage_count = (size_t) row_stage_count;
    stream_counts.row_receive_value_count =
        pc_fragment_schur ? (size_t) maximum_row_value_count : 0;
    stream_counts.raw_panel_count =
        input->runtime.wpanel_cache && plan->pr <= 1 && plan->pc <= 1
            ? panel_value_count
            : 0;
    stream_counts.l_index_count = panel_index_count;
    stream_counts.partner_index_count =
        plan->pr <= 1 ? panel_index_count
                      : (size_t) maximum_partner_index_count;
    stream_counts.row_receive_index_count =
        pc_fragment_schur ? (size_t) maximum_row_index_count : 0;
    stream_counts.row_send_map_count = stream_counts.row_stage_count;
    if (plan->pr > 1)
        stream_counts.row_send_map_count = SUPERLU_MAX(
            stream_counts.row_send_map_count, panel_value_count);
    stream_counts.lookahead_l_count = panel_value_count;
    stream_counts.lookahead_row_count =
        plan->pr <= 1 ? panel_value_count : 0;
    stream_counts.pc_fragment_schur = pc_fragment_schur;

    size_t stream_workspace_bytes = 0;
    size_t gemm_workspace_bytes = 0;
    size_t max_supernode = (size_t) SUPERLU_MAX(
        1, input->runtime.max_supernode_size);
    uint64_t gemm_value_limit = SUPERLU_MAX(
        (uint64_t) 1, structure->factor_value_entries);
    size_t gemm_buffer_values = input->runtime.gemm_buffer_values;
    if (gemm_value_limit < (uint64_t) gemm_buffer_values)
        gemm_buffer_values = (size_t) gemm_value_limit;
    if (!dSymLDLV2StreamWorkspaceBytes(
            &stream_counts, sizeof(double), sizeof(int_t), sizeof(int),
            &stream_workspace_bytes) ||
        max_supernode > SIZE_MAX / max_supernode ||
        !dSymLDLV2GemmWorkspaceBytes(
            max_supernode * max_supernode,
            gemm_buffer_values, sizeof(double),
            &gemm_workspace_bytes))
        goto overflow_allocated;
    uint64_t per_stream_workspace = dSymLDLV2AddBytes(
        (uint64_t) stream_workspace_bytes,
        (uint64_t) gemm_workspace_bytes);
    int pooled_staging = input->runtime.pinned_staging &&
                         input->runtime.pooled_pinned_staging;
    uint64_t staging_copies = pooled_staging
                                  ? 1
                                  : (uint64_t) requested_streams;
    uint64_t receive_values = dSymLDLV2AddBytes(
        (uint64_t) stream_counts.partner_value_count,
        dSymLDLV2AddBytes(
            (uint64_t) stream_counts.row_receive_value_count,
            (uint64_t) stream_counts.row_stage_count));
    uint64_t partner_send_values = 0;
    if (stream_counts.partner_send_stage_count > 0)
        partner_send_values = pooled_staging
            ? (uint64_t) stream_counts.partner_send_stage_count
            : factor_maximum / sizeof(double);
    uint64_t host_fragment_value_staging = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(staging_copies, receive_values),
        partner_send_values);
    uint64_t host_fragment_index_staging = dSymLDLV2MultiplyBytes(
        dSymLDLV2MultiplyBytes(
            (uint64_t) requested_streams,
            (uint64_t) stream_counts.partner_index_count),
        sizeof(int_t));
    uint64_t host_fragment_staging = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(host_fragment_value_staging,
                               sizeof(double)),
        host_fragment_index_staging);
    uint64_t pinned_staging = input->runtime.pinned_staging
        ? dSymLDLV2MultiplyBytes(host_fragment_value_staging,
                                 sizeof(double))
        : 0;
    uint64_t l_panel_value_bytes = dSymLDLV2MultiplyBytes(
        maximum_l_panel_value_count, sizeof(double));
    uint64_t l_panel_index_bytes = dSymLDLV2MultiplyBytes(
        maximum_l_panel_index_count, sizeof(int_t));
    uint64_t host_lookahead = dSymLDLV2MultiplyBytes(
        (uint64_t) SUPERLU_MAX(1, input->runtime.lookahead_depth),
        dSymLDLV2AddBytes(l_panel_value_bytes, l_panel_index_bytes));
    uint64_t host_diagonal = dSymLDLV2MultiplyBytes(
        dSymLDLV2MultiplyBytes(
            2 * (uint64_t) SUPERLU_MAX(
                    1, input->runtime.lookahead_depth),
            dSymLDLV2MultiplyBytes(max_supernode, max_supernode)),
        sizeof(double));
    uint64_t host_communication = dSymLDLV2AddBytes(
        dSymLDLV2AddBytes(host_lookahead, host_diagonal),
        host_fragment_staging);
    uint64_t vector_entries =
        ((uint64_t) structure->n + (uint64_t) plan->pr - 1) /
        (uint64_t) plan->pr;
    uint64_t solve_value_entries = dSymLDLV2MultiplyBytes(
        vector_entries,
        (uint64_t) SUPERLU_MAX(1, input->runtime.solve_nrhs));
    uint64_t solve_vector_workspace = dSymLDLV2MultiplyBytes(
        dSymLDLV2MultiplyBytes(solve_value_entries, sizeof(double)), 4);
    uint64_t cached_solve_rows =
        panel_metadata_maximum / sizeof(int_t);
    uint64_t host_solve_cache = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(
            dSymLDLV2MultiplyBytes(
                cached_solve_rows,
                (uint64_t) SUPERLU_MAX(1, input->runtime.solve_nrhs)),
            sizeof(double)),
        dSymLDLV2MultiplyBytes(
            cached_solve_rows, sizeof(unsigned char)));
    uint64_t host_solve_workspace = dSymLDLV2AddBytes(
        solve_vector_workspace,
        dSymLDLV2AddBytes(host_solve_cache, host_plan_metadata));
    uint64_t gpu_solve_workspace = dSymLDLV2AddBytes(
        solve_vector_workspace, panel_metadata_maximum);
    uint64_t nvshmem_workspace = SUPERLU_MAX(
        dSymLDLV2MultiplyBytes(solve_value_entries, sizeof(double)),
        l_panel_value_bytes);
    int estimated_streams = requested_streams;
    int insufficient_gpu_memory = 0;
    if (input->runtime.gpu_offload &&
        input->available.gpu_memory_known)
    {
        uint64_t usable_gpu = dSymLDLV2UsableBytes(
            input->available.available_gpu_bytes,
            input->gpu_reserve_bytes);
        uint64_t gpu_fixed_base = dSymLDLV2AddBytes(
            dSymLDLV2AddBytes(factor_maximum, gpu_metadata_maximum),
            dSymLDLV2AddBytes(gpu_solve_workspace, nvshmem_workspace));
        uint64_t minimum_stream_workspace = dSymLDLV2MultiplyBytes(
            2, per_stream_workspace);
        uint64_t minimum_tracked = dSymLDLV2AddBytes(
            gpu_fixed_base, minimum_stream_workspace);
        uint64_t minimum_total = dSymLDLV2AddBytes(
            minimum_tracked,
            dSymLDLV2RuntimeGPUOverhead(minimum_tracked));
        /* The current GPU setup requires capacity for at least two streams. */
        if (minimum_total == UINT64_MAX || minimum_total > usable_gpu)
            insufficient_gpu_memory = 1;
        else
        {
            estimated_streams = 1;
            for (int streams = requested_streams; streams >= 1; --streams)
            {
                uint64_t stream_bytes = dSymLDLV2MultiplyBytes(
                    (uint64_t) streams, per_stream_workspace);
                uint64_t tracked = dSymLDLV2AddBytes(
                    gpu_fixed_base, stream_bytes);
                uint64_t total = dSymLDLV2AddBytes(
                    tracked, dSymLDLV2RuntimeGPUOverhead(tracked));
                if (total != UINT64_MAX && total <= usable_gpu)
                {
                    estimated_streams = streams;
                    break;
                }
            }
        }
    }
    uint64_t gpu_factor_workspace = dSymLDLV2MultiplyBytes(
        (uint64_t) estimated_streams, (uint64_t) gemm_workspace_bytes);
    uint64_t gpu_communication = dSymLDLV2MultiplyBytes(
        (uint64_t) estimated_streams, (uint64_t) stream_workspace_bytes);
    uint64_t gpu_workspace = dSymLDLV2AddBytes(
        gpu_factor_workspace, gpu_communication);
    uint64_t input_nnz = structure->input_nnz > 0
                             ? structure->input_nnz
                             : symbolic_row_entries;
    uint64_t source_matrix_workspace = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(
            input_nnz, sizeof(double) + sizeof(int_t)),
        dSymLDLV2MultiplyBytes((uint64_t) structure->n + 1,
                               sizeof(int_t)));
    uint64_t symbolic_array_entries = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(6, (uint64_t) structure->n),
        dSymLDLV2MultiplyBytes(3, (uint64_t) structure->nsupers));
    uint64_t symbolic_index_workspace = dSymLDLV2AddBytes(
        dSymLDLV2MultiplyBytes(symbolic_row_entries, 2 * sizeof(int_t)),
        dSymLDLV2MultiplyBytes(symbolic_array_entries, sizeof(int_t)));
    uint64_t host_symbolic_workspace = dSymLDLV2AddBytes(
        source_matrix_workspace, symbolic_index_workspace);
    uint64_t host_distribution_workspace = dSymLDLV2AddBytes(
        host_symbolic_workspace,
        dSymLDLV2AddBytes(factor_maximum, host_metadata_maximum));
    /* Conversion and MPI work arrays are not retained in the symbolic data. */
    host_distribution_workspace = dSymLDLV2AddBytes(
        host_distribution_workspace, host_distribution_workspace / 2);

    dSymLDLV2MemoryEstimate *memory = &candidate->memory;
    memset(memory, 0, sizeof(*memory));
    memory->host_persistent_factor = factor_maximum;
    memory->gpu_persistent_factor = factor_maximum;
    memory->host_persistent_metadata = host_metadata_maximum;
    memory->gpu_persistent_metadata = gpu_metadata_maximum;
    memory->host_symbolic_workspace_high_water =
        host_symbolic_workspace;
    memory->host_distribution_workspace_high_water =
        host_distribution_workspace;
    memory->host_factor_workspace_high_water = SUPERLU_MAX(
        l_panel_value_bytes,
        dSymLDLV2MultiplyBytes(
            dSymLDLV2MultiplyBytes(max_supernode, max_supernode),
            sizeof(double)));
    memory->gpu_factor_workspace_high_water = gpu_factor_workspace;
    memory->host_communication_staging = host_communication;
    memory->gpu_communication_staging = gpu_communication;
    memory->pinned_staging = pinned_staging;
    memory->host_solve_workspace = host_solve_workspace;
    memory->gpu_solve_workspace = gpu_solve_workspace;
    memory->nvshmem_symmetric = nvshmem_workspace;
    memory->host_high_water_per_rank = factor_maximum;
    memory->host_high_water_per_rank = dSymLDLV2AddBytes(
        memory->host_high_water_per_rank, host_metadata_maximum);
    memory->host_high_water_per_rank = dSymLDLV2AddBytes(
        memory->host_high_water_per_rank,
        memory->host_factor_workspace_high_water);
    memory->host_high_water_per_rank = dSymLDLV2AddBytes(
        memory->host_high_water_per_rank, host_communication);
    memory->host_high_water_per_rank = dSymLDLV2AddBytes(
        memory->host_high_water_per_rank, host_solve_workspace);
    memory->host_high_water_per_rank = SUPERLU_MAX(
        memory->host_high_water_per_rank,
        host_distribution_workspace);
    memory->host_high_water_per_node = dSymLDLV2MultiplyBytes(
        memory->host_high_water_per_rank,
        (uint64_t) SUPERLU_MAX(1, input->available.ranks_per_node));
    memory->gpu_high_water_per_rank = factor_maximum;
    memory->gpu_high_water_per_rank = dSymLDLV2AddBytes(
        memory->gpu_high_water_per_rank, gpu_metadata_maximum);
    memory->gpu_high_water_per_rank = dSymLDLV2AddBytes(
        memory->gpu_high_water_per_rank, gpu_workspace);
    memory->gpu_high_water_per_rank = dSymLDLV2AddBytes(
        memory->gpu_high_water_per_rank, gpu_solve_workspace);
    memory->gpu_high_water_per_rank = dSymLDLV2AddBytes(
        memory->gpu_high_water_per_rank, nvshmem_workspace);
    memory->gpu_runtime_overhead = dSymLDLV2RuntimeGPUOverhead(
        memory->gpu_high_water_per_rank);
    memory->gpu_high_water_per_rank = dSymLDLV2AddBytes(
        memory->gpu_high_water_per_rank, memory->gpu_runtime_overhead);
    if (memory->host_persistent_factor == UINT64_MAX ||
        memory->gpu_persistent_factor == UINT64_MAX ||
        memory->host_persistent_metadata == UINT64_MAX ||
        memory->gpu_persistent_metadata == UINT64_MAX ||
        memory->host_symbolic_workspace_high_water == UINT64_MAX ||
        memory->host_distribution_workspace_high_water == UINT64_MAX ||
        memory->host_factor_workspace_high_water == UINT64_MAX ||
        memory->gpu_factor_workspace_high_water == UINT64_MAX ||
        memory->host_communication_staging == UINT64_MAX ||
        memory->gpu_communication_staging == UINT64_MAX ||
        memory->pinned_staging == UINT64_MAX ||
        memory->host_solve_workspace == UINT64_MAX ||
        memory->gpu_solve_workspace == UINT64_MAX ||
        memory->nvshmem_symmetric == UINT64_MAX ||
        memory->gpu_runtime_overhead == UINT64_MAX ||
        memory->host_high_water_per_rank == UINT64_MAX ||
        memory->host_high_water_per_node == UINT64_MAX ||
        memory->gpu_high_water_per_rank == UINT64_MAX)
        goto overflow_allocated;

    if (!dSymLDLV2EvaluateRuntimeModel(
            input, plan, requested_streams, estimated_streams,
            &candidate->performance, error, error_size))
        goto runtime_model_failed;

    candidate->status = DSYMLDL_V2_GRID_FEASIBLE;
    if (input->runtime.gpu_offload &&
        input->available.gpu_memory_known &&
        (insufficient_gpu_memory ||
         memory->gpu_high_water_per_rank >
             dSymLDLV2UsableBytes(input->available.available_gpu_bytes,
                                  input->gpu_reserve_bytes)))
        candidate->status = DSYMLDL_V2_GRID_REJECT_GPU_MEMORY;
    else if (input->available.host_memory_known &&
             memory->host_high_water_per_node >
                 dSymLDLV2UsableBytes(
                     input->available.available_host_bytes_per_node,
                     input->host_reserve_bytes_per_node))
        candidate->status = DSYMLDL_V2_GRID_REJECT_HOST_MEMORY;

    free(factor);
    free(metadata);
    free(panel_factor);
    free(panel_metadata);
    free(panel_row_counts);
    free(panel_block_counts);
    free(partner_value_counts);
    free(partner_row_counts);
    free(partner_block_counts);
    free(row_value_counts);
    free(row_row_counts);
    free(row_block_counts);
    return 1;

overflow_allocated:
    free(factor);
    free(metadata);
    free(panel_factor);
    free(panel_metadata);
    free(panel_row_counts);
    free(panel_block_counts);
    free(partner_value_counts);
    free(partner_row_counts);
    free(partner_block_counts);
    free(row_value_counts);
    free(row_row_counts);
    free(row_block_counts);
overflow:
    dSymLDLV2SetModelError(error, error_size,
                           "SymLDL grid model dimensions overflow.");
    return 0;
invalid_allocated:
    free(factor);
    free(metadata);
    free(panel_factor);
    free(panel_metadata);
    free(panel_row_counts);
    free(panel_block_counts);
    free(partner_value_counts);
    free(partner_row_counts);
    free(partner_block_counts);
    free(row_value_counts);
    free(row_row_counts);
    free(row_block_counts);
    dSymLDLV2SetModelError(error, error_size,
                           "SymLDL grid model block metadata is inconsistent.");
    return 0;
allocation_failed:
    free(factor);
    free(metadata);
    free(panel_factor);
    free(panel_metadata);
    free(panel_row_counts);
    free(panel_block_counts);
    free(partner_value_counts);
    free(partner_row_counts);
    free(partner_block_counts);
    free(row_value_counts);
    free(row_row_counts);
    free(row_block_counts);
    dSymLDLV2SetModelError(error, error_size,
                           "SymLDL grid model allocation failed.");
    return 0;

runtime_model_failed:
    free(factor);
    free(metadata);
    free(panel_factor);
    free(panel_metadata);
    free(panel_row_counts);
    free(panel_block_counts);
    free(partner_value_counts);
    free(partner_row_counts);
    free(partner_block_counts);
    free(row_value_counts);
    free(row_row_counts);
    free(row_block_counts);
    return 0;
}
