#include "dsymldl_v2_grid_report.h"
#include "dsymldl_v2_grid_calibration.h"
#include "dsymldl_v2_workspace_size.h"

#ifdef GPU_ACC
#include "gpu_wrapper.h"
#if defined(HAVE_CUDA) && defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define DSYMLDL_V2_HAVE_MPI_EXT_HEADER 1
#endif
#endif
#endif

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DSYMLDL_V2_GIB ((uint64_t) 1024 * 1024 * 1024)

static dSymLDLV2MemoryEstimate dSymLDLV2CurrentGridMemory;
static int dSymLDLV2CurrentGridMemoryValid;
static uint64_t dSymLDLV2CurrentGridGPUFreeBaseline;
static int dSymLDLV2CurrentGridGPUFreeBaselineValid;

static void dSymLDLV2SetReportError(char *error, size_t error_size,
                                    const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2BooleanEnvironment(const char *name, int fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0')
        return fallback != 0;
    if (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 ||
        strcmp(value, "TRUE") == 0 || strcmp(value, "yes") == 0 ||
        strcmp(value, "YES") == 0 || strcmp(value, "on") == 0 ||
        strcmp(value, "ON") == 0)
        return 1;
    if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 ||
        strcmp(value, "FALSE") == 0 || strcmp(value, "no") == 0 ||
        strcmp(value, "NO") == 0 || strcmp(value, "off") == 0 ||
        strcmp(value, "OFF") == 0)
        return 0;
    fprintf(stderr, "%s must be a boolean value.\n", name);
    ABORT("Invalid boolean environment value.");
    return fallback != 0;
}

static int dSymLDLV2ParseUnsignedEnvironment(const char *name,
                                              uint64_t *value)
{
    const char *text = getenv(name);
    char *end = NULL;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0')
        return 1;
    if (text[0] == '-')
        return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0')
        return 0;
    *value = (uint64_t) parsed;
    return 1;
}

static int dSymLDLV2ParseDimensionEnvironment(const char *name,
                                               int *dimension,
                                               char *error,
                                               size_t error_size)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0')
        return 1;
    return dSymLDLV2ParseGridDimension(value, dimension, error, error_size);
}

static int dSymLDLV2ParseLimitEnvironment(const char *name, int *limit,
                                           char *error, size_t error_size)
{
    const char *value = getenv(name);
    int parsed;
    if (value == NULL || value[0] == '\0')
        return 1;
    if (!dSymLDLV2ParseGridDimension(value, &parsed, error, error_size) ||
        parsed == SUPERLU_GRID_AUTO)
    {
        dSymLDLV2SetReportError(error, error_size,
                                "Grid limits must be positive integers.");
        return 0;
    }
    *limit = parsed;
    return 1;
}

static uint64_t dSymLDLV2DefaultGPUReserve(uint64_t available,
                                           int ranks_per_gpu)
{
    uint64_t fractional = available / 10;
    uint64_t shared_fixed =
        (2 * DSYMLDL_V2_GIB + (uint64_t) SUPERLU_MAX(1, ranks_per_gpu) - 1) /
        (uint64_t) SUPERLU_MAX(1, ranks_per_gpu);
    return SUPERLU_MAX(shared_fixed, fractional);
}

static uint64_t dSymLDLV2DefaultHostReserve(uint64_t available)
{
    uint64_t fractional = available / 10;
    return SUPERLU_MAX(4 * DSYMLDL_V2_GIB, fractional);
}

static double dSymLDLV2GiB(uint64_t bytes)
{
    return (double) bytes / (double) DSYMLDL_V2_GIB;
}

static void dSymLDLV2GridTopologyDestroy(dSymLDLV2GridTopology *topology)
{
    if (topology == NULL)
        return;
    free((void *) topology->node_of_rank);
    memset(topology, 0, sizeof(*topology));
}

static int dSymLDLV2DiscoverGridTopology(
    MPI_Comm communicator, dSymLDLV2RankOrder rank_order,
    dSymLDLV2GridTopology *topology, char *error, size_t error_size)
{
    int communicator_rank;
    int communicator_size;
    int *leaders = NULL;
    int *node_of_rank = NULL;

    if (topology == NULL)
    {
        dSymLDLV2SetReportError(error, error_size,
                                "SymLDL grid topology output is missing.");
        return 0;
    }
    memset(topology, 0, sizeof(*topology));
    MPI_Comm_rank(communicator, &communicator_rank);
    MPI_Comm_size(communicator, &communicator_size);
    if (communicator_size < 1 ||
        (size_t) communicator_size > SIZE_MAX / sizeof(int))
    {
        dSymLDLV2SetReportError(error, error_size,
                                "SymLDL communicator size is invalid.");
        return 0;
    }
    leaders = (int *) malloc((size_t) communicator_size * sizeof(int));
    if (leaders == NULL)
    {
        dSymLDLV2SetReportError(error, error_size,
                                "SymLDL grid topology allocation failed.");
        return 0;
    }

#if MPI_VERSION >= 3
    MPI_Comm shared = MPI_COMM_NULL;
    int shared_rank = 0;
    int leader = communicator_rank;
    if (MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED,
                            communicator_rank, MPI_INFO_NULL,
                            &shared) != MPI_SUCCESS)
        goto discovery_failed;
    MPI_Comm_rank(shared, &shared_rank);
    if (shared_rank != 0)
        leader = -1;
    MPI_Bcast(&leader, 1, MPI_INT, 0, shared);
    MPI_Comm_free(&shared);
    shared = MPI_COMM_NULL;
    if (MPI_Allgather(&leader, 1, MPI_INT, leaders, 1, MPI_INT,
                      communicator) != MPI_SUCCESS)
        goto discovery_failed;
#else
    char local_name[MPI_MAX_PROCESSOR_NAME];
    char *names = NULL;
    int name_length = 0;
    if ((size_t) communicator_size >
        SIZE_MAX / (size_t) MPI_MAX_PROCESSOR_NAME)
        goto discovery_failed;
    names = (char *) calloc((size_t) communicator_size,
                            (size_t) MPI_MAX_PROCESSOR_NAME);
    if (names == NULL)
        goto discovery_failed;
    memset(local_name, 0, sizeof(local_name));
    MPI_Get_processor_name(local_name, &name_length);
    if (MPI_Allgather(local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      communicator) != MPI_SUCCESS)
    {
        free(names);
        goto discovery_failed;
    }
    for (int rank = 0; rank < communicator_size; ++rank)
    {
        leaders[rank] = rank;
        for (int prior = 0; prior < rank; ++prior)
            if (strcmp(names + (size_t) rank * MPI_MAX_PROCESSOR_NAME,
                       names + (size_t) prior * MPI_MAX_PROCESSOR_NAME) == 0)
            {
                leaders[rank] = leaders[prior];
                break;
            }
    }
    free(names);
#endif

    node_of_rank = (int *) malloc(
        (size_t) communicator_size * sizeof(int));
    if (node_of_rank == NULL)
        goto discovery_failed;
    int node_count = 0;
    for (int rank = 0; rank < communicator_size; ++rank)
    {
        int node = -1;
        for (int prior = 0; prior < rank; ++prior)
            if (leaders[prior] == leaders[rank])
            {
                node = node_of_rank[prior];
                break;
            }
        if (node < 0)
            node = node_count++;
        node_of_rank[rank] = node;
    }
    free(leaders);
    topology->communicator_size = communicator_size;
    topology->node_count = node_count;
    topology->topology_known = 1;
    topology->rank_order = rank_order;
    topology->node_of_rank = node_of_rank;
    return 1;

discovery_failed:
#if MPI_VERSION >= 3
    if (shared != MPI_COMM_NULL)
        MPI_Comm_free(&shared);
#endif
    free(leaders);
    free(node_of_rank);
    dSymLDLV2SetReportError(error, error_size,
                            "SymLDL process-node topology discovery failed.");
    return 0;
}

int dSymLDLV2GridReportEnabled(void)
{
    return dSymLDLV2BooleanEnvironment("SYMLDL_V2_GRID_REPORT", 0);
}

int dSymLDLV2GetCurrentGridPrediction(dSymLDLV2MemoryEstimate *memory)
{
    if (!dSymLDLV2CurrentGridMemoryValid || memory == NULL)
        return 0;
    *memory = dSymLDLV2CurrentGridMemory;
    return 1;
}

void dSymLDLV2SetCurrentGridPrediction(
    const dSymLDLV2MemoryEstimate *memory)
{
    dSymLDLV2CurrentGridMemoryValid = memory != NULL;
    dSymLDLV2CurrentGridGPUFreeBaseline = 0;
    dSymLDLV2CurrentGridGPUFreeBaselineValid = 0;
    dSymLDLV2SetCurrentGridGPUStreamsUsed(0);
    if (memory == NULL)
        dSymLDLV2SetCurrentGridGPUStreamCap(0);
    if (memory != NULL)
    {
        dSymLDLV2CurrentGridMemory = *memory;
#ifdef GPU_ACC
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (gpuMemGetInfo(&free_bytes, &total_bytes) == gpuSuccess)
        {
            dSymLDLV2CurrentGridGPUFreeBaseline = (uint64_t) free_bytes;
            dSymLDLV2CurrentGridGPUFreeBaselineValid = 1;
        }
#endif
    }
    else
        memset(&dSymLDLV2CurrentGridMemory, 0,
               sizeof(dSymLDLV2CurrentGridMemory));
}

int dSymLDLV2GetCurrentGridGPUUsage(uint64_t *bytes)
{
    if (bytes == NULL)
        return 0;
    *bytes = 0;
#ifdef GPU_ACC
    if (dSymLDLV2CurrentGridGPUFreeBaselineValid)
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (gpuMemGetInfo(&free_bytes, &total_bytes) == gpuSuccess)
        {
            if (dSymLDLV2CurrentGridGPUFreeBaseline >
                (uint64_t) free_bytes)
                *bytes = dSymLDLV2CurrentGridGPUFreeBaseline -
                         (uint64_t) free_bytes;
            return 1;
        }
    }
#endif
    return 0;
}

int dSymLDLV2ApplyGridEnvironment(
    dSymLDLV2GridRequest *request, char *error, size_t error_size)
{
    if (request == NULL)
    {
        dSymLDLV2SetReportError(error, error_size,
                                "SymLDL grid request is missing.");
        return 0;
    }

    if (!dSymLDLV2ParseLimitEnvironment(
            "SYMLDL_V2_GRID_MAX_PR", &request->max_pr,
            error, error_size) ||
        !dSymLDLV2ParseLimitEnvironment(
            "SYMLDL_V2_GRID_MAX_PC", &request->max_pc,
            error, error_size) ||
        !dSymLDLV2ParseLimitEnvironment(
            "SYMLDL_V2_GRID_MAX_PZ", &request->max_pz,
            error, error_size) ||
        !dSymLDLV2ParseUnsignedEnvironment(
            "SYMLDL_V2_GRID_GPU_BUDGET_BYTES",
            &request->gpu_memory_budget_bytes) ||
        !dSymLDLV2ParseUnsignedEnvironment(
            "SYMLDL_V2_GRID_HOST_BUDGET_BYTES",
            &request->host_memory_budget_bytes) ||
        !dSymLDLV2ParseUnsignedEnvironment(
            "SYMLDL_V2_GRID_GPU_RESERVE_BYTES",
            &request->gpu_memory_reserve_bytes) ||
        !dSymLDLV2ParseUnsignedEnvironment(
            "SYMLDL_V2_GRID_HOST_RESERVE_BYTES",
            &request->host_memory_reserve_bytes))
    {
        if (error != NULL && error[0] == '\0')
            dSymLDLV2SetReportError(
                error, error_size,
                "SymLDL grid memory override is invalid.");
        return 0;
    }
    return 1;
}

void dSymLDLV2GridRuntimeConfigInit(
    const superlu_dist_options_t *options,
    dSymLDLV2GridRuntimeConfig *runtime)
{
    if (runtime == NULL)
        return;
    memset(runtime, 0, sizeof(*runtime));
    if (options == NULL)
        return;
#ifdef GPU_ACC
    runtime->gpu_offload =
        sp_ienv_dist(10, (superlu_dist_options_t *) options) != 0;
#endif
    int_t lookahead = getNumLookAhead((superlu_dist_options_t *) options);
    if (lookahead < 1 || lookahead > INT_MAX)
        ABORT("The number of lookahead panels is outside the supported range.");
    runtime->lookahead_depth = (int) lookahead;
    runtime->solve_nrhs = 1;
    runtime->gpu_streams = SUPERLU_MIN(
        runtime->lookahead_depth,
        SUPERLU_MIN(DSYMLDL_V2_MAX_GPU_STREAMS,
                    SUPERLU_MAX(1, sp_ienv_dist(
                        9, (superlu_dist_options_t *) options))));
    runtime->max_supernode_size = SUPERLU_MAX(
        1, sp_ienv_dist(3, (superlu_dist_options_t *) options));
    size_t max_supernode = (size_t) runtime->max_supernode_size;
    if (max_supernode > SIZE_MAX / max_supernode)
        ABORT("SymLDL maximum supernode workspace size overflows.");
    runtime->gemm_buffer_values = SUPERLU_MAX(
        max_supernode * max_supernode,
        (size_t) SUPERLU_MAX(
            1, sp_ienv_dist(8, (superlu_dist_options_t *) options)));
    runtime->pinned_staging =
        dSymLDLV2BooleanEnvironment("GPU3DV2_PINNED_STAGING", 1);
    runtime->pooled_pinned_staging =
        dSymLDLV2BooleanEnvironment("GPU3DV2_PINNED_STAGING_POOL", 0);
    runtime->wpanel_cache =
        dSymLDLV2BooleanEnvironment("GPU3DV2_WPANEL_CACHE", 0);
    runtime->pc_fragment_schur =
        dSymLDLV2BooleanEnvironment("GPU3DV2_PC_FRAGMENT_SCHUR", 1);
    runtime->pc_fragment_ldl_native = dSymLDLV2BooleanEnvironment(
        "GPU3DV2_PC_FRAGMENT_LDL_NATIVE",
        runtime->pc_fragment_schur ? 1 : 0);
    runtime->rank_order =
        getenv("SUPERLU_RANKORDER") != NULL &&
                strcmp(getenv("SUPERLU_RANKORDER"), "XY") == 0
            ? DSYMLDL_V2_RANK_ORDER_XY_MAJOR
            : DSYMLDL_V2_RANK_ORDER_Z_MAJOR;
    const char *cuda_aware = getenv("SUPERLU_CUDA_AWARE_MPI");
    const char *mpich_gpu = getenv("MPICH_GPU_SUPPORT_ENABLED");
    if (cuda_aware != NULL && cuda_aware[0] != '\0')
        runtime->cuda_aware_mpi = dSymLDLV2BooleanEnvironment(
            "SUPERLU_CUDA_AWARE_MPI", 0);
    else if (mpich_gpu != NULL && mpich_gpu[0] != '\0')
        runtime->cuda_aware_mpi = dSymLDLV2BooleanEnvironment(
            "MPICH_GPU_SUPPORT_ENABLED", 0);
#if defined(DSYMLDL_V2_HAVE_MPI_EXT_HEADER) && \
    defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    else
        runtime->cuda_aware_mpi = MPIX_Query_cuda_support() != 0;
#endif
}

static int dSymLDLV2GridRequestCollectivelyEqual(
    MPI_Comm communicator, const dSymLDLV2GridRequest *request,
    char *error, size_t error_size)
{
    int dimensions[6] = {
        request->pr, request->pc, request->pz,
        request->max_pr, request->max_pc, request->max_pz
    };
    int minimum_dimensions[6];
    int maximum_dimensions[6];
    uint64_t memory[4] = {
        request->host_memory_budget_bytes,
        request->gpu_memory_budget_bytes,
        request->host_memory_reserve_bytes,
        request->gpu_memory_reserve_bytes
    };
    uint64_t minimum_memory[4];
    uint64_t maximum_memory[4];

    MPI_Allreduce(dimensions, minimum_dimensions, 6, MPI_INT, MPI_MIN,
                  communicator);
    MPI_Allreduce(dimensions, maximum_dimensions, 6, MPI_INT, MPI_MAX,
                  communicator);
    MPI_Allreduce(memory, minimum_memory, 4, MPI_UINT64_T, MPI_MIN,
                  communicator);
    MPI_Allreduce(memory, maximum_memory, 4, MPI_UINT64_T, MPI_MAX,
                  communicator);
    if (memcmp(minimum_dimensions, maximum_dimensions,
               sizeof(minimum_dimensions)) != 0 ||
        memcmp(minimum_memory, maximum_memory,
               sizeof(minimum_memory)) != 0)
    {
        dSymLDLV2SetReportError(
            error, error_size,
            "SymLDL grid request differs between MPI ranks.");
        return 0;
    }
    return 1;
}

static int dSymLDLV2GridRuntimeCollectivelyEqual(
    MPI_Comm communicator, const dSymLDLV2GridRuntimeConfig *runtime,
    char *error, size_t error_size)
{
    int settings[12] = {
        runtime->gpu_offload,
        runtime->gpu_streams,
        runtime->lookahead_depth,
        runtime->solve_nrhs,
        runtime->max_supernode_size,
        runtime->pinned_staging,
        runtime->pooled_pinned_staging,
        runtime->wpanel_cache,
        runtime->pc_fragment_schur,
        runtime->pc_fragment_ldl_native,
        runtime->cuda_aware_mpi,
        (int) runtime->rank_order
    };
    int minimum_settings[12];
    int maximum_settings[12];
    uint64_t gemm_values = (uint64_t) runtime->gemm_buffer_values;
    uint64_t minimum_gemm_values;
    uint64_t maximum_gemm_values;

    MPI_Allreduce(settings, minimum_settings, 12, MPI_INT, MPI_MIN,
                  communicator);
    MPI_Allreduce(settings, maximum_settings, 12, MPI_INT, MPI_MAX,
                  communicator);
    MPI_Allreduce(&gemm_values, &minimum_gemm_values, 1, MPI_UINT64_T,
                  MPI_MIN, communicator);
    MPI_Allreduce(&gemm_values, &maximum_gemm_values, 1, MPI_UINT64_T,
                  MPI_MAX, communicator);
    if (memcmp(minimum_settings, maximum_settings,
               sizeof(minimum_settings)) != 0 ||
        minimum_gemm_values != maximum_gemm_values)
    {
        dSymLDLV2SetReportError(
            error, error_size,
            "SymLDL grid runtime configuration differs between MPI ranks.");
        return 0;
    }
    return 1;
}

int dSymLDLV2SelectGridForStructure(
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2StructuralSummary *structure, MPI_Comm communicator,
    const dSymLDLV2GridRequest *request,
    const dSymLDLV2GridRuntimeConfig *runtime,
    int calibrate,
    dSymLDLV2GridSelection *selection, char *error, size_t error_size)
{
    int communicator_size;
    unsigned long long local_candidate_count;
    unsigned long long minimum_candidate_count;
    unsigned long long maximum_candidate_count;
    dSymLDLV2GridModelInput model;
    dSymLDLV2AvailableMemory available;
    dSymLDLV2GridTopology topology;

    memset(&topology, 0, sizeof(topology));

    if (partition_input == NULL || structure == NULL || request == NULL ||
        runtime == NULL || selection == NULL)
    {
        dSymLDLV2SetReportError(error, error_size,
                                "SymLDL grid selection input is incomplete.");
        return 0;
    }

    if (!dSymLDLV2GridRequestCollectivelyEqual(
            communicator, request, error, error_size))
        return 0;
    if (!dSymLDLV2GridRuntimeCollectivelyEqual(
            communicator, runtime, error, error_size))
        return 0;

    MPI_Comm_size(communicator, &communicator_size);
    if (!dSymLDLV2DetectAvailableMemory(communicator, &available,
                                        error, error_size))
        return 0;
    if (request->gpu_memory_budget_bytes > 0)
    {
        available.available_gpu_bytes = request->gpu_memory_budget_bytes;
        available.gpu_memory_known = 1;
    }
    if (request->host_memory_budget_bytes > 0)
    {
        available.available_host_bytes_per_node =
            request->host_memory_budget_bytes;
        available.host_memory_known = 1;
    }
    int local_success = dSymLDLV2EnumerateGridCandidates(
        request, communicator_size, selection, error, error_size);
    int all_success = 0;
    MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (!all_success)
    {
        if (local_success)
            dSymLDLV2SetReportError(
                error, error_size,
                "SymLDL grid candidate enumeration failed on another MPI rank.");
        dSymLDLV2GridSelectionDestroy(selection);
        return 0;
    }
    local_candidate_count = (unsigned long long) selection->candidate_count;
    MPI_Allreduce(&local_candidate_count, &minimum_candidate_count, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
    MPI_Allreduce(&local_candidate_count, &maximum_candidate_count, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
    if (minimum_candidate_count != maximum_candidate_count)
    {
        dSymLDLV2GridSelectionDestroy(selection);
        dSymLDLV2SetReportError(
            error, error_size,
            "SymLDL grid candidate counts differ between MPI ranks.");
        return 0;
    }
    if (!dSymLDLV2DiscoverGridTopology(
            communicator, runtime->rank_order, &topology,
            error, error_size))
    {
        dSymLDLV2GridSelectionDestroy(selection);
        return 0;
    }
    selection->available_gpu_bytes = available.available_gpu_bytes;
    selection->available_host_bytes_per_node =
        available.available_host_bytes_per_node;
    selection->input_nnz = structure->input_nnz;
    selection->symbolic_row_entries = structure->symbolic_row_entries;
    selection->factor_value_entries = structure->factor_value_entries;
    selection->ranks_per_node = available.ranks_per_node;
    selection->ranks_per_gpu = available.ranks_per_gpu;
    selection->node_count = topology.node_count;
    selection->topology_known = topology.topology_known;
    selection->rank_order_xy =
        runtime->rank_order == DSYMLDL_V2_RANK_ORDER_XY_MAJOR;

    memset(&model, 0, sizeof(model));
    model.partition_input = partition_input;
    model.structure = structure;
    model.available = available;
    model.topology = &topology;
    model.runtime = *runtime;
    model.gpu_reserve_bytes = SUPERLU_MAX(
        request->gpu_memory_reserve_bytes,
        dSymLDLV2DefaultGPUReserve(available.available_gpu_bytes,
                                   available.ranks_per_gpu));
    model.host_reserve_bytes_per_node =
        request->host_memory_reserve_bytes > 0
            ? request->host_memory_reserve_bytes
            : dSymLDLV2DefaultHostReserve(
                  available.available_host_bytes_per_node);
    selection->usable_gpu_bytes =
        available.gpu_memory_known
            ? (available.available_gpu_bytes > model.gpu_reserve_bytes
                   ? available.available_gpu_bytes - model.gpu_reserve_bytes
                   : 0)
            : 0;
    selection->usable_host_bytes_per_node =
        available.host_memory_known
            ? (available.available_host_bytes_per_node >
                       model.host_reserve_bytes_per_node
                   ? available.available_host_bytes_per_node -
                         model.host_reserve_bytes_per_node
                   : 0)
            : 0;

    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        dSymLDLV2PartitionPlan plan;
        memset(&plan, 0, sizeof(plan));
        local_success = dSymLDLV2BuildPartitionPlan(
            partition_input, candidate->pr, candidate->pc, candidate->pz,
            &plan, error, error_size);
        if (local_success)
            local_success = dSymLDLV2EvaluateGridCandidate(
                &model, &plan, candidate, error, error_size);
        MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN,
                      communicator);
        dSymLDLV2PartitionPlanDestroy(&plan);
        if (!all_success)
        {
            if (local_success)
                dSymLDLV2SetReportError(
                    error, error_size,
                    "SymLDL grid candidate evaluation failed on another MPI rank.");
            dSymLDLV2GridSelectionDestroy(selection);
            dSymLDLV2GridTopologyDestroy(&topology);
            return 0;
        }
    }

    int selected = dSymLDLV2SelectGrid(selection, error, error_size);
    if (selected && calibrate && dSymLDLV2GridCalibrationEnabled())
    {
        dSymLDLV2CalibrationProfile profile;
        dSymLDLV2CalibrationProfileInit(&profile);
        char calibration_error[256] = {0};
        double budget = dSymLDLV2CalibrationBudgetSeconds(
            topology.node_count);
        if (dSymLDLV2CalibrateGridModel(
                communicator, &topology, runtime, partition_input,
                selection, budget,
                &profile, calibration_error, sizeof(calibration_error)))
        {
            if (!dSymLDLV2SelectGridCalibrated(
                    selection, &profile, error, error_size))
            {
                selected = dSymLDLV2SelectGrid(
                    selection, error, error_size);
                profile.complete = 0;
                snprintf(profile.diagnostic, sizeof(profile.diagnostic),
                         "calibrated ranking failed; using structural selection");
                selection->calibration = profile;
                selection->confidence =
                    DSYMLDL_V2_GRID_CONFIDENCE_UNCALIBRATED_FALLBACK;
                if (selected && error != NULL && error_size > 0)
                    error[0] = '\0';
            }
        }
        else
        {
            selection->calibration = profile;
            selection->confidence =
                DSYMLDL_V2_GRID_CONFIDENCE_UNCALIBRATED_FALLBACK;
            if (error != NULL && error_size > 0)
                error[0] = '\0';
        }
    }
    else if (selected && calibrate)
    {
        selection->confidence =
            DSYMLDL_V2_GRID_CONFIDENCE_UNCALIBRATED_FALLBACK;
        snprintf(selection->calibration.diagnostic,
                 sizeof(selection->calibration.diagnostic),
                 "calibration disabled by SYMLDL_V2_GRID_CALIBRATION");
    }
    int selected_configuration[4] = {-1, -1, -1, -1};
    if (selected)
    {
        const dSymLDLV2GridCandidate *choice =
            &selection->candidates[selection->selected_index];
        selected_configuration[0] = choice->pr;
        selected_configuration[1] = choice->pc;
        selected_configuration[2] = choice->pz;
        selected_configuration[3] =
            choice->performance.estimated_gpu_streams;
    }
    int minimum_configuration[4];
    int maximum_configuration[4];
    MPI_Allreduce(selected_configuration, minimum_configuration, 4, MPI_INT,
                  MPI_MIN, communicator);
    MPI_Allreduce(selected_configuration, maximum_configuration, 4, MPI_INT,
                  MPI_MAX, communicator);
    if (memcmp(minimum_configuration, maximum_configuration,
               sizeof(minimum_configuration)) != 0)
    {
        dSymLDLV2GridSelectionDestroy(selection);
        dSymLDLV2GridTopologyDestroy(&topology);
        dSymLDLV2SetReportError(
            error, error_size,
            "SymLDL grid selection differs between MPI ranks.");
        return 0;
    }
    dSymLDLV2GridTopologyDestroy(&topology);
    return selected;
}

static int dSymLDLV2ReportCandidateBetter(
    const dSymLDLV2GridCandidate *left,
    const dSymLDLV2GridCandidate *right)
{
    if (right == NULL)
        return 1;
    if (left->performance.predicted_seconds > 0.0 ||
        right->performance.predicted_seconds > 0.0)
    {
        if (left->performance.predicted_seconds !=
            right->performance.predicted_seconds)
            return left->performance.predicted_seconds <
                   right->performance.predicted_seconds;
    }
    if (left->performance.worst_case_regret !=
        right->performance.worst_case_regret)
        return left->performance.worst_case_regret <
               right->performance.worst_case_regret;
    if (left->performance.summed_regret !=
        right->performance.summed_regret)
        return left->performance.summed_regret <
               right->performance.summed_regret;
    if (left->pz != right->pz)
        return left->pz < right->pz;
    if (left->pr != right->pr)
        return left->pr < right->pr;
    return left->pc < right->pc;
}

static const char *dSymLDLV2CalibrationSourceName(
    dSymLDLV2CalibrationSource source)
{
    switch (source)
    {
    case DSYMLDL_V2_CALIBRATION_MEASURED:
        return "measured";
    case DSYMLDL_V2_CALIBRATION_DERIVED:
        return "derived";
    default:
        return "unavailable";
    }
}

void dSymLDLV2PrintGridSelection(
    const dSymLDLV2GridSelection *selection,
    int current_pr, int current_pc, int current_pz, int report_only)
{
    if (selection == NULL || selection->candidates == NULL)
        return;

    int selected = selection->selected_index != DSYMLDL_V2_GRID_NO_SELECTION;
    printf("SymLDL automatic grid selection%s:\n",
           report_only ? " (report only)" : "");
    printf("  ranks=%d nodes=%d ranks_per_node=%d ranks_per_gpu=%d rank_order=%s\n",
           selection->total_ranks, selection->node_count,
           selection->ranks_per_node, selection->ranks_per_gpu,
           selection->rank_order_xy ? "XY" : "Z");
    if (selected)
    {
        const dSymLDLV2GridCandidate *choice =
            &selection->candidates[selection->selected_index];
        printf("  selected=%dx%dx%d confidence=%s pareto=%zu/%zu\n",
               choice->pr, choice->pc, choice->pz,
               dSymLDLV2GridConfidenceString(selection->confidence),
               selection->pareto_count, selection->candidate_count);
        if (selection->calibration.complete)
            printf("  predicted_factor=%.4f s [%.4f, %.4f] calibration=%.3f s%s\n",
                   choice->performance.predicted_seconds,
                   choice->performance.predicted_seconds_lower,
                   choice->performance.predicted_seconds_upper,
                   selection->calibration.elapsed_seconds,
                   selection->calibration.cache_hit ? " (cached)" : "");
        else if (selection->calibration.diagnostic[0] != '\0')
            printf("  calibration=%s\n",
                   selection->calibration.diagnostic);
        printf("  predicted_peak: GPU/rank=%.2f GiB host/node=%.2f GiB streams=%d/%d\n",
               dSymLDLV2GiB(choice->memory.gpu_high_water_per_rank),
               dSymLDLV2GiB(choice->memory.host_high_water_per_node),
               choice->performance.estimated_gpu_streams,
               choice->performance.requested_gpu_streams);

        size_t alternatives[5];
        for (int position = 0; position < 5; ++position)
            alternatives[position] = DSYMLDL_V2_GRID_NO_SELECTION;
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            const dSymLDLV2GridCandidate *candidate =
                &selection->candidates[i];
            if (i == selection->selected_index ||
                candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
                candidate->performance.pareto_dominated ||
                (selection->calibration.complete &&
                 !candidate->performance.plausible_alternative))
                continue;
            for (int position = 0; position < 5; ++position)
            {
                if (alternatives[position] ==
                        DSYMLDL_V2_GRID_NO_SELECTION ||
                    dSymLDLV2ReportCandidateBetter(
                        candidate,
                        &selection->candidates[alternatives[position]]))
                {
                    for (int shift = 4; shift > position; --shift)
                        alternatives[shift] = alternatives[shift - 1];
                    alternatives[position] = i;
                    break;
                }
            }
        }
        if (alternatives[0] != DSYMLDL_V2_GRID_NO_SELECTION)
        {
            printf("  alternatives:");
            for (int position = 0; position < 5; ++position)
            {
                if (alternatives[position] ==
                    DSYMLDL_V2_GRID_NO_SELECTION)
                    continue;
                const dSymLDLV2GridCandidate *alternative =
                    &selection->candidates[alternatives[position]];
                if (selection->calibration.complete)
                    printf(" %dx%dx%d(%.4f s [%.4f, %.4f])",
                           alternative->pr, alternative->pc,
                           alternative->pz,
                           alternative->performance.predicted_seconds,
                           alternative->performance.predicted_seconds_lower,
                           alternative->performance.predicted_seconds_upper);
                else
                    printf(" %dx%dx%d(regret=%.3f)",
                           alternative->pr, alternative->pc,
                           alternative->pz,
                           alternative->performance.worst_case_regret);
            }
            printf("\n");
        }
    }
    else
        printf("  no feasible automatic candidate\n");

    if (dSymLDLV2BooleanEnvironment("SYMLDL_V2_GRID_REPORT_DETAIL", 0))
    {
        printf("\n  structure: input_nnz=%" PRIu64
               " symbolic_rows=%" PRIu64 " factor_values=%" PRIu64 "\n",
               selection->input_nnz, selection->symbolic_row_entries,
               selection->factor_value_entries);
        printf("  available: GPU/rank=");
        if (selection->available_gpu_bytes > 0)
            printf("%.2f GiB", dSymLDLV2GiB(
                       selection->available_gpu_bytes));
        else
            printf("unknown");
        printf(" host/node=");
        if (selection->available_host_bytes_per_node > 0)
            printf("%.2f GiB", dSymLDLV2GiB(
                       selection->available_host_bytes_per_node));
        else
            printf("unknown");
        printf(" topology=%s\n",
               selection->topology_known ? "discovered" : "unknown");

        if (selection->calibration.complete)
        {
            printf("\n  calibration: backend=%s budget=%.2f s elapsed=%.3f s OMP=%d ranks/GPU=%d device=%s\n",
                   selection->calibration.backend_cuda ? "CUDA" : "CPU",
                   selection->calibration.budget_seconds,
                   selection->calibration.elapsed_seconds,
                   selection->calibration.omp_threads,
                   selection->calibration.ranks_per_gpu,
                   selection->calibration.device_identity[0] != '\0'
                       ? selection->calibration.device_identity : "none");
            for (int metric = 0;
                 metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT; ++metric)
            {
                const dSymLDLV2CalibrationCoefficient *coefficient =
                    &selection->calibration.coefficient[metric];
                if (coefficient->source ==
                    DSYMLDL_V2_CALIBRATION_UNAVAILABLE)
                    continue;
                printf("    %-24s %12.5e [%12.5e,%12.5e] samples=%d dispersion=%.3f source=%s\n",
                       dSymLDLV2RuntimeMetricName(
                           (dSymLDLV2RuntimeMetricKind) metric),
                       coefficient->seconds_per_unit,
                       coefficient->lower_seconds_per_unit,
                       coefficient->upper_seconds_per_unit,
                       coefficient->samples,
                       coefficient->relative_dispersion,
                       dSymLDLV2CalibrationSourceName(coefficient->source));
            }
        }

        if (selection->calibration.complete)
            printf("\n  grid      predicted interval             rank plausible pareto GPU/rank host/node streams status\n");
        else
            printf("\n  grid      regret  sum-regret pareto GPU/rank host/node streams status\n");
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            const dSymLDLV2GridCandidate *candidate =
                &selection->candidates[i];
            const char *status = i == selection->selected_index
                                     ? "selected"
                                     : dSymLDLV2GridCandidateStatusString(
                                           candidate->status);
            char current = candidate->pr == current_pr &&
                                   candidate->pc == current_pc &&
                                   candidate->pz == current_pz
                               ? '*'
                               : ' ';
            if (candidate->status == DSYMLDL_V2_GRID_FEASIBLE)
            {
                if (selection->calibration.complete)
                    printf(" %c%dx%dx%-3d %9.4f [%9.4f,%9.4f] %4d %9s %6s %7.2f %9.2f %3d/%-3d %s\n",
                           current, candidate->pr, candidate->pc,
                           candidate->pz,
                           candidate->performance.predicted_seconds,
                           candidate->performance.predicted_seconds_lower,
                           candidate->performance.predicted_seconds_upper,
                           candidate->performance.calibrated_rank,
                           candidate->performance.plausible_alternative
                               ? "yes" : "no",
                           candidate->performance.pareto_dominated
                               ? "no" : "yes",
                           dSymLDLV2GiB(
                               candidate->memory.gpu_high_water_per_rank),
                           dSymLDLV2GiB(
                               candidate->memory.host_high_water_per_node),
                           candidate->performance.estimated_gpu_streams,
                           candidate->performance.requested_gpu_streams,
                           status);
                else
                    printf(" %c%dx%dx%-3d %7.3f %10.3f %6s %7.2f %9.2f %3d/%-3d %s\n",
                           current, candidate->pr, candidate->pc,
                           candidate->pz,
                           candidate->performance.worst_case_regret,
                           candidate->performance.summed_regret,
                           candidate->performance.pareto_dominated
                               ? "no" : "yes",
                           dSymLDLV2GiB(
                               candidate->memory.gpu_high_water_per_rank),
                           dSymLDLV2GiB(
                               candidate->memory.host_high_water_per_node),
                           candidate->performance.estimated_gpu_streams,
                           candidate->performance.requested_gpu_streams,
                           status);
            }
            else
            {
                if (selection->calibration.complete)
                    printf(" %c%dx%dx%-3d %9s %21s %4s %9s %6s %7.2f %9.2f %7s %s\n",
                           current, candidate->pr, candidate->pc,
                           candidate->pz, "-", "-", "-", "-", "-",
                           dSymLDLV2GiB(
                               candidate->memory.gpu_high_water_per_rank),
                           dSymLDLV2GiB(
                               candidate->memory.host_high_water_per_node),
                           "-", status);
                else
                    printf(" %c%dx%dx%-3d %7s %10s %6s %7.2f %9.2f %7s %s\n",
                           current, candidate->pr, candidate->pc,
                           candidate->pz, "-", "-", "-",
                           dSymLDLV2GiB(
                               candidate->memory.gpu_high_water_per_rank),
                           dSymLDLV2GiB(
                               candidate->memory.host_high_water_per_node),
                           "-", status);
            }
        }

        printf("\n  Structural runtime exposures (raw units):\n");
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            const dSymLDLV2GridCandidate *candidate =
                &selection->candidates[i];
            if (candidate->performance.total_factor_flops <= 0.0)
                continue;
            printf("  %dx%dx%d total_flops=%.6e\n",
                   candidate->pr, candidate->pc, candidate->pz,
                   candidate->performance.total_factor_flops);
            for (int metric = 0;
                 metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT; ++metric)
            {
                const dSymLDLV2RuntimeMetric *value =
                    &candidate->performance.metric[metric];
                if (!value->active)
                    continue;
                printf("    %-24s total=%12.5e critical=%12.5e waiting=%12.5e",
                       dSymLDLV2RuntimeMetricName(
                           (dSymLDLV2RuntimeMetricKind) metric),
                       value->total, value->critical, value->waiting);
                if (!selection->calibration.complete)
                    printf(" regret=%.3f", value->normalized_regret);
                printf("\n");
            }
        }

        printf("\n  Detailed memory estimate (GiB):\n");
        printf("  grid      factor gpu-meta gpu-work gpu-comm gpu-solve nvshmem host-meta pinned host-comm host-dist host-solve host/rank\n");
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            const dSymLDLV2GridCandidate *candidate =
                &selection->candidates[i];
            const dSymLDLV2MemoryEstimate *memory = &candidate->memory;
            printf("  %dx%dx%-3d %6.2f %8.2f %8.2f %8.2f %9.2f %7.2f %9.2f %6.2f %9.2f %9.2f %10.2f %9.2f\n",
                   candidate->pr, candidate->pc, candidate->pz,
                   dSymLDLV2GiB(memory->gpu_persistent_factor),
                   dSymLDLV2GiB(memory->gpu_persistent_metadata),
                   dSymLDLV2GiB(memory->gpu_factor_workspace_high_water),
                   dSymLDLV2GiB(memory->gpu_communication_staging),
                   dSymLDLV2GiB(memory->gpu_solve_workspace),
                   dSymLDLV2GiB(memory->nvshmem_symmetric),
                   dSymLDLV2GiB(memory->host_persistent_metadata),
                   dSymLDLV2GiB(memory->pinned_staging),
                   dSymLDLV2GiB(memory->host_communication_staging),
                   dSymLDLV2GiB(
                       memory->host_distribution_workspace_high_water),
                   dSymLDLV2GiB(memory->host_solve_workspace),
                   dSymLDLV2GiB(memory->host_high_water_per_rank));
        }
    }
    if (current_pr > 0 && current_pc > 0 && current_pz > 0 &&
        dSymLDLV2BooleanEnvironment("SYMLDL_V2_GRID_REPORT_DETAIL", 0))
        printf("  * current explicit grid\n");

    size_t warning_index = DSYMLDL_V2_GRID_NO_SELECTION;
    for (size_t i = 0; i < selection->candidate_count; ++i)
        if (selection->candidates[i].pr == current_pr &&
            selection->candidates[i].pc == current_pc &&
            selection->candidates[i].pz == current_pz)
            warning_index = i;
    if (warning_index == DSYMLDL_V2_GRID_NO_SELECTION && selected)
        warning_index = selection->selected_index;
    if (warning_index != DSYMLDL_V2_GRID_NO_SELECTION)
    {
        const dSymLDLV2GridCandidate *candidate =
            &selection->candidates[warning_index];
        if (selection->usable_gpu_bytes > 0 &&
            candidate->memory.gpu_high_water_per_rank >=
                9 * (selection->usable_gpu_bytes / 10))
            printf("SymLDL warning: predicted GPU use for %dx%dx%d approaches the usable memory budget.\n",
                   candidate->pr, candidate->pc, candidate->pz);
        if (selection->usable_host_bytes_per_node > 0 &&
            candidate->memory.host_high_water_per_node >=
                9 * (selection->usable_host_bytes_per_node / 10))
            printf("SymLDL warning: predicted host use for %dx%dx%d approaches the usable memory budget.\n",
                   candidate->pr, candidate->pc, candidate->pz);
    }
    fflush(stdout);
}

int dSymLDLV2ReportGridCandidates(
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2StructuralSummary *structure, MPI_Comm communicator,
    int current_pr, int current_pc, int current_pz,
    const dSymLDLV2GridRuntimeConfig *runtime,
    char *error, size_t error_size)
{
    int rank;
    int local_success;
    int all_success;
    dSymLDLV2GridRequest request;
    dSymLDLV2GridSelection selection;
    dSymLDLV2MemoryEstimate automatic_prediction;
    int automatic_prediction_valid =
        dSymLDLV2GetCurrentGridPrediction(&automatic_prediction);
    int automatic_stream_cap = dSymLDLV2GetCurrentGridGPUStreamCap();

    dSymLDLV2GridSelectionInit(&selection);
    dSymLDLV2SetCurrentGridPrediction(NULL);
    dSymLDLV2SetCurrentGridGPUStreamCap(automatic_stream_cap);
    dSymLDLV2GridRequestInit(&request);
    MPI_Comm_rank(communicator, &rank);

    local_success = dSymLDLV2ParseDimensionEnvironment(
            "SYMLDL_V2_GRID_REPORT_PR", &request.pr, error, error_size) &&
        dSymLDLV2ParseDimensionEnvironment(
            "SYMLDL_V2_GRID_REPORT_PC", &request.pc, error, error_size) &&
        dSymLDLV2ParseDimensionEnvironment(
            "SYMLDL_V2_GRID_REPORT_PZ", &request.pz, error, error_size) &&
        dSymLDLV2ApplyGridEnvironment(&request, error, error_size);
    MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (!all_success)
    {
        if (local_success)
            dSymLDLV2SetReportError(
                error, error_size,
                "SymLDL grid report configuration is invalid on another MPI rank.");
        else if (error != NULL && error[0] == '\0')
            dSymLDLV2SetReportError(error, error_size,
                                    "SymLDL grid memory override is invalid.");
        return 0;
    }

    int selected = dSymLDLV2SelectGridForStructure(
        partition_input, structure, communicator, &request, runtime,
        dSymLDLV2BooleanEnvironment(
            "SYMLDL_V2_GRID_REPORT_CALIBRATE", 0),
        &selection, error, error_size);
    int report_available = selection.candidates != NULL &&
                           selection.candidate_count > 0;

    for (size_t i = 0; i < selection.candidate_count; ++i)
    {
        const dSymLDLV2GridCandidate *candidate = &selection.candidates[i];
        if (candidate->pr == current_pr && candidate->pc == current_pc &&
            candidate->pz == current_pz &&
            candidate->status == DSYMLDL_V2_GRID_FEASIBLE)
        {
            dSymLDLV2SetCurrentGridPrediction(&candidate->memory);
            break;
        }
    }

    if (rank == 0)
    {
        dSymLDLV2PrintGridSelection(&selection, current_pr, current_pc,
                                    current_pz, 1);
        if (!selected && dSymLDLV2CurrentGridMemoryValid)
            printf("SymLDL warning: the explicit grid exceeds the configured automatic-selection budget; execution remains explicit.\n");
        else if (!selected && !dSymLDLV2CurrentGridMemoryValid)
            printf("SymLDL warning: the explicit grid is outside the report candidate constraints; execution remains explicit.\n");
    }

    if (automatic_stream_cap > 0 && automatic_prediction_valid)
        dSymLDLV2SetCurrentGridPrediction(&automatic_prediction);
    dSymLDLV2SetCurrentGridGPUStreamCap(automatic_stream_cap);

    dSymLDLV2GridSelectionDestroy(&selection);
    if (report_available && error != NULL && error_size > 0)
        error[0] = '\0';
    return report_available;
}
