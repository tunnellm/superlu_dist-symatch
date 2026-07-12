#ifndef DSYMLDL_V2_GRID_SELECTOR_H
#define DSYMLDL_V2_GRID_SELECTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SUPERLU_GRID_AUTO 0
#define DSYMLDL_V2_GRID_NO_SELECTION ((size_t) -1)
#define DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS 64

typedef enum {
    DSYMLDL_V2_GRID_FEASIBLE = 0,
    DSYMLDL_V2_GRID_REJECT_PRODUCT,
    DSYMLDL_V2_GRID_REJECT_PZ,
    DSYMLDL_V2_GRID_REJECT_LIMIT,
    DSYMLDL_V2_GRID_REJECT_HOST_MEMORY,
    DSYMLDL_V2_GRID_REJECT_GPU_MEMORY,
    DSYMLDL_V2_GRID_REJECT_IMPLEMENTATION
} dSymLDLV2GridCandidateStatus;

typedef struct {
    int pr;
    int pc;
    int pz;
    /* Zero means no additional dimension limit. */
    int max_pr;
    int max_pc;
    int max_pz;
    /* Host values are per node; GPU values are per MPI rank. */
    uint64_t host_memory_budget_bytes;
    uint64_t gpu_memory_budget_bytes;
    uint64_t host_memory_reserve_bytes;
    uint64_t gpu_memory_reserve_bytes;
} dSymLDLV2GridRequest;

typedef struct {
    uint64_t host_persistent_factor;
    uint64_t gpu_persistent_factor;
    uint64_t host_persistent_metadata;
    uint64_t gpu_persistent_metadata;
    uint64_t host_symbolic_workspace_high_water;
    uint64_t host_distribution_workspace_high_water;
    uint64_t host_factor_workspace_high_water;
    uint64_t gpu_factor_workspace_high_water;
    uint64_t host_communication_staging;
    uint64_t gpu_communication_staging;
    uint64_t pinned_staging;
    uint64_t host_solve_workspace;
    uint64_t gpu_solve_workspace;
    uint64_t nvshmem_symmetric;
    uint64_t gpu_runtime_overhead;
    uint64_t host_high_water_per_rank;
    uint64_t host_high_water_per_node;
    uint64_t gpu_high_water_per_rank;
} dSymLDLV2MemoryEstimate;

typedef enum {
    DSYMLDL_V2_RUNTIME_GPU_FLOPS = 0,
    DSYMLDL_V2_RUNTIME_CPU_FLOPS,
    DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES,
    DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES,
    DSYMLDL_V2_RUNTIME_TASK_LAUNCHES,
    DSYMLDL_V2_RUNTIME_GPU_SYNCHRONIZATIONS,
    DSYMLDL_V2_RUNTIME_PROCESS_SYNCHRONIZATIONS,
    DSYMLDL_V2_RUNTIME_INTRA_NODE_MESSAGES,
    DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES,
    DSYMLDL_V2_RUNTIME_INTER_NODE_MESSAGES,
    DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES,
    DSYMLDL_V2_RUNTIME_HOST_TO_DEVICE_BYTES,
    DSYMLDL_V2_RUNTIME_DEVICE_TO_HOST_BYTES,
    DSYMLDL_V2_RUNTIME_METRIC_COUNT
} dSymLDLV2RuntimeMetricKind;

typedef struct {
    double total;
    double critical;
    double waiting;
    double normalized_regret;
    int active;
} dSymLDLV2RuntimeMetric;

typedef enum {
    DSYMLDL_V2_GRID_CONFIDENCE_NONE = 0,
    DSYMLDL_V2_GRID_CONFIDENCE_DOMINANT,
    DSYMLDL_V2_GRID_CONFIDENCE_ROBUST_COMPROMISE,
    DSYMLDL_V2_GRID_CONFIDENCE_AMBIGUOUS_TIE,
    DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ROBUST,
    DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ESTIMATE,
    DSYMLDL_V2_GRID_CONFIDENCE_UNCALIBRATED_FALLBACK
} dSymLDLV2GridSelectionConfidence;

typedef struct {
    dSymLDLV2RuntimeMetric metric[DSYMLDL_V2_RUNTIME_METRIC_COUNT];
    double total_factor_flops;
    double worst_case_regret;
    double summed_regret;
    double predicted_seconds;
    double predicted_seconds_lower;
    double predicted_seconds_upper;
    uint64_t communication_size_histogram[
        DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS];
    int pareto_dominated;
    int pareto_dominator;
    int calibrated_rank;
    int plausible_alternative;
    int robust_winner;
    int requested_gpu_streams;
    int estimated_gpu_streams;
} dSymLDLV2PerformanceEstimate;

typedef struct {
    int pr;
    int pc;
    int pz;
    dSymLDLV2MemoryEstimate memory;
    dSymLDLV2PerformanceEstimate performance;
    dSymLDLV2GridCandidateStatus status;
} dSymLDLV2GridCandidate;

typedef enum {
    DSYMLDL_V2_CALIBRATION_UNAVAILABLE = 0,
    DSYMLDL_V2_CALIBRATION_MEASURED,
    DSYMLDL_V2_CALIBRATION_DERIVED
} dSymLDLV2CalibrationSource;

typedef struct {
    double seconds_per_unit;
    double lower_seconds_per_unit;
    double upper_seconds_per_unit;
    double relative_dispersion;
    int samples;
    dSymLDLV2CalibrationSource source;
} dSymLDLV2CalibrationCoefficient;

typedef struct {
    dSymLDLV2CalibrationCoefficient
        coefficient[DSYMLDL_V2_RUNTIME_METRIC_COUNT];
    double elapsed_seconds;
    double budget_seconds;
    int attempted;
    int complete;
    int cache_hit;
    int backend_cuda;
    int communicator_size;
    int node_count;
    int ranks_per_node;
    int ranks_per_gpu;
    int omp_threads;
    uint64_t topology_hash;
    char device_identity[96];
    char mpi_identity[96];
    char diagnostic[160];
} dSymLDLV2CalibrationProfile;

typedef struct {
    int total_ranks;
    uint64_t input_nnz;
    uint64_t symbolic_row_entries;
    uint64_t factor_value_entries;
    uint64_t available_gpu_bytes;
    uint64_t available_host_bytes_per_node;
    uint64_t usable_gpu_bytes;
    uint64_t usable_host_bytes_per_node;
    int ranks_per_node;
    int ranks_per_gpu;
    int node_count;
    int topology_known;
    int rank_order_xy;
    size_t pareto_count;
    dSymLDLV2GridSelectionConfidence confidence;
    dSymLDLV2CalibrationProfile calibration;
    dSymLDLV2GridCandidate *candidates;
    size_t candidate_count;
    size_t selected_index;
} dSymLDLV2GridSelection;

void dSymLDLV2GridRequestInit(dSymLDLV2GridRequest *request);

void dSymLDLV2GridSelectionInit(dSymLDLV2GridSelection *selection);

int dSymLDLV2ParseGridDimension(const char *text, int *dimension,
                                char *error, size_t error_size);

/* selection must be initialized before its first enumeration. */
int dSymLDLV2EnumerateGridCandidates(
    const dSymLDLV2GridRequest *request, int communicator_size,
    dSymLDLV2GridSelection *selection, char *error, size_t error_size);

int dSymLDLV2SelectGrid(dSymLDLV2GridSelection *selection,
                        char *error, size_t error_size);

int dSymLDLV2SelectGridCalibrated(
    dSymLDLV2GridSelection *selection,
    const dSymLDLV2CalibrationProfile *profile,
    char *error, size_t error_size);

const char *dSymLDLV2RuntimeMetricName(
    dSymLDLV2RuntimeMetricKind kind);

const char *dSymLDLV2GridConfidenceString(
    dSymLDLV2GridSelectionConfidence confidence);

const char *dSymLDLV2GridCandidateStatusString(
    dSymLDLV2GridCandidateStatus status);

void dSymLDLV2GridSelectionDestroy(dSymLDLV2GridSelection *selection);

#ifdef __cplusplus
}
#endif

#endif
