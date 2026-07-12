#include "dsymldl_v2_grid_calibration.h"

#if defined(GPU_ACC) && defined(HAVE_CUDA)
#include "dsymldl_v2_grid_calibration_gpu.h"
#include "gpu_wrapper.h"
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define DSYMLDL_V2_CALIBRATION_MAX_SAMPLES 64
#define DSYMLDL_V2_CALIBRATION_COMM_SIZES 4

static dSymLDLV2CalibrationProfile dSymLDLV2CachedProfile;
static int dSymLDLV2CachedProfileValid;
static int dSymLDLV2CachedGPUOffload;

static void dSymLDLV2SetCalibrationError(char *error, size_t error_size,
                                         const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2CalibrationBoolean(const char *name, int fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0')
        return fallback;
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 && strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

void dSymLDLV2CalibrationProfileInit(
    dSymLDLV2CalibrationProfile *profile)
{
    if (profile != NULL)
        memset(profile, 0, sizeof(*profile));
}

int dSymLDLV2GridCalibrationEnabled(void)
{
    return dSymLDLV2CalibrationBoolean("SYMLDL_V2_GRID_CALIBRATION", 1);
}

double dSymLDLV2CalibrationBudgetSeconds(int node_count)
{
    double nodes = node_count > 0 ? (double) node_count : 1.0;
    double budget = 2.0 + 0.5 * (log(nodes) / log(2.0));
    return fmax(2.0, fmin(5.0, budget));
}

static int dSymLDLV2CompareDouble(const void *left, const void *right)
{
    double a = *(const double *) left;
    double b = *(const double *) right;
    return a < b ? -1 : a > b;
}

static double dSymLDLV2Median(const double *values, int count)
{
    double copy[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    if (count <= 0)
        return 0.0;
    if (count > DSYMLDL_V2_CALIBRATION_MAX_SAMPLES)
        count = DSYMLDL_V2_CALIBRATION_MAX_SAMPLES;
    memcpy(copy, values, (size_t) count * sizeof(double));
    qsort(copy, (size_t) count, sizeof(double), dSymLDLV2CompareDouble);
    return count & 1 ? copy[count / 2]
                     : 0.5 * (copy[count / 2 - 1] + copy[count / 2]);
}

static void dSymLDLV2SetMeasuredCoefficient(
    const double *samples, int count,
    dSymLDLV2CalibrationCoefficient *coefficient)
{
    double deviations[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    double point = dSymLDLV2Median(samples, count);
    for (int i = 0; i < count; ++i)
        deviations[i] = fabs(samples[i] - point);
    double scaled_mad = 1.4826 * dSymLDLV2Median(deviations, count);
    double width = fmax(3.0 * scaled_mad, 0.05 * point);
    memset(coefficient, 0, sizeof(*coefficient));
    coefficient->seconds_per_unit = point;
    coefficient->lower_seconds_per_unit = fmax(0.0, point - width);
    coefficient->upper_seconds_per_unit = point + width;
    coefficient->relative_dispersion =
        point > 0.0 ? scaled_mad / point : 0.0;
    coefficient->samples = count;
    coefficient->source = DSYMLDL_V2_CALIBRATION_MEASURED;
}

static void dSymLDLV2SetZeroCoefficient(
    dSymLDLV2CalibrationCoefficient *coefficient)
{
    memset(coefficient, 0, sizeof(*coefficient));
    coefficient->source = DSYMLDL_V2_CALIBRATION_DERIVED;
}

static int dSymLDLV2CalibrateCPU(
    MPI_Comm communicator, int representative_size, double deadline,
    dSymLDLV2CalibrationCoefficient *flops,
    dSymLDLV2CalibrationCoefficient *bytes)
{
    int n = SUPERLU_MAX(64, SUPERLU_MIN(256, representative_size));
    size_t matrix_values = (size_t) n * (size_t) n;
    size_t memory_bytes = 16u * 1024u * 1024u;
    double *a = (double *) malloc(matrix_values * sizeof(double));
    double *b = (double *) malloc(matrix_values * sizeof(double));
    double *c = (double *) malloc(matrix_values * sizeof(double));
    unsigned char *source = (unsigned char *) malloc(memory_bytes);
    unsigned char *destination = (unsigned char *) malloc(memory_bytes);
    double flop_samples[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    double byte_samples[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    int samples = 0;
    int rank;
    MPI_Comm_rank(communicator, &rank);
    int local_ok = a != NULL && b != NULL && c != NULL && source != NULL &&
                   destination != NULL;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, communicator);
    if (!all_ok)
    {
        free(a); free(b); free(c); free(source); free(destination);
        return 0;
    }
    for (size_t i = 0; i < matrix_values; ++i)
        a[i] = b[i] = 1.0 / (double) (1 + (i % 31));
    memset(c, 0, matrix_values * sizeof(double));
    memset(source, 1, memory_bytes);
    memset(destination, 0, memory_bytes);

    superlu_dgemm("N", "N", n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    do
    {
        double begin = MPI_Wtime();
        superlu_dgemm("N", "N", n, n, n, 1.0, a, n, b, n, 0.0, c, n);
        double elapsed = MPI_Wtime() - begin;
        double maximum = 0.0;
        MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                      communicator);
        flop_samples[samples] = maximum /
            (2.0 * (double) n * (double) n * (double) n);

        begin = MPI_Wtime();
        memcpy(destination, source, memory_bytes);
        elapsed = MPI_Wtime() - begin;
        MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                      communicator);
        byte_samples[samples] = maximum / (2.0 * (double) memory_bytes);
        ++samples;
    } while (samples < 7 ||
             (samples < DSYMLDL_V2_CALIBRATION_MAX_SAMPLES &&
              MPI_Wtime() < deadline));

    /* Prevent an optimizing compiler from proving the copies dead. */
    if (destination[(size_t) rank % memory_bytes] == 255)
        source[0] = destination[0];
    dSymLDLV2SetMeasuredCoefficient(flop_samples, samples, flops);
    dSymLDLV2SetMeasuredCoefficient(byte_samples, samples, bytes);
    free(a); free(b); free(c); free(source); free(destination);
    return 1;
}

static void dSymLDLV2FitAlphaBeta(
    const size_t *sizes, const double *times, double *alpha, double *beta)
{
    double slopes[DSYMLDL_V2_CALIBRATION_COMM_SIZES *
                  DSYMLDL_V2_CALIBRATION_COMM_SIZES];
    int slope_count = 0;
    for (int i = 0; i < DSYMLDL_V2_CALIBRATION_COMM_SIZES; ++i)
        for (int j = i + 1; j < DSYMLDL_V2_CALIBRATION_COMM_SIZES; ++j)
        {
            double slope = (times[j] - times[i]) /
                           (2.0 * (double) (sizes[j] - sizes[i]));
            slopes[slope_count++] = fmax(0.0, slope);
        }
    *beta = dSymLDLV2Median(slopes, slope_count);
    double intercepts[DSYMLDL_V2_CALIBRATION_COMM_SIZES];
    for (int i = 0; i < DSYMLDL_V2_CALIBRATION_COMM_SIZES; ++i)
        intercepts[i] = fmax(
            0.0, 0.5 * times[i] - *beta * (double) sizes[i]);
    *alpha = dSymLDLV2Median(intercepts,
                             DSYMLDL_V2_CALIBRATION_COMM_SIZES);
}

static int dSymLDLV2CalibrateCommunication(
    MPI_Comm communicator, MPI_Comm traffic_communicator, double deadline,
    const size_t *sizes,
    dSymLDLV2CalibrationCoefficient *messages,
    dSymLDLV2CalibrationCoefficient *bytes)
{
    int traffic_rank = 0;
    int traffic_size = 0;
    MPI_Comm_rank(traffic_communicator, &traffic_rank);
    MPI_Comm_size(traffic_communicator, &traffic_size);
    if (traffic_size <= 1)
    {
        dSymLDLV2SetZeroCoefficient(messages);
        dSymLDLV2SetZeroCoefficient(bytes);
        return 1;
    }
    unsigned char *send_buffer = (unsigned char *) malloc(
        sizes[DSYMLDL_V2_CALIBRATION_COMM_SIZES - 1]);
    unsigned char *receive_buffer = (unsigned char *) malloc(
        sizes[DSYMLDL_V2_CALIBRATION_COMM_SIZES - 1]);
    int local_ok = send_buffer != NULL && receive_buffer != NULL;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, communicator);
    if (!all_ok)
    {
        free(send_buffer); free(receive_buffer);
        return 0;
    }
    memset(send_buffer, traffic_rank, sizes[3]);
    double alpha_samples[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    double beta_samples[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    int samples = 0;
    int destination = (traffic_rank + 1) % traffic_size;
    int source = (traffic_rank + traffic_size - 1) % traffic_size;
    do
    {
        double times[DSYMLDL_V2_CALIBRATION_COMM_SIZES];
        for (int i = 0; i < DSYMLDL_V2_CALIBRATION_COMM_SIZES; ++i)
        {
            MPI_Barrier(traffic_communicator);
            double begin = MPI_Wtime();
            MPI_Sendrecv(send_buffer, (int) sizes[i], MPI_BYTE,
                         destination, 931 + i, receive_buffer,
                         (int) sizes[i], MPI_BYTE, source, 931 + i,
                         traffic_communicator, MPI_STATUS_IGNORE);
            double elapsed = MPI_Wtime() - begin;
            MPI_Allreduce(&elapsed, &times[i], 1, MPI_DOUBLE, MPI_MAX,
                          communicator);
        }
        dSymLDLV2FitAlphaBeta(sizes, times,
                              &alpha_samples[samples],
                              &beta_samples[samples]);
        ++samples;
    } while (samples < 7 ||
             (samples < DSYMLDL_V2_CALIBRATION_MAX_SAMPLES &&
              MPI_Wtime() < deadline));
    dSymLDLV2SetMeasuredCoefficient(alpha_samples, samples, messages);
    dSymLDLV2SetMeasuredCoefficient(beta_samples, samples, bytes);
    free(send_buffer); free(receive_buffer);
    return 1;
}

static void dSymLDLV2RepresentativeMessageSizes(
    const dSymLDLV2GridSelection *selection, size_t *sizes)
{
    uint64_t histogram[DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS] = {0};
    uint64_t count = 0;
    sizes[0] = 8;
    sizes[1] = 4096;
    sizes[2] = 65536;
    sizes[3] = 1048576;
    if (selection == NULL || selection->candidates == NULL)
        return;
    for (size_t candidate_index = 0;
         candidate_index < selection->candidate_count; ++candidate_index)
    {
        const dSymLDLV2GridCandidate *candidate =
            &selection->candidates[candidate_index];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        for (int bin = 0; bin < DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS; ++bin)
        {
            uint64_t entries =
                candidate->performance.communication_size_histogram[bin];
            histogram[bin] = histogram[bin] > UINT64_MAX - entries
                                 ? UINT64_MAX : histogram[bin] + entries;
            count = count > UINT64_MAX - entries
                        ? UINT64_MAX : count + entries;
        }
    }
    if (count == 0)
        return;
    static const double quantiles[3] = {0.25, 0.50, 0.90};
    for (int q = 0; q < 3; ++q)
    {
        uint64_t target = (uint64_t) ceil(quantiles[q] * (double) count);
        uint64_t cumulative = 0;
        int bin = 0;
        while (bin + 1 < DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS)
        {
            cumulative += histogram[bin];
            if (cumulative >= target)
                break;
            ++bin;
        }
        uint64_t bytes = bin >= 63 ? UINT64_MAX : (UINT64_C(1) << bin);
        bytes = SUPERLU_MAX(UINT64_C(8),
                            SUPERLU_MIN(UINT64_C(8) * 1024 * 1024, bytes));
        sizes[q + 1] = (size_t) bytes;
    }
    for (int i = 1; i < DSYMLDL_V2_CALIBRATION_COMM_SIZES; ++i)
        if (sizes[i] <= sizes[i - 1])
            sizes[i] = SUPERLU_MIN(
                (size_t) 8 * 1024 * 1024, sizes[i - 1] * 2);
    if (sizes[1] <= sizes[0] || sizes[2] <= sizes[1] ||
        sizes[3] <= sizes[2])
    {
        sizes[0] = 8;
        sizes[1] = 4096;
        sizes[2] = 65536;
        sizes[3] = 1048576;
    }
}

static int dSymLDLV2RepresentativeSupernodeSize(
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2GridRuntimeConfig *runtime)
{
    if (partition_input == NULL || partition_input->xsup == NULL ||
        partition_input->nsupers <= 0)
        return SUPERLU_MAX(64, runtime->max_supernode_size);
    long double weighted_width = 0.0;
    long double weight = 0.0;
    for (int_t k = 0; k < partition_input->nsupers; ++k)
    {
        int_t width = partition_input->xsup[k + 1] -
                      partition_input->xsup[k];
        if (width <= 0)
            continue;
        long double cubic = (long double) width * (long double) width *
                            (long double) width;
        weight += cubic;
        weighted_width += cubic * (long double) width;
    }
    if (weight <= 0.0)
        return SUPERLU_MAX(64, runtime->max_supernode_size);
    long double representative = weighted_width / weight;
    if (representative > (long double) INT_MAX)
        return INT_MAX;
    return SUPERLU_MAX(64, (int) llround(representative));
}

static int dSymLDLV2CalibrateBarrier(
    MPI_Comm communicator, double deadline,
    dSymLDLV2CalibrationCoefficient *coefficient)
{
    double samples[DSYMLDL_V2_CALIBRATION_MAX_SAMPLES];
    int count = 0;
    do
    {
        double begin = MPI_Wtime();
        MPI_Barrier(communicator);
        double elapsed = MPI_Wtime() - begin;
        MPI_Allreduce(&elapsed, &samples[count], 1, MPI_DOUBLE, MPI_MAX,
                      communicator);
        ++count;
    } while (count < 9 ||
             (count < DSYMLDL_V2_CALIBRATION_MAX_SAMPLES &&
              MPI_Wtime() < deadline));
    dSymLDLV2SetMeasuredCoefficient(samples, count, coefficient);
    return 1;
}

static int dSymLDLV2ProfileMatchesCache(
    const dSymLDLV2CalibrationProfile *profile,
    const dSymLDLV2GridTopology *topology,
    const dSymLDLV2GridRuntimeConfig *runtime, int communicator_size,
    int ranks_per_node)
{
    return dSymLDLV2CachedProfileValid && profile != NULL &&
           dSymLDLV2CachedProfile.complete &&
           dSymLDLV2CachedGPUOffload == runtime->gpu_offload &&
           dSymLDLV2CachedProfile.communicator_size == communicator_size &&
           dSymLDLV2CachedProfile.node_count == topology->node_count &&
           dSymLDLV2CachedProfile.ranks_per_node == ranks_per_node &&
           dSymLDLV2CachedProfile.ranks_per_gpu == profile->ranks_per_gpu &&
           dSymLDLV2CachedProfile.omp_threads == profile->omp_threads &&
           dSymLDLV2CachedProfile.topology_hash == profile->topology_hash &&
           strcmp(dSymLDLV2CachedProfile.device_identity,
                  profile->device_identity) == 0 &&
           strcmp(dSymLDLV2CachedProfile.mpi_identity,
                  profile->mpi_identity) == 0;
}

int dSymLDLV2CalibrateGridModel(
    MPI_Comm communicator, const dSymLDLV2GridTopology *topology,
    const dSymLDLV2GridRuntimeConfig *runtime,
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2GridSelection *selection,
    double budget_seconds, dSymLDLV2CalibrationProfile *profile,
    char *error, size_t error_size)
{
    int rank, size;
    int all_success = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    dSymLDLV2CalibrationProfileInit(profile);
    profile->attempted = 1;
    profile->budget_seconds = budget_seconds;
    profile->communicator_size = size;
    profile->node_count = topology != NULL ? topology->node_count : 0;
    profile->ranks_per_node = topology != NULL && topology->node_count > 0
                                  ? size / topology->node_count
                                  : size;
    profile->ranks_per_gpu = selection != NULL
                                 ? selection->ranks_per_gpu : 0;
#ifdef _OPENMP
    profile->omp_threads = omp_get_max_threads();
#else
    profile->omp_threads = 1;
#endif
    profile->topology_hash = UINT64_C(1469598103934665603);
    if (topology != NULL && topology->node_of_rank != NULL)
        for (int communicator_rank = 0; communicator_rank < size;
             ++communicator_rank)
        {
            profile->topology_hash ^=
                (uint64_t) (unsigned int)
                    topology->node_of_rank[communicator_rank];
            profile->topology_hash *= UINT64_C(1099511628211);
        }
    int mpi_version_length = 0;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    memset(mpi_version, 0, sizeof(mpi_version));
    MPI_Get_library_version(mpi_version, &mpi_version_length);
    snprintf(profile->mpi_identity, sizeof(profile->mpi_identity), "%.*s",
             (int) sizeof(profile->mpi_identity) - 1, mpi_version);
#if defined(GPU_ACC) && defined(HAVE_CUDA)
    if (runtime->gpu_offload)
    {
        int device_count = 0;
        struct cudaDeviceProp properties;
        if (gpuGetDeviceCount(&device_count) == gpuSuccess &&
            device_count > 0 &&
            gpuGetDeviceProperties(&properties, rank % device_count) ==
                gpuSuccess)
            snprintf(profile->device_identity,
                     sizeof(profile->device_identity), "%.*s",
                     (int) sizeof(profile->device_identity) - 1,
                     properties.name);
    }
#else
    snprintf(profile->device_identity, sizeof(profile->device_identity),
             "CPU");
#endif
    if (dSymLDLV2ProfileMatchesCache(
            profile, topology, runtime, size, profile->ranks_per_node))
    {
        *profile = dSymLDLV2CachedProfile;
        profile->cache_hit = 1;
        profile->elapsed_seconds = 0.0;
        return 1;
    }

    MPI_Comm calibration_comm = MPI_COMM_NULL;
    MPI_Comm shared_comm = MPI_COMM_NULL;
    MPI_Comm inter_comm = MPI_COMM_NULL;
    double started = MPI_Wtime();
    size_t communication_sizes[DSYMLDL_V2_CALIBRATION_COMM_SIZES];
    dSymLDLV2RepresentativeMessageSizes(selection, communication_sizes);
    int success = MPI_Comm_dup(communicator, &calibration_comm) == MPI_SUCCESS;
    MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_MIN, communicator);
    if (!all_success)
        goto failed;
    MPI_Comm_set_errhandler(calibration_comm, MPI_ERRORS_RETURN);
#if MPI_VERSION >= 3
    success = MPI_Comm_split_type(
        calibration_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
        &shared_comm) == MPI_SUCCESS;
#else
    success = topology != NULL && topology->node_of_rank != NULL &&
              MPI_Comm_split(calibration_comm, topology->node_of_rank[rank],
                             rank, &shared_comm) == MPI_SUCCESS;
#endif
    MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_MIN, communicator);
    if (!all_success)
        goto failed;
    MPI_Comm_set_errhandler(shared_comm, MPI_ERRORS_RETURN);
    int shared_rank = 0;
    MPI_Comm_rank(shared_comm, &shared_rank);
    success = MPI_Comm_split(calibration_comm, shared_rank, rank,
                             &inter_comm) == MPI_SUCCESS;
    MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_MIN, communicator);
    if (!all_success)
        goto failed;
    MPI_Comm_set_errhandler(inter_comm, MPI_ERRORS_RETURN);

    int representative_size = dSymLDLV2RepresentativeSupernodeSize(
        partition_input, runtime);
    double slice = budget_seconds / (runtime->gpu_offload ? 5.0 : 4.0);
    success = dSymLDLV2CalibrateCPU(
        calibration_comm, representative_size, started + slice,
        &profile->coefficient[DSYMLDL_V2_RUNTIME_CPU_FLOPS],
        &profile->coefficient[DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES]);
    success = success && dSymLDLV2CalibrateCommunication(
        calibration_comm, shared_comm, started + 2.0 * slice,
        communication_sizes,
        &profile->coefficient[DSYMLDL_V2_RUNTIME_INTRA_NODE_MESSAGES],
        &profile->coefficient[DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES]);
    success = success && dSymLDLV2CalibrateCommunication(
        calibration_comm, inter_comm, started + 3.0 * slice,
        communication_sizes,
        &profile->coefficient[DSYMLDL_V2_RUNTIME_INTER_NODE_MESSAGES],
        &profile->coefficient[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES]);
    success = success && dSymLDLV2CalibrateBarrier(
        calibration_comm, started + 4.0 * slice,
        &profile->coefficient[
            DSYMLDL_V2_RUNTIME_PROCESS_SYNCHRONIZATIONS]);

#if defined(GPU_ACC) && defined(HAVE_CUDA)
    if (success && runtime->gpu_offload)
    {
        int device_count = 0;
        success = gpuGetDeviceCount(&device_count) == gpuSuccess &&
                  device_count > 0;
        if (success)
        {
            int device_id = rank % device_count;
            success = dSymLDLV2CalibrateCUDADevice(
                device_id, representative_size,
                fmax(0.25, budget_seconds - (MPI_Wtime() - started)),
                &profile->coefficient[DSYMLDL_V2_RUNTIME_GPU_FLOPS],
                &profile->coefficient[
                    DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES],
                &profile->coefficient[DSYMLDL_V2_RUNTIME_TASK_LAUNCHES],
                &profile->coefficient[
                    DSYMLDL_V2_RUNTIME_GPU_SYNCHRONIZATIONS],
                &profile->coefficient[
                    DSYMLDL_V2_RUNTIME_HOST_TO_DEVICE_BYTES],
                &profile->coefficient[
                    DSYMLDL_V2_RUNTIME_DEVICE_TO_HOST_BYTES],
                error, error_size);
            profile->backend_cuda = success;
        }
        MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN,
                      calibration_comm);
        if (success)
        {
            for (int metric = DSYMLDL_V2_RUNTIME_GPU_FLOPS;
                 metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT; ++metric)
            {
                dSymLDLV2CalibrationCoefficient *coefficient =
                    &profile->coefficient[metric];
                if (coefficient->source == DSYMLDL_V2_CALIBRATION_UNAVAILABLE)
                    continue;
                MPI_Allreduce(MPI_IN_PLACE, &coefficient->seconds_per_unit,
                              1, MPI_DOUBLE, MPI_MAX, calibration_comm);
                MPI_Allreduce(MPI_IN_PLACE,
                              &coefficient->lower_seconds_per_unit,
                              1, MPI_DOUBLE, MPI_MAX, calibration_comm);
                MPI_Allreduce(MPI_IN_PLACE,
                              &coefficient->upper_seconds_per_unit,
                              1, MPI_DOUBLE, MPI_MAX, calibration_comm);
            }
        }
    }
#else
    if (runtime->gpu_offload)
        success = 0;
#endif
    if (!runtime->gpu_offload)
    {
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[DSYMLDL_V2_RUNTIME_GPU_FLOPS]);
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES]);
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[DSYMLDL_V2_RUNTIME_TASK_LAUNCHES]);
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[
                DSYMLDL_V2_RUNTIME_GPU_SYNCHRONIZATIONS]);
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[DSYMLDL_V2_RUNTIME_HOST_TO_DEVICE_BYTES]);
        dSymLDLV2SetZeroCoefficient(
            &profile->coefficient[DSYMLDL_V2_RUNTIME_DEVICE_TO_HOST_BYTES]);
    }
    MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_MIN,
                  calibration_comm);
    if (!all_success)
        goto failed;

    profile->elapsed_seconds = MPI_Wtime() - started;
    profile->complete = 1;
    snprintf(profile->diagnostic, sizeof(profile->diagnostic),
             "measured on %d rank%s across %d node%s",
             size, size == 1 ? "" : "s", profile->node_count,
             profile->node_count == 1 ? "" : "s");
    dSymLDLV2CachedProfile = *profile;
    dSymLDLV2CachedProfile.cache_hit = 0;
    dSymLDLV2CachedProfileValid = 1;
    dSymLDLV2CachedGPUOffload = runtime->gpu_offload;
    MPI_Comm_free(&inter_comm);
    MPI_Comm_free(&shared_comm);
    MPI_Comm_free(&calibration_comm);
    return 1;

failed:
    profile->elapsed_seconds = MPI_Wtime() - started;
    profile->complete = 0;
    snprintf(profile->diagnostic, sizeof(profile->diagnostic),
             "calibration incomplete; using structural selection");
    dSymLDLV2SetCalibrationError(
        error, error_size,
        "SymLDL grid calibration was incomplete; structural selection will be used.");
    if (inter_comm != MPI_COMM_NULL) MPI_Comm_free(&inter_comm);
    if (shared_comm != MPI_COMM_NULL) MPI_Comm_free(&shared_comm);
    if (calibration_comm != MPI_COMM_NULL) MPI_Comm_free(&calibration_comm);
    return 0;
}
