#include "dsymldl_v2_grid_calibration_gpu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kSamples = 11;

double median(double *values, int count)
{
    std::sort(values, values + count);
    return count & 1 ? values[count / 2]
                     : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

void set_coefficient(const double *input, int count,
                     dSymLDLV2CalibrationCoefficient *coefficient)
{
    double values[kSamples];
    double deviations[kSamples];
    std::memcpy(values, input, static_cast<size_t>(count) * sizeof(double));
    double point = median(values, count);
    for (int i = 0; i < count; ++i)
        deviations[i] = std::fabs(input[i] - point);
    double mad = 1.4826 * median(deviations, count);
    double width = std::max(3.0 * mad, 0.05 * point);
    std::memset(coefficient, 0, sizeof(*coefficient));
    coefficient->seconds_per_unit = point;
    coefficient->lower_seconds_per_unit = std::max(0.0, point - width);
    coefficient->upper_seconds_per_unit = point + width;
    coefficient->relative_dispersion = point > 0.0 ? mad / point : 0.0;
    coefficient->samples = count;
    coefficient->source = DSYMLDL_V2_CALIBRATION_MEASURED;
}

bool elapsed(cudaEvent_t begin, cudaEvent_t end, cudaStream_t stream,
             double units, double *result)
{
    float milliseconds = 0.0f;
    if (cudaEventRecord(end, stream) != cudaSuccess ||
        cudaEventSynchronize(end) != cudaSuccess ||
        cudaEventElapsedTime(&milliseconds, begin, end) != cudaSuccess)
        return false;
    *result = (1.0e-3 * static_cast<double>(milliseconds)) / units;
    return std::isfinite(*result) && *result >= 0.0;
}

}  // namespace

extern "C" int dSymLDLV2CalibrateCUDADevice(
    int device_id, int representative_size, double budget_seconds,
    dSymLDLV2CalibrationCoefficient *gpu_flops,
    dSymLDLV2CalibrationCoefficient *gpu_local_bytes,
    dSymLDLV2CalibrationCoefficient *launches,
    dSymLDLV2CalibrationCoefficient *gpu_synchronizations,
    dSymLDLV2CalibrationCoefficient *host_to_device,
    dSymLDLV2CalibrationCoefficient *device_to_host,
    char *error, size_t error_size)
{
    int n = std::max(64, std::min(256, representative_size));
    size_t matrix_values = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t matrix_bytes = matrix_values * sizeof(double);
    size_t transfer_bytes = 32u * 1024u * 1024u;
    double *a = nullptr;
    double *b = nullptr;
    double *c = nullptr;
    void *device_transfer = nullptr;
    void *device_transfer_source = nullptr;
    void *host_transfer = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cublasHandle_t handle = nullptr;
    double flop_samples[kSamples];
    double local_byte_samples[kSamples];
    double launch_samples[kSamples];
    double sync_samples[kSamples];
    double h2d_samples[kSamples];
    double d2h_samples[kSamples];
    double alpha = 1.0;
    double beta = 0.0;
    int sample_count = 0;
    int ok = 0;
    double deadline = MPI_Wtime() + std::max(0.25, budget_seconds);

#define CUDA_OK(call) do { if ((call) != cudaSuccess) goto cleanup; } while (0)
#define CUBLAS_OK(call) do { if ((call) != CUBLAS_STATUS_SUCCESS) goto cleanup; } while (0)
    CUDA_OK(cudaSetDevice(device_id));
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));
    CUBLAS_OK(cublasCreate(&handle));
    CUBLAS_OK(cublasSetStream(handle, stream));
    CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&a), matrix_bytes));
    CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&b), matrix_bytes));
    CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&c), matrix_bytes));
    CUDA_OK(cudaMalloc(&device_transfer, transfer_bytes));
    CUDA_OK(cudaMalloc(&device_transfer_source, transfer_bytes));
    CUDA_OK(cudaMallocHost(&host_transfer, transfer_bytes));
    std::memset(host_transfer, 1, transfer_bytes);
    CUDA_OK(cudaMemsetAsync(a, 0, matrix_bytes, stream));
    CUDA_OK(cudaMemsetAsync(b, 0, matrix_bytes, stream));
    CUDA_OK(cudaMemsetAsync(c, 0, matrix_bytes, stream));
    CUDA_OK(cudaMemsetAsync(device_transfer_source, 0, transfer_bytes,
                            stream));
    CUBLAS_OK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha, a, n, b, n, &beta, c, n));
    CUDA_OK(cudaStreamSynchronize(stream));

    do
    {
        int sample = sample_count;
        CUDA_OK(cudaEventRecord(begin, stream));
        CUBLAS_OK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              n, n, n, &alpha, a, n, b, n, &beta, c, n));
        if (!elapsed(begin, end, stream,
                     2.0 * static_cast<double>(n) * n * n,
                     &flop_samples[sample]))
            goto cleanup;

        CUDA_OK(cudaEventRecord(begin, stream));
        CUDA_OK(cudaMemcpyAsync(device_transfer, device_transfer_source,
                                transfer_bytes,
                                cudaMemcpyDeviceToDevice, stream));
        if (!elapsed(begin, end, stream,
                     2.0 * static_cast<double>(transfer_bytes),
                     &local_byte_samples[sample]))
            goto cleanup;

        double wall = MPI_Wtime();
        constexpr int launches_per_sample = 256;
        for (int launch = 0; launch < launches_per_sample; ++launch)
            CUDA_OK(cudaMemsetAsync(device_transfer, launch & 1, 1, stream));
        launch_samples[sample] =
            (MPI_Wtime() - wall) / static_cast<double>(launches_per_sample);
        CUDA_OK(cudaStreamSynchronize(stream));

        wall = MPI_Wtime();
        CUDA_OK(cudaStreamSynchronize(stream));
        sync_samples[sample] = MPI_Wtime() - wall;

        CUDA_OK(cudaEventRecord(begin, stream));
        CUDA_OK(cudaMemcpyAsync(device_transfer, host_transfer,
                                transfer_bytes, cudaMemcpyHostToDevice,
                                stream));
        if (!elapsed(begin, end, stream, static_cast<double>(transfer_bytes),
                     &h2d_samples[sample]))
            goto cleanup;

        CUDA_OK(cudaEventRecord(begin, stream));
        CUDA_OK(cudaMemcpyAsync(host_transfer, device_transfer,
                                transfer_bytes, cudaMemcpyDeviceToHost,
                                stream));
        if (!elapsed(begin, end, stream, static_cast<double>(transfer_bytes),
                     &d2h_samples[sample]))
            goto cleanup;
        ++sample_count;
    } while (sample_count < 7 ||
             (sample_count < kSamples && MPI_Wtime() < deadline));

    set_coefficient(flop_samples, sample_count, gpu_flops);
    set_coefficient(local_byte_samples, sample_count, gpu_local_bytes);
    set_coefficient(launch_samples, sample_count, launches);
    set_coefficient(sync_samples, sample_count, gpu_synchronizations);
    set_coefficient(h2d_samples, sample_count, host_to_device);
    set_coefficient(d2h_samples, sample_count, device_to_host);
    ok = 1;

cleanup:
    if (!ok && error != nullptr && error_size > 0)
    {
        std::snprintf(error, error_size,
                      "SymLDL CUDA grid calibration failed.");
        error[error_size - 1] = '\0';
    }
    if (handle != nullptr) cublasDestroy(handle);
    if (end != nullptr) cudaEventDestroy(end);
    if (begin != nullptr) cudaEventDestroy(begin);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (host_transfer != nullptr) cudaFreeHost(host_transfer);
    if (device_transfer_source != nullptr) cudaFree(device_transfer_source);
    if (device_transfer != nullptr) cudaFree(device_transfer);
    if (c != nullptr) cudaFree(c);
    if (b != nullptr) cudaFree(b);
    if (a != nullptr) cudaFree(a);
#undef CUBLAS_OK
#undef CUDA_OK
    return ok;
}
