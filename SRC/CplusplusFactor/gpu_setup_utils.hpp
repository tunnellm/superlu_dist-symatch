#pragma once

#include <cstdlib>

#include "superlu_ddefs.h"
#include "gpuCommon.hpp"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "mpi.h"

#define SUPERLU_GPU_USABLE_MEM_FRACTION 0.9

static inline int superlu_gpu_processes_per_device(MPI_Comm baseCommunicator)
{
    const char *env = std::getenv("MPI_PROCESS_PER_GPU");
    if (env != NULL)
    {
        int envCount = std::atoi(env);
        return SUPERLU_MAX(envCount, 1);
    }

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess)
        return 1;

    MPI_Comm sharedComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED,
                            0, MPI_INFO_NULL, &sharedComm) != MPI_SUCCESS)
        return 1;

    MPI_Comm deviceComm = MPI_COMM_NULL;
    int procsPerGpu = 1;
    if (MPI_Comm_split(sharedComm, device_id, 0, &deviceComm) == MPI_SUCCESS)
    {
        MPI_Comm_size(deviceComm, &procsPerGpu);
        MPI_Comm_free(&deviceComm);
    }
    MPI_Comm_free(&sharedComm);

    return SUPERLU_MAX(procsPerGpu, 1);
}

static inline size_t superlu_gpu_memory_per_process(MPI_Comm baseCommunicator)
{
    size_t mfree, mtotal;
    cudaMemGetInfo(&mfree, &mtotal);
    return (size_t)(SUPERLU_GPU_USABLE_MEM_FRACTION * (double)mfree) /
        superlu_gpu_processes_per_device(baseCommunicator);
}

static inline int superlu_gpu_getrf_workspace_size(int ldt, int enabled)
{
    if (!enabled)
        return 0;

    int workspaceSize = 0;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    cusolverDnDgetrf_bufferSize(cusolverH, ldt, ldt, NULL, ldt,
                                &workspaceSize);
    cusolverDnDestroy(cusolverH);
    return workspaceSize;
}

static inline void superlu_gpu_create_cusolver_handle(
    cusolverDnHandle_t *handle, int enabled)
{
    if (!enabled)
    {
        *handle = NULL;
        return;
    }
    cusolverDnCreate(handle);
}

static inline void superlu_gpu_destroy_cusolver_handle(
    cusolverDnHandle_t handle)
{
    if (handle != NULL)
        cusolverDnDestroy(handle);
}

#endif
