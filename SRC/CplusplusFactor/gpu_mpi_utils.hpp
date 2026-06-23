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

static inline bool superlu_sym_v2_tiny_aggregate()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;
    const char *env = std::getenv("GPU3DV2_TINY_AGGREGATE");
    if (env == NULL || env[0] == '\0')
    {
        cached = 0;
        return false;
    }
    const int parsed = superlu_env_truthy(env);
    if (parsed < 0)
        ABORT("GPU3DV2_TINY_AGGREGATE must be a boolean value.");
    cached = parsed;
    return cached != 0;
}

static inline size_t superlu_sym_v2_tiny_aggregate_bytes()
{
    static size_t cached = 0;
    if (cached != 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_TINY_AGG_BYTES");
    if (env == NULL || env[0] == '\0')
    {
        cached = 64u * 1024u;
        return cached;
    }
    char *end = NULL;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || value < sizeof(double) ||
        value > static_cast<unsigned long long>(
            std::numeric_limits<size_t>::max()))
        ABORT("GPU3DV2_TINY_AGG_BYTES must be a positive byte count.");
    cached = static_cast<size_t>(value);
    return cached;
}

static inline size_t superlu_sym_v2_tiny_aggregate_total_bytes()
{
    static size_t cached = 0;
    if (cached != 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_TINY_AGG_TOTAL_BYTES");
    if (env == NULL || env[0] == '\0')
    {
        cached = 256u * 1024u;
        return cached;
    }
    char *end = NULL;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || value < sizeof(double) ||
        value > static_cast<unsigned long long>(
            std::numeric_limits<size_t>::max()))
        ABORT("GPU3DV2_TINY_AGG_TOTAL_BYTES must be a positive byte count.");
    cached = static_cast<size_t>(value);
    return cached;
}

static inline int superlu_sym_v2_tiny_aggregate_min_messages()
{
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *env = std::getenv("GPU3DV2_TINY_AGG_MIN_MESSAGES");
    if (env == NULL || env[0] == '\0')
    {
        cached = 4;
        return cached;
    }
    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 1 || value > 2147483647L)
        ABORT("GPU3DV2_TINY_AGG_MIN_MESSAGES must be a positive integer.");
    cached = static_cast<int>(value);
    return cached;
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
