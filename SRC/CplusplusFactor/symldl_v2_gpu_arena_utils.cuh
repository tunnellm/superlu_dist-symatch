#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

static inline size_t symldl_v2_arena_align(size_t value)
{
    const size_t alignment = 256;
    const size_t mask = alignment - 1;
    if (value > static_cast<size_t>(-1) - mask)
        ABORT("SymFact V2 GPU arena alignment overflows.");
    return (value + mask) & ~mask;
}

static inline size_t symldl_v2_arena_advance(
    size_t offset, size_t count, size_t elem_size, const char *what)
{
    offset = symldl_v2_arena_align(offset);
    if (elem_size != 0 && count > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    size_t bytes = count * elem_size;
    if (offset > static_cast<size_t>(-1) - bytes)
        ABORT(what);
    return offset + bytes;
}

static inline size_t symldl_v2_cuda_bytes(int_t count, size_t elem_size,
                                          const char *what)
{
    if (count <= 0)
        return 0;
    size_t n = static_cast<size_t>(count);
    if (elem_size != 0 && n > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return n * elem_size;
}

static inline void symldl_v2_cuda_malloc_optional(void **ptr, int_t count,
                                                  size_t elem_size,
                                                  const char *what)
{
    size_t bytes = symldl_v2_cuda_bytes(count, elem_size, what);
    if (bytes == 0)
    {
        *ptr = NULL;
        return;
    }
    gpuErrchk(cudaMalloc(ptr, bytes));
}

#endif
