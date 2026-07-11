#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"
#include "symldl_v2_gpu_stream_workspace_impl.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
static inline size_t symldl_v2_gemm_workspace_stride(size_t dfbuf_elems,
                                                     size_t gemm_elems)
{
    size_t stride = 0;
    if (!dSymLDLV2GemmWorkspaceBytes(
            dfbuf_elems, gemm_elems, sizeof(Ftype), &stride))
        ABORT("SymFact V2 GEMM workspace arena size overflows.");
    return stride;
}

template <typename Ftype>
static inline size_t symldl_v2_stream_workspace_estimate(
    xLUstruct_t<Ftype> *lu, int_t ldt, size_t gemm_elems)
{
    SymV2GpuStreamWorkspaceSpec<Ftype> estimate_spec =
        symldl_v2_make_stream_workspace_spec(lu, 0, false);
    size_t ldt_size = static_cast<size_t>(ldt);
    if (ldt_size != 0 && ldt_size > static_cast<size_t>(-1) / ldt_size)
        ABORT("SymFact V2 diagonal workspace size overflows.");
    size_t dfbuf_elems = ldt_size * ldt_size;
    size_t stream_bytes =
        symldl_v2_stream_workspace_bytes(lu, estimate_spec);
    size_t gemm_bytes =
        symldl_v2_gemm_workspace_stride<Ftype>(dfbuf_elems, gemm_elems);
    if (stream_bytes > static_cast<size_t>(-1) - gemm_bytes)
        ABORT("SymFact V2 total stream workspace size overflows.");
    return stream_bytes + gemm_bytes;
}

template <typename Ftype>
static void symldl_v2_setup_gemm_workspace(
    xLUstruct_t<Ftype> *lu, int num_bufs, size_t dfbuf_elems,
    size_t *sum_diag_size, size_t *sum_gemm_size)
{
    if (lu->useSymV2Solve() && superlu_sym_v2_workspace_arena_enabled())
    {
        size_t stride = symldl_v2_gemm_workspace_stride<Ftype>(
            dfbuf_elems, static_cast<size_t>(lu->A_gpu.gemmBufferSize));
        size_t nbufs = static_cast<size_t>(num_bufs);
        if (stride != 0 && nbufs > static_cast<size_t>(-1) / stride)
            ABORT("SymFact V2 GEMM workspace arena size overflows.");
        lu->symV2GemmArenaBytes = stride * nbufs;
        if (lu->symV2GemmArenaBytes > 0)
            gpuErrchk(cudaMalloc(&lu->symV2GemmArenaGPU,
                                 lu->symV2GemmArenaBytes));

        for (int i = 0; i < num_bufs; ++i)
        {
            char *base = static_cast<char *>(lu->symV2GemmArenaGPU) +
                         static_cast<size_t>(i) * stride;
            size_t offset = 0;
            size_t begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(
                offset, dfbuf_elems, sizeof(Ftype),
                "SymFact V2 diagonal buffer arena overflows.");
            lu->A_gpu.dFBufs[i] = reinterpret_cast<Ftype *>(base + begin);
            begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(
                offset, static_cast<size_t>(lu->A_gpu.gemmBufferSize),
                sizeof(Ftype), "SymFact V2 GEMM buffer arena overflows.");
            if (offset > stride)
                ABORT("SymFact V2 GEMM workspace layout is invalid.");
            lu->A_gpu.gpuGemmBuffs[i] =
                reinterpret_cast<Ftype *>(base + begin);
            *sum_diag_size += dfbuf_elems;
            *sum_gemm_size += lu->A_gpu.gemmBufferSize;
        }
        return;
    }

    for (int i = 0; i < num_bufs; ++i)
    {
        gpuErrchk(cudaMalloc(&(lu->A_gpu.dFBufs[i]),
                             dfbuf_elems * sizeof(Ftype)));
        gpuErrchk(cudaMalloc(&(lu->A_gpu.gpuGemmBuffs[i]),
                             lu->A_gpu.gemmBufferSize * sizeof(Ftype)));
        *sum_diag_size += dfbuf_elems;
        *sum_gemm_size += lu->A_gpu.gemmBufferSize;
    }
}

#endif
