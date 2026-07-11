#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"
#include "symldl_v2_gpu_fragment_stream_buffers_impl.cuh"
#include "symldl_v2_gpu_stream_workspace_spec_impl.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_clear_gpu_stream_workspace(xLUstruct_t<Ftype> *lu,
                                                 int stream)
{
    lu->A_gpu.LvalRecvBufs[stream] = NULL;
    lu->A_gpu.UvalRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLvalRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLStageBufs[stream] = NULL;
    lu->A_gpu.symPartnerLSendStageBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragStageBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragValRecvBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragIdxRecvBufs[stream] = NULL;
    lu->A_gpu.symV2RowFragSendMapStageBufs[stream] = NULL;
    lu->A_gpu.symV2RawPanelBufs[stream] = NULL;
    lu->A_gpu.LidxRecvBufs[stream] = NULL;
    lu->A_gpu.UidxRecvBufs[stream] = NULL;
    lu->A_gpu.symPartnerLidxRecvBufs[stream] = NULL;
    lu->A_gpu.diagFactWork[stream] = NULL;
    lu->A_gpu.diagFactInfo[stream] = NULL;
    lu->A_gpu.lookAheadLGemmBuffer[stream] = NULL;
    lu->A_gpu.lookAheadUGemmBuffer[stream] = NULL;
}

template <typename Ftype>
static void symldl_v2_setup_gpu_stream_workspace(
    xLUstruct_t<Ftype> *lu, int stream,
    const SymV2GpuStreamWorkspaceSpec<Ftype> &spec)
{
    symldl_v2_clear_gpu_stream_workspace(lu, stream);

    if (lu->useSymV2Solve() && superlu_sym_v2_workspace_arena_enabled())
    {
        size_t stride = symldl_v2_stream_workspace_bytes(lu, spec);
        if (lu->symV2StreamArenaGPU == NULL)
        {
            size_t nstreams = static_cast<size_t>(lu->A_gpu.numCudaStreams);
            if (stride != 0 &&
                nstreams > static_cast<size_t>(-1) / stride)
                ABORT("SymFact V2 stream workspace arena size overflows.");
            lu->symV2StreamArenaBytes = stride * nstreams;
            gpuErrchk(cudaMalloc(&lu->symV2StreamArenaGPU,
                                 lu->symV2StreamArenaBytes));
        }

        char *stream_base =
            static_cast<char *>(lu->symV2StreamArenaGPU) +
            static_cast<size_t>(stream) * stride;
        size_t offset = 0;
        auto take = [&](size_t count, size_t elem_size,
                        const char *what) -> void *
        {
            size_t begin = symldl_v2_arena_align(offset);
            offset = symldl_v2_arena_advance(offset, count, elem_size, what);
            if (offset > stride)
                ABORT("SymFact V2 stream workspace layout is invalid.");
            return stream_base + begin;
        };

        lu->A_gpu.LvalRecvBufs[stream] = static_cast<Ftype *>(take(
            static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLvalCount)),
            sizeof(Ftype), "SymFact V2 stream L receive values"));
        if (spec.u_val_count > 0)
            lu->A_gpu.UvalRecvBufs[stream] = static_cast<Ftype *>(take(
                static_cast<size_t>(spec.u_val_count), sizeof(Ftype),
                "SymFact V2 stream row-side receive values"));
        lu->A_gpu.symPartnerLvalRecvBufs[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxSymPartnerLvalCount)),
                sizeof(Ftype), "SymFact V2 stream partner values"));
        lu->A_gpu.symPartnerLStageBufs[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, spec.partner_stage_count)),
                sizeof(Ftype), "SymFact V2 stream partner staging"));
        if (spec.need_partner_send_stage)
            lu->A_gpu.symPartnerLSendStageBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, lu->maxSymPartnerLSendStageCount)),
                    sizeof(Ftype),
                    "SymFact V2 stream partner send staging"));
        if (spec.pc_fragment_schur)
        {
            lu->A_gpu.symV2RowFragStageBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_stage_count)),
                    sizeof(Ftype),
                    "SymFact V2 stream row fragment staging"));
            lu->A_gpu.symV2RowFragValRecvBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_recv_val_count)),
                    sizeof(Ftype),
                    "SymFact V2 stream row fragment values"));
        }
        if (spec.raw_panel_count > 0)
            lu->A_gpu.symV2RawPanelBufs[stream] =
                static_cast<Ftype *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.raw_panel_count)),
                    sizeof(Ftype), "SymFact V2 stream W panel"));
        lu->A_gpu.LidxRecvBufs[stream] = static_cast<int_t *>(take(
            static_cast<size_t>(SUPERLU_MAX((int_t)1, lu->maxLidxCount)),
            sizeof(int_t), "SymFact V2 stream L receive indices"));
        if (spec.u_idx_count > 0)
            lu->A_gpu.UidxRecvBufs[stream] = static_cast<int_t *>(take(
                static_cast<size_t>(spec.u_idx_count), sizeof(int_t),
                "SymFact V2 stream row-side receive indices"));
        lu->A_gpu.symPartnerLidxRecvBufs[stream] =
            static_cast<int_t *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxSymPartnerLidxCount)),
                sizeof(int_t), "SymFact V2 stream partner indices"));
        if (spec.pc_fragment_schur)
        {
            lu->A_gpu.symV2RowFragIdxRecvBufs[stream] =
                static_cast<int_t *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_recv_idx_count)),
                    sizeof(int_t),
                    "SymFact V2 stream row fragment indices"));
        }
        if (spec.row_send_map_count > 0)
        {
            lu->A_gpu.symV2RowFragSendMapStageBufs[stream] =
                static_cast<int_t *>(take(
                    static_cast<size_t>(SUPERLU_MAX(
                        (int_t)1, spec.row_send_map_count)),
                    sizeof(int_t),
                    "SymFact V2 stream L-fragment send maps"));
        }
        if (spec.need_diag_work)
        {
            lu->A_gpu.diagFactWork[stream] = static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(1, spec.diag_work_count)),
                sizeof(Ftype), "SymFact V2 stream diagonal work"));
            lu->A_gpu.diagFactInfo[stream] = static_cast<int *>(take(
                1, sizeof(int), "SymFact V2 stream diagonal info"));
        }
        lu->A_gpu.lookAheadLGemmBuffer[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, lu->maxLvalCount)),
                sizeof(Ftype), "SymFact V2 stream lookahead L"));
        /* The inherited U-named members hold row-side scratch for L^T
           updates; they do not represent a materialized U factor. */
        lu->A_gpu.lookAheadUGemmBuffer[stream] =
            static_cast<Ftype *>(take(
                static_cast<size_t>(SUPERLU_MAX(
                    (int_t)1, spec.lookahead_u_count)),
                sizeof(Ftype), "SymFact V2 stream row-side lookahead"));
        return;
    }

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.LvalRecvBufs[stream], lu->maxLvalCount,
        sizeof(Ftype), "L value receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.UvalRecvBufs[stream], spec.u_val_count,
        sizeof(Ftype),
        "SymFact V2 row-side value receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.LidxRecvBufs[stream], lu->maxLidxCount,
        sizeof(int_t), "L index receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.UidxRecvBufs[stream], spec.u_idx_count,
        sizeof(int_t),
        "SymFact V2 row-side index receive buffer allocation overflows.");
    symldl_v2_setup_gpu_fragment_stream_buffers(lu, stream);
    if (spec.need_diag_work)
    {
        gpuErrchk(cudaMalloc(&lu->A_gpu.diagFactWork[stream],
                             sizeof(Ftype) *
                                 static_cast<size_t>(spec.diag_work_count)));
        gpuErrchk(cudaMalloc(&lu->A_gpu.diagFactInfo[stream],
                             sizeof(int)));
    }
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.lookAheadLGemmBuffer[stream],
        lu->maxLvalCount, sizeof(Ftype),
        "Lookahead L buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.lookAheadUGemmBuffer[stream],
        spec.lookahead_u_count, sizeof(Ftype),
        "SymFact V2 row-side lookahead buffer allocation overflows.");
}

#endif
