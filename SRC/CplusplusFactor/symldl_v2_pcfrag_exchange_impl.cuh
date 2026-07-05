#pragma once

#include <algorithm>
#include <climits>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "gpu_mpi_utils.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_l_fragment_map_impl.cuh"

#ifdef HAVE_CUDA

static inline int symldl_v2_mpi_count(size_t count, const char *what)
{
    if (count > static_cast<size_t>(INT_MAX))
        ABORT(what);
    return static_cast<int>(count);
}

static inline int symldl_v2_mpi_count(int_t count, const char *what)
{
    if (count < 0)
        ABORT(what);
    return symldl_v2_mpi_count(static_cast<size_t>(count), what);
}

static inline size_t symldl_v2_square_count(int_t n, const char *what)
{
    if (n < 0)
        ABORT(what);
    size_t dim = static_cast<size_t>(n);
    if (dim != 0 && dim > static_cast<size_t>(-1) / dim)
        ABORT(what);
    return dim * dim;
}

static inline size_t symldl_v2_square_bytes(int_t n, size_t elem_size,
                                            const char *what)
{
    size_t count = symldl_v2_square_count(n, what);
    if (elem_size != 0 && count > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return count * elem_size;
}

static __global__ void symldl_v2_lfrag_pack_kernel(const double *lpanel,
                                                   double *sendbuf,
                                                   const int_t *sendmap,
                                                   int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    sendbuf[idx] = (src < 0) ? (double) (-src - 1) : lpanel[src];
}

static __global__ void symldl_v2_lfrag_pack_raw_kernel(
    const double *lpanel, double *sendbuf, const int_t *sendmap, int count,
    int_t panel_ld, const double *diag, int_t diag_ld)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    if (src < 0)
    {
        sendbuf[idx] = (double) (-src - 1);
        return;
    }

    int_t row = src % panel_ld;
    int_t col = src / panel_ld;
    double sum = 0.0;
    for (int_t p = 0; p < diag_ld; ++p)
        sum += lpanel[row + p * panel_ld] * diag[p + col * diag_ld];
    sendbuf[idx] = sum;
}

static __global__ void symldl_v2_row_down_pack_segments_kernel(
    const double *lpanel,
    double *sendbuf,
    const int_t *base_sendmap,
    size_t sendmap_base,
    const SymV2RowDownSendSegmentGPU *segments,
    int nsegments,
    int_t ksupc,
    int_t dst_lda)
{
    int seg_id = blockIdx.x;
    if (seg_id >= nsegments)
        return;

    SymV2RowDownSendSegmentGPU seg = segments[seg_id];
    int_t nrows = seg.nrows;
    if (nrows <= 0 || ksupc <= 0 || dst_lda <= 0)
        return;

    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        size_t map_pos = (seg.map_offset - sendmap_base) +
                         static_cast<size_t>(row) +
                         static_cast<size_t>(col) *
                             static_cast<size_t>(nrows);
        int_t src = base_sendmap[map_pos];
        sendbuf[seg.dst_row_offset + row + col * dst_lda] =
            (src < 0) ? (double) (-src - 1) : lpanel[src];
    }
}

static inline void symldl_v2_trace_pcfrag_exchange(
    gridinfo3d_t *grid3d, int_t k, int stream_offset, int_t partner_rows,
    int partner_recv_total, int_t row_rows, int row_recv_total,
    int row_send_total)
{
    if (!superlu_sym_v2_trace_pcfrag())
        return;

    static int printed = 0;
    if (printed >= 64)
        return;

    std::fprintf(stderr,
                 "[symv2-pcfrag] rank %d exchange k %d stream %d partner_rows %lld partner_recv %d row_rows %lld row_recv %d row_send %d\n",
                 grid3d != NULL ? grid3d->iam : -1,
                 static_cast<int>(k), stream_offset,
                 static_cast<long long>(partner_rows), partner_recv_total,
                 static_cast<long long>(row_rows), row_recv_total,
                 row_send_total);
    std::fflush(stderr);
    ++printed;
}

static __global__ void symldl_v2_lfrag_assemble_kernel(
    const double *stage,
    double *frag,
    const int_t *recv_map,
    int pieces,
    int_t ksupc,
    int_t frag_lda)
{
    int piece = blockIdx.x;
    if (piece >= pieces)
        return;

    int_t dst_offset = recv_map[3 * piece];
    int_t nrows = recv_map[3 * piece + 1];
    int_t src_offset = recv_map[3 * piece + 2];
    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        frag[dst_offset + row + col * frag_lda] =
            stage[src_offset + row + col * nrows];
    }
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PrepackLFragmentsGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment prepack is not implemented.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("GPU3DVERSION=2 raw L-fragment prepack requires GPU offload.");
    if (k < 0 || k >= nsupers || mycol != symV2PanelRoot(k))
        return 0;

    if (Pr <= 1)
    {
        int_t lk = symV2PanelIndex(k);
        if (lk < 0)
            return 0;
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (!lpanel.isEmpty() && symldl_v2_use_wpanel_cache(grid3d))
        {
            if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
                stream_offset = 0;
            if (static_cast<size_t>(stream_offset) >=
                    symV2RawPanelNodes.size() ||
                A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
                A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
                ABORT("SymFact V2 W-panel ring is not initialized.");
            cudaStream_t stream = A_gpu.cuStreams[stream_offset];
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
                static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
            gpuErrchk(cudaEventRecord(
                A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
            symV2RawPanelNodes[stream_offset] = k;
        }
        return 0;
    }

    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLPrepacked.empty())
        ABORT("SymFact V2 raw L-fragment prepack buffers are not allocated.");

    int_t lk = symV2PanelIndex(k);
    if (lk < 0 ||
        static_cast<size_t>(lk) >= symV2PartnerLPrepacked.size())
        ABORT("SymFact V2 raw L-fragment prepack has an invalid local panel.");

    symV2PartnerLPrepacked[static_cast<size_t>(lk)] = 0;
    if (symV2UsePcFragmentSchurPanel(k) ||
        symV2PartnerLSendBufPoolCount == 0)
        return 0;

    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];
    xlpanel_t<double> &lpanel = lPanelVec[lk];

    if (lpanel.isEmpty())
    {
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
        symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
            static_cast<unsigned char>(stream_offset + 1);
        return 0;
    }

    if (symldl_v2_use_wpanel_cache(grid3d))
    {
        if (static_cast<size_t>(stream_offset) >=
                symV2RawPanelNodes.size() ||
            A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
            A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
            ABORT("SymFact V2 W-panel ring is not initialized.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
            static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
        symV2RawPanelNodes[stream_offset] = k;
    }

    bool packed_any = false;
    for (int pc = 0; pc < Pc; ++pc)
    {
        size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc);
        if (flat >= symV2PartnerLSendSizes.size())
            ABORT("SymFact V2 raw L-fragment prepack size is missing.");
        int size = symV2PartnerLSendSizes[flat];
        if (size <= 0)
            continue;

        bool active_dest = false;
        for (int pr = 0; pr < Pr; ++pr)
        {
            size_t active_pos =
                flat * static_cast<size_t>(Pr) + static_cast<size_t>(pr);
            if (active_pos >= symV2PartnerLSendRowActive.size())
                ABORT("SymFact V2 raw L-fragment prepack row mask is missing.");
            if (symV2PartnerLSendRowActive[active_pos])
            {
                active_dest = true;
                break;
            }
        }
        if (!active_dest)
            continue;

        double *sendbuf = symV2PartnerLSendBufsGPU[flat];
        int_t *sendmap = symL2LSendMapsGPU[flat];
        if (sendbuf == NULL || sendmap == NULL)
            ABORT("SymFact V2 raw L-fragment prepack buffer is missing.");

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        symldl_v2_lfrag_pack_kernel<<<blocks, threads, 0, stream>>>(
            lpanel.gpuPanel.val, sendbuf, sendmap, size);
        packed_any = true;
    }
    if (packed_any)
        gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaEventRecord(
        A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
    symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
        static_cast<unsigned char>(stream_offset + 1);
    return 0;
}

#include "symldl_v2_l_fragment_exchange_impl.cuh"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment exchange is not implemented.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2LFragmentExchangeGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("SymFact V2 requires GPU offload.");
    if (Pr <= 1)
        return 0;
    if (k < 0 || k >= nsupers)
        return 0;
    if (!symV2UsePcFragmentSchurPanel(k))
    {
        symV2RouteProfileNoteLFragmentExchange();
        return symldl_v2_l_fragment_exchange(this, k, stream_offset);
    }
    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    symV2RouteProfileNotePcFragmentExchange();

    const bool cuda_aware = superlu_cuda_aware_mpi();
    const bool async_factor = superlu_sym_v2_async_factor();
    const bool pcfrag_async_exchange =
        async_factor &&
        superlu_sym_v2_pcfrag_async_prereqs_enabled() &&
        (superlu_sym_v2_pcfrag_async_exchange() ||
         superlu_sym_v2_pcfrag_async_pipeline());
    if (cuda_aware)
        ABORT("SymFact V2 Pc-fragment exchange is fail-closed for CUDA-aware MPI.");
    if (!superlu_sym_v2_pc_fragment_ldl_native())
        ABORT("SymFact V2 Pc-fragment exchange requires LDL-native row movement.");
    if (!superlu_sym_v2_row_l_plan_v2() ||
        !superlu_sym_v2_row_l_plan_v2_exchange() ||
        superlu_sym_v2_row_l_plan_v2_dryrun() ||
        !superlu_sym_v2_row_l_plan_v2_aggregate_dest() ||
        !superlu_sym_v2_row_l_direct_recv() ||
        !superlu_sym_v2_row_l_compressed_plan() ||
        !superlu_sym_v2_row_l_lazy_sendmap() ||
        !superlu_sym_v2_row_l_pack_all_dest() ||
        !superlu_sym_v2_row_l_separate_send_staging())
        ABORT("SymFact V2 Pc-fragment exchange requires lazy LDL-native row-down planning.");
    if (async_factor && !pcfrag_async_exchange)
        ABORT("SymFact V2 Pc-fragment async factor requires a Pc-fragment async exchange mode.");
    if (grid3d == NULL || grid3d->rscp.comm == MPI_COMM_NULL)
        ABORT("SymFact V2 row-down communicator is missing.");

    cudaStream_t stream = async_factor
                              ? A_gpu.lookAheadUStream[stream_offset]
                              : A_gpu.cuStreams[stream_offset];
    int tag_ub = symFactTagUb;
    int_t kcol = symV2PanelRoot(k);
    int_t ksupc = SuperSize(k);
    int_t lk = symV2PanelIndex(k);
    if (kcol < 0 || kcol >= Pc || ksupc <= 0)
        ABORT("SymFact V2 Pc-fragment panel metadata is invalid.");

    if (symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symL2LSendMapsGPU.empty() ||
        symV2PartnerLRecvSizes.empty() ||
        symV2PartnerLRecvIndex.empty() ||
        symV2PartnerLRecvMap.empty() ||
        symV2PartnerLRecvMapsGPU.empty() ||
        symV2RowFragRecvSizes.empty() ||
        symV2RowFragRecvIndex.empty() ||
        symV2RowDownSendSizes.empty() ||
        symV2RowDownSendSegsGPU.empty())
        ABORT("SymFact V2 Pc-fragment exchange buffers are not allocated.");

    std::vector<int> &send_sizes = symV2ExchangeSendSizesScratch;
    std::vector<int> &recv_sizes = symV2ExchangeRecvSizesScratch;
    std::vector<int> &recv_offsets = symV2ExchangeRecvOffsetsScratch;
    std::vector<MPI_Request> &recv_reqs = symV2ExchangeRecvReqsScratch;
    std::vector<MPI_Request> &send_reqs = symV2ExchangeSendReqsScratch;
    if (send_sizes.size() != static_cast<size_t>(Pc))
        send_sizes.assign(static_cast<size_t>(Pc), 0);
    if (recv_sizes.size() != static_cast<size_t>(Pr))
        recv_sizes.assign(static_cast<size_t>(Pr), 0);
    if (recv_offsets.size() != static_cast<size_t>(Pr))
        recv_offsets.assign(static_cast<size_t>(Pr), -1);
    std::fill(send_sizes.begin(), send_sizes.end(), 0);
    std::fill(recv_sizes.begin(), recv_sizes.end(), 0);
    std::fill(recv_offsets.begin(), recv_offsets.end(), -1);
    recv_reqs.clear();
    send_reqs.clear();

    auto partner_send_buffer = [&](size_t flat, int size) -> double *
    {
        if (flat >= symV2PartnerLSendSizes.size() ||
            flat >= symV2PartnerLHostSendScratchOffsets.size())
            ABORT("SymFact V2 partner send slot is invalid.");
        if (size < 0 || symV2PartnerLSendSizes[flat] != size)
            ABORT("SymFact V2 partner send size is invalid.");
        if (size == 0)
            return NULL;
        if (A_gpu.symPartnerLSendStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner send staging buffer is missing.");
        size_t offset = symV2PartnerLHostSendScratchOffsets[flat];
        size_t count = static_cast<size_t>(size);
        if (offset + count > static_cast<size_t>(maxSymPartnerLSendStageCount) ||
            offset + count < offset)
            ABORT("SymFact V2 partner send staging buffer is too small.");
        return A_gpu.symPartnerLSendStageBufs[stream_offset] + offset;
    };

    auto partner_host_send_buffer = [&](size_t flat) -> double *
    {
        if (flat < symV2PartnerLHostSendBufsPinned.size() &&
            symV2PartnerLHostSendBufsPinned[flat] != NULL)
            return symV2PartnerLHostSendBufsPinned[flat];
        if (flat < symV2PartnerLHostSendBufs.size() &&
            !symV2PartnerLHostSendBufs[flat].empty())
            return symV2PartnerLHostSendBufs[flat].data();
        return NULL;
    };

    size_t partner_recv_base =
        static_cast<size_t>(k) * static_cast<size_t>(Pr);
    if (partner_recv_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvSizes.size() ||
        partner_recv_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvMap.size() ||
        partner_recv_base + static_cast<size_t>(Pr) >
            symV2PartnerLRecvMapsGPU.size())
        ABORT("SymFact V2 partner receive metadata is missing.");

    int partner_recv_total = 0;
    for (int pr = 0; pr < Pr; ++pr)
    {
        int count = symV2PartnerLRecvSizes[
            partner_recv_base + static_cast<size_t>(pr)];
        if (count < 0)
            ABORT("SymFact V2 partner receive size is invalid.");
        recv_sizes[static_cast<size_t>(pr)] = count;
        int src = PNUM(pr, kcol, grid);
        if (count > 0 && src != iam)
        {
            recv_offsets[static_cast<size_t>(pr)] = partner_recv_total;
            if (partner_recv_total >
                std::numeric_limits<int>::max() - count)
                ABORT("SymFact V2 partner receive size overflows.");
            partner_recv_total += count;
        }
    }
    if (partner_recv_total > maxSymPartnerLvalCount)
        ABORT("SymFact V2 partner receive exceeds staging buffer.");
    double *partner_recv_host = NULL;
    if (partner_recv_total > 0)
    {
        if (static_cast<size_t>(stream_offset) >=
                symPartnerLvalRecvBufs.size() ||
            symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner receive staging buffer is missing.");
        if (A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner device staging buffer is missing.");
        partner_recv_host = symPartnerLvalRecvBufs[stream_offset];
    }
    for (int pr = 0; pr < Pr; ++pr)
    {
        int count = recv_sizes[static_cast<size_t>(pr)];
        int src = PNUM(pr, kcol, grid);
        if (count <= 0 || src == iam)
            continue;
        MPI_Request req;
        MPI_Irecv(partner_recv_host + recv_offsets[static_cast<size_t>(pr)],
                  count, MPI_DOUBLE, src, SLU_MPI_TAG(5, k),
                  grid->comm, &req);
        recv_reqs.push_back(req);
    }

    if (async_factor && mycol == kcol &&
        static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
        symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = symPanelReadyEventIds[static_cast<size_t>(k)];
        if (event_id >= A_gpu.numCudaStreams)
            ABORT("SymFact V2 transformed-panel event is invalid.");
        gpuErrchk(cudaStreamWaitEvent(stream, A_gpu.panelReadyEvents[event_id],
                                      0));
    }

    bool packed_partner = false;
    if (mycol == kcol)
    {
        if (lk < 0 || static_cast<size_t>(lk) >= symV2PanelCount())
            ABORT("SymFact V2 partner source panel is invalid.");
        if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
            symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 partner source diagonal block is missing.");
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        for (int pc = 0; pc < Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                          static_cast<size_t>(pc);
            if (flat >= symV2PartnerLSendSizes.size() ||
                flat >= symL2LSendMapsGPU.size())
                ABORT("SymFact V2 partner send metadata is missing.");
            int count = symV2PartnerLSendSizes[flat];
            send_sizes[static_cast<size_t>(pc)] = count;
            if (count <= 0)
                continue;

            bool active = false;
            for (int pr = 0; pr < Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner send mask is missing.");
                active = active || symV2PartnerLSendRowActive[active_pos];
            }
            if (!active)
                continue;
            if (lpanel.isEmpty())
                ABORT("SymFact V2 partner source L panel is missing.");
            double *sendbuf = partner_send_buffer(flat, count);
            int_t *sendmap = symldl_v2_partner_send_map_gpu(
                this, flat, count, stream_offset, stream);
            if (sendbuf == NULL || sendmap == NULL)
                ABORT("SymFact V2 partner send map is missing.");
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            symldl_v2_lfrag_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                lpanel.gpuPanel.val, sendbuf, sendmap, count, lpanel.LDA(),
                symV2DiagBlocksGPU[k], ksupc);
            packed_partner = true;
        }
        if (packed_partner)
        {
            gpuErrchk(cudaGetLastError());
            for (int pc = 0; pc < Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(pc);
                int count = send_sizes[static_cast<size_t>(pc)];
                if (count <= 0)
                    continue;
                bool active_remote = false;
                for (int pr = 0; pr < Pr; ++pr)
                {
                    size_t active_pos = flat * static_cast<size_t>(Pr) +
                                        static_cast<size_t>(pr);
                    if (active_pos >= symV2PartnerLSendRowActive.size())
                        ABORT("SymFact V2 partner send mask is missing.");
                    if (symV2PartnerLSendRowActive[active_pos] &&
                        PNUM(pr, pc, grid) != iam)
                    {
                        active_remote = true;
                        break;
                    }
                }
                if (!active_remote)
                    continue;
                double *hostbuf = partner_host_send_buffer(flat);
                if (hostbuf == NULL)
                    ABORT("SymFact V2 partner host send staging is missing.");
                gpuErrchk(cudaMemcpyAsync(
                    hostbuf, partner_send_buffer(flat, count),
                    sizeof(double) * static_cast<size_t>(count),
                    cudaMemcpyDeviceToHost, stream));
            }
            gpuErrchk(cudaStreamSynchronize(stream));
        }
    }

    if (mycol == kcol)
    {
        if (lk < 0)
            ABORT("SymFact V2 partner source panel is invalid.");
        for (int pc = 0; pc < Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                          static_cast<size_t>(pc);
            int count = send_sizes[static_cast<size_t>(pc)];
            if (count <= 0)
                continue;
            double *hostbuf = partner_host_send_buffer(flat);
            double *sendbuf = partner_send_buffer(flat, count);
            if (hostbuf == NULL || sendbuf == NULL)
                ABORT("SymFact V2 partner send staging is missing.");
            for (int pr = 0; pr < Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner send mask is missing.");
                if (!symV2PartnerLSendRowActive[active_pos])
                    continue;
                int dest = PNUM(pr, pc, grid);
                if (dest == iam)
                    continue;
                MPI_Request req;
                MPI_Isend(hostbuf, count, MPI_DOUBLE, dest,
                          SLU_MPI_TAG(5, k), grid->comm, &req);
                send_reqs.push_back(req);
            }
        }
    }

    if (!recv_reqs.empty())
        MPI_Waitall(static_cast<int>(recv_reqs.size()),
                    recv_reqs.data(), MPI_STATUSES_IGNORE);
    if (partner_recv_total > 0)
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLStageBufs[stream_offset], partner_recv_host,
            sizeof(double) * static_cast<size_t>(partner_recv_total),
            cudaMemcpyHostToDevice, stream));

    const std::vector<int_t> &partner_index =
        symV2PartnerLRecvIndex[static_cast<size_t>(k)];
    int_t empty_header[LPANEL_HEADER_SIZE] = {0, 0, 0, ksupc};
    if (!partner_index.empty())
    {
        if (partner_index[3] != ksupc)
            ABORT("SymFact V2 partner fragment index has wrong width.");
        if (static_cast<int_t>(partner_index.size()) >
            maxSymPartnerLidxCount)
            ABORT("SymFact V2 partner fragment index exceeds buffer.");
        if (A_gpu.symPartnerLidxRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner index buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLidxRecvBufs[stream_offset],
            partner_index.data(),
            sizeof(int_t) * partner_index.size(),
            cudaMemcpyHostToDevice, stream));
    }
    else if (A_gpu.symPartnerLidxRecvBufs[stream_offset] != NULL)
    {
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLidxRecvBufs[stream_offset], empty_header,
            sizeof(int_t) * LPANEL_HEADER_SIZE,
            cudaMemcpyHostToDevice, stream));
    }

    int_t partner_nrows = partner_index.empty() ? 0 : partner_index[1];
    if (partner_nrows > 0)
    {
        if (static_cast<int64_t>(partner_nrows) *
                static_cast<int64_t>(ksupc) >
            static_cast<int64_t>(maxSymPartnerLvalCount))
            ABORT("SymFact V2 partner fragment value buffer is too small.");
        if (A_gpu.symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner value buffer is missing.");
        gpuErrchk(cudaMemsetAsync(
            A_gpu.symPartnerLvalRecvBufs[stream_offset], 0,
            sizeof(double) * static_cast<size_t>(partner_nrows) *
                static_cast<size_t>(ksupc),
            stream));
        for (int pr = 0; pr < Pr; ++pr)
        {
            int count = recv_sizes[static_cast<size_t>(pr)];
            if (count <= 0)
                continue;
            size_t pos = partner_recv_base + static_cast<size_t>(pr);
            if (symV2PartnerLRecvMap[pos].size() % 3 != 0)
                ABORT("SymFact V2 partner receive map is invalid.");
            int pieces = static_cast<int>(symV2PartnerLRecvMap[pos].size() / 3);
            if (pieces <= 0)
                continue;
            int_t *recv_map = symV2PartnerLRecvMapsGPU[pos];
            if (recv_map == NULL)
                ABORT("SymFact V2 partner receive map is missing.");
            int src = PNUM(pr, kcol, grid);
            double *stage = NULL;
            if (src == iam)
            {
                if (mycol != kcol || lk < 0)
                    ABORT("SymFact V2 partner self fragment source is invalid.");
                size_t flat = static_cast<size_t>(lk) *
                                  static_cast<size_t>(Pc) +
                              static_cast<size_t>(mycol);
                if (flat >= symV2PartnerLSendSizes.size() ||
                    symV2PartnerLSendSizes[flat] != count)
                    ABORT("SymFact V2 partner self fragment size is invalid.");
                stage = partner_send_buffer(flat, count);
            }
            else
            {
                int offset = recv_offsets[static_cast<size_t>(pr)];
                if (offset < 0)
                    ABORT("SymFact V2 partner receive offset is invalid.");
                stage = A_gpu.symPartnerLStageBufs[stream_offset] + offset;
            }
            symldl_v2_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                stage, A_gpu.symPartnerLvalRecvBufs[stream_offset],
                recv_map, pieces, ksupc, partner_nrows);
            gpuErrchk(cudaGetLastError());
        }
    }

    const std::vector<int_t> &row_index =
        symV2RowFragRecvIndex[static_cast<size_t>(k)];
    int_t row_nrows = row_index.empty() ? 0 : row_index[1];
    if (!row_index.empty())
    {
        if (row_index[3] != ksupc)
            ABORT("SymFact V2 row fragment index has wrong width.");
        if (static_cast<int_t>(row_index.size()) >
            maxSymV2RowFragIdxRecvCount)
            ABORT("SymFact V2 row fragment index exceeds buffer.");
        if (A_gpu.symV2RowFragIdxRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row index buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
            row_index.data(), sizeof(int_t) * row_index.size(),
            cudaMemcpyHostToDevice, stream));
    }
    else if (A_gpu.symV2RowFragIdxRecvBufs[stream_offset] != NULL)
    {
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RowFragIdxRecvBufs[stream_offset], empty_header,
            sizeof(int_t) * LPANEL_HEADER_SIZE,
            cudaMemcpyHostToDevice, stream));
    }

    auto row_count_destination = [&](int pc_dest) -> int
    {
        if (lk < 0)
            return 0;
        size_t slot = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc_dest);
        if (slot >= symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send size is missing.");
        return symV2RowDownSendSizes[slot];
    };

    int_t *row_down_sendmap_gpu = symL2LSendMapPoolGPU;
    size_t row_down_sendmap_base = 0;
    auto ensure_row_down_sendmap_gpu = [&]() -> int_t *
    {
        if (row_down_sendmap_gpu != NULL)
            return row_down_sendmap_gpu;
        row_down_sendmap_gpu = symldl_v2_panel_send_maps_gpu(
            this, lk, stream_offset, stream, &row_down_sendmap_base);
        return row_down_sendmap_gpu;
    };

    auto row_pack_destination = [&](int pc_dest, double *dst_buf) -> int
    {
        if (mycol != kcol)
            ABORT("SymFact V2 row-down pack called on a non-source rank.");
        if (lk < 0 || static_cast<size_t>(lk) >= symV2PanelCount())
            ABORT("SymFact V2 row-down source panel is invalid.");
        if (dst_buf == NULL)
            ABORT("SymFact V2 row-down destination buffer is missing.");
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (lpanel.isEmpty())
            ABORT("SymFact V2 row-down source L panel is missing.");
        size_t slot = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc_dest);
        if (slot >= symV2RowDownSendSizes.size() ||
            slot >= symV2RowDownSendSegsGPU.size() ||
            slot >= symV2RowDownSendSegCounts.size())
            ABORT("SymFact V2 row-down send segment slot is invalid.");
        int total = symV2RowDownSendSizes[slot];
        if (total <= 0)
            return 0;
        if (total > maxSymV2RowFragValSendCount)
            ABORT("SymFact V2 row-down packed destination exceeds send buffer.");
        if (total % ksupc != 0)
            ABORT("SymFact V2 row-down packed destination has wrong width.");
        int nsegments = symV2RowDownSendSegCounts[slot];
        SymV2RowDownSendSegmentGPU *segments =
            symV2RowDownSendSegsGPU[slot];
        int_t *sendmap = ensure_row_down_sendmap_gpu();
        if (nsegments <= 0 || segments == NULL || sendmap == NULL)
            ABORT("SymFact V2 row-down send descriptors are missing.");
        int_t dst_lda = static_cast<int_t>(total / ksupc);
        symldl_v2_row_down_pack_segments_kernel
            <<<nsegments, 256, 0, stream>>>(
                lpanel.gpuPanel.val, dst_buf, sendmap,
                row_down_sendmap_base, segments, nsegments, ksupc,
                dst_lda);
        gpuErrchk(cudaGetLastError());
        return total;
    };

    int row_recv_total = 0;
    int row_src_pc = static_cast<int>(kcol);
    size_t row_recv_base =
        static_cast<size_t>(k) * static_cast<size_t>(Pc);
    if (row_recv_base + static_cast<size_t>(Pc) >
            symV2RowFragRecvSizes.size())
        ABORT("SymFact V2 row-fragment receive sizes are missing.");
    for (int pc = 0; pc < Pc; ++pc)
    {
        int count = symV2RowFragRecvSizes[
            row_recv_base + static_cast<size_t>(pc)];
        if (count < 0 ||
            row_recv_total > std::numeric_limits<int>::max() - count)
            ABORT("SymFact V2 row-fragment receive size is invalid.");
        row_recv_total += count;
    }
    if (row_recv_total > maxSymV2RowFragStageCount ||
        row_recv_total > maxSymV2RowFragValRecvCount)
        ABORT("SymFact V2 row-fragment receive exceeds staging buffer.");

    MPI_Request row_recv_req = MPI_REQUEST_NULL;
    double *row_recv_host = NULL;
    int row_send_total = 0;
    if (row_recv_total > 0 && row_src_pc != mycol)
    {
        if (static_cast<size_t>(stream_offset) >=
                symV2RowFragHostRecvBufs.size() ||
            symV2RowFragHostRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment host receive staging is missing.");
        row_recv_host = symV2RowFragHostRecvBufs[stream_offset];
        MPI_Irecv(row_recv_host, row_recv_total, MPI_DOUBLE, row_src_pc,
                  SLU_MPI_TAG(5, k), grid3d->rscp.comm, &row_recv_req);
    }

    if (mycol == kcol)
    {
        if (static_cast<size_t>(stream_offset) >=
                symV2RowFragHostSendBufs.size() ||
            symV2RowFragHostSendBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment host send staging is missing.");
        if (A_gpu.symV2RowFragStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment device staging buffer is missing.");
        if (symV2RowFragSendCountsScratch.size() !=
            static_cast<size_t>(Pc))
            symV2RowFragSendCountsScratch.assign(
                static_cast<size_t>(Pc), 0);
        if (symV2RowFragSendOffsetsScratch.size() !=
            static_cast<size_t>(Pc))
            symV2RowFragSendOffsetsScratch.assign(
                static_cast<size_t>(Pc), -1);
        std::fill(symV2RowFragSendCountsScratch.begin(),
                  symV2RowFragSendCountsScratch.end(), 0);
        std::fill(symV2RowFragSendOffsetsScratch.begin(),
                  symV2RowFragSendOffsetsScratch.end(), -1);

        for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
        {
            if (pc_dest == mycol)
                continue;
            int count = row_count_destination(pc_dest);
            if (count <= 0)
                continue;
            if (count > maxSymV2RowFragValSendCount ||
                row_send_total > maxSymV2RowFragValSendCount - count)
                ABORT("SymFact V2 row-fragment send exceeds staging buffer.");
            symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)] =
                row_send_total;
            symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)] =
                count;
            row_send_total += count;
        }
        for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
        {
            int count =
                symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)];
            if (count <= 0)
                continue;
            int offset =
                symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)];
            if (offset < 0)
                ABORT("SymFact V2 row-fragment send offset is invalid.");
            int packed = row_pack_destination(
                pc_dest, A_gpu.symV2RowFragStageBufs[stream_offset] + offset);
            if (packed != count)
                ABORT("SymFact V2 row-fragment pack size mismatch.");
        }
        if (row_send_total > 0)
        {
            gpuErrchk(cudaMemcpyAsync(
                symV2RowFragHostSendBufs[stream_offset],
                A_gpu.symV2RowFragStageBufs[stream_offset],
                sizeof(double) * static_cast<size_t>(row_send_total),
                cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            for (int pc_dest = 0; pc_dest < Pc; ++pc_dest)
            {
                int count = symV2RowFragSendCountsScratch[
                    static_cast<size_t>(pc_dest)];
                if (count <= 0)
                    continue;
                int offset = symV2RowFragSendOffsetsScratch[
                    static_cast<size_t>(pc_dest)];
                MPI_Request req;
                MPI_Isend(symV2RowFragHostSendBufs[stream_offset] + offset,
                          count, MPI_DOUBLE, pc_dest, SLU_MPI_TAG(5, k),
                          grid3d->rscp.comm, &req);
                send_reqs.push_back(req);
            }
        }
    }

    if (row_recv_req != MPI_REQUEST_NULL)
        MPI_Wait(&row_recv_req, MPI_STATUS_IGNORE);
    if (row_recv_total > 0 && row_src_pc != mycol)
    {
        if (A_gpu.symV2RowFragValRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment value buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RowFragValRecvBufs[stream_offset], row_recv_host,
            sizeof(double) * static_cast<size_t>(row_recv_total),
            cudaMemcpyHostToDevice, stream));
    }
    else if (row_recv_total > 0)
    {
        if (A_gpu.symV2RowFragValRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment value buffer is missing.");
        int packed = row_pack_destination(
            mycol, A_gpu.symV2RowFragValRecvBufs[stream_offset]);
        if (packed != row_recv_total)
            ABORT("SymFact V2 row-fragment self pack size mismatch.");
    }
    if (row_nrows > 0)
    {
        int64_t expected =
            static_cast<int64_t>(row_nrows) * static_cast<int64_t>(ksupc);
        if (row_recv_total != expected)
            ABORT("SymFact V2 row-fragment receive size does not match its layout.");
    }

    symldl_v2_trace_pcfrag_exchange(
        grid3d, k, stream_offset, partner_nrows, partner_recv_total,
        row_nrows, row_recv_total, row_send_total);

    if (!send_reqs.empty())
    {
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
        send_reqs.clear();
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PanelBcastGPU(int_t k, int_t offset)
{
    if (!useSymV2Solve())
        ABORT("GPU3DVERSION=2 requires SymFact=YES.");

    double t0 = SuperLU_timer_();
    int_t sym_panel_root = symV2PanelRoot(k);
    bool pc_fragment_schur = symV2UsePcFragmentSchurPanel(k);
    symV2RouteProfileNotePanelBcast(pc_fragment_schur);
    xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);

    if (Pr > 1)
        dSymV2LFragmentExchangeGPU(k, offset);

    bool local_singleton_panel =
        Pr == 1 && Pc == 1 &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1;

    const bool pcfrag_async_exchange_panel_ready =
        pc_fragment_schur && Pr > 1 && Pc > 1 &&
        !superlu_cuda_aware_mpi() &&
        superlu_sym_v2_pc_fragment_ldl_native() &&
        superlu_sym_v2_row_l_plan_v2_exchange() &&
        superlu_sym_v2_row_l_direct_recv() &&
        superlu_sym_v2_row_l_compressed_plan() &&
        superlu_sym_v2_row_l_lazy_sendmap() &&
        (superlu_sym_v2_pcfrag_async_exchange() ||
         superlu_sym_v2_pcfrag_async_pipeline());

    if (superlu_sym_v2_async_factor() &&
        pcfrag_async_exchange_panel_ready &&
        mycol == sym_panel_root &&
        LidxSendCounts[k] > 0 && k >= 0 &&
        static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
        symPanelReadyEventIds[k] >= 0)
    {
        int event_id = symPanelReadyEventIds[k];
        if (event_id >= A_gpu.numCudaStreams)
            ABORT("SymFact V2 panel-ready event is invalid.");
        symPanelReadyEventIds[k] = -1;
    }
    else if (superlu_sym_v2_async_factor() &&
             !local_singleton_panel && mycol == sym_panel_root &&
             LidxSendCounts[k] > 0 && k >= 0 &&
             static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
             symPanelReadyEventIds[k] >= 0)
    {
        int event_id = symPanelReadyEventIds[k];
        if (event_id >= A_gpu.numCudaStreams)
            ABORT("SymFact V2 panel-ready event is invalid.");
        gpuErrchk(cudaEventSynchronize(A_gpu.panelReadyEvents[event_id]));
        symPanelReadyEventIds[k] = -1;
    }

    if (LidxSendCounts[k] > 0 && grid3d->rscp.Np > 1 && !pc_fragment_schur)
    {
        int lidx_count = symldl_v2_mpi_count(
            LidxSendCounts[k],
            "SymFact V2 L-panel index count exceeds MPI limit.");
        int lval_count = symldl_v2_mpi_count(
            LvalSendCounts[k],
            "SymFact V2 L-panel value count exceeds MPI limit.");
        superlu_gpu_mpi_bcast(k_lpanel.gpuPanel.index, k_lpanel.index,
                              sizeof(int_t), lidx_count, mpi_int_t,
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
        superlu_gpu_mpi_bcast(k_lpanel.gpuPanel.val, k_lpanel.val,
                              sizeof(Ftype), lval_count,
                              get_mpi_type<Ftype>(),
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
        if (superlu_cuda_aware_mpi())
        {
            gpuErrchk(cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index,
                                 sizeof(int_t) *
                                     static_cast<size_t>(lidx_count),
                                 cudaMemcpyDeviceToHost));
        }
    }

    if (Pr == 1 && Pc > 1 && LidxSendCounts[k] > 0)
    {
        int_t ksupc = SuperSize(k);
        if (symV2DiagBlocks.size() != static_cast<size_t>(nsupers) ||
            symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers))
            ABORT("SymFact V2 diagonal block vector has invalid size.");
        if (mycol == sym_panel_root && symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 device diagonal block is missing.");
        if (symV2DiagBlocks[k] == NULL)
        {
            symV2DiagBlocks[k] = (Ftype *) SUPERLU_MALLOC(
                symldl_v2_square_bytes(
                    ksupc, sizeof(Ftype),
                    "SymFact V2 diagonal block allocation overflows."));
            if (symV2DiagBlocks[k] == NULL)
                ABORT("Malloc fails for SymFact V2 diagonal block.");
        }
        if (symV2DiagBlocksGPU[k] == NULL)
            gpuErrchk(cudaMalloc(
                (void **) &symV2DiagBlocksGPU[k],
                symldl_v2_square_bytes(
                    ksupc, sizeof(Ftype),
                    "SymFact V2 device diagonal block allocation overflows.")));

        int diag_count = symldl_v2_mpi_count(
            symldl_v2_square_count(ksupc,
                                   "SymFact V2 diagonal block count overflows."),
            "SymFact V2 diagonal block count exceeds MPI limit.");
        superlu_gpu_mpi_bcast(symV2DiagBlocksGPU[k], symV2DiagBlocks[k],
                              sizeof(Ftype), diag_count,
                              get_mpi_type<Ftype>(),
                              static_cast<int>(sym_panel_root),
                              grid3d->rscp.comm);
    }

    SCT->tPanelBcast += (SuperLU_timer_() - t0);
    return 0;
}

#endif
