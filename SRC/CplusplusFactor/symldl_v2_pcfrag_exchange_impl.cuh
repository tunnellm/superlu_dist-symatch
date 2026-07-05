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


#include "symldl_v2_fragment_exchange_common.cuh"
#include "symldl_v2_fragment_prepack_impl.cuh"
#include "symldl_v2_l_fragment_exchange_impl.cuh"
#include "symldl_v2_pcfrag_row_exchange_impl.cuh"

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

    SymLDLV2RowFragmentExchangeResult row_exchange =
        symldl_v2_pcfrag_exchange_row_fragments(
            this, k, stream_offset, stream, kcol, ksupc, lk, send_reqs);

    symldl_v2_trace_pcfrag_exchange(
        grid3d, k, stream_offset, partner_nrows, partner_recv_total,
        row_exchange.row_nrows, row_exchange.row_recv_total,
        row_exchange.row_send_total);

    if (!send_reqs.empty())
    {
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
        send_reqs.clear();
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    return 0;
}

#include "symldl_v2_panel_bcast_impl.cuh"

#endif
