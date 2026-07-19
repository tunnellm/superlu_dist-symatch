#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_fragment_exchange_common.cuh"
#include "symldl_v2_l_fragment_map_impl.cuh"

#ifdef HAVE_CUDA

struct SymLDLV2PartnerExchangeResult
{
    int_t partner_nrows;
    int partner_recv_total;
};

static SymLDLV2PartnerExchangeResult
symldl_v2_pcfrag_exchange_partner_l(
    xLUstruct_t<double> *lu, int_t k, int stream_offset,
    cudaStream_t stream, bool async_factor, int_t kcol, int_t ksupc,
    int_t lk, std::vector<MPI_Request> &send_reqs)
{
    int tag_ub = lu->symFactTagUb;
    std::vector<int> &send_sizes = lu->symV2ExchangeSendSizesScratch;
    std::vector<int> &recv_sizes = lu->symV2ExchangeRecvSizesScratch;
    std::vector<int> &recv_offsets = lu->symV2ExchangeRecvOffsetsScratch;
    std::vector<MPI_Request> &recv_reqs = lu->symV2ExchangeRecvReqsScratch;
    if (send_sizes.size() != static_cast<size_t>(lu->Pc))
        send_sizes.assign(static_cast<size_t>(lu->Pc), 0);
    if (recv_sizes.size() != static_cast<size_t>(lu->Pr))
        recv_sizes.assign(static_cast<size_t>(lu->Pr), 0);
    if (recv_offsets.size() != static_cast<size_t>(lu->Pr))
        recv_offsets.assign(static_cast<size_t>(lu->Pr), -1);
    std::fill(send_sizes.begin(), send_sizes.end(), 0);
    std::fill(recv_sizes.begin(), recv_sizes.end(), 0);
    std::fill(recv_offsets.begin(), recv_offsets.end(), -1);
    recv_reqs.clear();
    send_reqs.clear();

    auto partner_send_buffer = [&](size_t flat, int size) -> double *
    {
        if (flat >= lu->symV2PartnerLSendSizes.size() ||
            flat >= lu->symV2PartnerLHostSendScratchOffsets.size())
            ABORT("SymFact V2 partner send slot is invalid.");
        if (size < 0 || lu->symV2PartnerLSendSizes[flat] != size)
            ABORT("SymFact V2 partner send size is invalid.");
        if (size == 0)
            return NULL;
        if (lu->A_gpu.symPartnerLSendStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner send staging buffer is missing.");
        size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
        size_t count = static_cast<size_t>(size);
        if (offset + count >
                static_cast<size_t>(lu->maxSymPartnerLSendStageCount) ||
            offset + count < offset)
            ABORT("SymFact V2 partner send staging buffer is too small.");
        return lu->A_gpu.symPartnerLSendStageBufs[stream_offset] + offset;
    };

    auto partner_host_send_buffer = [&](size_t flat) -> double *
    {
        if (flat < lu->symV2PartnerLHostSendBufsPinned.size() &&
            lu->symV2PartnerLHostSendBufsPinned[flat] != NULL)
            return lu->symV2PartnerLHostSendBufsPinned[flat];
        if (flat < lu->symV2PartnerLHostSendBufs.size() &&
            !lu->symV2PartnerLHostSendBufs[flat].empty())
            return lu->symV2PartnerLHostSendBufs[flat].data();
        return NULL;
    };

    size_t partner_recv_base =
        static_cast<size_t>(k) * static_cast<size_t>(lu->Pr);
    if (partner_recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvSizes.size() ||
        partner_recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvMap.size() ||
        partner_recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvMapsGPU.size())
        ABORT("SymFact V2 partner receive metadata is missing.");

    int partner_recv_total = 0;
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        int count = lu->symV2PartnerLRecvSizes[
            partner_recv_base + static_cast<size_t>(pr)];
        if (count < 0)
            ABORT("SymFact V2 partner receive size is invalid.");
        recv_sizes[static_cast<size_t>(pr)] = count;
        int src = PNUM(pr, kcol, lu->grid);
        if (count > 0 && src != lu->iam)
        {
            recv_offsets[static_cast<size_t>(pr)] = partner_recv_total;
            if (partner_recv_total >
                std::numeric_limits<int>::max() - count)
                ABORT("SymFact V2 partner receive size overflows.");
            partner_recv_total += count;
        }
    }
    if (partner_recv_total > lu->maxSymPartnerLvalCount)
        ABORT("SymFact V2 partner receive exceeds staging buffer.");
    double *partner_recv_host = NULL;
    if (partner_recv_total > 0)
    {
        if (static_cast<size_t>(stream_offset) >=
                lu->symPartnerLvalRecvBufs.size() ||
            lu->symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner receive staging buffer is missing.");
        if (lu->A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner device staging buffer is missing.");
        partner_recv_host = lu->symPartnerLvalRecvBufs[stream_offset];
    }
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        int count = recv_sizes[static_cast<size_t>(pr)];
        int src = PNUM(pr, kcol, lu->grid);
        if (count <= 0 || src == lu->iam)
            continue;
        MPI_Request req;
        MPI_Irecv(partner_recv_host + recv_offsets[static_cast<size_t>(pr)],
                  count, MPI_DOUBLE, src, SLU_MPI_TAG(5, k),
                  lu->grid->comm, &req);
        recv_reqs.push_back(req);
    }

    if (async_factor && lu->mycol == kcol &&
        static_cast<size_t>(k) < lu->symPanelReadyEventIds.size() &&
        lu->symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = lu->symPanelReadyEventIds[static_cast<size_t>(k)];
        if (event_id >= lu->A_gpu.numCudaStreams)
            ABORT("SymFact V2 transformed-panel event is invalid.");
        gpuErrchk(cudaStreamWaitEvent(
            stream, lu->A_gpu.panelReadyEvents[event_id], 0));
    }

    bool packed_partner = false;
    if (lu->mycol == kcol)
    {
        if (lk < 0 || static_cast<size_t>(lk) >= lu->symV2PanelCount())
            ABORT("SymFact V2 partner source panel is invalid.");
        if (lu->symV2DiagBlocksGPU.size() !=
                static_cast<size_t>(lu->nsupers) ||
            lu->symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 partner source diagonal block is missing.");
        xlpanel_t<double> &lpanel = lu->lPanelVec[lk];
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            if (flat >= lu->symV2PartnerLSendSizes.size() ||
                flat >= lu->symL2LSendMapsGPU.size())
                ABORT("SymFact V2 partner send metadata is missing.");
            int count = lu->symV2PartnerLSendSizes[flat];
            send_sizes[static_cast<size_t>(pc)] = count;
            if (count <= 0)
                continue;

            bool active = false;
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner send mask is missing.");
                active = active ||
                         lu->symV2PartnerLSendRowActive[active_pos];
            }
            if (!active)
                continue;
            if (lpanel.isEmpty())
                ABORT("SymFact V2 partner source L panel is missing.");
            double *sendbuf = partner_send_buffer(flat, count);
            int_t *sendmap = symldl_v2_partner_send_map_gpu(
                lu, flat, count, stream_offset, stream);
            if (sendbuf == NULL || sendmap == NULL)
                ABORT("SymFact V2 partner send map is missing.");
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            symldl_v2_lfrag_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                lpanel.gpuPanel.val, sendbuf, sendmap, count, lpanel.LDA(),
                lu->symV2DiagBlocksGPU[k], ksupc);
            packed_partner = true;
        }
        if (packed_partner)
        {
            gpuErrchk(cudaGetLastError());
            for (int pc = 0; pc < lu->Pc; ++pc)
            {
                size_t flat =
                    static_cast<size_t>(lk) *
                        static_cast<size_t>(lu->Pc) +
                    static_cast<size_t>(pc);
                int count = send_sizes[static_cast<size_t>(pc)];
                if (count <= 0)
                    continue;
                bool active_remote = false;
                for (int pr = 0; pr < lu->Pr; ++pr)
                {
                    size_t active_pos = flat *
                                            static_cast<size_t>(lu->Pr) +
                                        static_cast<size_t>(pr);
                    if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                        ABORT("SymFact V2 partner send mask is missing.");
                    if (lu->symV2PartnerLSendRowActive[active_pos] &&
                        PNUM(pr, pc, lu->grid) != lu->iam)
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

    if (lu->mycol == kcol)
    {
        if (lk < 0)
            ABORT("SymFact V2 partner source panel is invalid.");
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            int count = send_sizes[static_cast<size_t>(pc)];
            if (count <= 0)
                continue;
            double *hostbuf = partner_host_send_buffer(flat);
            double *sendbuf = partner_send_buffer(flat, count);
            if (hostbuf == NULL || sendbuf == NULL)
                ABORT("SymFact V2 partner send staging is missing.");
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner send mask is missing.");
                if (!lu->symV2PartnerLSendRowActive[active_pos])
                    continue;
                int dest = PNUM(pr, pc, lu->grid);
                if (dest == lu->iam)
                    continue;
                MPI_Request req;
                MPI_Isend(hostbuf, count, MPI_DOUBLE, dest,
                          SLU_MPI_TAG(5, k), lu->grid->comm, &req);
                send_reqs.push_back(req);
                lu->symV2RouteProfileNotePartnerSend(
                    static_cast<size_t>(count), 1,
                    static_cast<size_t>(count));
            }
        }
    }

    if (!recv_reqs.empty())
        MPI_Waitall(static_cast<int>(recv_reqs.size()),
                    recv_reqs.data(), MPI_STATUSES_IGNORE);
    if (partner_recv_total > 0)
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLStageBufs[stream_offset],
            partner_recv_host,
            sizeof(double) * static_cast<size_t>(partner_recv_total),
            cudaMemcpyHostToDevice, stream));

    const std::vector<int_t> &partner_index =
        lu->symV2PartnerLRecvIndex[static_cast<size_t>(k)];
    int_t empty_header[LPANEL_HEADER_SIZE] = {0, 0, 0, ksupc};
    if (!partner_index.empty())
    {
        if (partner_index[3] != ksupc)
            ABORT("SymFact V2 partner fragment index has wrong width.");
        if (static_cast<int_t>(partner_index.size()) >
            lu->maxSymPartnerLidxCount)
            ABORT("SymFact V2 partner fragment index exceeds buffer.");
        if (lu->A_gpu.symPartnerLidxRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner index buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLidxRecvBufs[stream_offset],
            partner_index.data(), sizeof(int_t) * partner_index.size(),
            cudaMemcpyHostToDevice, stream));
    }
    else if (lu->A_gpu.symPartnerLidxRecvBufs[stream_offset] != NULL)
    {
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLidxRecvBufs[stream_offset], empty_header,
            sizeof(int_t) * LPANEL_HEADER_SIZE,
            cudaMemcpyHostToDevice, stream));
    }

    int_t partner_nrows = partner_index.empty() ? 0 : partner_index[1];
    if (partner_nrows > 0)
    {
        if (static_cast<int64_t>(partner_nrows) *
                static_cast<int64_t>(ksupc) >
            static_cast<int64_t>(lu->maxSymPartnerLvalCount))
            ABORT("SymFact V2 partner fragment value buffer is too small.");
        if (lu->A_gpu.symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 partner value buffer is missing.");
        gpuErrchk(cudaMemsetAsync(
            lu->A_gpu.symPartnerLvalRecvBufs[stream_offset], 0,
            sizeof(double) * static_cast<size_t>(partner_nrows) *
                static_cast<size_t>(ksupc),
            stream));
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            int count = recv_sizes[static_cast<size_t>(pr)];
            if (count <= 0)
                continue;
            size_t pos = partner_recv_base + static_cast<size_t>(pr);
            if (lu->symV2PartnerLRecvMap[pos].size() % 3 != 0)
                ABORT("SymFact V2 partner receive map is invalid.");
            int pieces = static_cast<int>(
                lu->symV2PartnerLRecvMap[pos].size() / 3);
            if (pieces <= 0)
                continue;
            int_t *recv_map = lu->symV2PartnerLRecvMapsGPU[pos];
            if (recv_map == NULL)
                ABORT("SymFact V2 partner receive map is missing.");
            int src = PNUM(pr, kcol, lu->grid);
            double *stage = NULL;
            if (src == lu->iam)
            {
                if (lu->mycol != kcol || lk < 0)
                    ABORT("SymFact V2 partner self fragment source is invalid.");
                size_t flat = static_cast<size_t>(lk) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(lu->mycol);
                if (flat >= lu->symV2PartnerLSendSizes.size() ||
                    lu->symV2PartnerLSendSizes[flat] != count)
                    ABORT("SymFact V2 partner self fragment size is invalid.");
                stage = partner_send_buffer(flat, count);
            }
            else
            {
                int offset = recv_offsets[static_cast<size_t>(pr)];
                if (offset < 0)
                    ABORT("SymFact V2 partner receive offset is invalid.");
                stage = lu->A_gpu.symPartnerLStageBufs[stream_offset] + offset;
            }
            symldl_v2_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                stage, lu->A_gpu.symPartnerLvalRecvBufs[stream_offset],
                recv_map, pieces, ksupc, partner_nrows);
            gpuErrchk(cudaGetLastError());
        }
    }

    SymLDLV2PartnerExchangeResult result;
    result.partner_nrows = partner_nrows;
    result.partner_recv_total = partner_recv_total;
    return result;
}

#endif
