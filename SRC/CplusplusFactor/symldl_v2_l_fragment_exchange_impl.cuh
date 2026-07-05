#pragma once

#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static int_t symldl_v2_l_fragment_exchange(
    xLUstruct_t<double> *lu, int_t k, int_t stream_offset)
{
    if (lu->options->SymFact != YES || lu->symGPU3DVersion != 2)
        return 0;
    if (!lu->superlu_acc_offload)
        ABORT("GPU3DVERSION=2 requires GPU offload.");
    if (lu->Pr <= 1)
        return 0;
    if (k < 0 || k >= lu->nsupers)
        return 0;

    if (lu->symV2PartnerLSendSizes.empty() ||
        lu->symV2PartnerLSendRowActive.empty() ||
        lu->symL2LSendMapsGPU.empty() ||
        lu->symV2PartnerLRecvSizes.empty() ||
        lu->symV2PartnerLRecvIndex.empty() ||
        lu->symV2PartnerLRecvMap.empty() ||
        lu->symV2PartnerLRecvMapsGPU.empty())
        ABORT("SymFact V2 L-fragment buffers are not allocated.");

    if (stream_offset < 0 || stream_offset >= lu->A_gpu.numCudaStreams)
        stream_offset = 0;

    const bool cuda_aware = superlu_cuda_aware_mpi();
    cudaStream_t stream = superlu_sym_v2_async_factor()
                              ? lu->A_gpu.lookAheadUStream[stream_offset]
                              : lu->A_gpu.cuStreams[stream_offset];
    int_t kcol = lu->symV2PanelRoot(k);
    int_t ksupc = lu->supersize(k);
    int_t lk = lu->symV2PanelIndex(k);
    int tag_ub = lu->symFactTagUb;
    if (kcol < 0 || kcol >= lu->Pc || ksupc <= 0)
        ABORT("SymFact V2 L-fragment panel metadata is invalid.");

    if (lu->symV2ExchangeSendSizesScratch.size() !=
            static_cast<size_t>(lu->Pc))
        lu->symV2ExchangeSendSizesScratch.assign(
            static_cast<size_t>(lu->Pc), 0);
    if (lu->symV2ExchangeRecvSizesScratch.size() !=
            static_cast<size_t>(lu->Pr))
        lu->symV2ExchangeRecvSizesScratch.assign(
            static_cast<size_t>(lu->Pr), 0);
    if (lu->symV2ExchangeRecvOffsetsScratch.size() !=
            static_cast<size_t>(lu->Pr))
        lu->symV2ExchangeRecvOffsetsScratch.assign(
            static_cast<size_t>(lu->Pr), -1);
    std::vector<int> &send_sizes = lu->symV2ExchangeSendSizesScratch;
    std::vector<int> &recv_sizes = lu->symV2ExchangeRecvSizesScratch;
    std::vector<int> &recv_offsets = lu->symV2ExchangeRecvOffsetsScratch;
    std::vector<MPI_Request> &recv_reqs = lu->symV2ExchangeRecvReqsScratch;
    std::vector<MPI_Request> &send_reqs = lu->symV2ExchangeSendReqsScratch;
    std::fill(send_sizes.begin(), send_sizes.end(), 0);
    std::fill(recv_sizes.begin(), recv_sizes.end(), 0);
    std::fill(recv_offsets.begin(), recv_offsets.end(), -1);
    recv_reqs.clear();
    send_reqs.clear();

    auto partner_send_buffer = [&](size_t flat, int count) -> double *
    {
        if (flat >= lu->symV2PartnerLSendSizes.size() ||
            lu->symV2PartnerLSendSizes[flat] != count)
            ABORT("SymFact V2 L-fragment send size is invalid.");
        if (count <= 0)
            return NULL;
        if (superlu_sym_v2_pc_fragment_ldl_native())
        {
            if (lu->A_gpu.symPartnerLSendStageBufs[stream_offset] == NULL)
                ABORT("SymFact V2 L-fragment send staging is missing.");
            if (flat >= lu->symV2PartnerLHostSendScratchOffsets.size())
                ABORT("SymFact V2 L-fragment send offset is missing.");
            size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
            size_t size = static_cast<size_t>(count);
            if (offset + size >
                    static_cast<size_t>(lu->maxSymPartnerLSendStageCount) ||
                offset + size < offset)
                ABORT("SymFact V2 L-fragment send staging is too small.");
            return lu->A_gpu.symPartnerLSendStageBufs[stream_offset] + offset;
        }
        if (flat >= lu->symV2PartnerLSendBufsGPU.size())
            ABORT("SymFact V2 L-fragment send slot is invalid.");
        return lu->symV2PartnerLSendBufsGPU[flat];
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

    size_t recv_base =
        static_cast<size_t>(k) * static_cast<size_t>(lu->Pr);
    if (recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvSizes.size() ||
        recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvMap.size() ||
        recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2PartnerLRecvMapsGPU.size())
        ABORT("SymFact V2 L-fragment receive metadata is missing.");

    int recv_total = 0;
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        int count = lu->symV2PartnerLRecvSizes[
            recv_base + static_cast<size_t>(pr)];
        if (count < 0 ||
            recv_total > std::numeric_limits<int>::max() - count)
            ABORT("SymFact V2 L-fragment receive size is invalid.");
        recv_sizes[static_cast<size_t>(pr)] = count;
        int src = PNUM(pr, kcol, lu->grid);
        if (count > 0 && src != lu->iam)
        {
            recv_offsets[static_cast<size_t>(pr)] = recv_total;
            recv_total += count;
        }
    }
    if (recv_total > lu->maxSymPartnerLvalCount)
        ABORT("SymFact V2 L-fragment receive exceeds staging buffer.");

    double *recv_host = NULL;
    if (recv_total > 0 && !cuda_aware)
    {
        if (static_cast<size_t>(stream_offset) >=
                lu->symPartnerLvalRecvBufs.size() ||
            lu->symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 L-fragment host receive staging is missing.");
        recv_host = lu->symPartnerLvalRecvBufs[stream_offset];
    }

    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        int count = recv_sizes[static_cast<size_t>(pr)];
        int src = PNUM(pr, kcol, lu->grid);
        if (count <= 0 || src == lu->iam)
            continue;
        MPI_Request req;
        double *dst = cuda_aware
                          ? lu->A_gpu.symPartnerLStageBufs[stream_offset] +
                                recv_offsets[static_cast<size_t>(pr)]
                          : recv_host + recv_offsets[static_cast<size_t>(pr)];
        MPI_Irecv(dst, count, MPI_DOUBLE, src, SLU_MPI_TAG(5, k),
                  lu->grid->comm, &req);
        recv_reqs.push_back(req);
    }

    if (lu->mycol == kcol)
    {
        if (lk < 0 || lk >= lu->symV2PanelCount())
            ABORT("SymFact V2 L-fragment source panel is invalid.");
        if (lu->symV2DiagBlocksGPU.size() !=
                static_cast<size_t>(lu->nsupers) ||
            lu->symV2DiagBlocksGPU[static_cast<size_t>(k)] == NULL)
            ABORT("SymFact V2 device diagonal block is missing.");
        xlpanel_t<double> &lpanel = lu->lPanelVec[lk];
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            if (flat >= lu->symV2PartnerLSendSizes.size() ||
                flat >= lu->symL2LSendMapsGPU.size())
                ABORT("SymFact V2 L-fragment send metadata is missing.");
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
                    ABORT("SymFact V2 L-fragment send mask is missing.");
                active = active ||
                         lu->symV2PartnerLSendRowActive[active_pos];
            }
            if (!active)
                continue;
            if (lpanel.isEmpty())
                ABORT("SymFact V2 L-fragment source panel is missing.");

            double *sendbuf = partner_send_buffer(flat, count);
            int_t *sendmap = lu->symL2LSendMapsGPU[flat];
            if (sendbuf == NULL || sendmap == NULL)
                ABORT("SymFact V2 L-fragment send map is missing.");

            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            symldl_v2_lfrag_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                lpanel.gpuPanel.val, sendbuf, sendmap, count, lpanel.LDA(),
                lu->symV2DiagBlocksGPU[static_cast<size_t>(k)], ksupc);
            gpuErrchk(cudaGetLastError());

            if (!cuda_aware)
            {
                bool active_remote = false;
                for (int pr = 0; pr < lu->Pr; ++pr)
                {
                    size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                        static_cast<size_t>(pr);
                    if (lu->symV2PartnerLSendRowActive[active_pos] &&
                        PNUM(pr, pc, lu->grid) != lu->iam)
                        active_remote = true;
                }
                if (active_remote)
                {
                    double *hostbuf = partner_host_send_buffer(flat);
                    if (hostbuf == NULL)
                        ABORT("SymFact V2 L-fragment host send staging is missing.");
                    gpuErrchk(cudaMemcpyAsync(
                        hostbuf, sendbuf,
                        sizeof(double) * static_cast<size_t>(count),
                        cudaMemcpyDeviceToHost, stream));
                }
            }
        }
        gpuErrchk(cudaStreamSynchronize(stream));

        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            int count = send_sizes[static_cast<size_t>(pc)];
            if (count <= 0)
                continue;
            double *sendbuf = partner_send_buffer(flat, count);
            double *hostbuf = cuda_aware ? NULL : partner_host_send_buffer(flat);
            if (sendbuf == NULL || (!cuda_aware && hostbuf == NULL))
                ABORT("SymFact V2 L-fragment send staging is missing.");
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 L-fragment send mask is missing.");
                if (!lu->symV2PartnerLSendRowActive[active_pos])
                    continue;
                int dest = PNUM(pr, pc, lu->grid);
                if (dest == lu->iam)
                    continue;
                MPI_Request req;
                MPI_Isend(cuda_aware ? sendbuf : hostbuf, count, MPI_DOUBLE,
                          dest, SLU_MPI_TAG(5, k), lu->grid->comm, &req);
                send_reqs.push_back(req);
            }
        }
    }

    if (!recv_reqs.empty())
        MPI_Waitall(static_cast<int>(recv_reqs.size()), recv_reqs.data(),
                    MPI_STATUSES_IGNORE);
    if (!cuda_aware && recv_total > 0)
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLStageBufs[stream_offset], recv_host,
            sizeof(double) * static_cast<size_t>(recv_total),
            cudaMemcpyHostToDevice, stream));

    const std::vector<int_t> &index =
        lu->symV2PartnerLRecvIndex[static_cast<size_t>(k)];
    int_t empty_header[LPANEL_HEADER_SIZE] = {0, 0, 0, ksupc};
    if (lu->A_gpu.symPartnerLidxRecvBufs[stream_offset] == NULL)
        ABORT("SymFact V2 L-fragment index buffer is missing.");
    if (index.empty())
    {
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLidxRecvBufs[stream_offset], empty_header,
            sizeof(int_t) * LPANEL_HEADER_SIZE, cudaMemcpyHostToDevice,
            stream));
    }
    else
    {
        if (index[3] != ksupc)
            ABORT("SymFact V2 L-fragment index has wrong width.");
        if (static_cast<int_t>(index.size()) > lu->maxSymPartnerLidxCount)
            ABORT("SymFact V2 L-fragment index exceeds buffer.");
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symPartnerLidxRecvBufs[stream_offset], index.data(),
            sizeof(int_t) * index.size(), cudaMemcpyHostToDevice, stream));
    }

    int_t frag_nrows = index.empty() ? 0 : index[1];
    if (frag_nrows > 0)
    {
        if (lu->A_gpu.symPartnerLvalRecvBufs[stream_offset] == NULL ||
            lu->A_gpu.symPartnerLStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 L-fragment value buffers are missing.");
        if (static_cast<int64_t>(frag_nrows) *
                static_cast<int64_t>(ksupc) >
            static_cast<int64_t>(lu->maxSymPartnerLvalCount))
            ABORT("SymFact V2 L-fragment values exceed buffer.");
        gpuErrchk(cudaMemsetAsync(
            lu->A_gpu.symPartnerLvalRecvBufs[stream_offset], 0,
            sizeof(double) * static_cast<size_t>(frag_nrows) *
                static_cast<size_t>(ksupc),
            stream));

        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            int count = recv_sizes[static_cast<size_t>(pr)];
            if (count <= 0)
                continue;
            size_t pos = recv_base + static_cast<size_t>(pr);
            if (lu->symV2PartnerLRecvMap[pos].size() % 3 != 0)
                ABORT("SymFact V2 L-fragment receive map is invalid.");
            int pieces =
                static_cast<int>(lu->symV2PartnerLRecvMap[pos].size() / 3);
            if (pieces <= 0)
                continue;
            int_t *recv_map = lu->symV2PartnerLRecvMapsGPU[pos];
            if (recv_map == NULL)
                ABORT("SymFact V2 L-fragment receive map is missing.");

            int src = PNUM(pr, kcol, lu->grid);
            double *stage = NULL;
            if (src == lu->iam)
            {
                if (lu->mycol != kcol || lk < 0)
                    ABORT("SymFact V2 L-fragment self source is invalid.");
                size_t flat = static_cast<size_t>(lk) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(lu->mycol);
                if (flat >= lu->symV2PartnerLSendSizes.size() ||
                    lu->symV2PartnerLSendSizes[flat] != count)
                    ABORT("SymFact V2 L-fragment self size is invalid.");
                stage = partner_send_buffer(flat, count);
            }
            else
            {
                int offset = recv_offsets[static_cast<size_t>(pr)];
                if (offset < 0)
                    ABORT("SymFact V2 L-fragment receive offset is invalid.");
                stage = lu->A_gpu.symPartnerLStageBufs[stream_offset] +
                        offset;
            }
            symldl_v2_lfrag_assemble_kernel<<<pieces, 256, 0, stream>>>(
                stage, lu->A_gpu.symPartnerLvalRecvBufs[stream_offset],
                recv_map, pieces, ksupc, frag_nrows);
            gpuErrchk(cudaGetLastError());
        }
    }

    if (!send_reqs.empty())
    {
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
        send_reqs.clear();
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    return 0;
}

#endif
