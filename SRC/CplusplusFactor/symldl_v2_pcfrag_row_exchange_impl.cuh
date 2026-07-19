#pragma once

#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_fragment_exchange_common.cuh"
#include "symldl_v2_l_fragment_map_impl.cuh"

#ifdef HAVE_CUDA

struct SymLDLV2RowFragmentExchangeResult
{
    int_t row_nrows;
    int row_recv_total;
    int row_send_total;
};

static SymLDLV2RowFragmentExchangeResult
symldl_v2_pcfrag_exchange_row_fragments(
    xLUstruct_t<double> *lu, int_t k, int stream_offset,
    cudaStream_t stream, int_t kcol, int_t ksupc, int_t lk,
    std::vector<MPI_Request> &send_reqs)
{
    int tag_ub = lu->symFactTagUb;
    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    int_t row_nrows = row_index.empty() ? 0 : row_index[1];
    int_t empty_header[LPANEL_HEADER_SIZE] = {0, 0, 0, ksupc};
    if (!row_index.empty())
    {
        if (row_index[3] != ksupc)
            ABORT("SymFact V2 row fragment index has wrong width.");
        if (static_cast<int_t>(row_index.size()) >
            lu->maxSymV2RowFragIdxRecvCount)
            ABORT("SymFact V2 row fragment index exceeds buffer.");
        if (lu->A_gpu.symV2RowFragIdxRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row index buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symV2RowFragIdxRecvBufs[stream_offset],
            row_index.data(), sizeof(int_t) * row_index.size(),
            cudaMemcpyHostToDevice, stream));
    }
    else if (lu->A_gpu.symV2RowFragIdxRecvBufs[stream_offset] != NULL)
    {
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symV2RowFragIdxRecvBufs[stream_offset], empty_header,
            sizeof(int_t) * LPANEL_HEADER_SIZE,
            cudaMemcpyHostToDevice, stream));
    }

    auto row_count_destination = [&](int pc_dest) -> int
    {
        if (lk < 0)
            return 0;
        size_t slot = static_cast<size_t>(lk) * static_cast<size_t>(lu->Pc) +
                      static_cast<size_t>(pc_dest);
        if (slot >= lu->symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send size is missing.");
        return lu->symV2RowDownSendSizes[slot];
    };

    int_t *row_down_sendmap_gpu = lu->symL2LSendMapPoolGPU;
    size_t row_down_sendmap_base = 0;
    auto ensure_row_down_sendmap_gpu = [&]() -> int_t *
    {
        if (row_down_sendmap_gpu != NULL)
            return row_down_sendmap_gpu;
        row_down_sendmap_gpu = symldl_v2_panel_send_maps_gpu(
            lu, lk, stream_offset, stream, &row_down_sendmap_base);
        return row_down_sendmap_gpu;
    };

    auto row_pack_destination = [&](int pc_dest, double *dst_buf) -> int
    {
        if (lu->mycol != kcol)
            ABORT("SymFact V2 row-down pack called on a non-source rank.");
        if (lk < 0 || static_cast<size_t>(lk) >= lu->symV2PanelCount())
            ABORT("SymFact V2 row-down source panel is invalid.");
        if (dst_buf == NULL)
            ABORT("SymFact V2 row-down destination buffer is missing.");
        xlpanel_t<double> &lpanel = lu->lPanelVec[lk];
        if (lpanel.isEmpty())
            ABORT("SymFact V2 row-down source L panel is missing.");
        size_t slot = static_cast<size_t>(lk) * static_cast<size_t>(lu->Pc) +
                      static_cast<size_t>(pc_dest);
        if (slot >= lu->symV2RowDownSendSizes.size() ||
            slot >= lu->symV2RowDownSendSegsGPU.size() ||
            slot >= lu->symV2RowDownSendSegCounts.size())
            ABORT("SymFact V2 row-down send segment slot is invalid.");
        int total = lu->symV2RowDownSendSizes[slot];
        if (total <= 0)
            return 0;
        if (total > lu->maxSymV2RowFragValSendCount)
            ABORT("SymFact V2 row-down packed destination exceeds send buffer.");
        if (total % ksupc != 0)
            ABORT("SymFact V2 row-down packed destination has wrong width.");
        int nsegments = lu->symV2RowDownSendSegCounts[slot];
        SymV2RowDownSendSegmentGPU *segments =
            lu->symV2RowDownSendSegsGPU[slot];
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
        static_cast<size_t>(k) * static_cast<size_t>(lu->Pc);
    if (row_recv_base + static_cast<size_t>(lu->Pc) >
            lu->symV2RowFragRecvSizes.size())
        ABORT("SymFact V2 row-fragment receive sizes are missing.");
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        int count = lu->symV2RowFragRecvSizes[
            row_recv_base + static_cast<size_t>(pc)];
        if (count < 0 ||
            row_recv_total > std::numeric_limits<int>::max() - count)
            ABORT("SymFact V2 row-fragment receive size is invalid.");
        row_recv_total += count;
    }
    if (row_recv_total > lu->maxSymV2RowFragStageCount ||
        row_recv_total > lu->maxSymV2RowFragValRecvCount)
        ABORT("SymFact V2 row-fragment receive exceeds staging buffer.");

    MPI_Request row_recv_req = MPI_REQUEST_NULL;
    double *row_recv_host = NULL;
    int row_send_total = 0;
    if (row_recv_total > 0 && row_src_pc != lu->mycol)
    {
        if (static_cast<size_t>(stream_offset) >=
                lu->symV2RowFragHostRecvBufs.size() ||
            lu->symV2RowFragHostRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment host receive staging is missing.");
        row_recv_host = lu->symV2RowFragHostRecvBufs[stream_offset];
        MPI_Irecv(row_recv_host, row_recv_total, MPI_DOUBLE, row_src_pc,
                  SLU_MPI_TAG(5, k), lu->grid3d->rscp.comm, &row_recv_req);
    }

    if (lu->mycol == kcol)
    {
        if (static_cast<size_t>(stream_offset) >=
                lu->symV2RowFragHostSendBufs.size() ||
            lu->symV2RowFragHostSendBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment host send staging is missing.");
        if (lu->A_gpu.symV2RowFragStageBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment device staging buffer is missing.");
        if (lu->symV2RowFragSendCountsScratch.size() !=
            static_cast<size_t>(lu->Pc))
            lu->symV2RowFragSendCountsScratch.assign(
                static_cast<size_t>(lu->Pc), 0);
        if (lu->symV2RowFragSendOffsetsScratch.size() !=
            static_cast<size_t>(lu->Pc))
            lu->symV2RowFragSendOffsetsScratch.assign(
                static_cast<size_t>(lu->Pc), -1);
        std::fill(lu->symV2RowFragSendCountsScratch.begin(),
                  lu->symV2RowFragSendCountsScratch.end(), 0);
        std::fill(lu->symV2RowFragSendOffsetsScratch.begin(),
                  lu->symV2RowFragSendOffsetsScratch.end(), -1);

        for (int pc_dest = 0; pc_dest < lu->Pc; ++pc_dest)
        {
            if (pc_dest == lu->mycol)
                continue;
            int count = row_count_destination(pc_dest);
            if (count <= 0)
                continue;
            if (count > lu->maxSymV2RowFragValSendCount ||
                row_send_total > lu->maxSymV2RowFragValSendCount - count)
                ABORT("SymFact V2 row-fragment send exceeds staging buffer.");
            lu->symV2RowFragSendOffsetsScratch[
                static_cast<size_t>(pc_dest)] = row_send_total;
            lu->symV2RowFragSendCountsScratch[
                static_cast<size_t>(pc_dest)] = count;
            row_send_total += count;
        }
        for (int pc_dest = 0; pc_dest < lu->Pc; ++pc_dest)
        {
            int count =
                lu->symV2RowFragSendCountsScratch[static_cast<size_t>(pc_dest)];
            if (count <= 0)
                continue;
            int offset =
                lu->symV2RowFragSendOffsetsScratch[static_cast<size_t>(pc_dest)];
            if (offset < 0)
                ABORT("SymFact V2 row-fragment send offset is invalid.");
            int packed = row_pack_destination(
                pc_dest, lu->A_gpu.symV2RowFragStageBufs[stream_offset] + offset);
            if (packed != count)
                ABORT("SymFact V2 row-fragment pack size mismatch.");
        }
        if (row_send_total > 0)
        {
            gpuErrchk(cudaMemcpyAsync(
                lu->symV2RowFragHostSendBufs[stream_offset],
                lu->A_gpu.symV2RowFragStageBufs[stream_offset],
                sizeof(double) * static_cast<size_t>(row_send_total),
                cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            for (int pc_dest = 0; pc_dest < lu->Pc; ++pc_dest)
            {
                int count = lu->symV2RowFragSendCountsScratch[
                    static_cast<size_t>(pc_dest)];
                if (count <= 0)
                    continue;
                int offset = lu->symV2RowFragSendOffsetsScratch[
                    static_cast<size_t>(pc_dest)];
                MPI_Request req;
                MPI_Isend(lu->symV2RowFragHostSendBufs[stream_offset] + offset,
                          count, MPI_DOUBLE, pc_dest, SLU_MPI_TAG(5, k),
                          lu->grid3d->rscp.comm, &req);
                send_reqs.push_back(req);
                lu->symV2RouteProfileNoteRowSend(
                    static_cast<size_t>(count), 1,
                    static_cast<size_t>(count));
            }
        }
    }

    if (row_recv_req != MPI_REQUEST_NULL)
        MPI_Wait(&row_recv_req, MPI_STATUS_IGNORE);
    if (row_recv_total > 0 && row_src_pc != lu->mycol)
    {
        if (lu->A_gpu.symV2RowFragValRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment value buffer is missing.");
        gpuErrchk(cudaMemcpyAsync(
            lu->A_gpu.symV2RowFragValRecvBufs[stream_offset], row_recv_host,
            sizeof(double) * static_cast<size_t>(row_recv_total),
            cudaMemcpyHostToDevice, stream));
    }
    else if (row_recv_total > 0)
    {
        if (lu->A_gpu.symV2RowFragValRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 row-fragment value buffer is missing.");
        int packed = row_pack_destination(
            lu->mycol, lu->A_gpu.symV2RowFragValRecvBufs[stream_offset]);
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

    SymLDLV2RowFragmentExchangeResult result;
    result.row_nrows = row_nrows;
    result.row_recv_total = row_recv_total;
    result.row_send_total = row_send_total;
    return result;
}

#endif
