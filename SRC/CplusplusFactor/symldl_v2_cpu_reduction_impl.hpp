#pragma once

#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_blas.hpp"

template <typename Ftype>
static size_t symldl_v2_cpu_post_reduction_chunks(
    xLUstruct_t<Ftype> *lu, Ftype *buffer, int_t count, int peer, int tag,
    bool send, size_t request_begin)
{
    if (count < 0)
        ABORT("SymFact V2 CPU reduction count is invalid.");
    size_t offset = 0;
    size_t request_count = request_begin;
    const size_t max_chunk =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (offset < static_cast<size_t>(count))
    {
        size_t remaining = static_cast<size_t>(count) - offset;
        int chunk = static_cast<int>(SUPERLU_MIN(remaining, max_chunk));
        if (request_count >= lu->symV2CpuRequests.size())
            ABORT("SymFact V2 CPU reduction request workspace is undersized.");
        if (send)
            MPI_Isend(buffer + offset, chunk, get_mpi_type<Ftype>(), peer,
                      tag, lu->grid3d->zscp.comm,
                      &lu->symV2CpuRequests[request_count]);
        else
            MPI_Irecv(buffer + offset, chunk, get_mpi_type<Ftype>(), peer,
                      tag, lu->grid3d->zscp.comm,
                      &lu->symV2CpuRequests[request_count]);
        ++request_count;
        offset += static_cast<size_t>(chunk);
    }
    size_t chunks = request_count - request_begin;
    if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 CPU reduction has too many MPI chunks.");
    if (chunks > 1)
        lu->symV2CpuOversizedMpiChunks +=
            static_cast<uint64_t>(chunks);
    return request_count;
}

template <typename Ftype>
static void symldl_v2_cpu_window_reduction_transfer(
    xLUstruct_t<Ftype> *lu, Ftype *buffer, int_t count, int peer, int tag,
    bool send)
{
    if (count < 0)
        ABORT("SymFact V2 CPU window reduction count is invalid.");
    size_t offset = 0;
    const size_t limit =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (offset < static_cast<size_t>(count))
    {
        int chunk = static_cast<int>(SUPERLU_MIN(
            limit, static_cast<size_t>(count) - offset));
        int result = MPI_SUCCESS;
        if (send)
            result = MPI_Send(buffer + offset, chunk, get_mpi_type<Ftype>(),
                              peer, tag, lu->grid3d->zscp.comm);
        else
            result = MPI_Recv(buffer + offset, chunk, get_mpi_type<Ftype>(),
                              peer, tag, lu->grid3d->zscp.comm,
                              MPI_STATUS_IGNORE);
        if (result != MPI_SUCCESS)
            ABORT("SymFact V2 CPU window ancestor transfer failed.");
        offset += static_cast<size_t>(chunk);
        if (static_cast<size_t>(count) > limit)
            ++lu->symV2CpuOversizedMpiChunks;
    }
}

template <typename Ftype>
static int_t symldl_v2_cpu_window_ancestor_reduction(
    xLUstruct_t<Ftype> *lu, int_t ilvl, int_t *my_node_count,
    int_t **tree_perm)
{
    int_t level_count = log2i(lu->grid3d->zscp.Np) + 1;
    int_t my_layer = lu->grid3d->zscp.Iam;
    int_t stride = static_cast<int_t>(1) << ilvl;
    int_t group = static_cast<int_t>(1) << (ilvl + 1);
    int sender = 0;
    int receiver = 0;
    if ((my_layer % group) == 0)
    {
        sender = my_layer + stride;
        receiver = my_layer;
    }
    else
    {
        sender = my_layer;
        receiver = my_layer - stride;
    }

    double start = SuperLU_timer_();
    for (int_t ancestor_level = ilvl + 1;
         ancestor_level < level_count; ++ancestor_level)
    {
        int_t node_count = my_node_count[ancestor_level];
        int_t *nodes = tree_perm[ancestor_level];
        for (int_t position = 0; position < node_count; ++position)
        {
            int_t k = nodes[position];
            if (lu->symV2PanelRoot(k) != lu->mycol)
                continue;
            int_t local_panel = lu->symV2PanelIndex(k);
            if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
                lu->symV2PanelGid(local_panel) != k)
                ABORT("SymFact V2 CPU window reduction panel is invalid.");
            xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
            if (panel.isEmpty())
                continue;
            int_t count = panel.nzvalSize();
            if (my_layer == sender)
            {
                symldl_v2_cpu_window_reduction_transfer(
                    lu, panel.blkPtr(0), count, receiver,
                    static_cast<int>(k), true);
                uint64_t bytes = static_cast<uint64_t>(count) * sizeof(Ftype);
                lu->symV2CpuReductionBytes += bytes;
                lu->SCT->commVolRed += bytes;
            }
            else
            {
                symldl_v2_cpu_window_reduction_transfer(
                    lu, lu->LvalRecvBufs[0], count, sender,
                    static_cast<int>(k), false);
                symldl_v2_cpu_axpy<Ftype>(
                    count, one<Ftype>(), lu->LvalRecvBufs[0], 1,
                    panel.blkPtr(0), 1);
            }
        }
    }
    double elapsed = SuperLU_timer_() - start;
    lu->symV2CpuReductionTime += elapsed;
    lu->SCT->ancsReduce += elapsed;
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_cpu_ancestor_reduction(
    xLUstruct_t<Ftype> *lu, int_t ilvl, int_t *my_node_count,
    int_t **tree_perm)
{
    int_t level_count = log2i(lu->grid3d->zscp.Np) + 1;
    int_t my_layer = lu->grid3d->zscp.Iam;
    int tag_ub = lu->symFactTagUb;
    int_t stride = static_cast<int_t>(1) << ilvl;
    int_t group = static_cast<int_t>(1) << (ilvl + 1);
    int sender = 0;
    int receiver = 0;
    if ((my_layer % group) == 0)
    {
        sender = my_layer + stride;
        receiver = my_layer;
    }
    else
    {
        sender = my_layer;
        receiver = my_layer - stride;
    }

    double start = SuperLU_timer_();
    int slots = static_cast<int>(lu->symV2CpuReductionPanelSlots.size());
    if (slots <= 0 || static_cast<size_t>(slots) > lu->LvalRecvBufs.size())
        ABORT("SymFact V2 CPU reduction slots are unavailable.");
#pragma omp parallel
    {
#pragma omp single
        {
            for (int_t ancestor_level = ilvl + 1;
                 ancestor_level < level_count; ++ancestor_level)
            {
                int_t node_count = my_node_count[ancestor_level];
                int_t *nodes = tree_perm[ancestor_level];
                int_t position = 0;
                while (position < node_count)
                {
                    int active = 0;
                    while (position < node_count && active < slots)
                    {
                        int_t k = nodes[position++];
                        if (lu->symV2PanelRoot(k) != lu->mycol)
                            continue;
                        int_t local_panel = lu->symV2PanelIndex(k);
                        if (local_panel < 0 ||
                            local_panel >= lu->symV2PanelCount() ||
                            lu->symV2PanelGid(local_panel) != k)
                            ABORT("SymFact V2 CPU reduction panel is invalid.");
                        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
                        if (panel.isEmpty())
                            continue;
                        lu->symV2CpuReductionPanelSlots[
                            static_cast<size_t>(active++)] = local_panel;
                    }
                    if (active == 0)
                        continue;

                    size_t requests = 0;
                    for (int slot = 0; slot < active; ++slot)
                    {
                        int_t local_panel = lu->symV2CpuReductionPanelSlots[
                            static_cast<size_t>(slot)];
                        int_t k = lu->symV2PanelGid(local_panel);
                        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
                        int_t count = panel.nzvalSize();
                        int tag = SLU_MPI_TAG(6, k);
                        Ftype *buffer = my_layer == sender
                                            ? panel.blkPtr(0)
                                            : lu->LvalRecvBufs[slot];
                        size_t begin = requests;
                        requests = symldl_v2_cpu_post_reduction_chunks(
                            lu, buffer, count,
                            my_layer == sender ? receiver : sender, tag,
                            my_layer == sender, requests);
                        size_t chunks = requests - begin;
                        if (chunks > static_cast<size_t>(
                                         std::numeric_limits<int>::max()))
                            ABORT("SymFact V2 CPU reduction has too many chunks.");
                        lu->symV2CpuReductionChunksRemaining[
                            static_cast<size_t>(slot)] =
                            static_cast<int>(chunks);
                        for (size_t request = begin; request < requests;
                             ++request)
                            lu->symV2CpuRequestPeers[request] = slot;
                        if (my_layer == sender)
                        {
                            uint64_t bytes = static_cast<uint64_t>(count) *
                                             static_cast<uint64_t>(sizeof(Ftype));
                            lu->symV2CpuReductionBytes += bytes;
                            lu->SCT->commVolRed += bytes;
                        }
                    }
                    if (requests > static_cast<size_t>(
                                       std::numeric_limits<int>::max()) ||
                        requests > lu->symV2CpuWaitIndices.size())
                        ABORT("SymFact V2 CPU reduction has too many requests.");

                    size_t pending = requests;
                    while (pending > 0)
                    {
                        int completed = 0;
                        if (MPI_Testsome(
                                static_cast<int>(requests),
                                lu->symV2CpuRequests.data(), &completed,
                                lu->symV2CpuWaitIndices.data(),
                                lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                            ABORT("SymFact V2 CPU reduction progress failed.");
                        ++lu->symV2CpuMpiTestsomeCalls;
                        if (completed == MPI_UNDEFINED)
                            ABORT("SymFact V2 CPU reduction lost active requests.");
                        if (completed == 0)
                        {
                            if (MPI_Waitsome(
                                    static_cast<int>(requests),
                                    lu->symV2CpuRequests.data(), &completed,
                                    lu->symV2CpuWaitIndices.data(),
                                    lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                                ABORT("SymFact V2 CPU reduction wait failed.");
                            ++lu->symV2CpuMpiWaitsomeCalls;
                        }
                        if (completed == MPI_UNDEFINED || completed <= 0 ||
                            static_cast<size_t>(completed) > pending)
                            ABORT("SymFact V2 CPU reduction made no progress.");
                        lu->symV2CpuMpiCompletions +=
                            static_cast<uint64_t>(completed);
                        for (int item = 0; item < completed; ++item)
                        {
                            int request = lu->symV2CpuWaitIndices[
                                static_cast<size_t>(item)];
                            if (request < 0 ||
                                static_cast<size_t>(request) >= requests)
                                ABORT("SymFact V2 CPU reduction completion is invalid.");
                            int slot = lu->symV2CpuRequestPeers[
                                static_cast<size_t>(request)];
                            if (slot < 0 || slot >= active)
                                ABORT("SymFact V2 CPU reduction slot is invalid.");
                            int &remaining =
                                lu->symV2CpuReductionChunksRemaining[
                                    static_cast<size_t>(slot)];
                            if (remaining <= 0)
                                ABORT("SymFact V2 CPU reduction completed twice.");
                            --remaining;
                            if (remaining == 0 && my_layer == receiver)
                            {
                                int_t local_panel =
                                    lu->symV2CpuReductionPanelSlots[
                                        static_cast<size_t>(slot)];
#pragma omp task firstprivate(slot, local_panel) shared(lu)
                                {
                                    xlpanel_t<Ftype> &panel =
                                        lu->lPanelVec[local_panel];
                                    symldl_v2_cpu_axpy<Ftype>(
                                        panel.nzvalSize(), one<Ftype>(),
                                        lu->LvalRecvBufs[slot], 1,
                                        panel.blkPtr(0), 1);
                                }
                            }
                        }
                        pending -= static_cast<size_t>(completed);
                    }
#pragma omp taskwait
                    for (size_t request = 0; request < requests; ++request)
                    {
                        if (lu->symV2CpuRequests[request] != MPI_REQUEST_NULL)
                            ABORT("SymFact V2 CPU reduction request remains active.");
                        lu->symV2CpuRequestPeers[request] = -1;
                    }
                }
            }
        }
    }
    double elapsed = SuperLU_timer_() - start;
    lu->symV2CpuReductionTime += elapsed;
    lu->SCT->ancsReduce += elapsed;
    return 0;
}
