#pragma once

#include <climits>
#include <cstring>

#include "xlupanels.hpp"

template <typename T>
static void symldl_v2_cpu_bcast_chunks(
    T *buffer, size_t count, MPI_Datatype datatype, int root, MPI_Comm comm,
    uint64_t *oversized_chunks)
{
    size_t offset = 0;
    const size_t limit = static_cast<size_t>(INT_MAX);
    size_t chunks = count == 0 ? 0 : (count - 1) / limit + 1;
    if (chunks > 1)
        *oversized_chunks += static_cast<uint64_t>(chunks);
    while (offset < count)
    {
        int chunk = static_cast<int>(SUPERLU_MIN(limit, count - offset));
        if (MPI_Bcast(buffer + offset, chunk, datatype, root, comm) !=
            MPI_SUCCESS)
            ABORT("SymFact V2 CPU panel broadcast failed.");
        offset += static_cast<size_t>(chunk);
    }
}

template <typename Ftype>
static void symldl_v2_cpu_capture_raw_panel(
    xLUstruct_t<Ftype> *lu, int_t k, int slot)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuRawPanelBufs.size())
        ABORT("SymFact V2 CPU raw-panel slot is invalid.");
    if (lu->mycol != lu->symV2PanelRoot(k))
        return;

    int_t value_count = lu->LvalSendCounts[static_cast<size_t>(k)];
    if (value_count <= 0)
        return;
    if (static_cast<size_t>(value_count) > lu->symV2CpuRawPanelCapacity)
        ABORT("SymFact V2 CPU raw-panel workspace is undersized.");

    int_t lk = lu->symV2PanelIndex(k);
    if (lk < 0 || lk >= lu->symV2PanelCount())
        ABORT("SymFact V2 CPU panel owner has no local panel.");
    xlpanel_t<Ftype> &panel = lu->lPanelVec[lk];
    if (panel.isEmpty() || panel.nzvalSize() != value_count)
        ABORT("SymFact V2 CPU source panel has invalid values.");
    std::memcpy(lu->symV2CpuRawPanelBufs[slot], panel.val,
                static_cast<size_t>(value_count) * sizeof(Ftype));
}

template <typename Ftype>
static xlpanel_t<Ftype> symldl_v2_cpu_exchange_panel(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, Ftype **raw_values)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuRawPanelBufs.size() ||
        static_cast<size_t>(slot) >= lu->LidxRecvBufs.size() ||
        static_cast<size_t>(slot) >= lu->LvalRecvBufs.size())
        ABORT("SymFact V2 CPU panel slot is invalid.");

    int_t index_count = lu->LidxSendCounts[static_cast<size_t>(k)];
    int_t value_count = lu->LvalSendCounts[static_cast<size_t>(k)];
    if (index_count < 0 || value_count < 0)
        ABORT("SymFact V2 CPU panel count is invalid.");
    if (static_cast<size_t>(value_count) > lu->symV2CpuRawPanelCapacity)
        ABORT("SymFact V2 CPU raw-panel workspace is undersized.");

    int root = static_cast<int>(lu->symV2PanelRoot(k));
    int_t *index = lu->LidxRecvBufs[slot];
    Ftype *values = lu->LvalRecvBufs[slot];
    Ftype *raw = lu->symV2CpuRawPanelBufs[slot];
    if (lu->mycol == root)
    {
        int_t lk = lu->symV2PanelIndex(k);
        if (lk < 0 || lk >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU panel owner has no local panel.");
        xlpanel_t<Ftype> &local = lu->lPanelVec[lk];
        if (local.isEmpty())
            ABORT("SymFact V2 CPU panel owner has an empty panel.");
        index = local.index;
        values = local.val;
    }

    double start = SuperLU_timer_();
    if (lu->grid3d->rscp.Np > 1)
    {
        if (index_count > 0)
            symldl_v2_cpu_bcast_chunks(
                index, static_cast<size_t>(index_count), mpi_int_t, root,
                lu->grid3d->rscp.comm, &lu->symV2CpuOversizedMpiChunks);
        if (value_count > 0)
        {
            symldl_v2_cpu_bcast_chunks(
                values, static_cast<size_t>(value_count),
                get_mpi_type<Ftype>(), root, lu->grid3d->rscp.comm,
                &lu->symV2CpuOversizedMpiChunks);
            symldl_v2_cpu_bcast_chunks(
                raw, static_cast<size_t>(value_count),
                get_mpi_type<Ftype>(), root, lu->grid3d->rscp.comm,
                &lu->symV2CpuOversizedMpiChunks);
        }
    }
    double elapsed = SuperLU_timer_() - start;
    lu->symV2CpuPanelExchangeTime += elapsed;
    lu->SCT->tPanelBcast += elapsed;

    *raw_values = raw;
    if (index_count == 0 || value_count == 0)
        return xlpanel_t<Ftype>();
    return xlpanel_t<Ftype>(index, values);
}
