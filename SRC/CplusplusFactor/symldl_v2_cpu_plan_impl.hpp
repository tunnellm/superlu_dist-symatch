#pragma once

#include <algorithm>
#include <climits>
#include <limits>
#include <utility>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

struct SymLDLV2CpuCachedRowBlock
{
    int_t gid;
    size_t cols_begin;
    int_t len;
    bool needed;
};

static SymLDLV2PartnerMetaPayload symldl_v2_cpu_route_row_demands(
    const std::vector<std::vector<int_t> > &send_payloads, MPI_Comm comm,
    uint64_t *oversized_chunks)
{
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (send_payloads.size() != static_cast<size_t>(comm_size))
        ABORT("SymFact V2 CPU row demand route has wrong size.");

    std::vector<unsigned long long> send_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<unsigned long long> recv_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<size_t> send_displs(static_cast<size_t>(comm_size), 0);
    size_t send_total = 0;
    bool local_fits = true;
    for (int destination = 0; destination < comm_size; ++destination)
    {
        const std::vector<int_t> &payload =
            send_payloads[static_cast<size_t>(destination)];
        if (payload.size() >
            static_cast<size_t>(std::numeric_limits<unsigned long long>::max()) ||
            payload.size() > std::numeric_limits<size_t>::max() - send_total)
            ABORT("SymFact V2 CPU row demand payload size overflows.");
        send_counts[static_cast<size_t>(destination)] =
            static_cast<unsigned long long>(payload.size());
        send_displs[static_cast<size_t>(destination)] = send_total;
        send_total += payload.size();
        local_fits = local_fits &&
            payload.size() <= static_cast<size_t>(INT_MAX) &&
            send_displs[static_cast<size_t>(destination)] <=
                static_cast<size_t>(INT_MAX);
    }

    std::vector<int_t> send_payload(send_total);
    for (int destination = 0; destination < comm_size; ++destination)
    {
        const std::vector<int_t> &payload =
            send_payloads[static_cast<size_t>(destination)];
        std::copy(payload.begin(), payload.end(),
                  send_payload.begin() +
                      send_displs[static_cast<size_t>(destination)]);
    }

    if (MPI_Alltoall(send_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     recv_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     comm) != MPI_SUCCESS)
        ABORT("SymFact V2 CPU row demand size exchange failed.");

    SymLDLV2PartnerMetaPayload result;
    result.comm_size = comm_size;
    result.counts.assign(static_cast<size_t>(comm_size), 0);
    result.displs.assign(static_cast<size_t>(comm_size), 0);
    size_t recv_total = 0;
    for (int source = 0; source < comm_size; ++source)
    {
        unsigned long long count = recv_counts[static_cast<size_t>(source)];
        if (count > static_cast<unsigned long long>(
                        std::numeric_limits<size_t>::max()) ||
            static_cast<size_t>(count) >
                std::numeric_limits<size_t>::max() - recv_total)
            ABORT("SymFact V2 CPU row demand payload size overflows.");
        result.counts[static_cast<size_t>(source)] =
            static_cast<size_t>(count);
        result.displs[static_cast<size_t>(source)] = recv_total;
        recv_total += static_cast<size_t>(count);
        local_fits = local_fits &&
            count <= static_cast<unsigned long long>(INT_MAX) &&
            result.displs[static_cast<size_t>(source)] <=
                static_cast<size_t>(INT_MAX);
    }
    local_fits = local_fits &&
        send_total <= static_cast<size_t>(INT_MAX) &&
        recv_total <= static_cast<size_t>(INT_MAX);

    int local_fits_int = local_fits ? 1 : 0;
    int all_fit = 0;
    if (MPI_Allreduce(&local_fits_int, &all_fit, 1, MPI_INT, MPI_MIN,
                      comm) != MPI_SUCCESS)
        ABORT("SymFact V2 CPU row demand limit exchange failed.");
    if (!all_fit)
        return symldl_v2_allgather_metadata(
            send_payload, comm, oversized_chunks);

    std::vector<int> send_counts_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> send_displs_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> recv_counts_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> recv_displs_int(static_cast<size_t>(comm_size), 0);
    for (int peer = 0; peer < comm_size; ++peer)
    {
        send_counts_int[static_cast<size_t>(peer)] =
            static_cast<int>(send_counts[static_cast<size_t>(peer)]);
        send_displs_int[static_cast<size_t>(peer)] =
            static_cast<int>(send_displs[static_cast<size_t>(peer)]);
        recv_counts_int[static_cast<size_t>(peer)] =
            static_cast<int>(result.counts[static_cast<size_t>(peer)]);
        recv_displs_int[static_cast<size_t>(peer)] =
            static_cast<int>(result.displs[static_cast<size_t>(peer)]);
    }
    result.payload.assign(recv_total, 0);
    if (MPI_Alltoallv(
            send_payload.empty() ? NULL : send_payload.data(),
            send_counts_int.data(), send_displs_int.data(), mpi_int_t,
            result.payload.empty() ? NULL : result.payload.data(),
            recv_counts_int.data(), recv_displs_int.data(), mpi_int_t,
            comm) != MPI_SUCCESS)
        ABORT("SymFact V2 CPU row demand exchange failed.");
    return result;
}

template <typename Ftype>
static std::vector<int_t> symldl_v2_cpu_fragment_index(
    xlpanel_t<Ftype> &panel)
{
    int_t first = panel.haveDiag() ? 1 : 0;
    int_t nblocks = panel.nblocks() - first;
    int_t nrows = panel.nzrows() -
                  (panel.haveDiag() ? panel.nbrow(0) : 0);
    if (nblocks <= 0 || nrows <= 0)
        return std::vector<int_t>();

    size_t index_size = static_cast<size_t>(LPANEL_HEADER_SIZE) +
                        2 * static_cast<size_t>(nblocks) + 1 +
                        static_cast<size_t>(nrows);
    std::vector<int_t> index(index_size, 0);
    index[0] = nblocks;
    index[1] = nrows;
    index[2] = 0;
    index[3] = panel.ncols();
    int_t gid_pos = LPANEL_HEADER_SIZE;
    int_t prefix_pos = LPANEL_HEADER_SIZE + nblocks;
    int_t row_pos = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
    index[prefix_pos] = 0;
    for (int_t block = 0; block < nblocks; ++block)
    {
        int_t source_block = first + block;
        int_t rows = panel.nbrow(source_block);
        index[gid_pos + block] = panel.gid(source_block);
        index[prefix_pos + block + 1] =
            index[prefix_pos + block] + rows;
        std::copy(panel.rowList(source_block),
                  panel.rowList(source_block) + rows,
                  index.begin() + row_pos);
        row_pos += rows;
    }
    return index;
}

template <typename Ftype>
static void symldl_v2_build_cpu_partner_send_plan(
    xLUstruct_t<Ftype> *lu)
{
    int_t local_panels = lu->symV2PanelCount();
    size_t slots = symldl_v2_checked_product(
        static_cast<size_t>(local_panels), static_cast<size_t>(lu->Pc),
        "SymFact V2 CPU partner plan overflows.");
    if (lu->symL2LSendMeta.size() != slots ||
        lu->symV2PartnerLSendSizes.size() != slots)
        ABORT("SymFact V2 CPU partner plan tables are not initialized.");
    size_t active_slots = symldl_v2_checked_product(
        slots, static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU partner active mask overflows.");
    lu->symV2CpuPartnerSendRowActive.assign(active_slots, 0);

    lu->symV2CpuPartnerSegOffsets.assign(slots + 1, 0);
    lu->symV2CpuPartnerSendOffsets.assign(slots, 0);
    lu->symV2CpuPartnerSendSizes.assign(slots, 0);
    lu->symV2CpuPartnerSegments.clear();
    lu->symV2CpuPartnerRowPermutations.clear();

    std::vector<std::pair<int_t, int_t> > blocks;
    std::vector<std::pair<int_t, int_t> > rows;
    for (int_t local_panel = 0; local_panel < local_panels; ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        size_t panel_value_offset = 0;
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t slot = static_cast<size_t>(local_panel) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            lu->symV2CpuPartnerSegOffsets[slot] =
                lu->symV2CpuPartnerSegments.size();
            lu->symV2CpuPartnerSendOffsets[slot] = panel_value_offset;
            lu->symL2LSendMeta[slot].clear();
            lu->symV2PartnerLSendSizes[slot] = 0;
            if (panel.isEmpty())
                continue;

            blocks.clear();
            int_t first = panel.haveDiag() ? 1 : 0;
            for (int_t block = first; block < panel.nblocks(); ++block)
                if (lu->symV2PanelRoot(panel.gid(block)) == pc)
                    blocks.push_back(
                        std::make_pair(panel.gid(block), block));
            std::sort(blocks.begin(), blocks.end());

            int_t packed_rows = 0;
            for (size_t b = 0; b < blocks.size(); ++b)
            {
                int_t source_block = blocks[b].second;
                int_t row_count = panel.nbrow(source_block);
                int_t *row_ids = panel.rowList(source_block);
                bool rows_are_sorted = true;
                for (int_t row = 1; row < row_count; ++row)
                    rows_are_sorted = rows_are_sorted &&
                                      row_ids[row - 1] < row_ids[row];
                if (!rows_are_sorted)
                {
                    rows.clear();
                    rows.reserve(static_cast<size_t>(row_count));
                    for (int_t row = 0; row < row_count; ++row)
                        rows.push_back(std::make_pair(row_ids[row], row));
                    std::sort(rows.begin(), rows.end());
                }

                SymLDLV2CpuPackSegment segment;
                segment.source_block = source_block;
                segment.row_count = row_count;
                segment.packed_row_offset = packed_rows;
                segment.row_permutation_offset = rows_are_sorted
                    ? SYM_LDL_V2_CPU_CONTIGUOUS_ROWS
                    : lu->symV2CpuPartnerRowPermutations.size();
                lu->symV2CpuPartnerSegments.push_back(segment);
                lu->symL2LSendMeta[slot].push_back(blocks[b].first);
                lu->symL2LSendMeta[slot].push_back(row_count);
                for (int_t row = 0; row < row_count; ++row)
                {
                    lu->symL2LSendMeta[slot].push_back(
                        rows_are_sorted ? row_ids[row] : rows[row].first);
                    if (!rows_are_sorted)
                        lu->symV2CpuPartnerRowPermutations.push_back(
                            rows[row].second);
                }
                packed_rows += row_count;
            }

            size_t values = symldl_v2_checked_product(
                static_cast<size_t>(packed_rows),
                static_cast<size_t>(panel.ncols()),
                "SymFact V2 CPU partner values overflow.");
            if (values > lu->symV2CpuPartnerSendCapacity ||
                panel_value_offset >
                    lu->symV2CpuPartnerSendCapacity - values)
                ABORT("SymFact V2 CPU partner-send workspace is undersized.");
            lu->symV2CpuPartnerSendSizes[slot] = values;
            if (values <=
                static_cast<size_t>(std::numeric_limits<int>::max()))
                lu->symV2PartnerLSendSizes[slot] =
                    static_cast<int>(values);
            if (values > 0)
                for (int pr = 0; pr < lu->Pr; ++pr)
                    lu->symV2CpuPartnerSendRowActive[
                        slot * static_cast<size_t>(lu->Pr) + pr] = 1;
            panel_value_offset += values;
        }
    }
    lu->symV2CpuPartnerSegOffsets[slots] =
        lu->symV2CpuPartnerSegments.size();
}

template <typename Ftype>
static void symldl_v2_build_cpu_partner_receive_plan(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &metadata)
{
    size_t receive_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU compact receive plan overflows.");
    lu->symV2PartnerLRecvSizes.assign(receive_slots, 0);
    lu->symV2CpuPartnerRecvSizes.assign(receive_slots, 0);
    lu->symV2PartnerLRecvIndexBySrc.assign(
        receive_slots, std::vector<int_t>());

    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = static_cast<size_t>(metadata.displs[rank]);
        size_t end = position + static_cast<size_t>(metadata.counts[rank]);
        int source_pr =
            symldl_v2_partner_metadata_source_row(metadata, rank, lu);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 CPU partner metadata is truncated.");
            int_t target_pc = metadata.payload[position++];
            int_t k = metadata.payload[position++];
            int_t length = metadata.payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 CPU partner metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (target_pc == lu->mycol)
            {
                int_t nblocks = 0;
                int_t nrows = 0;
                size_t scan = position;
                while (scan < record_end)
                {
                    if (scan + 2 > record_end)
                        ABORT("SymFact V2 CPU partner block is truncated.");
                    ++nblocks;
                    ++scan;
                    int_t rows = metadata.payload[scan++];
                    if (rows < 0 || scan + static_cast<size_t>(rows) > record_end)
                        ABORT("SymFact V2 CPU partner block is invalid.");
                    if (nrows > std::numeric_limits<int_t>::max() - rows)
                        ABORT("SymFact V2 CPU partner row count overflows.");
                    nrows += rows;
                    scan += static_cast<size_t>(rows);
                }
                size_t slot = static_cast<size_t>(k) * lu->Pr + source_pr;
                if (!lu->symV2PartnerLRecvIndexBySrc[slot].empty())
                    ABORT("SymFact V2 CPU partner metadata is duplicated.");
                std::vector<int_t> &index =
                    lu->symV2PartnerLRecvIndexBySrc[slot];
                index.assign(static_cast<size_t>(LPANEL_HEADER_SIZE) +
                                 2 * static_cast<size_t>(nblocks) + 1 +
                                 static_cast<size_t>(nrows),
                             0);
                index[0] = nblocks;
                index[1] = nrows;
                index[2] = 0;
                index[3] = lu->supersize(k);
                size_t values = symldl_v2_checked_product(
                    static_cast<size_t>(nrows),
                    static_cast<size_t>(lu->supersize(k)),
                    "SymFact V2 CPU partner receive size overflows.");
                lu->symV2CpuPartnerRecvSizes[slot] = values;
                if (values <=
                    static_cast<size_t>(std::numeric_limits<int>::max()))
                    lu->symV2PartnerLRecvSizes[slot] =
                        static_cast<int>(values);
                int_t gid_pos = LPANEL_HEADER_SIZE;
                int_t prefix_pos = LPANEL_HEADER_SIZE + nblocks;
                int_t row_pos = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
                index[prefix_pos] = 0;
                scan = position;
                for (int_t block = 0; block < nblocks; ++block)
                {
                    index[gid_pos + block] = metadata.payload[scan++];
                    int_t rows = metadata.payload[scan++];
                    index[prefix_pos + block + 1] =
                        index[prefix_pos + block] + rows;
                    std::copy(metadata.payload.begin() + scan,
                              metadata.payload.begin() + scan + rows,
                              index.begin() + row_pos);
                    row_pos += rows;
                    scan += static_cast<size_t>(rows);
                }
            }
            position = record_end;
        }
    }
}

struct SymLDLV2CpuPartnerBlockRef
{
    int_t gid;
    int source_pr;
    int_t source_block;
};

template <typename Ftype>
static void symldl_v2_build_cpu_partner_aggregate_plan(
    xLUstruct_t<Ftype> *lu)
{
    size_t compact_count = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU partner aggregate table overflows.");
    lu->symV2CpuPartnerAssembledIndex.assign(
        static_cast<size_t>(lu->nsupers), std::vector<int_t>());
    lu->symV2CpuPartnerAssembleMaps.assign(
        compact_count, std::vector<int_t>());

    std::vector<SymLDLV2CpuPartnerBlockRef> blocks;
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        blocks.clear();
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t source_pos = static_cast<size_t>(k) * lu->Pr + pr;
            if (source_pos >= lu->symV2PartnerLRecvIndexBySrc.size())
                ABORT("SymFact V2 CPU partner source index is missing.");
            const std::vector<int_t> &source =
                lu->symV2PartnerLRecvIndexBySrc[source_pos];
            if (source.empty())
                continue;
            xlpanel_t<Ftype> panel(
                const_cast<int_t *>(source.data()), (Ftype *) NULL);
            for (int_t block = 0; block < panel.nblocks(); ++block)
            {
                SymLDLV2CpuPartnerBlockRef ref;
                ref.gid = panel.gid(block);
                ref.source_pr = pr;
                ref.source_block = block;
                blocks.push_back(ref);
            }
        }
        if (blocks.empty())
            continue;
        std::stable_sort(
            blocks.begin(), blocks.end(),
            [](const SymLDLV2CpuPartnerBlockRef &left,
               const SymLDLV2CpuPartnerBlockRef &right)
            {
                if (left.gid != right.gid)
                    return left.gid < right.gid;
                return left.source_pr < right.source_pr;
            });

        int_t total_rows = 0;
        for (size_t block = 0; block < blocks.size(); ++block)
        {
            size_t source_pos = static_cast<size_t>(k) * lu->Pr +
                                blocks[block].source_pr;
            const std::vector<int_t> &source =
                lu->symV2PartnerLRecvIndexBySrc[source_pos];
            xlpanel_t<Ftype> panel(
                const_cast<int_t *>(source.data()), (Ftype *) NULL);
            int_t rows = panel.nbrow(blocks[block].source_block);
            if (rows < 0 || total_rows > std::numeric_limits<int_t>::max() - rows)
                ABORT("SymFact V2 CPU partner aggregate rows overflow.");
            total_rows += rows;
        }
        size_t index_size = static_cast<size_t>(LPANEL_HEADER_SIZE) +
                            2 * blocks.size() + 1 +
                            static_cast<size_t>(total_rows);
        std::vector<int_t> &aggregate =
            lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(k)];
        aggregate.assign(index_size, 0);
        aggregate[0] = static_cast<int_t>(blocks.size());
        aggregate[1] = total_rows;
        aggregate[2] = 0;
        aggregate[3] = lu->supersize(k);
        int_t gid_pos = LPANEL_HEADER_SIZE;
        int_t prefix_pos = LPANEL_HEADER_SIZE + aggregate[0];
        int_t row_pos = LPANEL_HEADER_SIZE + 2 * aggregate[0] + 1;
        aggregate[prefix_pos] = 0;

        for (size_t block = 0; block < blocks.size(); ++block)
        {
            const SymLDLV2CpuPartnerBlockRef &ref = blocks[block];
            size_t source_pos = static_cast<size_t>(k) * lu->Pr +
                                ref.source_pr;
            const std::vector<int_t> &source =
                lu->symV2PartnerLRecvIndexBySrc[source_pos];
            xlpanel_t<Ftype> panel(
                const_cast<int_t *>(source.data()), (Ftype *) NULL);
            int_t rows = panel.nbrow(ref.source_block);
            int_t destination_row = aggregate[prefix_pos + block];
            aggregate[gid_pos + block] = ref.gid;
            aggregate[prefix_pos + block + 1] = destination_row + rows;
            std::copy(panel.rowList(ref.source_block),
                      panel.rowList(ref.source_block) + rows,
                      aggregate.begin() + row_pos);
            row_pos += rows;

            std::vector<int_t> &map =
                lu->symV2CpuPartnerAssembleMaps[source_pos];
            map.push_back(destination_row);
            map.push_back(rows);
            map.push_back(panel.stRow(ref.source_block));
        }
        size_t values = symldl_v2_checked_product(
            static_cast<size_t>(total_rows),
            static_cast<size_t>(lu->supersize(k)),
            "SymFact V2 CPU assembled partner values overflow.");
        if (values > lu->symV2CpuRawPanelCapacity)
            ABORT("SymFact V2 CPU assembled partner workspace is undersized.");
    }
}

template <typename Ftype>
static void symldl_v2_build_cpu_row_receive_plan(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &source_row_metadata,
    const SymLDLV2PartnerMetaPayload &target_column_metadata)
{
    double phase_start = SuperLU_timer_();
    const std::vector<int_t> &payload = source_row_metadata.payload;
    lu->symV2CpuRowMetadataBlocks = 0;
    lu->symV2CpuRowDemandRawBlocks = 0;
    lu->symV2CpuRowDemandUniqueBlocks = 0;
    lu->symV2CpuRowRecvIndexEntries = 0;
    lu->symV2CpuRowLocalDemandEntries = 0;
    lu->symV2CpuRowReceivedDemandEntries = 0;
    std::vector<std::vector<SymLDLV2CpuCachedRowBlock> > row_blocks(
        static_cast<size_t>(lu->nsupers));
    for (int rank = 0; rank < source_row_metadata.comm_size; ++rank)
    {
        size_t position =
            static_cast<size_t>(source_row_metadata.displs[rank]);
        size_t end = position +
                     static_cast<size_t>(source_row_metadata.counts[rank]);
        int source_pr =
            symldl_v2_partner_metadata_source_row(
                source_row_metadata, rank, lu);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 CPU row metadata is truncated.");
            int_t target_pc = payload[position++];
            int_t k = payload[position++];
            int_t length = payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 CPU row metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (source_pr == lu->myrow)
            {
                while (position < record_end)
                {
                    if (position + 2 > record_end)
                        ABORT("SymFact V2 CPU row block is truncated.");
                    SymLDLV2CpuCachedRowBlock block;
                    block.gid = payload[position++];
                    block.len = payload[position++];
                    block.cols_begin = position;
                    block.needed = false;
                    if (block.len < 0 ||
                        position + static_cast<size_t>(block.len) >
                            record_end)
                        ABORT("SymFact V2 CPU row block is invalid.");
                    row_blocks[static_cast<size_t>(k)].push_back(block);
                    ++lu->symV2CpuRowMetadataBlocks;
                    position += static_cast<size_t>(block.len);
                }
            }
            position = record_end;
        }
    }

    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        std::vector<SymLDLV2CpuCachedRowBlock> &blocks =
            row_blocks[static_cast<size_t>(k)];
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CpuCachedRowBlock &left,
                     const SymLDLV2CpuCachedRowBlock &right)
                  {
                      return left.gid < right.gid;
                  });
        for (size_t block = 1; block < blocks.size(); ++block)
            if (blocks[block - 1].gid == blocks[block].gid)
                ABORT("SymFact V2 CPU row block metadata is duplicated.");
    }
    lu->symV2CpuRowMetadataPlanTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();

    std::vector<std::vector<int_t> > needed(
        static_cast<size_t>(lu->nsupers));
    const std::vector<int_t> &target_payload =
        target_column_metadata.payload;
    for (int rank = 0; rank < target_column_metadata.comm_size; ++rank)
    {
        size_t position =
            static_cast<size_t>(target_column_metadata.displs[rank]);
        size_t end = position +
                     static_cast<size_t>(
                         target_column_metadata.counts[rank]);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 CPU row-demand metadata is truncated.");
            int_t target_pc = target_payload[position++];
            int_t k = target_payload[position++];
            int_t length = target_payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 CPU row-demand metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (target_pc == lu->mycol)
            {
                std::vector<SymLDLV2CpuCachedRowBlock> &rows =
                    row_blocks[static_cast<size_t>(k)];
                while (position < record_end)
                {
                    if (position + 2 > record_end)
                        ABORT("SymFact V2 CPU row-demand block is truncated.");
                    int_t gj = target_payload[position++];
                    int_t count = target_payload[position++];
                    if (count < 0 ||
                        position + static_cast<size_t>(count) > record_end)
                        ABORT("SymFact V2 CPU row-demand block is invalid.");
                    if (!rows.empty() &&
                        lu->symV2PanelRoot(gj) == lu->mycol)
                    {
                        int_t local_panel = lu->symV2PanelIndex(gj);
                        if (local_panel >= 0 &&
                            local_panel < lu->symV2PanelCount() &&
                            lu->symV2PanelGid(local_panel) == gj)
                        {
                            xlpanel_t<Ftype> &destination =
                                lu->lPanelVec[local_panel];
                            if (!destination.isEmpty())
                            {
                                typename std::vector<
                                    SymLDLV2CpuCachedRowBlock>::iterator
                                    row = std::lower_bound(
                                        rows.begin(), rows.end(), gj,
                                        [](const SymLDLV2CpuCachedRowBlock &block,
                                           int_t gid)
                                        {
                                            return block.gid < gid;
                                        });
                                int_t destination_block = 0;
                                while (row != rows.end() &&
                                       destination_block <
                                           destination.nblocks())
                                {
                                    int_t destination_gid =
                                        destination.gid(destination_block);
                                    if (row->gid < destination_gid)
                                    {
                                        ++row;
                                        continue;
                                    }
                                    if (destination_gid < row->gid)
                                    {
                                        ++destination_block;
                                        continue;
                                    }
                                    ++lu->symV2CpuRowDemandRawBlocks;
                                    if (!row->needed)
                                    {
                                        row->needed = true;
                                        needed[static_cast<size_t>(k)]
                                            .push_back(row->gid);
                                        ++lu->symV2CpuRowDemandUniqueBlocks;
                                    }
                                    ++row;
                                    ++destination_block;
                                }
                            }
                        }
                    }
                    position += static_cast<size_t>(count);
                }
            }
            position = record_end;
        }
    }
    lu->symV2CpuRowDemandPlanTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();

    lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                     std::vector<int_t>());
    lu->symV2RowFragRecvSizes.assign(
        symldl_v2_checked_product(static_cast<size_t>(lu->nsupers),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 CPU row plan overflows."),
        0);
    std::vector<std::vector<int_t> > local_demands(
        static_cast<size_t>(lu->Pc));
    std::vector<size_t> chunk_values(static_cast<size_t>(lu->Pc), 0);
    std::vector<const SymLDLV2CpuCachedRowBlock *> selected_rows;
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        std::vector<int_t> &blocks = needed[static_cast<size_t>(k)];
        if (blocks.empty())
            continue;
        size_t unique_blocks = blocks.size();

        blocks.clear();
        selected_rows.clear();
        selected_rows.reserve(unique_blocks);
        std::fill(chunk_values.begin(), chunk_values.end(), 0);

        const std::vector<SymLDLV2CpuCachedRowBlock> &rows =
            row_blocks[static_cast<size_t>(k)];
        int_t row_count = 0;
        for (size_t block = 0; block < rows.size(); ++block)
        {
            const SymLDLV2CpuCachedRowBlock &row = rows[block];
            if (row.gid < 0 || row.gid >= lu->nsupers)
                ABORT("SymFact V2 CPU row metadata block is invalid.");
            if (!row.needed)
                continue;
            blocks.push_back(row.gid);
            selected_rows.push_back(&row);
            if (row_count > std::numeric_limits<int_t>::max() - row.len)
                ABORT("SymFact V2 CPU row index overflows.");
            row_count += row.len;
            int chunk_pc = static_cast<int>(lu->symV2PanelRoot(row.gid));
            if (chunk_pc < 0 || chunk_pc >= lu->Pc)
                ABORT("SymFact V2 CPU row source column is invalid.");
            size_t block_values = symldl_v2_checked_product(
                static_cast<size_t>(row.len),
                static_cast<size_t>(lu->supersize(k)),
                "SymFact V2 CPU row receive chunk size overflows.");
            if (chunk_values[static_cast<size_t>(chunk_pc)] >
                std::numeric_limits<size_t>::max() - block_values)
                ABORT("SymFact V2 CPU row receive chunk size overflows.");
            chunk_values[static_cast<size_t>(chunk_pc)] += block_values;
        }
        if (blocks.size() != unique_blocks)
            ABORT("SymFact V2 CPU row demand cannot find its source block.");
        size_t index_size = static_cast<size_t>(LPANEL_HEADER_SIZE) +
                            2 * blocks.size() + 1 +
                            static_cast<size_t>(row_count);
        lu->symV2CpuRowRecvIndexEntries += index_size;
        std::vector<int_t> &index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
        index.assign(index_size, 0);
        index[0] = static_cast<int_t>(blocks.size());
        index[1] = row_count;
        index[2] = 0;
        index[3] = lu->supersize(k);
        int_t gid_pos = LPANEL_HEADER_SIZE;
        int_t prefix_pos = gid_pos + index[0];
        int_t row_pos = prefix_pos + index[0] + 1;
        index[prefix_pos] = 0;
        for (int_t block = 0; block < index[0]; ++block)
        {
            const SymLDLV2CpuCachedRowBlock &row =
                *selected_rows[static_cast<size_t>(block)];
            index[gid_pos + block] = row.gid;
            index[prefix_pos + block + 1] =
                index[prefix_pos + block] + row.len;
            std::copy(payload.begin() + row.cols_begin,
                      payload.begin() + row.cols_begin + row.len,
                      index.begin() + row_pos);
            row_pos += row.len;
        }
        size_t values = symldl_v2_checked_product(
            static_cast<size_t>(row_count),
            static_cast<size_t>(lu->supersize(k)),
            "SymFact V2 CPU row receive size overflows.");
        if (values > lu->symV2CpuRowRecvCapacity)
            ABORT("SymFact V2 CPU row receive exceeds workspace.");
        for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
        {
            size_t chunk = chunk_values[static_cast<size_t>(chunk_pc)];
            if (chunk <=
                static_cast<size_t>(std::numeric_limits<int>::max()))
                lu->symV2RowFragRecvSizes[
                    static_cast<size_t>(k) * lu->Pc + chunk_pc] =
                    static_cast<int>(chunk);
        }

        int source_pc = static_cast<int>(lu->symV2PanelRoot(k));
        if (source_pc < 0 || source_pc >= lu->Pc)
            ABORT("SymFact V2 CPU row demand source is invalid.");
        std::vector<int_t> &source_demands =
            local_demands[static_cast<size_t>(source_pc)];
        source_demands.push_back(k);
        source_demands.push_back(static_cast<int_t>(blocks.size()));
        source_demands.insert(source_demands.end(), blocks.begin(), blocks.end());
    }
    for (int source_pc = 0; source_pc < lu->Pc; ++source_pc)
        lu->symV2CpuRowLocalDemandEntries +=
            local_demands[static_cast<size_t>(source_pc)].size();
    lu->symV2CpuRowRecvLayoutTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();

    int row_comm_size = 0;
    int row_comm_rank = -1;
    MPI_Comm_size(lu->grid3d->rscp.comm, &row_comm_size);
    MPI_Comm_rank(lu->grid3d->rscp.comm, &row_comm_rank);
    if (row_comm_size != lu->Pc || row_comm_rank != lu->mycol)
        ABORT("SymFact V2 CPU row communicator is invalid.");
    SymLDLV2PartnerMetaPayload gathered_demands =
        symldl_v2_cpu_route_row_demands(
            local_demands, lu->grid3d->rscp.comm,
            &lu->symV2CpuOversizedMpiChunks);
    const std::vector<int_t> &all_demands = gathered_demands.payload;
    lu->symV2CpuRowReceivedDemandEntries = all_demands.size();

    size_t slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->symV2PanelCount()),
        static_cast<size_t>(lu->Pc),
        "SymFact V2 CPU row send plan overflows.");
    std::vector<std::vector<int_t> > requests(slots);
    for (int destination_pc = 0; destination_pc < row_comm_size;
         ++destination_pc)
    {
        size_t position = gathered_demands.displs[destination_pc];
        size_t end = position + gathered_demands.counts[destination_pc];
        while (position < end)
        {
            if (position + 2 > end)
                ABORT("SymFact V2 CPU row demand is truncated.");
            int_t k = all_demands[position++];
            int_t count = all_demands[position++];
            if (k < 0 || k >= lu->nsupers || count < 0 ||
                position + static_cast<size_t>(count) > end)
                ABORT("SymFact V2 CPU row demand is invalid.");
            if (lu->symV2PanelRoot(k) == lu->mycol)
            {
                int_t local_panel = lu->symV2PanelIndex(k);
                if (local_panel < 0 ||
                    local_panel >= lu->symV2PanelCount() ||
                    lu->symV2PanelGid(local_panel) != k)
                    ABORT("SymFact V2 CPU row demand has no source panel.");
                size_t slot = static_cast<size_t>(local_panel) * lu->Pc +
                              static_cast<size_t>(destination_pc);
                if (!requests[slot].empty())
                    ABORT("SymFact V2 CPU row demand is duplicated.");
                requests[slot].assign(all_demands.begin() + position,
                                      all_demands.begin() + position + count);
            }
            position += static_cast<size_t>(count);
        }
    }
    lu->symV2CpuRowDemandExchangeTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();

    lu->symV2CpuRowSegOffsets.assign(slots + 1, 0);
    lu->symV2CpuRowSendOffsets.assign(slots, 0);
    lu->symV2CpuRowSendSizes.assign(slots, 0);
    lu->symV2CpuRowSegments.clear();
    lu->symV2CpuRowPermutations.clear();
    size_t max_panel_values = 0;
    std::vector<size_t> partner_segment_by_block;
    for (int_t local_panel = 0; local_panel < lu->symV2PanelCount();
         ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        partner_segment_by_block.assign(
            static_cast<size_t>(panel.isEmpty() ? 0 : panel.nblocks()),
            std::numeric_limits<size_t>::max());
        for (int source_pc = 0; source_pc < lu->Pc; ++source_pc)
        {
            size_t partner_slot = static_cast<size_t>(local_panel) * lu->Pc +
                                  static_cast<size_t>(source_pc);
            for (size_t segment_id =
                     lu->symV2CpuPartnerSegOffsets[partner_slot];
                 segment_id <
                     lu->symV2CpuPartnerSegOffsets[partner_slot + 1];
                 ++segment_id)
            {
                const SymLDLV2CpuPackSegment &segment =
                    lu->symV2CpuPartnerSegments[segment_id];
                if (segment.source_block < 0 ||
                    segment.source_block >= panel.nblocks())
                    ABORT("SymFact V2 CPU partner segment is invalid.");
                size_t source_block =
                    static_cast<size_t>(segment.source_block);
                if (partner_segment_by_block[source_block] !=
                    std::numeric_limits<size_t>::max())
                    ABORT("SymFact V2 CPU partner segment is duplicated.");
                partner_segment_by_block[source_block] = segment_id;
            }
        }
        size_t panel_values = 0;
        for (int destination_pc = 0; destination_pc < lu->Pc;
             ++destination_pc)
        {
            size_t slot = static_cast<size_t>(local_panel) * lu->Pc +
                          static_cast<size_t>(destination_pc);
            lu->symV2CpuRowSegOffsets[slot] =
                lu->symV2CpuRowSegments.size();
            lu->symV2CpuRowSendOffsets[slot] = panel_values;
            int_t packed_rows = 0;
            for (size_t request = 0; request < requests[slot].size();
                 ++request)
            {
                int_t gid = requests[slot][request];
                int_t source_block = panel.find(gid);
                if (source_block == GLOBAL_BLOCK_NOT_FOUND)
                    ABORT("SymFact V2 CPU row source block is missing.");
                size_t partner_segment_id =
                    partner_segment_by_block[static_cast<size_t>(source_block)];
                if (partner_segment_id ==
                    std::numeric_limits<size_t>::max())
                    ABORT("SymFact V2 CPU row source segment is missing.");
                const SymLDLV2CpuPackSegment &partner_segment =
                    lu->symV2CpuPartnerSegments[partner_segment_id];
                if (partner_segment.row_count != panel.nbrow(source_block))
                    ABORT("SymFact V2 CPU row source segment is invalid.");
                SymLDLV2CpuPackSegment segment = partner_segment;
                segment.packed_row_offset = packed_rows;
                if (segment.row_permutation_offset !=
                    SYM_LDL_V2_CPU_CONTIGUOUS_ROWS)
                {
                    size_t permutation_begin = segment.row_permutation_offset;
                    size_t permutation_end = permutation_begin +
                                             static_cast<size_t>(
                                                 segment.row_count);
                    if (permutation_end < permutation_begin ||
                        permutation_end >
                            lu->symV2CpuPartnerRowPermutations.size())
                        ABORT("SymFact V2 CPU partner permutation is invalid.");
                    segment.row_permutation_offset =
                        lu->symV2CpuRowPermutations.size();
                    lu->symV2CpuRowPermutations.insert(
                        lu->symV2CpuRowPermutations.end(),
                        lu->symV2CpuPartnerRowPermutations.begin() +
                            permutation_begin,
                        lu->symV2CpuPartnerRowPermutations.begin() +
                            permutation_end);
                }
                lu->symV2CpuRowSegments.push_back(segment);
                packed_rows += segment.row_count;
            }
            size_t values = symldl_v2_checked_product(
                static_cast<size_t>(packed_rows),
                static_cast<size_t>(panel.isEmpty() ? 0 : panel.ncols()),
                "SymFact V2 CPU row send size overflows.");
            if (panel_values > std::numeric_limits<size_t>::max() - values)
                ABORT("SymFact V2 CPU row send size overflows.");
            lu->symV2CpuRowSendSizes[slot] = values;
            panel_values += values;
        }
        max_panel_values = SUPERLU_MAX(max_panel_values, panel_values);
    }
    lu->symV2CpuRowSegOffsets[slots] = lu->symV2CpuRowSegments.size();
    symldl_v2_resize_cpu_slot_buffers(
        lu->symV2CpuRowSendBufs, lu->symV2CpuRowSendCapacity,
        max_panel_values,
        "SymFact V2 CPU compact row-send workspace overflows.",
        "Malloc fails for SymFact V2 CPU compact row-send workspace.");
    lu->symV2CpuRowSendPlanTime += SuperLU_timer_() - phase_start;
}

template <typename Ftype>
static void symldl_v2_build_cpu_fragment_plan(xLUstruct_t<Ftype> *lu)
{
    if (!lu->symV2UsesCpuFactor() || (lu->Pr <= 1 && lu->Pc <= 1))
        return;
    SymLDLV2CpuExchangeRoute route = symldl_v2_cpu_exchange_route(lu);
    double plan_start = SuperLU_timer_();
    double phase_start = plan_start;
    symldl_v2_build_cpu_partner_send_plan(lu);
    lu->symV2CpuPartnerSendPlanTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();
    SymLDLV2PartnerMetaPayload metadata;
    SymLDLV2ScopedPartnerMetaPayload scoped_metadata;
    const SymLDLV2PartnerMetaPayload *source_row_metadata = NULL;
    const SymLDLV2PartnerMetaPayload *target_column_metadata = NULL;
    if (superlu_sym_v2_scoped_fragment_metadata())
    {
        scoped_metadata = symldl_v2_collect_scoped_partner_l_metadata(lu);
        source_row_metadata = &scoped_metadata.source_row;
        target_column_metadata = &scoped_metadata.target_column;
    }
    else
    {
        metadata = symldl_v2_collect_partner_l_metadata(lu);
        source_row_metadata = &metadata;
        target_column_metadata = &metadata;
    }
    lu->symV2CpuMetadataGatherTime += SuperLU_timer_() - phase_start;
    phase_start = SuperLU_timer_();
    symldl_v2_build_cpu_partner_receive_plan(
        lu, *target_column_metadata);
    if (symldl_v2_cpu_scheduler_kind() ==
        SYM_LDL_V2_CPU_SCHEDULER_WINDOW)
        symldl_v2_build_cpu_partner_aggregate_plan(lu);
    lu->symV2CpuPartnerRecvPlanTime += SuperLU_timer_() - phase_start;
    if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT)
    {
        phase_start = SuperLU_timer_();
        symldl_v2_build_cpu_row_receive_plan(
            lu, *source_row_metadata, *target_column_metadata);
        lu->symV2CpuRowPlanTime += SuperLU_timer_() - phase_start;
    }
    phase_start = SuperLU_timer_();
    symldl_v2_resize_cpu_request_workspace(lu);
    lu->symV2CpuRequestPlanTime += SuperLU_timer_() - phase_start;
    lu->symV2CpuPlanBuildTime += SuperLU_timer_() - plan_start;
}
