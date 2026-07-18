#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_cached_block_impl.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

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
    size_t size_count = symldl_v2_checked_product(
        symldl_v2_checked_product(static_cast<size_t>(lu->nsupers),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 CPU receive plan overflows."),
        static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU receive plan overflows.");
    std::vector<unsigned long long> local_sizes(size_count, 0);
    std::vector<unsigned long long> global_sizes(size_count, 0);
    for (int_t local_panel = 0;
         local_panel < lu->symV2PanelCount(); ++local_panel)
    {
        int_t k = lu->symV2PanelGid(local_panel);
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_slot = static_cast<size_t>(local_panel) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(pc);
            size_t size_slot = (static_cast<size_t>(k) *
                                    static_cast<size_t>(lu->Pc) +
                                static_cast<size_t>(pc)) *
                                   static_cast<size_t>(lu->Pr) +
                               static_cast<size_t>(lu->myrow);
            local_sizes[size_slot] =
                static_cast<unsigned long long>(
                    lu->symV2CpuPartnerSendSizes[send_slot]);
        }
    }
    size_t reduced = 0;
    const size_t mpi_limit =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (reduced < size_count)
    {
        int chunk = static_cast<int>(SUPERLU_MIN(
            mpi_limit, size_count - reduced));
        MPI_Allreduce(local_sizes.data() + reduced,
                      global_sizes.data() + reduced, chunk,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, lu->grid->comm);
        reduced += static_cast<size_t>(chunk);
    }

    size_t receive_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU compact receive plan overflows.");
    lu->symV2PartnerLRecvSizes.assign(receive_slots, 0);
    lu->symV2CpuPartnerRecvSizes.assign(receive_slots, 0);
    lu->symV2PartnerLRecvIndexBySrc.assign(
        receive_slots, std::vector<int_t>());
    for (int_t k = 0; k < lu->nsupers; ++k)
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t source = (static_cast<size_t>(k) *
                                 static_cast<size_t>(lu->Pc) +
                             static_cast<size_t>(lu->mycol)) *
                                static_cast<size_t>(lu->Pr) +
                            static_cast<size_t>(pr);
            size_t receive_slot = static_cast<size_t>(k) * lu->Pr + pr;
            if (global_sizes[source] >
                static_cast<unsigned long long>(
                    std::numeric_limits<size_t>::max()))
                ABORT("SymFact V2 CPU partner receive size overflows.");
            size_t values = static_cast<size_t>(global_sizes[source]);
            lu->symV2CpuPartnerRecvSizes[receive_slot] = values;
            if (values <=
                static_cast<size_t>(std::numeric_limits<int>::max()))
                lu->symV2PartnerLRecvSizes[receive_slot] =
                    static_cast<int>(values);
        }

    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = static_cast<size_t>(metadata.displs[rank]);
        size_t end = position + static_cast<size_t>(metadata.counts[rank]);
        int source_pr = MYROW(rank, lu->grid);
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
                size_t expected = symldl_v2_checked_product(
                    static_cast<size_t>(nrows),
                    static_cast<size_t>(lu->supersize(k)),
                    "SymFact V2 CPU partner receive size overflows.");
                if (expected != lu->symV2CpuPartnerRecvSizes[slot])
                    ABORT("SymFact V2 CPU partner metadata size mismatch.");
            }
            position = record_end;
        }
    }
}

template <typename Ftype>
static void symldl_v2_build_cpu_row_receive_plan(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &metadata)
{
    const std::vector<int_t> &payload = metadata.payload;
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> > row_blocks(
        static_cast<size_t>(lu->nsupers));
    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = static_cast<size_t>(metadata.displs[rank]);
        size_t end = position + static_cast<size_t>(metadata.counts[rank]);
        int source_pr = MYROW(rank, lu->grid);
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
                    SymLDLV2CachedPartnerBlock block;
                    block.gid = payload[position++];
                    block.len = payload[position++];
                    block.cols_begin = position;
                    if (block.len < 0 ||
                        position + static_cast<size_t>(block.len) >
                            record_end)
                        ABORT("SymFact V2 CPU row block is invalid.");
                    block.cols.assign(payload.begin() + position,
                                      payload.begin() + position + block.len);
                    row_blocks[static_cast<size_t>(k)].push_back(block);
                    position += static_cast<size_t>(block.len);
                }
            }
            position = record_end;
        }
    }

    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        std::vector<SymLDLV2CachedPartnerBlock> &blocks =
            row_blocks[static_cast<size_t>(k)];
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CachedPartnerBlock &left,
                     const SymLDLV2CachedPartnerBlock &right)
                  {
                      return left.gid < right.gid;
                  });
        for (size_t block = 1; block < blocks.size(); ++block)
            if (blocks[block - 1].gid == blocks[block].gid)
                ABORT("SymFact V2 CPU row block metadata is duplicated.");
    }

    std::vector<std::vector<int_t> > needed(
        static_cast<size_t>(lu->nsupers));
    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = static_cast<size_t>(metadata.displs[rank]);
        size_t end = position + static_cast<size_t>(metadata.counts[rank]);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 CPU row-demand metadata is truncated.");
            int_t target_pc = payload[position++];
            int_t k = payload[position++];
            int_t length = payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 CPU row-demand metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (target_pc == lu->mycol)
            {
                const std::vector<SymLDLV2CachedPartnerBlock> &rows =
                    row_blocks[static_cast<size_t>(k)];
                while (position < record_end)
                {
                    if (position + 2 > record_end)
                        ABORT("SymFact V2 CPU row-demand block is truncated.");
                    int_t gj = payload[position++];
                    int_t count = payload[position++];
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
                                    SymLDLV2CachedPartnerBlock>::const_iterator
                                    row = std::lower_bound(
                                        rows.begin(), rows.end(), gj,
                                        [](const SymLDLV2CachedPartnerBlock &block,
                                           int_t gid)
                                        {
                                            return block.gid < gid;
                                        });
                                for (; row != rows.end(); ++row)
                                    if (destination.find(row->gid) !=
                                        GLOBAL_BLOCK_NOT_FOUND)
                                        needed[static_cast<size_t>(k)]
                                            .push_back(row->gid);
                            }
                        }
                    }
                    position += static_cast<size_t>(count);
                }
            }
            position = record_end;
        }
    }

    lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                     std::vector<int_t>());
    lu->symV2RowFragRecvSizes.assign(
        symldl_v2_checked_product(static_cast<size_t>(lu->nsupers),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 CPU row plan overflows."),
        0);
    std::vector<int_t> local_demands;
    std::vector<size_t> chunk_values(static_cast<size_t>(lu->Pc), 0);
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        std::vector<int_t> &blocks = needed[static_cast<size_t>(k)];
        if (blocks.empty())
            continue;
        std::sort(blocks.begin(), blocks.end());
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        std::fill(chunk_values.begin(), chunk_values.end(), 0);

        const std::vector<SymLDLV2CachedPartnerBlock> &rows =
            row_blocks[static_cast<size_t>(k)];
        int_t row_count = 0;
        for (size_t block = 0; block < blocks.size(); ++block)
        {
            typename std::vector<SymLDLV2CachedPartnerBlock>::const_iterator
                row = std::lower_bound(
                    rows.begin(), rows.end(), blocks[block],
                    [](const SymLDLV2CachedPartnerBlock &entry, int_t gid)
                    {
                        return entry.gid < gid;
                    });
            if (row == rows.end() || row->gid != blocks[block])
                ABORT("SymFact V2 CPU row demand cannot find its source block.");
            if (row_count > std::numeric_limits<int_t>::max() - row->len)
                ABORT("SymFact V2 CPU row index overflows.");
            row_count += row->len;
            int chunk_pc = static_cast<int>(lu->symV2PanelRoot(row->gid));
            if (chunk_pc < 0 || chunk_pc >= lu->Pc)
                ABORT("SymFact V2 CPU row source column is invalid.");
            size_t block_values = symldl_v2_checked_product(
                static_cast<size_t>(row->len),
                static_cast<size_t>(lu->supersize(k)),
                "SymFact V2 CPU row receive chunk size overflows.");
            if (chunk_values[static_cast<size_t>(chunk_pc)] >
                std::numeric_limits<size_t>::max() - block_values)
                ABORT("SymFact V2 CPU row receive chunk size overflows.");
            chunk_values[static_cast<size_t>(chunk_pc)] += block_values;
        }
        size_t index_size = static_cast<size_t>(LPANEL_HEADER_SIZE) +
                            2 * blocks.size() + 1 +
                            static_cast<size_t>(row_count);
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
            typename std::vector<SymLDLV2CachedPartnerBlock>::const_iterator
                row = std::lower_bound(
                    rows.begin(), rows.end(), blocks[block],
                    [](const SymLDLV2CachedPartnerBlock &entry, int_t gid)
                    {
                        return entry.gid < gid;
                    });
            index[gid_pos + block] = row->gid;
            index[prefix_pos + block + 1] =
                index[prefix_pos + block] + row->len;
            std::copy(row->cols.begin(), row->cols.end(),
                      index.begin() + row_pos);
            row_pos += row->len;
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

        local_demands.push_back(k);
        local_demands.push_back(static_cast<int_t>(blocks.size()));
        local_demands.insert(local_demands.end(), blocks.begin(), blocks.end());
    }

    int row_comm_size = 0;
    MPI_Comm_size(lu->grid3d->rscp.comm, &row_comm_size);
    if (row_comm_size != lu->Pc)
        ABORT("SymFact V2 CPU row communicator has wrong size.");
    SymLDLV2PartnerMetaPayload gathered_demands =
        symldl_v2_allgather_metadata(
            local_demands, lu->grid3d->rscp.comm,
            &lu->symV2CpuOversizedMpiChunks);
    const std::vector<int_t> &all_demands = gathered_demands.payload;

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

    lu->symV2CpuRowSegOffsets.assign(slots + 1, 0);
    lu->symV2CpuRowSendOffsets.assign(slots, 0);
    lu->symV2CpuRowSendSizes.assign(slots, 0);
    lu->symV2CpuRowSegments.clear();
    lu->symV2CpuRowPermutations.clear();
    size_t max_panel_values = 0;
    std::vector<std::pair<int_t, int_t> > sorted_rows;
    for (int_t local_panel = 0; local_panel < lu->symV2PanelCount();
         ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
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
                int_t rows = panel.nbrow(source_block);
                int_t *row_ids = panel.rowList(source_block);
                bool rows_are_sorted = true;
                for (int_t row = 1; row < rows; ++row)
                    rows_are_sorted = rows_are_sorted &&
                                      row_ids[row - 1] < row_ids[row];
                if (!rows_are_sorted)
                {
                    sorted_rows.clear();
                    sorted_rows.reserve(static_cast<size_t>(rows));
                    for (int_t row = 0; row < rows; ++row)
                        sorted_rows.push_back(
                            std::make_pair(row_ids[row], row));
                    std::sort(sorted_rows.begin(), sorted_rows.end());
                }
                SymLDLV2CpuPackSegment segment;
                segment.source_block = source_block;
                segment.row_count = rows;
                segment.packed_row_offset = packed_rows;
                segment.row_permutation_offset = rows_are_sorted
                    ? SYM_LDL_V2_CPU_CONTIGUOUS_ROWS
                    : lu->symV2CpuRowPermutations.size();
                lu->symV2CpuRowSegments.push_back(segment);
                if (!rows_are_sorted)
                    for (int_t row = 0; row < rows; ++row)
                        lu->symV2CpuRowPermutations.push_back(
                            sorted_rows[static_cast<size_t>(row)].second);
                packed_rows += rows;
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
}

template <typename Ftype>
static void symldl_v2_build_cpu_fragment_plan(xLUstruct_t<Ftype> *lu)
{
    if (!lu->symV2UsesCpuFactor() || (lu->Pr <= 1 && lu->Pc <= 1))
        return;
    double plan_start = SuperLU_timer_();
    symldl_v2_build_cpu_partner_send_plan(lu);
    SymLDLV2PartnerMetaPayload metadata =
        symldl_v2_collect_partner_l_metadata(lu);
    symldl_v2_build_cpu_partner_receive_plan(lu, metadata);
    symldl_v2_build_cpu_row_receive_plan(lu, metadata);
    symldl_v2_resize_cpu_request_workspace(lu);
    lu->symV2CpuPlanBuildTime += SuperLU_timer_() - plan_start;
}
