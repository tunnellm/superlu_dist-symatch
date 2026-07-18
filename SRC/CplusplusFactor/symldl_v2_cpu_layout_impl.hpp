#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "xlupanels.hpp"

template <typename Ftype>
static void symldl_v2_cpu_canonicalize_l_panel_rows(
    xLUstruct_t<Ftype> *lu)
{
    if (!lu->symV2UsesCpuFactor())
        return;
    if (!lu->symV2CpuCanonicalBlocks.empty() ||
        !lu->symV2CpuCanonicalToOriginalRows.empty() ||
        lu->symV2CpuRowsCanonical)
        ABORT("SymFact V2 CPU row layout is already initialized.");

    std::vector<std::pair<int_t, int_t> > order(
        static_cast<size_t>(lu->ldt));
    std::vector<int_t> sorted_rows(static_cast<size_t>(lu->ldt));
    std::vector<Ftype> sorted_values(symldl_v2_checked_product(
        static_cast<size_t>(lu->ldt), static_cast<size_t>(lu->ldt),
        "SymFact V2 CPU row-layout workspace overflows."));

    for (int_t local_panel = 0;
         local_panel < lu->symV2PanelCount(); ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        if (panel.isEmpty())
            continue;
        int_t columns = panel.ncols();
        if (columns <= 0 || columns > lu->ldt)
            ABORT("SymFact V2 CPU panel width exceeds row-layout workspace.");
        int_t first = panel.haveDiag() ? 1 : 0;
        for (int_t block = first; block < panel.nblocks(); ++block)
        {
            int_t rows = panel.nbrow(block);
            if (rows <= 0 || rows > lu->ldt)
                ABORT("SymFact V2 CPU panel block exceeds row-layout workspace.");
            int_t *row_ids = panel.rowList(block);
            bool sorted = true;
            for (int_t row = 1; row < rows; ++row)
                sorted = sorted && row_ids[row - 1] < row_ids[row];
            if (sorted)
                continue;

            SymLDLV2CpuCanonicalBlock record;
            record.local_panel = local_panel;
            record.block = block;
            record.permutation_offset =
                lu->symV2CpuCanonicalToOriginalRows.size();

            for (int_t row = 0; row < rows; ++row)
                order[static_cast<size_t>(row)] =
                    std::make_pair(row_ids[row], row);
            std::sort(order.begin(), order.begin() + rows);
            for (int_t row = 1; row < rows; ++row)
                if (order[static_cast<size_t>(row - 1)].first ==
                    order[static_cast<size_t>(row)].first)
                    ABORT("SymFact V2 CPU panel block has duplicate rows.");

            Ftype *values = panel.blkPtr(block);
            int_t lda = panel.LDA();
            for (int_t row = 0; row < rows; ++row)
            {
                int_t source = order[static_cast<size_t>(row)].second;
                lu->symV2CpuCanonicalToOriginalRows.push_back(source);
                sorted_rows[static_cast<size_t>(row)] =
                    order[static_cast<size_t>(row)].first;
                for (int_t column = 0; column < columns; ++column)
                    sorted_values[
                        static_cast<size_t>(row) +
                        static_cast<size_t>(rows) * column] =
                        values[source + static_cast<size_t>(lda) * column];
            }
            std::copy(sorted_rows.begin(), sorted_rows.begin() + rows,
                      row_ids);
            for (int_t column = 0; column < columns; ++column)
                std::copy(
                    sorted_values.begin() +
                        static_cast<size_t>(rows) * column,
                    sorted_values.begin() +
                        static_cast<size_t>(rows) * (column + 1),
                    values + static_cast<size_t>(lda) * column);
            lu->symV2CpuCanonicalBlocks.push_back(record);
        }
    }
    lu->symV2CpuRowsCanonical = true;
}

template <typename Ftype>
static void symldl_v2_cpu_restore_l_panel_rows(xLUstruct_t<Ftype> *lu)
{
    if (!lu->symV2UsesCpuFactor() || !lu->symV2CpuRowsCanonical)
        return;
    if (lu->bigV == NULL || lu->indirectRow == NULL)
        ABORT("SymFact V2 CPU row-layout restore workspace is missing.");

    Ftype *values_scratch = lu->bigV;
    int_t *rows_scratch = lu->indirectRow;
    for (size_t entry = 0; entry < lu->symV2CpuCanonicalBlocks.size();
         ++entry)
    {
        const SymLDLV2CpuCanonicalBlock &record =
            lu->symV2CpuCanonicalBlocks[entry];
        if (record.local_panel < 0 ||
            record.local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU row-layout panel is invalid.");
        xlpanel_t<Ftype> &panel = lu->lPanelVec[record.local_panel];
        if (panel.isEmpty() || record.block < 0 ||
            record.block >= panel.nblocks())
            ABORT("SymFact V2 CPU row-layout block is invalid.");
        int_t rows = panel.nbrow(record.block);
        int_t columns = panel.ncols();
        if (rows <= 0 || rows > lu->ldt || columns <= 0 ||
            columns > lu->ldt ||
            record.permutation_offset >
                lu->symV2CpuCanonicalToOriginalRows.size() ||
            static_cast<size_t>(rows) >
                lu->symV2CpuCanonicalToOriginalRows.size() -
                    record.permutation_offset)
            ABORT("SymFact V2 CPU row-layout restore exceeds workspace.");

        int_t *row_ids = panel.rowList(record.block);
        Ftype *values = panel.blkPtr(record.block);
        int_t lda = panel.LDA();
        for (int_t sorted_row = 0; sorted_row < rows; ++sorted_row)
        {
            int_t original_row = lu->symV2CpuCanonicalToOriginalRows[
                record.permutation_offset +
                static_cast<size_t>(sorted_row)];
            if (original_row < 0 || original_row >= rows)
                ABORT("SymFact V2 CPU row-layout permutation is invalid.");
            rows_scratch[original_row] = row_ids[sorted_row];
            for (int_t column = 0; column < columns; ++column)
                values_scratch[
                    original_row + static_cast<size_t>(rows) * column] =
                    values[sorted_row + static_cast<size_t>(lda) * column];
        }
        std::copy(rows_scratch, rows_scratch + rows, row_ids);
        for (int_t column = 0; column < columns; ++column)
            std::copy(
                values_scratch + static_cast<size_t>(rows) * column,
                values_scratch + static_cast<size_t>(rows) * (column + 1),
                values + static_cast<size_t>(lda) * column);
    }
    lu->symV2CpuRowsCanonical = false;
}
