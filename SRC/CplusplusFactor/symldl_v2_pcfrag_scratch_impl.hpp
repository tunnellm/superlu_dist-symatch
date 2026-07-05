#pragma once

#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

template <typename Ftype>
static void symldl_v2_compute_pcfrag_scratch(
    xLUstruct_t<Ftype> *lu,
    LUStruct_type<Ftype> *)
{
    if (!lu->useSymV2Solve())
    {
        lu->maxSymPartnerLvalCount = lu->maxLvalCount;
        lu->maxSymPartnerLidxCount = lu->maxLidxCount;
        lu->maxSymPartnerLSendStageCount = 0;
        return;
    }

    lu->maxSymPartnerLvalCount = 0;
    lu->maxSymPartnerLidxCount = 0;
    lu->maxSymPartnerLSendStageCount = 0;
    lu->maxSymV2RowFragStageCount = 0;
    lu->maxSymV2RowFragValRecvCount = 0;
    lu->maxSymV2RowFragIdxRecvCount = 0;
    lu->maxSymV2RowFragValSendCount = 0;

    if (lu->Pr <= 1)
    {
        lu->maxSymPartnerLvalCount = lu->maxLvalCount;
        lu->maxSymPartnerLidxCount = lu->maxLidxCount;
        return;
    }

    size_t partner_count_size = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L count table overflows.");
    size_t row_source_count_size = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 row-fragment count table overflows.");
    if (partner_count_size > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        row_source_count_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 fragment count table exceeds MPI limit.");

    std::vector<long long> local_partner_val(partner_count_size, 0);
    std::vector<long long> global_partner_val(partner_count_size, 0);
    std::vector<long long> local_partner_meta(partner_count_size, 0);
    std::vector<long long> global_partner_meta(partner_count_size, 0);
    std::vector<long long> local_row_val(row_source_count_size, 0);
    std::vector<long long> global_row_val(row_source_count_size, 0);
    std::vector<long long> local_row_meta(row_source_count_size, 0);
    std::vector<long long> global_row_meta(row_source_count_size, 0);
    long long local_row_send_val = 0;
    long long max_row_send_val = 0;

    for (int_t li = 0; li < lu->symV2PanelCount(); ++li)
    {
        int_t k0 = lu->symV2PanelGid(li);
        if (k0 < 0 || k0 >= lu->nsupers ||
            lu->isNodeInMyGrid == NULL || lu->isNodeInMyGrid[k0] != 1)
            continue;

        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[li];
        if (lpanel.isEmpty())
            continue;

        int_t knsupc = lu->supersize(k0);
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
        for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = static_cast<int>(lu->symV2PanelRoot(ik));
            if (ikcol < 0 || ikcol >= lu->Pc)
                ABORT("SymFact V2 partner-L target process column is invalid.");
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;

            long long block_values =
                static_cast<long long>(len) *
                static_cast<long long>(knsupc);
            size_t col_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pc) +
                             static_cast<size_t>(ikcol);
            size_t row_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(lu->myrow);
            local_partner_val[col_pos] += block_values;
            local_row_val[row_pos] += block_values;
            local_partner_meta[col_pos] += static_cast<long long>(len) + 2;
            local_row_meta[row_pos] += static_cast<long long>(len) + 2;
            local_row_send_val =
                SUPERLU_MAX(local_row_send_val, block_values);
        }
    }

    MPI_Allreduce(local_partner_val.data(), global_partner_val.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_partner_meta.data(), global_partner_meta.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_row_val.data(), global_row_val.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_row_meta.data(), global_row_meta.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(&local_row_send_val, &max_row_send_val, 1,
                  MPI_LONG_LONG, MPI_MAX, lu->grid->comm);

    long long max_partner_val = 0;
    long long max_partner_meta = 0;
    long long max_row_val = 0;
    long long max_row_meta = 0;
    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t pos = static_cast<size_t>(k0) *
                             static_cast<size_t>(lu->Pc) +
                         static_cast<size_t>(pc);
            max_partner_val =
                SUPERLU_MAX(max_partner_val, global_partner_val[pos]);
            max_partner_meta =
                SUPERLU_MAX(max_partner_meta, global_partner_meta[pos]);
        }
        if (superlu_sym_v2_pc_fragment_schur() &&
            superlu_sym_v2_pc_fragment_ldl_native() && lu->Pc > 1)
        {
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(pr);
                max_row_val =
                    SUPERLU_MAX(max_row_val, global_row_val[pos]);
                max_row_meta =
                    SUPERLU_MAX(max_row_meta, global_row_meta[pos]);
            }
        }
    }

    long long max_partner_idx =
        (max_partner_meta > 0) ? max_partner_meta + LPANEL_HEADER_SIZE + 1 : 0;
    long long max_row_idx =
        (max_row_meta > 0) ? max_row_meta + LPANEL_HEADER_SIZE + 1 : 0;
    if (superlu_sym_v2_pc_fragment_schur() &&
        superlu_sym_v2_pc_fragment_ldl_native() && lu->Pc > 1)
    {
        long long row_send_multiplier = static_cast<long long>(lu->Pc - 1);
        if (max_row_val > 0 &&
            row_send_multiplier >
                std::numeric_limits<long long>::max() / max_row_val)
            ABORT("SymFact V2 row-fragment send size overflows.");
        max_row_send_val =
            SUPERLU_MAX(max_row_send_val, max_row_val * row_send_multiplier);
    }

    if (max_partner_val > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_partner_idx > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_val > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_idx > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_send_val > static_cast<long long>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 fragment staging size exceeds int_t range.");

    lu->maxSymPartnerLvalCount = static_cast<int_t>(max_partner_val);
    lu->maxSymPartnerLidxCount = static_cast<int_t>(max_partner_idx);
    lu->maxSymPartnerLSendStageCount = lu->maxLvalCount;
    lu->maxSymV2RowFragStageCount = static_cast<int_t>(max_row_val);
    lu->maxSymV2RowFragValRecvCount = static_cast<int_t>(max_row_val);
    lu->maxSymV2RowFragIdxRecvCount = static_cast<int_t>(max_row_idx);
    lu->maxSymV2RowFragValSendCount = static_cast<int_t>(max_row_send_val);
}
