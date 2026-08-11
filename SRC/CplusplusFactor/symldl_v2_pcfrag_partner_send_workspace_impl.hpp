#pragma once

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_build_partner_l_send_maps(xLUstruct_t<Ftype> *lu)
{
    const bool gpu_pc_fragment =
        lu->symV2UsesGpuFactor() &&
        symldl_v2_use_gpu_pc_fragment_schur(lu->grid3d);
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload ||
        (lu->Pr <= 1 && !gpu_pc_fragment))
        return;

    int_t local_cols = lu->symV2PanelCount();
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(local_cols), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L map table overflows.");
    if (lu->symL2LSendMeta.size() != l2l_slots ||
        lu->symV2PartnerLMapOffsets.size() != l2l_slots ||
        lu->symV2PartnerLSendSizes.size() != l2l_slots)
        ABORT("SymFact V2 partner-L send tables are not initialized.");

    std::vector<size_t> map_counts(l2l_slots, 0);
    std::vector<size_t> meta_counts(l2l_slots, 0);
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[lk];
        if (lpanel.isEmpty())
            continue;
        int_t jb = lu->symV2PanelGid(lk);
        int_t knsupc = lu->supersize(jb);
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
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(ikcol);
            size_t map_add = symldl_v2_checked_product(
                static_cast<size_t>(len), static_cast<size_t>(knsupc),
                "SymFact V2 partner-L send map overflows.");
            if (map_counts[flat] >
                std::numeric_limits<size_t>::max() - map_add)
                ABORT("SymFact V2 partner-L send map overflows.");
            map_counts[flat] += map_add;
            size_t meta_add = static_cast<size_t>(len) + 2;
            if (meta_counts[flat] >
                std::numeric_limits<size_t>::max() - meta_add)
                ABORT("SymFact V2 partner-L metadata overflows.");
            meta_counts[flat] += meta_add;
        }
    }

    size_t total_partner_send = 0;
    size_t max_panel_scratch = 0;
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        size_t panel_scratch = 0;
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            lu->symV2PartnerLHostSendScratchOffsets[flat] = panel_scratch;
            if (panel_scratch >
                std::numeric_limits<size_t>::max() - map_counts[flat])
                ABORT("SymFact V2 partner-L send staging overflows.");
            panel_scratch += map_counts[flat];
        }
        max_panel_scratch = SUPERLU_MAX(max_panel_scratch, panel_scratch);
    }
    if (max_panel_scratch >
        static_cast<size_t>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 partner-L send staging exceeds int_t range.");
    lu->maxSymPartnerLSendStageCount =
        static_cast<int_t>(max_panel_scratch);

    for (size_t flat = 0; flat < l2l_slots; ++flat)
    {
        if (map_counts[flat] >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 partner-L send map exceeds MPI limit.");
        if (meta_counts[flat] >
            static_cast<size_t>(std::numeric_limits<int_t>::max()))
            ABORT("SymFact V2 partner-L metadata is too large.");
        lu->symV2PartnerLMapOffsets[flat] = total_partner_send;
        total_partner_send += map_counts[flat];
        if (total_partner_send < lu->symV2PartnerLMapOffsets[flat])
            ABORT("SymFact V2 partner-L send map overflows.");
        lu->symV2PartnerLSendSizes[flat] =
            static_cast<int>(map_counts[flat]);
        if (meta_counts[flat] > 0)
            lu->symL2LSendMeta[flat].resize(meta_counts[flat]);
        if (map_counts[flat] > 0 &&
            !superlu_sym_v2_pinned_staging())
            lu->symV2PartnerLHostSendBufs[flat].resize(map_counts[flat]);
    }
    lu->symV2PartnerLPackedMaps.assign(total_partner_send, 0);
    lu->symL2LSendMapPoolCount = total_partner_send;
    lu->symV2PartnerLSendBufPoolCount =
        superlu_sym_v2_pc_fragment_ldl_native() ? 0 : total_partner_send;
    if (superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool() &&
        !superlu_cuda_aware_mpi() && max_panel_scratch > 0)
        lu->symV2PartnerLHostSendPoolPinnedCount = max_panel_scratch;

    std::vector<size_t> map_write_offsets = lu->symV2PartnerLMapOffsets;
    std::vector<size_t> meta_write_offsets(l2l_slots, 0);
    std::vector<std::pair<int_t, int_t> > row_order;
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[lk];
        if (lpanel.isEmpty())
            continue;
        int_t jb = lu->symV2PanelGid(lk);
        int_t knsupc = lu->supersize(jb);
        int_t nsupr = lpanel.LDA();
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
        dLocalLU_t *Llu = lu->host_Llu;
        int_t *raw_lsub = (Llu != NULL && Llu->Lrowind_bc_ptr != NULL)
                              ? Llu->Lrowind_bc_ptr[lk]
                              : NULL;
        int_t *raw_lloc = (Llu != NULL && Llu->Lindval_loc_bc_ptr != NULL)
                              ? Llu->Lindval_loc_bc_ptr[lk]
                              : NULL;
        if (raw_lsub == NULL || raw_lloc == NULL)
            ABORT("SymFact V2 partner-L send maps require local L metadata.");
        int_t raw_nb = 0;
        int_t raw_idx_i = 0;
        int_t raw_idx_v = 0;
        if (lu->myrow == lu->symV2DiagRoot(jb))
        {
            raw_nb = raw_lsub[0] - 1;
            raw_idx_i = raw_nb + 2;
            raw_idx_v = 2 * raw_nb + 3;
        }
        else
        {
            raw_nb = raw_lsub[0];
            raw_idx_i = raw_nb;
            raw_idx_v = 2 * raw_nb;
        }
        if (raw_nb < 0 || raw_lsub[1] != nsupr)
            ABORT("SymFact V2 partner-L local L metadata is inconsistent.");
        for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = static_cast<int>(lu->symV2PanelRoot(ik));
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;
            int_t *row_ids = lpanel.rowList(lb);
            bool rows_sorted = true;
            for (int_t i = 1; i < len; ++i)
            {
                if (row_ids[i - 1] > row_ids[i])
                {
                    rows_sorted = false;
                    break;
                }
            }
            if (!rows_sorted)
            {
                row_order.clear();
                row_order.reserve(static_cast<size_t>(len));
                for (int_t i = 0; i < len; ++i)
                    row_order.push_back(std::make_pair(row_ids[i], i));
                std::sort(row_order.begin(), row_order.end());
            }

            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(ikcol);
            std::vector<int_t> &meta = lu->symL2LSendMeta[flat];
            size_t meta_pos = meta_write_offsets[flat];
            if (meta_pos + static_cast<size_t>(len) + 2 > meta.size())
                ABORT("SymFact V2 partner-L metadata overrun.");
            meta[meta_pos++] = ik;
            meta[meta_pos++] = len;
            for (int_t i = 0; i < len; ++i)
                meta[meta_pos++] =
                    rows_sorted ? row_ids[i] : row_order[i].first;
            meta_write_offsets[flat] = meta_pos;

            size_t map_pos = map_write_offsets[flat];
            size_t map_end = lu->symV2PartnerLMapOffsets[flat] +
                             map_counts[flat];
            int_t raw_lb = lb - first_block;
            if (raw_lb < 0 || raw_lb >= raw_nb)
                ABORT("SymFact V2 partner-L local L block is invalid.");
            int_t raw_lptr = raw_lloc[raw_lb + raw_idx_i];
            int_t raw_vptr = raw_lloc[raw_lb + raw_idx_v];
            if (raw_lsub[raw_lptr] != ik ||
                raw_lsub[raw_lptr + 1] != len)
                ABORT("SymFact V2 partner-L local L block metadata mismatch.");
            for (int_t col = 0; col < knsupc; ++col)
            {
                for (int_t i = 0; i < len; ++i)
                {
                    if (map_pos >= map_end)
                        ABORT("SymFact V2 partner-L send map overrun.");
                    int_t src_row = rows_sorted ? i : row_order[i].second;
                    lu->symV2PartnerLPackedMaps[map_pos++] =
                        raw_vptr + src_row + col * nsupr;
                }
            }
            map_write_offsets[flat] = map_pos;

            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos =
                    flat * static_cast<size_t>(lu->Pr) +
                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner-L send mask is invalid.");
                lu->symV2PartnerLSendRowActive[active_pos] = 1;
            }
        }
    }

    for (size_t flat = 0; flat < l2l_slots; ++flat)
    {
        if (map_write_offsets[flat] !=
            lu->symV2PartnerLMapOffsets[flat] + map_counts[flat])
            ABORT("SymFact V2 partner-L send map size mismatch.");
        if (meta_write_offsets[flat] != meta_counts[flat])
            ABORT("SymFact V2 partner-L metadata size mismatch.");
    }
}

#endif
