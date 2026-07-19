#pragma once

#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

template <typename Ftype>
static void symldl_v2_initialize_pcfrag_plan_tables(
    xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve())
        return;

    const bool pc_fragment =
        symldl_v2_use_pc_fragment_schur(lu->grid3d) ||
        (lu->symV2UsesCpuFactor() && (lu->Pr > 1 || lu->Pc > 1));
    int_t local_cols = lu->symV2PanelCount();
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(local_cols), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L table size overflows.");
    lu->symL2LSendMeta.assign(l2l_slots, std::vector<int_t>());
    lu->symV2PartnerLSendSizes.assign(l2l_slots, 0);
    size_t partner_active = symldl_v2_checked_product(
        l2l_slots, static_cast<size_t>(lu->Pr),
        "SymFact V2 partner-L active table overflows.");
    lu->symV2PartnerLSendRowActive.assign(partner_active, 0);

    size_t partner_recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner receive table overflows.");
    lu->symV2PartnerLRecvSizes.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvActive.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                      std::vector<int_t>());
    lu->symV2PartnerLRecvIndexBySrc.assign(partner_recv_slots,
                                           std::vector<int_t>());
    lu->symV2UsePcFragmentSchur.assign(
        static_cast<size_t>(lu->nsupers), pc_fragment ? 1 : 0);
    if (pc_fragment)
    {
        size_t row_recv_slots = symldl_v2_checked_product(
            static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
            "SymFact V2 row receive table overflows.");
        lu->symV2RowFragRecvSizes.assign(row_recv_slots, 0);
        lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                         std::vector<int_t>());

        lu->symV2RowDownSendSizes.assign(l2l_slots, 0);
        lu->symV2RowDownSegOffsets.assign(l2l_slots + 1, 0);
        lu->symV2RowDownRecvSizes.assign(row_recv_slots, 0);
        lu->symV2RowDownPlanReady.assign(static_cast<size_t>(lu->nsupers), 0);
    }
    else
    {
        lu->symV2RowFragRecvSizes.clear();
        lu->symV2RowFragRecvIndex.clear();

        lu->symV2RowDownSendSizes.clear();
        lu->symV2RowDownSegOffsets.clear();
        lu->symV2RowDownRecvSizes.clear();
        lu->symV2RowDownPlanReady.clear();
    }
    lu->symV2RowDownSegs.clear();
}

template <typename Ftype>
static void symldl_v2_initialize_pcfrag_tables(xLUstruct_t<Ftype> *lu)
{
    symldl_v2_initialize_pcfrag_plan_tables(lu);

#ifdef HAVE_CUDA
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload)
        return;

    const bool pc_fragment = symldl_v2_use_pc_fragment_schur(lu->grid3d);
    int_t local_cols = lu->symV2PanelCount();
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(local_cols), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L table size overflows.");
    lu->symV2PartnerLSendBufsGPU.assign(l2l_slots, NULL);
    lu->symL2LSendMapsGPU.assign(l2l_slots, NULL);
    lu->symV2PartnerLHostSendBufs.assign(l2l_slots,
                                         std::vector<Ftype>());
    lu->symV2PartnerLHostSendBufsPinned.assign(l2l_slots, NULL);
    lu->symV2PartnerLHostSendScratchOffsets.assign(l2l_slots, 0);
    lu->symV2PartnerLMapOffsets.assign(l2l_slots, 0);
    lu->symV2PartnerLPrepacked.assign(static_cast<size_t>(local_cols), 0);
    lu->symPanelReadyEventIds.assign(static_cast<size_t>(lu->nsupers), -1);
    lu->symV2UsePcFragmentSchur.assign(
        static_cast<size_t>(lu->nsupers), pc_fragment ? 1 : 0);

    if (pc_fragment)
    {
        size_t row_active = symldl_v2_checked_product(
            l2l_slots, static_cast<size_t>(lu->Pc),
            "SymFact V2 row-fragment active table overflows.");
        lu->symV2RowFragSendActive.assign(row_active, 0);
    }
    else
    {
        lu->symV2RowFragSendActive.clear();
    }

    size_t partner_recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner receive table overflows.");
    lu->symV2PartnerLRecvMap.assign(partner_recv_slots,
                                    std::vector<int_t>());
    lu->symV2PartnerLRecvMapOffsets.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvMapsGPU.assign(partner_recv_slots, NULL);
    if (pc_fragment)
    {
        size_t row_recv_slots = symldl_v2_checked_product(
            static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
            "SymFact V2 row receive table overflows.");
        lu->symV2RowFragRecvMap.assign(row_recv_slots,
                                      std::vector<int_t>());
        lu->symV2RowFragRecvMapOffsets.assign(row_recv_slots, 0);
        lu->symV2RowFragRecvMapsGPU.assign(row_recv_slots, NULL);

        lu->symV2RowDownSendSegOffsets.assign(l2l_slots, 0);
        lu->symV2RowDownSendSegCounts.assign(l2l_slots, 0);
        lu->symV2RowDownSendSegsGPU.assign(l2l_slots, NULL);
    }
    else
    {
        lu->symV2RowFragRecvMap.clear();
        lu->symV2RowFragRecvMapOffsets.clear();
        lu->symV2RowFragRecvMapsGPU.clear();

        lu->symV2RowDownSendSegOffsets.clear();
        lu->symV2RowDownSendSegCounts.clear();
        lu->symV2RowDownSendSegsGPU.clear();
    }

    lu->symV2ExchangeSendSizesScratch.assign(static_cast<size_t>(lu->Pc), 0);
    lu->symV2ExchangeRecvSizesScratch.assign(static_cast<size_t>(lu->Pr), 0);
    lu->symV2ExchangeRecvOffsetsScratch.assign(static_cast<size_t>(lu->Pr), -1);
    lu->symV2ExchangeRecvReqsScratch.clear();
    lu->symV2ExchangeRecvReqsScratch.reserve(static_cast<size_t>(lu->Pr));
    lu->symV2ExchangeSendReqsScratch.clear();
    lu->symV2ExchangeSendReqsScratch.reserve(
        symldl_v2_checked_product(static_cast<size_t>(lu->Pr),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 exchange request table overflows."));
    lu->symV2ExchangeRecvPeersScratch.clear();
    lu->symV2ExchangeRecvPeersScratch.reserve(static_cast<size_t>(lu->Pr));
    lu->symV2ExchangeWaitIndicesScratch.assign(static_cast<size_t>(lu->Pr), 0);
    lu->symV2ExchangeWaitStatusesScratch.resize(static_cast<size_t>(lu->Pr));
    if (pc_fragment)
    {
        lu->symV2RowFragSendCountsScratch.assign(static_cast<size_t>(lu->Pc), 0);
        lu->symV2RowFragSendOffsetsScratch.assign(static_cast<size_t>(lu->Pc), 0);
    }
    else
    {
        lu->symV2RowFragSendCountsScratch.clear();
        lu->symV2RowFragSendOffsetsScratch.clear();
    }
    lu->symV2RowFragSendReqsScratch.clear();
#else
    (void) lu;
#endif
}
