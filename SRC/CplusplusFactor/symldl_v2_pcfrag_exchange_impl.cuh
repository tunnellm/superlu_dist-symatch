#pragma once

#include <algorithm>
#include <climits>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "gpu_mpi_utils.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_l_fragment_map_impl.cuh"

#ifdef HAVE_CUDA


#include "symldl_v2_fragment_exchange_common.cuh"
#include "symldl_v2_fragment_prepack_impl.cuh"
#include "symldl_v2_l_fragment_exchange_impl.cuh"
#include "symldl_v2_pcfrag_partner_exchange_impl.cuh"
#include "symldl_v2_pcfrag_row_exchange_impl.cuh"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment exchange is not implemented.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2LFragmentExchangeGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("SymFact V2 requires GPU offload.");
    if (Pr <= 1)
        return 0;
    if (k < 0 || k >= nsupers)
        return 0;
    if (!symV2UsePcFragmentSchurPanel(k))
    {
        symV2RouteProfileNoteLFragmentExchange();
        return symldl_v2_l_fragment_exchange(this, k, stream_offset);
    }
    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    symV2RouteProfileNotePcFragmentExchange();

    const bool cuda_aware = superlu_cuda_aware_mpi();
    const bool async_factor = superlu_sym_v2_async_factor();
    const bool pcfrag_async_exchange =
        async_factor &&
        superlu_sym_v2_pcfrag_async_prereqs_enabled() &&
        (superlu_sym_v2_pcfrag_async_exchange() ||
         superlu_sym_v2_pcfrag_async_pipeline());
    if (cuda_aware)
        ABORT("SymFact V2 Pc-fragment exchange is fail-closed for CUDA-aware MPI.");
    if (!superlu_sym_v2_pc_fragment_ldl_native())
        ABORT("SymFact V2 Pc-fragment exchange requires LDL-native row movement.");
    if (!superlu_sym_v2_row_l_plan_v2() ||
        !superlu_sym_v2_row_l_plan_v2_exchange() ||
        superlu_sym_v2_row_l_plan_v2_dryrun() ||
        !superlu_sym_v2_row_l_plan_v2_aggregate_dest() ||
        !superlu_sym_v2_row_l_direct_recv() ||
        !superlu_sym_v2_row_l_compressed_plan() ||
        !superlu_sym_v2_row_l_lazy_sendmap() ||
        !superlu_sym_v2_row_l_pack_all_dest() ||
        !superlu_sym_v2_row_l_separate_send_staging())
        ABORT("SymFact V2 Pc-fragment exchange requires lazy LDL-native row-down planning.");
    if (async_factor && !pcfrag_async_exchange)
        ABORT("SymFact V2 Pc-fragment async factor requires a Pc-fragment async exchange mode.");
    if (grid3d == NULL || grid3d->rscp.comm == MPI_COMM_NULL)
        ABORT("SymFact V2 row-down communicator is missing.");

    cudaStream_t stream = async_factor
                              ? A_gpu.lookAheadUStream[stream_offset]
                              : A_gpu.cuStreams[stream_offset];
    int_t kcol = symV2PanelRoot(k);
    int_t ksupc = SuperSize(k);
    int_t lk = symV2PanelIndex(k);
    if (kcol < 0 || kcol >= Pc || ksupc <= 0)
        ABORT("SymFact V2 Pc-fragment panel metadata is invalid.");

    if (symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symL2LSendMapsGPU.empty() ||
        symV2PartnerLRecvSizes.empty() ||
        symV2PartnerLRecvIndex.empty() ||
        symV2PartnerLRecvMap.empty() ||
        symV2PartnerLRecvMapsGPU.empty() ||
        symV2RowFragRecvSizes.empty() ||
        symV2RowFragRecvIndex.empty() ||
        symV2RowDownSendSizes.empty() ||
        symV2RowDownSendSegsGPU.empty())
        ABORT("SymFact V2 Pc-fragment exchange buffers are not allocated.");

    std::vector<MPI_Request> &send_reqs = symV2ExchangeSendReqsScratch;
    SymLDLV2PartnerExchangeResult partner_exchange =
        symldl_v2_pcfrag_exchange_partner_l(
            this, k, stream_offset, stream, async_factor, kcol, ksupc, lk,
            send_reqs);

    SymLDLV2RowFragmentExchangeResult row_exchange =
        symldl_v2_pcfrag_exchange_row_fragments(
            this, k, stream_offset, stream, kcol, ksupc, lk, send_reqs);

    symldl_v2_trace_pcfrag_exchange(
        grid3d, k, stream_offset, partner_exchange.partner_nrows,
        partner_exchange.partner_recv_total,
        row_exchange.row_nrows, row_exchange.row_recv_total,
        row_exchange.row_send_total);

    if (!send_reqs.empty())
    {
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
        send_reqs.clear();
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    return 0;
}

#include "symldl_v2_panel_bcast_impl.cuh"

#endif
