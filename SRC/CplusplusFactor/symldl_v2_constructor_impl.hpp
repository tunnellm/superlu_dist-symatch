#pragma once

#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_workspace_impl.hpp"
#include "symldl_v2_cpu_plan_impl.hpp"
#include "symldl_v2_plan_signature_impl.hpp"

// Keep SymLDL V2 constructor orchestration out of the legacy LU panel setup.

template <typename Ftype>
static void symldl_v2_constructor_build_l_panels(
    xLUstruct_t<Ftype> *lu,
    LUStruct_type<Ftype> *LUstruct,
    std::vector<int_t> &localLvalSendCounts,
    std::vector<int_t> &localLidxSendCounts)
{
    if (!lu->useSymV2Solve())
        return;

    symldl_v2_build_l_panels(lu, LUstruct,
                             localLvalSendCounts,
                             localLidxSendCounts);
}

template <typename Ftype>
static void symldl_v2_constructor_exchange_l_counts(
    xLUstruct_t<Ftype> *lu,
    const std::vector<int_t> &localLvalSendCounts,
    const std::vector<int_t> &localLidxSendCounts)
{
    if (!lu->useSymV2Solve())
        return;

    symldl_v2_exchange_l_panel_counts(lu,
                                      localLvalSendCounts,
                                      localLidxSendCounts);
}

template <typename Ftype>
static void symldl_v2_constructor_setup_factor_workspace(
    xLUstruct_t<Ftype> *lu,
    LUStruct_type<Ftype> *LUstruct)
{
    symldl_v2_compute_pcfrag_scratch(lu, LUstruct);
    if (lu->useSymV2Solve())
    {
        symldl_v2_allocate_factor_workspace(lu);
        symldl_v2_allocate_cpu_factor_workspace(lu);
    }
}

template <typename Ftype>
static void symldl_v2_constructor_setup_fragment_metadata(
    xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve())
        return;
    if (superlu_sym_v2_scoped_fragment_metadata_verify() &&
        !superlu_sym_v2_scoped_fragment_metadata())
        ABORT("SYMLDL_V2_SCOPED_FRAGMENT_METADATA_VERIFY requires SYMLDL_V2_SCOPED_FRAGMENT_METADATA=1.");

    symldl_v2_initialize_pcfrag_tables(lu);
    symldl_v2_build_cpu_fragment_plan(lu);
#ifdef HAVE_CUDA
    symldl_v2_build_partner_l_send_maps(lu);
    if (lu->superlu_acc_offload && lu->Pr > 1)
    {
        if (superlu_sym_v2_scoped_fragment_metadata())
        {
            SymLDLV2ScopedPartnerMetaPayload metadata =
                symldl_v2_collect_scoped_partner_l_metadata(lu);
            symldl_v2_build_partner_l_recv_maps(
                lu, metadata.target_column);
            symldl_v2_build_row_down_maps(
                lu, metadata.source_row, metadata.target_column);
        }
        else
        {
            SymLDLV2PartnerMetaPayload metadata =
                symldl_v2_collect_partner_l_metadata(lu);
            symldl_v2_build_partner_l_recv_maps(lu, metadata);
            symldl_v2_build_row_down_maps(lu, metadata, metadata);
        }
    }
#endif
    symldl_v2_print_logical_plan_signature(lu);
    symldl_v2_allocate_fragment_host_buffers(lu);
}

template <typename Ftype>
static void symldl_v2_constructor_materialize_gpu_metadata(
    xLUstruct_t<Ftype> *lu)
{
#ifdef HAVE_CUDA
    if (lu->useSymV2Solve())
        symldl_v2_materialize_pcfrag_metadata(lu);
#else
    (void) lu;
#endif
}
