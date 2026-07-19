#pragma once

#include <cinttypes>
#include <cstdio>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

struct SymLDLV2PlanSignature
{
    uint64_t hash;
    uint64_t items;
};

template <typename T>
static void symldl_v2_plan_signature_mix(
    SymLDLV2PlanSignature *signature, const T &value)
{
    const unsigned char *bytes =
        reinterpret_cast<const unsigned char *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        signature->hash ^= static_cast<uint64_t>(bytes[i]);
        signature->hash *= UINT64_C(1099511628211);
    }
    ++signature->items;
}

template <typename T>
static void symldl_v2_plan_signature_mix_vector(
    SymLDLV2PlanSignature *signature, const std::vector<T> &values)
{
    uint64_t size = static_cast<uint64_t>(values.size());
    symldl_v2_plan_signature_mix(signature, size);
    for (size_t i = 0; i < values.size(); ++i)
        symldl_v2_plan_signature_mix(signature, values[i]);
}

template <typename T>
static void symldl_v2_plan_signature_mix_nested_vector(
    SymLDLV2PlanSignature *signature,
    const std::vector<std::vector<T> > &values)
{
    uint64_t size = static_cast<uint64_t>(values.size());
    symldl_v2_plan_signature_mix(signature, size);
    for (size_t i = 0; i < values.size(); ++i)
        symldl_v2_plan_signature_mix_vector(signature, values[i]);
}

template <typename Ftype>
static void symldl_v2_print_plan_signature_component(
    xLUstruct_t<Ftype> *lu, const char *component,
    const SymLDLV2PlanSignature &signature)
{
    uint64_t hash_xor = 0;
    uint64_t hash_sum = 0;
    uint64_t item_sum = 0;
    MPI_Allreduce(&signature.hash, &hash_xor, 1, MPI_UINT64_T,
                  MPI_BXOR, lu->grid3d->comm);
    MPI_Allreduce(&signature.hash, &hash_sum, 1, MPI_UINT64_T,
                  MPI_SUM, lu->grid3d->comm);
    MPI_Allreduce(&signature.items, &item_sum, 1, MPI_UINT64_T,
                  MPI_SUM, lu->grid3d->comm);
    if (lu->grid3d->iam != 0)
        return;
    if (component == NULL)
        std::printf(
            "SymFact V2 logical plan signature: backend=%s xor=%016" PRIx64
            " sum=%016" PRIx64 " items=%" PRIu64 "\n",
            lu->symV2FactorBackendName(), hash_xor, hash_sum, item_sum);
    else
        std::printf(
            "SymFact V2 logical plan component: backend=%s name=%s"
            " xor=%016" PRIx64 " sum=%016" PRIx64 " items=%" PRIu64
            "\n",
            lu->symV2FactorBackendName(), component, hash_xor, hash_sum,
            item_sum);
}

template <typename Ftype, typename T>
static void symldl_v2_print_vector_plan_signature(
    xLUstruct_t<Ftype> *lu, const char *component,
    const std::vector<T> &values)
{
    SymLDLV2PlanSignature signature = {
        UINT64_C(1469598103934665603), 0
    };
    symldl_v2_plan_signature_mix_vector(&signature, values);
    symldl_v2_print_plan_signature_component(lu, component, signature);
}

template <typename Ftype, typename T>
static void symldl_v2_print_nested_vector_plan_signature(
    xLUstruct_t<Ftype> *lu, const char *component,
    const std::vector<std::vector<T> > &values)
{
    SymLDLV2PlanSignature signature = {
        UINT64_C(1469598103934665603), 0
    };
    symldl_v2_plan_signature_mix_nested_vector(&signature, values);
    symldl_v2_print_plan_signature_component(lu, component, signature);
}

template <typename Ftype>
static void symldl_v2_print_logical_plan_signature(xLUstruct_t<Ftype> *lu)
{
    if (!superlu_sym_v2_route_profile())
        return;

    SymLDLV2PlanSignature signature = {
        UINT64_C(1469598103934665603), 0
    };
    symldl_v2_plan_signature_mix(&signature, lu->nsupers);
    symldl_v2_plan_signature_mix(&signature, lu->Pr);
    symldl_v2_plan_signature_mix(&signature, lu->Pc);
    int pz = lu->grid3d->zscp.Np;
    symldl_v2_plan_signature_mix(&signature, pz);
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        int_t diag_root = lu->symV2DiagRoot(k);
        int_t panel_root = lu->symV2PanelRoot(k);
        symldl_v2_plan_signature_mix(&signature, diag_root);
        symldl_v2_plan_signature_mix(&signature, panel_root);
    }
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2UsePcFragmentSchur);
    symldl_v2_plan_signature_mix_nested_vector(
        &signature, lu->symL2LSendMeta);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2PartnerLSendSizes);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2PartnerLSendRowActive);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2PartnerLSendAnyActive);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2PartnerLRecvSizes);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2PartnerLRecvActive);
    symldl_v2_plan_signature_mix_nested_vector(
        &signature, lu->symV2PartnerLRecvIndexBySrc);
    symldl_v2_plan_signature_mix_vector(
        &signature, lu->symV2RowFragRecvSizes);
    symldl_v2_plan_signature_mix_nested_vector(
        &signature, lu->symV2RowFragRecvIndex);

    symldl_v2_print_plan_signature_component(lu, NULL, signature);
    if (!superlu_sym_v2_plan_signature_components())
        return;

    SymLDLV2PlanSignature component = {
        UINT64_C(1469598103934665603), 0
    };
    symldl_v2_plan_signature_mix(&component, lu->nsupers);
    symldl_v2_plan_signature_mix(&component, lu->Pr);
    symldl_v2_plan_signature_mix(&component, lu->Pc);
    symldl_v2_plan_signature_mix(&component, pz);
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        int_t diag_root = lu->symV2DiagRoot(k);
        int_t panel_root = lu->symV2PanelRoot(k);
        symldl_v2_plan_signature_mix(&component, diag_root);
        symldl_v2_plan_signature_mix(&component, panel_root);
    }
    symldl_v2_print_plan_signature_component(lu, "ownership", component);

    symldl_v2_print_vector_plan_signature(
        lu,
        "pcfrag-mode", lu->symV2UsePcFragmentSchur);
    symldl_v2_print_nested_vector_plan_signature(
        lu,
        "partner-send-meta", lu->symL2LSendMeta);
    symldl_v2_print_vector_plan_signature(
        lu,
        "partner-send-sizes", lu->symV2PartnerLSendSizes);
    symldl_v2_print_vector_plan_signature(
        lu,
        "partner-send-active", lu->symV2PartnerLSendRowActive);
    symldl_v2_print_vector_plan_signature(
        lu,
        "partner-send-any-active", lu->symV2PartnerLSendAnyActive);
    symldl_v2_print_vector_plan_signature(
        lu,
        "partner-recv-sizes", lu->symV2PartnerLRecvSizes);
    symldl_v2_print_vector_plan_signature(
        lu,
        "partner-recv-active", lu->symV2PartnerLRecvActive);
    symldl_v2_print_nested_vector_plan_signature(
        lu,
        "partner-recv-index", lu->symV2PartnerLRecvIndexBySrc);
    symldl_v2_print_vector_plan_signature(
        lu,
        "row-recv-sizes", lu->symV2RowFragRecvSizes);
    symldl_v2_print_nested_vector_plan_signature(
        lu,
        "row-recv-index", lu->symV2RowFragRecvIndex);
}
