#pragma once

#include <cstdio>
#include <cstring>
#include <limits>

static inline const char *symldl_v2_route_profile_label(
    SymV2RouteProfileCounter counter)
{
    switch (counter)
    {
    case SYM_V2_ROUTE_PANEL_BCAST:
        return "panel_bcast";
    case SYM_V2_ROUTE_PCFRAG_PANEL_BCAST:
        return "pcfrag_panels";
    case SYM_V2_ROUTE_LFRAG_EXCHANGE:
        return "lfrag_exchange";
    case SYM_V2_ROUTE_PCFRAG_EXCHANGE:
        return "pcfrag_exchange";
    case SYM_V2_ROUTE_PARTNER_L_EXCHANGE:
        return "partner_l_exchange";
    case SYM_V2_ROUTE_ROWFRAG_EXCHANGE:
        return "rowfrag_exchange";
    case SYM_V2_ROUTE_DUAL_FRAGMENT_LOOKAHEAD:
        return "dual_fragment_lookahead";
    case SYM_V2_ROUTE_DUAL_FRAGMENT_EXCLUDE:
        return "dual_fragment_exclude";
    default:
        return "unknown";
    }
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileReset()
{
    std::memset(&symV2RouteProfile, 0, sizeof(symV2RouteProfile));
    symV2CommunicationProfile = SymV2CommunicationProfile();
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNote(
    SymV2RouteProfileCounter counter)
{
    if (counter >= 0 && counter < SYM_V2_ROUTE_PROFILE_COUNTERS)
        ++symV2RouteProfile.counters[counter];
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNotePanelBcast(
    bool pc_fragment)
{
    symV2RouteProfileNote(SYM_V2_ROUTE_PANEL_BCAST);
    if (pc_fragment)
        symV2RouteProfileNote(SYM_V2_ROUTE_PCFRAG_PANEL_BCAST);
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNoteLFragmentExchange()
{
    symV2RouteProfileNote(SYM_V2_ROUTE_LFRAG_EXCHANGE);
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNotePcFragmentExchange()
{
    symV2RouteProfileNote(SYM_V2_ROUTE_PCFRAG_EXCHANGE);
    symV2RouteProfileNote(SYM_V2_ROUTE_PARTNER_L_EXCHANGE);
    symV2RouteProfileNote(SYM_V2_ROUTE_ROWFRAG_EXCHANGE);
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNoteDualFragmentLookahead()
{
    symV2RouteProfileNote(SYM_V2_ROUTE_DUAL_FRAGMENT_LOOKAHEAD);
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNoteDualFragmentExclude()
{
    symV2RouteProfileNote(SYM_V2_ROUTE_DUAL_FRAGMENT_EXCLUDE);
}

template <typename Ftype>
static unsigned long long symldl_v2_profile_byte_count(size_t values)
{
    if (values > static_cast<size_t>(
                     std::numeric_limits<unsigned long long>::max() /
                     sizeof(Ftype)))
        ABORT("SymFact V2 communication profile byte count overflows.");
    return static_cast<unsigned long long>(values) * sizeof(Ftype);
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNotePartnerSend(
    size_t values, size_t messages, size_t max_message_values)
{
    if (messages > static_cast<size_t>(
                       std::numeric_limits<unsigned long long>::max()))
        ABORT("SymFact V2 partner message count overflows.");
    symV2CommunicationProfile.partner_send_messages +=
        static_cast<unsigned long long>(messages);
    symV2CommunicationProfile.partner_send_bytes +=
        symldl_v2_profile_byte_count<Ftype>(values);
    symV2CommunicationProfile.partner_max_message_bytes = SUPERLU_MAX(
        symV2CommunicationProfile.partner_max_message_bytes,
        symldl_v2_profile_byte_count<Ftype>(max_message_values));
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfileNoteRowSend(
    size_t values, size_t messages, size_t max_message_values)
{
    if (messages > static_cast<size_t>(
                       std::numeric_limits<unsigned long long>::max()))
        ABORT("SymFact V2 row message count overflows.");
    symV2CommunicationProfile.row_send_messages +=
        static_cast<unsigned long long>(messages);
    symV2CommunicationProfile.row_send_bytes +=
        symldl_v2_profile_byte_count<Ftype>(values);
    symV2CommunicationProfile.row_max_message_bytes = SUPERLU_MAX(
        symV2CommunicationProfile.row_max_message_bytes,
        symldl_v2_profile_byte_count<Ftype>(max_message_values));
}

template <typename Ftype>
inline void xLUstruct_t<Ftype>::symV2RouteProfilePrint(
    const char *phase) const
{
    if (!superlu_sym_v2_route_profile())
        return;

    long long local[SYM_V2_ROUTE_PROFILE_COUNTERS];
    long long global[SYM_V2_ROUTE_PROFILE_COUNTERS];
    for (int i = 0; i < SYM_V2_ROUTE_PROFILE_COUNTERS; ++i)
    {
        local[i] = symV2RouteProfile.counters[i];
        global[i] = 0;
    }

    int root = 0;
    int rank = grid3d != NULL ? grid3d->iam : 0;
    if (grid3d != NULL)
    {
        MPI_Reduce(local, global, SYM_V2_ROUTE_PROFILE_COUNTERS,
                   MPI_LONG_LONG, MPI_SUM, root, grid3d->comm);
    }
    else
    {
        for (int i = 0; i < SYM_V2_ROUTE_PROFILE_COUNTERS; ++i)
            global[i] = local[i];
    }

    unsigned long long local_sum[4] = {
        symV2CommunicationProfile.partner_send_messages,
        symV2CommunicationProfile.partner_send_bytes,
        symV2CommunicationProfile.row_send_messages,
        symV2CommunicationProfile.row_send_bytes
    };
    unsigned long long global_sum[4] = {0, 0, 0, 0};
    unsigned long long global_rank_max[4] = {0, 0, 0, 0};
    unsigned long long local_message_max[2] = {
        symV2CommunicationProfile.partner_max_message_bytes,
        symV2CommunicationProfile.row_max_message_bytes
    };
    unsigned long long global_message_max[2] = {0, 0};
    unsigned long long metadata_calls_min =
        symV2PartnerMetadataGatherCalls;
    unsigned long long metadata_calls_max =
        symV2PartnerMetadataGatherCalls;
    unsigned long long metadata_local_bytes =
        symV2PartnerMetadataLocalBytes;
    unsigned long long metadata_received_bytes_max =
        symV2PartnerMetadataReceivedBytes;
    double metadata_time_max = symV2PartnerMetadataGatherTime;
    if (grid3d != NULL)
    {
        MPI_Reduce(local_sum, global_sum, 4, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, root, grid3d->comm);
        MPI_Reduce(local_sum, global_rank_max, 4,
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, root, grid3d->comm);
        MPI_Reduce(local_message_max, global_message_max, 2,
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, root, grid3d->comm);
        MPI_Reduce(&symV2PartnerMetadataGatherCalls, &metadata_calls_min, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_MIN, root, grid3d->comm);
        MPI_Reduce(&symV2PartnerMetadataGatherCalls, &metadata_calls_max, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, root, grid3d->comm);
        MPI_Reduce(&symV2PartnerMetadataLocalBytes, &metadata_local_bytes, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, root, grid3d->comm);
        MPI_Reduce(&symV2PartnerMetadataReceivedBytes,
                   &metadata_received_bytes_max, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, root, grid3d->comm);
        MPI_Reduce(&symV2PartnerMetadataGatherTime, &metadata_time_max, 1,
                   MPI_DOUBLE, MPI_MAX, root, grid3d->comm);
    }
    else
    {
        for (int i = 0; i < 4; ++i)
            global_sum[i] = global_rank_max[i] = local_sum[i];
        for (int i = 0; i < 2; ++i)
            global_message_max[i] = local_message_max[i];
    }

    if (rank != root)
        return;

    std::printf("SymFact V2 route profile");
    if (phase != NULL && phase[0] != '\0')
        std::printf(" (%s)", phase);
    std::printf(": backend=%s", symV2FactorBackendName());
    for (int i = 0; i < SYM_V2_ROUTE_PROFILE_COUNTERS; ++i)
        std::printf(" %s=%lld",
                    symldl_v2_route_profile_label(
                        static_cast<SymV2RouteProfileCounter>(i)),
                    global[i]);
    std::printf(" dual_fragment_schur=%lld\n",
                global[SYM_V2_ROUTE_DUAL_FRAGMENT_LOOKAHEAD] +
                global[SYM_V2_ROUTE_DUAL_FRAGMENT_EXCLUDE]);

    double partner_mean = global_sum[0] > 0
                              ? static_cast<double>(global_sum[1]) /
                                    static_cast<double>(global_sum[0])
                              : 0.0;
    double row_mean = global_sum[2] > 0
                          ? static_cast<double>(global_sum[3]) /
                                static_cast<double>(global_sum[2])
                          : 0.0;
    std::printf(
        "SymFact V2 communication profile (%s, sum/max-rank): partner_messages=%llu/%llu partner_bytes=%llu/%llu partner_mean_bytes=%.3f partner_max_message_bytes=%llu row_messages=%llu/%llu row_bytes=%llu/%llu row_mean_bytes=%.3f row_max_message_bytes=%llu\n",
        phase != NULL && phase[0] != '\0' ? phase : "unknown",
        global_sum[0], global_rank_max[0], global_sum[1],
        global_rank_max[1], partner_mean, global_message_max[0],
        global_sum[2], global_rank_max[2], global_sum[3],
        global_rank_max[3], row_mean, global_message_max[1]);
    std::printf(
        "SymFact V2 metadata profile (%s, min/max-rank-calls): gather_calls=%llu/%llu source_payload_bytes=%llu received_payload_bytes_max_rank=%llu gather_time_max_rank=%.6f\n",
        phase != NULL && phase[0] != '\0' ? phase : "unknown",
        metadata_calls_min, metadata_calls_max, metadata_local_bytes,
        metadata_received_bytes_max, metadata_time_max);
    std::fflush(stdout);
}
