#pragma once

#include <cstdio>
#include <cstring>

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

    if (rank != root)
        return;

    std::printf("SymFact V2 route profile");
    if (phase != NULL && phase[0] != '\0')
        std::printf(" (%s)", phase);
    std::printf(":");
    for (int i = 0; i < SYM_V2_ROUTE_PROFILE_COUNTERS; ++i)
        std::printf(" %s=%lld",
                    symldl_v2_route_profile_label(
                        static_cast<SymV2RouteProfileCounter>(i)),
                    global[i]);
    std::printf(" dual_fragment_schur=%lld\n",
                global[SYM_V2_ROUTE_DUAL_FRAGMENT_LOOKAHEAD] +
                global[SYM_V2_ROUTE_DUAL_FRAGMENT_EXCLUDE]);
    std::fflush(stdout);
}
