#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LookAheadUpdateGPU(
    int, int_t k, int_t, xlpanel_t<Ftype> &)
{
    if (Pr == 1)
        ABORT("SymFact GPU3DVERSION=2 LL lookahead update is not implemented.");
    if (symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact GPU3DVERSION=2 dual-fragment lookahead update is not implemented.");
    ABORT("SymFact GPU3DVERSION=2 L-fragment lookahead update is not implemented.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2SchurCompUpdateExcludeOneGPU(
    int, int_t k, int_t, xlpanel_t<Ftype> &)
{
    if (Pr == 1)
        ABORT("SymFact GPU3DVERSION=2 LL Schur update is not implemented.");
    if (symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact GPU3DVERSION=2 dual-fragment Schur update is not implemented.");
    ABORT("SymFact GPU3DVERSION=2 L-fragment Schur update is not implemented.");
    return 0;
}

#endif
