#pragma once

#include "xlupanels.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::pdgstrf3dSymV2()
{
    if (!useSymV2Solve())
        ABORT("pdgstrf3dSymV2 is only valid for SymFact GPU3DVERSION=2.");

    ABORT("SymFact GPU3DVERSION=2 factor loop is not implemented.");
    return -1;
}
