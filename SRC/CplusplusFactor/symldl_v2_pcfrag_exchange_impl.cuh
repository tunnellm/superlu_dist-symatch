#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PanelBcastGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 panel broadcast is not implemented.");
    return 0;
}

#endif
