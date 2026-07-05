#pragma once

// GPU panel accessors for xLUstruct_t.

#ifdef HAVE_CUDA
template <typename Ftype>
xupanel_t<Ftype> xLUstruct_t<Ftype>::getKUpanel(int_t k, int_t offset)
{
    if (!needsUPanelStorage())
        ABORT("SymFact GPU3DVERSION=2 does not materialize U panels.");
    return (
        myrow == krow(k) ?
        uPanelVec[g2lRow(k)] :
        xupanel_t<Ftype>(UidxRecvBufs[offset], UvalRecvBufs[offset],
            A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset])
    );
}

template <typename Ftype>
xlpanel_t<Ftype> xLUstruct_t<Ftype>::getKLpanel(int_t k, int_t offset)
{
    int_t panel_root = symV2PanelRoot(k);
    return (
        mycol == panel_root ?
        lPanelVec[symV2PanelIndex(k)] :
        xlpanel_t<Ftype>(LidxRecvBufs[offset], LvalRecvBufs[offset],
            A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset])
    );
}
#endif
