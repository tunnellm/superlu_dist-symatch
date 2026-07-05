#pragma once

template <typename Ftype>
inline bool xLUstruct_t<Ftype>::useSymV2Solve() const
{
    return std::is_same<Ftype, double>::value &&
           options != NULL &&
           options->SymFact == YES &&
           symGPU3DVersion == 2;
}

template <typename Ftype>
inline bool xLUstruct_t<Ftype>::needsUPanelStorage() const
{
    return !useSymV2Solve();
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelRoot(int_t k)
{
    return kcol(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2DiagRoot(int_t k)
{
    return krow(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2DiagProc(int_t k)
{
    return procIJ(k, k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelIndex(int_t k)
{
    return g2lCol(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowIndex(int_t k)
{
    return g2lRow(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelCount()
{
    return CEILING(nsupers, Pc);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowCount()
{
    return CEILING(nsupers, Pr);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelGid(int_t local_index)
{
    return local_index * Pc + mycol;
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowGid(int_t local_index)
{
    return local_index * Pr + myrow;
}

template <typename Ftype>
inline bool xLUstruct_t<Ftype>::symV2ScheduleActive() const
{
    return false;
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2ForestLevelCount() const
{
    return grid3d != NULL ? log2i(grid3d->zscp.Np) + 1 : maxLvl;
}

template <typename Ftype>
inline bool xLUstruct_t<Ftype>::symV2UsePcFragmentSchurPanel(int_t) const
{
    return false;
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelRoot(int_t k)
{
    return symldl_v2_panel_root(trf3Dpartition, k, grid);
}

template <>
inline int_t xLUstruct_t<double>::symV2DiagRoot(int_t k)
{
    return symldl_v2_diag_root(trf3Dpartition, k, grid);
}

template <>
inline int_t xLUstruct_t<double>::symV2DiagProc(int_t k)
{
    return symldl_v2_owner_2d(trf3Dpartition, k, grid);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelIndex(int_t k)
{
    return symldl_v2_panel_local_index(trf3Dpartition, k);
}

template <>
inline int_t xLUstruct_t<double>::symV2RowIndex(int_t k)
{
    return symldl_v2_row_local_index(trf3Dpartition, k);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelCount()
{
    return useSymV2Solve() && trf3Dpartition != NULL
               ? trf3Dpartition->symV2LocalPanelCount
               : CEILING(nsupers, Pc);
}

template <>
inline int_t xLUstruct_t<double>::symV2RowCount()
{
    return useSymV2Solve() && trf3Dpartition != NULL
               ? trf3Dpartition->symV2LocalRowCount
               : CEILING(nsupers, Pr);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelGid(int_t local_index)
{
    return useSymV2Solve() && trf3Dpartition != NULL &&
                   trf3Dpartition->symV2LocalPanelGids != NULL
               ? trf3Dpartition->symV2LocalPanelGids[local_index]
               : local_index * Pc + mycol;
}

template <>
inline int_t xLUstruct_t<double>::symV2RowGid(int_t local_index)
{
    return useSymV2Solve() && trf3Dpartition != NULL &&
                   trf3Dpartition->symV2LocalRowGids != NULL
               ? trf3Dpartition->symV2LocalRowGids[local_index]
               : local_index * Pr + myrow;
}

template <>
inline bool xLUstruct_t<double>::symV2ScheduleActive() const
{
    return useSymV2Solve() && trf3Dpartition != NULL &&
           trf3Dpartition->symV2ScheduleEnabled;
}

template <>
inline int_t xLUstruct_t<double>::symV2ForestLevelCount() const
{
    return symV2ScheduleActive()
               ? trf3Dpartition->maxLvl
               : (grid3d != NULL ? log2i(grid3d->zscp.Np) + 1 : maxLvl);
}

template <>
inline bool xLUstruct_t<double>::symV2UsePcFragmentSchurPanel(int_t k) const
{
    if (!useSymV2Solve())
        return false;
    if (k < 0 || k >= nsupers)
        return false;
    if (!symV2UsePcFragmentSchur.empty())
    {
        if (static_cast<size_t>(k) >= symV2UsePcFragmentSchur.size())
            return false;
        return symV2UsePcFragmentSchur[static_cast<size_t>(k)] != 0;
    }
    return Pr > 1 && Pc > 1 && superlu_sym_v2_pc_fragment_schur() &&
           superlu_sym_v2_pc_fragment_ldl_native();
}
