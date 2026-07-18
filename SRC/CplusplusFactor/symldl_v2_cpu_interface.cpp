#include "superlu_upacked.h"
#include "xlupanels.hpp"
#include "anc25d_comm_impl.hpp"
#include "l_panels_impl.hpp"
#include "u_panels_impl.hpp"
#include "lupanels_impl.hpp"
#include "symldl_v2_factor_impl.hpp"

extern "C"
{

dLUgpu_Handle dCreateLUgpuHandle(
    int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition,
    dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, SCT_t *SCT,
    superlu_dist_options_t *options, SuperLUStat_t *stat, double thresh,
    int *info)
{
    xLUstruct_t<double> *instance = new xLUstruct_t<double>(
        nsupers, ldt, trf3Dpartition, LUstruct, grid3d, SCT, options, stat,
        thresh, info);
    return reinterpret_cast<dLUgpu_Handle>(instance);
}

void dDestroyLUgpuHandle(dLUgpu_Handle handle)
{
    delete reinterpret_cast<xLUstruct_t<double> *>(handle);
}

int dCopyLUGPU2Host(dLUgpu_Handle handle, dLUstruct_t *)
{
    xLUstruct_t<double> *lu =
        reinterpret_cast<xLUstruct_t<double> *>(handle);
    if (!lu->useSymV2Solve())
        ABORT("The CPU C++ factor interface only supports SymFact V2.");
    return 0;
}

int pdgstrf3d_LUv2(dLUgpu_Handle handle)
{
    xLUstruct_t<double> *lu =
        reinterpret_cast<xLUstruct_t<double> *>(handle);
    return lu->pdgstrf3dSymV2();
}

}
