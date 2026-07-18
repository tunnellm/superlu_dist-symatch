#include "superlu_upacked.h"
#include "xlupanels.hpp"
#include "symldl_v2_workspace_impl.hpp"
#include "symldl_v2_factor_impl.hpp"

static void symldl_v2_prepare_cpu_runtime(xLUstruct_t<double> *lu)
{
#ifdef _OPENMP
    if (!lu->symV2UsesCpuFactor() || lu->symV2CpuOutputLocks != NULL)
        return;
    if (lu->symV2CpuOutputLockOffsets.empty())
        ABORT("SymFact V2 CPU output-lock plan is missing.");

    size_t lock_count = lu->symV2CpuOutputLockOffsets.back();
    if (lock_count == 0)
        return;
    omp_lock_t *locks = static_cast<omp_lock_t *>(SUPERLU_MALLOC(
        symldl_v2_checked_product(
            lock_count, sizeof(omp_lock_t),
            "SymFact V2 CPU output lock table overflows.")));
    if (locks == NULL)
        ABORT("Malloc fails for SymFact V2 CPU output locks.");
    for (size_t lock = 0; lock < lock_count; ++lock)
        omp_init_lock(&locks[lock]);
    lu->symV2CpuOutputLocks = locks;
#else
    (void) lu;
#endif
}

extern "C" int pdgstrf3d_symv2_factor_cpp(dLUgpu_Handle handle)
{
    xLUstruct_t<double> *lu =
        reinterpret_cast<xLUstruct_t<double> *>(handle);
    symldl_v2_prepare_cpu_runtime(lu);
    return lu->pdgstrf3dSymV2();
}

extern "C" void pdgstrf3d_symv2_destroy_cpu_runtime_cpp(
    dLUgpu_Handle handle)
{
#ifdef _OPENMP
    xLUstruct_t<double> *lu =
        reinterpret_cast<xLUstruct_t<double> *>(handle);
    if (lu == NULL || lu->symV2CpuOutputLocks == NULL)
        return;
    size_t lock_count = lu->symV2CpuOutputLockOffsets.empty()
        ? 0 : lu->symV2CpuOutputLockOffsets.back();
    omp_lock_t *locks =
        static_cast<omp_lock_t *>(lu->symV2CpuOutputLocks);
    for (size_t lock = 0; lock < lock_count; ++lock)
        omp_destroy_lock(&locks[lock]);
    SUPERLU_FREE(locks);
    lu->symV2CpuOutputLocks = NULL;
#else
    (void) handle;
#endif
}
