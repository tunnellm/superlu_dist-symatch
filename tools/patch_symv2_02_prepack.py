#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "SRC/CplusplusFactor/lupanels_GPU_impl.hpp"
s = PATH.read_text()
marker = "template <typename Ftype>\nint_t xLUstruct_t<Ftype>::dSymV2LFragmentExchangeGPU(\n"
presence = "dSymV2PrepackLFragmentsGPU("

code = r'''template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    ABORT("SymFact GPU3D V2 raw L-fragment prepack is implemented for double precision only.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("GPU3DVERSION=2 raw L-fragment prepack requires GPU offload.");
    if (k < 0 || k >= nsupers || mycol != symV2PanelRoot(k))
        return 0;
    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLPrepacked.empty())
        ABORT("SymFact V2 raw L-fragment prepack buffers are not allocated.");

    int_t lk = symV2PanelIndex(k);
    if (lk < 0 || static_cast<size_t>(lk) >= symV2PartnerLPrepacked.size())
        ABORT("SymFact V2 raw L-fragment prepack has an invalid local panel.");
    if (symV2PartnerLPrepacked[static_cast<size_t>(lk)])
        return 0;

    xlpanel_t<double> &lpanel = lPanelVec[lk];
    if (lpanel.isEmpty())
    {
        symV2PartnerLPrepacked[static_cast<size_t>(lk)] = 1;
        return 0;
    }
    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double pack_issue_t = SuperLU_timer_();
#endif
    bool packed_any = false;
    for (int pc = 0; pc < Pc; ++pc)
    {
        size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc);
        int size = symV2PartnerLSendSizes[flat];
        if (size <= 0)
            continue;
        bool active_dest = false;
        for (int pr = 0; pr < Pr; ++pr)
        {
            size_t active_pos = flat * static_cast<size_t>(Pr) + pr;
            if (active_pos >= symV2PartnerLSendRowActive.size())
                ABORT("SymFact V2 raw L-fragment prepack row mask is missing.");
            if (symV2PartnerLSendRowActive[active_pos])
            {
                active_dest = true;
                break;
            }
        }
        if (!active_dest)
            continue;
        double *sendbuf = symV2PartnerLSendBufsGPU[flat];
        int_t *sendmap = symL2LSendMapsGPU[flat];
        if (sendbuf == NULL || sendmap == NULL)
            ABORT("SymFact V2 raw L-fragment prepack buffer is missing.");
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sym_l2u_pack_kernel<<<blocks, threads, 0, stream>>>(
            lpanel.gpuPanel.val, sendbuf, sendmap, size);
        packed_any = true;
    }
    if (packed_any)
        gpuErrchk(cudaGetLastError());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_ISSUE,
                 SuperLU_timer_() - pack_issue_t);
#endif
    /* The following diagonal D2H copy waits for this same stream. */
    symV2PartnerLPrepacked[static_cast<size_t>(lk)] = 1;
    return 0;
}

'''

if presence not in s:
    if s.count(marker) != 1:
        raise RuntimeError(f"unexpected insertion marker count: {s.count(marker)}")
    PATH.write_text(s.replace(marker, code + marker, 1))
