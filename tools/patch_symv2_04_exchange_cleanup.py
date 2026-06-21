#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "SRC/CplusplusFactor/lupanels_GPU_impl.hpp"
s = PATH.read_text()

START = "template <>\ninline int_t xLUstruct_t<double>::dSymV2LFragmentExchangeGPU(\n"
END = "template <>\ninline int_t xLUstruct_t<double>::dSymStartL2UGPU(\n"
start = s.find(START)
end = s.find(END, start + len(START))
if start < 0 or end < 0:
    raise RuntimeError("exchange function markers not found")
region = s[start:end]

def replace(old, new, label):
    global region
    if new in region:
        return
    count = region.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: unexpected match count {count}")
    region = region.replace(old, new, 1)

replace(
    "        symV2PartnerLSendRowActive.empty() ||\n        symV2PartnerLPrepacked.empty() ||\n        symV2PartnerLRecvSizes.empty() ||\n",
    "        symV2PartnerLSendRowActive.empty() ||\n        symV2PartnerLRecvSizes.empty() ||\n",
    "allow ranks with no local panel slots",
)

replace(
    """#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double pack_stage_sync_t = SuperLU_timer_();
#endif
            gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                         SuperLU_timer_() - pack_stage_sync_t);
#endif
""",
    """#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double pack_stage_sync_t = SuperLU_timer_();
#endif
            if (issued_pack || !cuda_aware)
                gpuErrchk(cudaStreamSynchronize(stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
                         SuperLU_timer_() - pack_stage_sync_t);
#endif
""",
    "avoid redundant CUDA-aware prepack synchronization",
)

# In the non-CUDA-aware staging loop, do not copy a packed chunk to the host
# when its only active consumer is the same MPI rank. The self fragment is
# assembled directly from the device send buffer later in the exchange.
d2h_start = region.find("            if (!cuda_aware)\n")
d2h_end = region.find("#ifdef SLU_ENABLE_SYM_GPU3D_TIMING\n            double pack_stage_sync_t", d2h_start)
if d2h_start < 0 or d2h_end < 0:
    raise RuntimeError("D2H staging region not found")
d2h = region[d2h_start:d2h_end]
old = """                    bool active_dest = false;
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        size_t active_pos =
                            flat * static_cast<size_t>(Pr) +
                            static_cast<size_t>(pr);
                        if (active_pos >= symV2PartnerLSendRowActive.size())
                            ABORT("SymFact V2 true symmetric L-fragment send row mask is missing.");
                        if (symV2PartnerLSendRowActive[active_pos])
                        {
                            active_dest = true;
                            break;
                        }
                    }
                    if (!active_dest)
                        continue;
"""
new = """                    bool active_remote_dest = false;
                    for (int pr = 0; pr < Pr; ++pr)
                    {
                        size_t active_pos =
                            flat * static_cast<size_t>(Pr) +
                            static_cast<size_t>(pr);
                        if (active_pos >= symV2PartnerLSendRowActive.size())
                            ABORT("SymFact V2 true symmetric L-fragment send row mask is missing.");
                        if (symV2PartnerLSendRowActive[active_pos] &&
                            PNUM(pr, pc, grid) != iam)
                        {
                            active_remote_dest = true;
                            break;
                        }
                    }
                    if (!active_remote_dest)
                        continue;
"""
if new not in d2h:
    if d2h.count(old) != 1:
        raise RuntimeError(f"remote staging mask: unexpected match count {d2h.count(old)}")
    d2h = d2h.replace(old, new, 1)
    region = region[:d2h_start] + d2h + region[d2h_end:]

s = s[:start] + region + s[end:]
PATH.write_text(s)
