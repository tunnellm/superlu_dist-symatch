#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace(path, old, new):
    p = ROOT / path
    s = p.read_text()
    if new in s:
        return
    if s.count(old) != 1:
        raise RuntimeError(f"unexpected match count in {path}: {s.count(old)}")
    p.write_text(s.replace(old, new, 1))

replace(
    "SRC/CplusplusFactor/xlupanels.hpp",
    "    std::vector<unsigned char> symV2PartnerLSendRowActive;\n    std::vector<int> symV2PartnerLRecvSizes;\n",
    "    std::vector<unsigned char> symV2PartnerLSendRowActive;\n    std::vector<unsigned char> symV2PartnerLPrepacked;\n    std::vector<int> symV2PartnerLRecvSizes;\n",
)
replace(
    "SRC/CplusplusFactor/xlupanels.hpp",
    "    int_t dSymV2ComputePartnerScratchSize(LUStruct_type<Ftype> *LUstruct);\n    int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);\n",
    "    int_t dSymV2ComputePartnerScratchSize(LUStruct_type<Ftype> *LUstruct);\n    int_t dSymV2PrepackLFragmentsGPU(int_t k, int_t stream_offset);\n    int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);\n",
)
replace(
    "SRC/CplusplusFactor/lupanels_impl.hpp",
    "        symV2PartnerLSendRowActive.assign(\n            xlu_checked_product(l2u_slots, static_cast<size_t>(Pr),\n                                \"SymFact V2 partner-L send row activity\"),\n            0);\n        symPanelReadyEventIds.assign(nsupers, -1);\n",
    "        symV2PartnerLSendRowActive.assign(\n            xlu_checked_product(l2u_slots, static_cast<size_t>(Pr),\n                                \"SymFact V2 partner-L send row activity\"),\n            0);\n        symV2PartnerLPrepacked.assign(static_cast<size_t>(local_cols), 0);\n        symPanelReadyEventIds.assign(nsupers, -1);\n",
)
replace(
    "SRC/CplusplusFactor/lupanels_impl.hpp",
    "    symV2PartnerLSendRowActive.clear();\n    symV2PartnerLRecvSizes.clear();\n",
    "    symV2PartnerLSendRowActive.clear();\n    symV2PartnerLPrepacked.clear();\n    symV2PartnerLRecvSizes.clear();\n",
)
replace(
    "SRC/CplusplusFactor/lupanels_impl.hpp",
    "    if (symGPU3DVersion != 2)\n        dSymStartL2U(k, handle_offset);\n\n    int_t ksupc = SuperSize(k);\n",
    "    if (symGPU3DVersion != 2)\n        dSymStartL2U(k, handle_offset);\n#ifdef HAVE_CUDA\n    else if (superlu_acc_offload)\n        dSymV2PrepackLFragmentsGPU(k, handle_offset);\n#endif\n\n    int_t ksupc = SuperSize(k);\n",
)
