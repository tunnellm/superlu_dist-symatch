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
    "        symV2PartnerLSendRowActive.empty() ||\n        symV2PartnerLRecvSizes.empty() ||\n",
    "        symV2PartnerLSendRowActive.empty() ||\n        symV2PartnerLPrepacked.empty() ||\n        symV2PartnerLRecvSizes.empty() ||\n",
    "prepack validation",
)
replace(
    """    if (mycol == kcol_)
    {
        if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
            symV2DiagBlocksGPU[k] == NULL)
            ABORT("SymFact V2 true symmetric device diagonal block is missing.");

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double pack_issue_t = SuperLU_timer_();
#endif
        int_t lk = symV2PanelIndex(k);
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        bool packed_any = false;
""",
    """    if (mycol == kcol_)
    {
        int_t lk = symV2PanelIndex(k);
        bool prepacked =
            lk >= 0 &&
            static_cast<size_t>(lk) < symV2PartnerLPrepacked.size() &&
            symV2PartnerLPrepacked[static_cast<size_t>(lk)] != 0;
        if (!prepacked &&
            (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
             symV2DiagBlocksGPU[k] == NULL))
            ABORT("SymFact V2 true symmetric device diagonal block is missing.");

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double pack_issue_t = SuperLU_timer_();
#endif
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        bool packed_any = false;
        bool issued_pack = false;
""",
    "prepacked source state",
)
replace(
    """            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            sym_l2u_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                lpanel.gpuPanel.val, sendbuf, sendmap, size,
                lpanel.LDA(), symV2DiagBlocksGPU[k], ksupc);
            packed_any = true;
""",
    """            if (!prepacked)
            {
                int threads = 256;
                int blocks = (size + threads - 1) / threads;
                sym_l2u_pack_raw_kernel<<<blocks, threads, 0, stream>>>(
                    lpanel.gpuPanel.val, sendbuf, sendmap, size,
                    lpanel.LDA(), symV2DiagBlocksGPU[k], ksupc);
                issued_pack = true;
            }
            packed_any = true;
""",
    "skip reconstructed pack",
)
replace(
    """        if (packed_any)
        {
            gpuErrchk(cudaGetLastError());
            if (!cuda_aware)
""",
    """        if (packed_any)
        {
            if (issued_pack)
                gpuErrchk(cudaGetLastError());
            if (!cuda_aware)
""",
    "conditional pack check",
)
replace(
    """    std::vector<MPI_Request> recv_reqs;
    std::vector<MPI_Request> send_reqs;
    std::vector<std::vector<double> > recv_buffers(cuda_aware ? 0 : Pr);
    recv_reqs.reserve(Pr);
""",
    """    std::vector<MPI_Request> recv_reqs;
    std::vector<MPI_Request> send_reqs;
    double *recv_host_base = NULL;
    if (!cuda_aware && recv_total > 0)
    {
        if (static_cast<size_t>(stream_offset) >=
                symPartnerLvalRecvBufs.size() ||
            symPartnerLvalRecvBufs[stream_offset] == NULL)
            ABORT("SymFact V2 host receive staging buffer is missing.");
        recv_host_base = symPartnerLvalRecvBufs[stream_offset];
    }
    recv_reqs.reserve(Pr);
""",
    "persistent host receive buffer",
)
replace(
    """        int src = PNUM(pr, kcol_, grid);
        MPI_Request req;
        double *recv_ptr = NULL;
        if (cuda_aware)
        {
            recv_ptr = A_gpu.symPartnerLStageBufs[stream_offset] +
                       recv_offsets[pr];
        }
        else
        {
            recv_buffers[pr].resize(size);
            recv_ptr = recv_buffers[pr].data();
        }
""",
    """        int src = PNUM(pr, kcol_, grid);
        if (src == iam)
            continue;
        MPI_Request req;
        double *recv_ptr = NULL;
        if (cuda_aware)
        {
            recv_ptr = A_gpu.symPartnerLStageBufs[stream_offset] +
                       recv_offsets[pr];
        }
        else
        {
            recv_ptr = recv_host_base + recv_offsets[pr];
        }
""",
    "self receive bypass",
)
replace(
    """                int dest = PNUM(pr, pc, grid);
                MPI_Request req;
""",
    """                int dest = PNUM(pr, pc, grid);
                if (dest == iam)
                    continue;
                MPI_Request req;
""",
    "self send bypass",
)
needle = """    if (frag_nblocks > 0 && frag_nrows > 0)
    {
"""
insertion = """    if (!cuda_aware && !recv_reqs.empty())
    {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double h2d_issue_t = SuperLU_timer_();
#endif
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symPartnerLStageBufs[stream_offset], recv_host_base,
            sizeof(double) * static_cast<size_t>(recv_total),
            cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                     SuperLU_timer_() - h2d_issue_t);
#endif
    }

    if (frag_nblocks > 0 && frag_nrows > 0)
    {
"""
replace(needle, insertion, "single H2D receive staging copy")
replace(
    """            double *stage = A_gpu.symPartnerLStageBufs[stream_offset];
            if (cuda_aware)
            {
                stage += recv_offsets[pr];
            }
            else
            {
                double *recv_data = recv_buffers[pr].data();
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double h2d_issue_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpyAsync(
                    stage, recv_data,
                    sizeof(double) * static_cast<size_t>(count),
                    cudaMemcpyHostToDevice, stream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
                             SuperLU_timer_() - h2d_issue_t);
#endif
            }
""",
    """            int src = PNUM(pr, kcol_, grid);
            double *stage = NULL;
            if (src == iam)
            {
                if (mycol != kcol_)
                    ABORT("SymFact V2 self fragment has an invalid source column.");
                int_t send_lk = symV2PanelIndex(k);
                size_t self_flat =
                    static_cast<size_t>(send_lk) * static_cast<size_t>(Pc) +
                    static_cast<size_t>(mycol);
                if (self_flat >= symV2PartnerLSendBufsGPU.size() ||
                    self_flat >= symV2PartnerLSendSizes.size() ||
                    symV2PartnerLSendBufsGPU[self_flat] == NULL ||
                    symV2PartnerLSendSizes[self_flat] != count)
                    ABORT("SymFact V2 self fragment buffer is invalid.");
                stage = symV2PartnerLSendBufsGPU[self_flat];
            }
            else
            {
                stage = A_gpu.symPartnerLStageBufs[stream_offset] +
                        recv_offsets[pr];
            }
""",
    "self device assembly",
)

s = s[:start] + region + s[end:]
PATH.write_text(s)
