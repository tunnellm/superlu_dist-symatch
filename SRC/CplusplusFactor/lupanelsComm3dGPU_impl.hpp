#include "mpi.h"
// #include "cublasDefs.hhandle, "
#include "lupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "gpu_mpi_utils.hpp"
#include <vector>
#include <limits>

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2BatchAncestorReduceGPU(
    int_t numNodes, int_t *nodeList,
    int_t sender, int_t receiver, int_t reduction_level)
{
    if (numNodes <= 0 || nodeList == NULL)
        return 0;

    struct PanelPart
    {
        int_t k;
        int_t lk;
        int count;
    };
    std::vector<PanelPart> parts;
    parts.reserve(static_cast<size_t>(numNodes));
    for (int_t node = 0; node < numNodes; ++node)
    {
        const int_t k = nodeList[node];
        if (mycol != symV2PanelRoot(k))
            continue;
        const int_t lk = symV2PanelIndex(k);
        if (lk < 0 || lPanelVec[lk].isEmpty())
            continue;
        const int_t count_t = lPanelVec[lk].nzvalSize();
        if (count_t <= 0)
            continue;
        if (count_t > static_cast<int_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 ancestor panel exceeds MPI int count.");
        PanelPart part;
        part.k = k;
        part.lk = lk;
        part.count = static_cast<int>(count_t);
        parts.push_back(part);
    }
    long long local_sig[3] = {
        static_cast<long long>(parts.size()), 0, 0
    };
    for (size_t i = 0; i < parts.size(); ++i)
    {
        local_sig[1] += static_cast<long long>(parts[i].count);
        local_sig[2] = local_sig[2] * 1315423911LL +
                       static_cast<long long>(parts[i].k) * 65537LL +
                       static_cast<long long>(parts[i].count);
    }
    long long peer_sig[3] = {0, 0, 0};
    const int my_grid_for_sig = grid3d->zscp.Iam;
    const int peer = (my_grid_for_sig == sender) ? receiver : sender;
    MPI_Sendrecv(local_sig, 3, MPI_LONG_LONG, peer, nodeList[0],
                 peer_sig, 3, MPI_LONG_LONG, peer, nodeList[0],
                 grid3d->zscp.comm, MPI_STATUS_IGNORE);
    if (local_sig[0] != peer_sig[0] ||
        local_sig[1] != peer_sig[1] ||
        local_sig[2] != peer_sig[2])
        ABORT("SymFact V2 ancestor batch layout differs across z peers.");
    if (parts.empty())
        return 0;

    const bool cuda_aware = superlu_cuda_aware_mpi();
    const size_t byte_cap = superlu_sym_v2_ancestor_batch_bytes();
    const size_t elem_cap = SUPERLU_MAX(
        static_cast<size_t>(1), byte_cap / sizeof(Ftype));
    cudaStream_t stream = A_gpu.cuStreams[0];
    cublasHandle_t handle = A_gpu.cuHandles[0];
    cublasSetStream(handle, stream);
    Ftype one_value = one<Ftype>();

    size_t begin = 0;
    int chunk_id = 0;
    while (begin < parts.size())
    {
        size_t end = begin;
        size_t total = 0;
        while (end < parts.size())
        {
            const size_t next = static_cast<size_t>(parts[end].count);
            if (end > begin && (total > elem_cap - SUPERLU_MIN(next, elem_cap)))
                break;
            if (next > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                total + next > static_cast<size_t>(std::numeric_limits<int>::max()))
                break;
            total += next;
            ++end;
            if (total >= elem_cap)
                break;
        }
        if (end == begin)
            ABORT("SymFact V2 ancestor batch cap is smaller than one panel.");

        Ftype *device_pack = NULL;
        Ftype *host_pack = NULL;
        gpuErrchk(cudaMalloc((void **)&device_pack, total * sizeof(Ftype)));
        if (!cuda_aware)
            gpuErrchk(cudaMallocHost((void **)&host_pack,
                                     total * sizeof(Ftype)));

        std::vector<size_t> offsets(end - begin + 1, 0);
        for (size_t i = begin; i < end; ++i)
            offsets[i - begin + 1] =
                offsets[i - begin] + static_cast<size_t>(parts[i].count);

        const int my_grid = grid3d->zscp.Iam;
        const int tag = static_cast<int>(parts[begin].k);
        if (my_grid == sender)
        {
            for (size_t i = begin; i < end; ++i)
            {
                Ftype *dst = cuda_aware
                                 ? device_pack + offsets[i - begin]
                                 : host_pack + offsets[i - begin];
                gpuErrchk(cudaMemcpyAsync(
                    dst, lPanelVec[parts[i].lk].blkPtrGPU(0),
                    static_cast<size_t>(parts[i].count) * sizeof(Ftype),
                    cuda_aware ? cudaMemcpyDeviceToDevice
                               : cudaMemcpyDeviceToHost,
                    stream));
            }
            gpuErrchk(cudaStreamSynchronize(stream));
            MPI_Send(cuda_aware ? device_pack : host_pack,
                     static_cast<int>(total), get_mpi_type<Ftype>(),
                     receiver, tag, grid3d->zscp.comm);
            SCT->commVolRed += total * sizeof(Ftype);
        }
        else if (my_grid == receiver)
        {
            MPI_Status status;
            MPI_Recv(cuda_aware ? device_pack : host_pack,
                     static_cast<int>(total), get_mpi_type<Ftype>(),
                     sender, tag, grid3d->zscp.comm, &status);
            if (!cuda_aware)
                gpuErrchk(cudaMemcpyAsync(device_pack, host_pack,
                                           total * sizeof(Ftype),
                                           cudaMemcpyHostToDevice, stream));
            for (size_t i = begin; i < end; ++i)
            {
                myCublasAxpy<Ftype>(
                    handle, parts[i].count, &one_value,
                    device_pack + offsets[i - begin], 1,
                    lPanelVec[parts[i].lk].blkPtrGPU(0), 1);
            }
            gpuErrchk(cudaStreamSynchronize(stream));
        }

        if (host_pack != NULL)
            gpuErrchk(cudaFreeHost(host_pack));
        gpuErrchk(cudaFree(device_pack));
        begin = end;
        ++chunk_id;
    }
    (void)reduction_level;
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                         int_t **treePerm)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    bool sym_v2_l_only =
        (options != NULL && options->SymFact == YES && symGPU3DVersion == 2);

#if (DEBUGlevel >= 1)
    printf(".maxLvl %d\n", maxLvl); fflush(stdout);
    CHECK_MALLOC(grid3d->iam, "Enter ancestorReduction3dGPU()");
#endif
	
    int_t sender, receiver;
    if ((myGrid % (1 << (ilvl + 1))) == 0)
    {
        sender = myGrid + (1 << ilvl);
        receiver = myGrid;
    }
    else
    {
        sender = myGrid;
        receiver = myGrid - (1 << ilvl);
    }

    /*Reduce all the ancestors*/
    for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
    {
        /* code */
        // int_t atree = myTreeIdxs[alvl];
        int_t numNodes = myNodeCount[alvl];
        int_t *nodeList = treePerm[alvl];
        double treduce = SuperLU_timer_();
        

        if (sym_v2_l_only && superlu_sym_v2_batch_ancestor_reduce())
        {
            dSymV2BatchAncestorReduceGPU(numNodes, nodeList,
                                         sender, receiver, alvl);
        }
        else
        {
            /* Legacy panel-at-a-time reduction. */
            for (int_t node = 0; node < numNodes; ++node)
            {
                int_t k0 = nodeList[node];

                if (myGrid == sender)
                {
                    zSendLPanelGPU(k0, receiver);
                    if (!sym_v2_l_only)
                        zSendUPanelGPU(k0, receiver);
                }
                else
                {
                    Ftype alpha = one<Ftype>();
                    Ftype beta = one<Ftype>();
                    zRecvLPanelGPU(k0, sender, alpha, beta);
                    if (!sym_v2_l_only)
                        zRecvUPanelGPU(k0, sender, alpha, beta);
                }
            }
            cudaStreamSynchronize(A_gpu.cuStreams[0]);
        }
        // return 0;
        SCT->ancsReduce += SuperLU_timer_() - treduce;
    }
    
#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit ancestorReduction3dGPU()");
#endif
    
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendLPanelGPU(int_t k0, int_t receiverGrid)
{
    int_t panel_col = useSymV2Solve() ? symV2PanelRoot(k0) : kcol(k0);

	if (mycol == panel_col)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0) : g2lCol(k0);
        if (lk < 0)
            return 0;
        if (!lPanelVec[lk].isEmpty())
		{
            superlu_gpu_mpi_send(lPanelVec[lk].blkPtrGPU(0), LvalRecvBufs[0],
                    sizeof(Ftype), static_cast<int>(lPanelVec[lk].nzvalSize()),
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
				SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(Ftype);
			}
		}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvLPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    int_t panel_col = useSymV2Solve() ? symV2PanelRoot(k0) : kcol(k0);

    if (mycol == panel_col)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0) : g2lCol(k0);
        if (lk < 0)
            return 0;
        if (!lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
				superlu_gpu_mpi_recv(A_gpu.LvalRecvBufs[0], LvalRecvBufs[0],
						 sizeof(Ftype), static_cast<int>(lPanelVec[lk].nzvalSize()),
						 get_mpi_type<Ftype>(), senderGrid, k0,
						 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
            myCublasScal<Ftype>(handle, lPanelVec[lk].nzvalSize(), &alpha, lPanelVec[lk].blkPtrGPU(0), 1);
            myCublasAxpy<Ftype>(handle, lPanelVec[lk].nzvalSize(), &beta, A_gpu.LvalRecvBufs[0], 1, lPanelVec[lk].blkPtrGPU(0), 1);
			cudaStreamSynchronize(cuStream);
            
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendUPanelGPU(int_t k0, int_t receiverGrid)
{
    
	if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            superlu_gpu_mpi_send(uPanelVec[lk].blkPtrGPU(0), UvalRecvBufs[0],
                    sizeof(Ftype), static_cast<int>(uPanelVec[lk].nzvalSize()),
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
				SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(Ftype);
			}
		}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvUPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
				superlu_gpu_mpi_recv(A_gpu.UvalRecvBufs[0], UvalRecvBufs[0],
						 sizeof(Ftype), static_cast<int>(uPanelVec[lk].nzvalSize()),
						 get_mpi_type<Ftype>(), senderGrid, k0,
						 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
			myCublasScal<Ftype>(handle, uPanelVec[lk].nzvalSize(), &alpha, uPanelVec[lk].blkPtrGPU(0), 1);
			myCublasAxpy<Ftype>(handle, uPanelVec[lk].nzvalSize(), &beta, A_gpu.UvalRecvBufs[0], 1, uPanelVec[lk].blkPtrGPU(0), 1);
            cudaStreamSynchronize(cuStream);
		}
	}
	return 0;
}

#endif
