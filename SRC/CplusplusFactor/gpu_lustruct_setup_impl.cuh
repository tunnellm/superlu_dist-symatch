#pragma once

#include <algorithm>
#include <cstddef>
#include "superlu_ddefs.h"
#include "xlupanels.hpp"
#include "gpuCommon.hpp"
#include "gpu_setup_utils.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_gpu_workspace_impl.cuh"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::setLUstruct_GPU()
{
    int i, stream;
    
#if (DEBUGlevel >= 1)
    int iam = 0;
    CHECK_MALLOC(iam, "Enter setLUstruct_GPU()"); fflush(stdout);
#endif
	
    A_gpu.Pr = Pr;
    A_gpu.Pc = Pc;
    A_gpu.maxSuperSize = ldt;
    const bool sym_v2_mode = useSymV2Solve();
    const bool need_u_panel_storage = needsUPanelStorage();
    const int_t local_l_panel_count =
        sym_v2_mode ? symV2PanelCount() : CEILING(nsupers, Pc);
    const int_t local_u_panel_count =
        need_u_panel_storage ? (sym_v2_mode ? symV2RowCount()
                                            : CEILING(nsupers, Pr))
                             : 0;
    const int_t gpu_u_panel_count =
        need_u_panel_storage ? local_u_panel_count : 1;

    /* Sherry: this mapping may be inefficient on Frontier */
    /*Mapping to device*/
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // How many GPUs?
    int device_id = grid3d->iam % deviceCount;
    cudaSetDevice(device_id);

    double tRegion[5];
    size_t useableGPUMem = superlu_gpu_memory_per_process(grid3d->comm);
    /**
     *  Memory is divided into two parts data memory and buffer memory
     *  data memory is used for useful data
     *  bufferMemory is used for buffers
     * */
    size_t memReqData = 0;

    /*Memory for XSUP*/
    memReqData += (nsupers + 1) * sizeof(int_t);

    tRegion[0] = SuperLU_timer_();
    
    size_t totalNzvalSize = 0; /* too big for gemmBufferSize */
    size_t max_gemmCsize = 0;  /* Sherry added 2/20/2023 */
    size_t max_nzrow = 0;  /* Yang added 10/20/2023 */
    size_t max_nzcol = 0;  
    
    /*Memory for lpapenl and upanel Data*/
    for (i = 0; i < local_l_panel_count; ++i)
    {
        int_t k0 = sym_v2_mode ? symV2PanelGid(i) : i * Pc + mycol;
        if (k0 < nsupers && isNodeInMyGrid[k0] == 1)
        {
            memReqData +=
                (sym_v2_mode && superlu_sym_v2_panel_arena_enabled())
                    ? symldl_v2_l_panel_arena_bytes(lPanelVec[i])
                    : lPanelVec[i].totalSize();
            totalNzvalSize += lPanelVec[i].nzvalSize();
            if(lPanelVec[i].nzvalSize()>0)
                max_nzrow = SUPERLU_MAX(lPanelVec[i].nzrows(),max_nzrow);
	    //max_gemmCsize = SUPERoLU_MAX(max_gemmCsize, ???);
        }
    }
    for (i = 0; i < local_u_panel_count; ++i)
    {
        int_t k0 = sym_v2_mode ? symV2RowGid(i) : i * Pr + myrow;
        if (k0 < nsupers && isNodeInMyGrid[k0] == 1)
        {
            memReqData += uPanelVec[i].totalSize();
            totalNzvalSize += uPanelVec[i].nzvalSize();
            if(uPanelVec[i].nzvalSize()>0)
                max_nzcol = SUPERLU_MAX(uPanelVec[i].nzcols(),max_nzcol);
        }
    }
    max_gemmCsize = max_nzcol*max_nzrow;
    
    memReqData += local_l_panel_count * sizeof(lpanelGPU_t);
    memReqData += gpu_u_panel_count * sizeof(upanelGPU_t);

    memReqData += sizeof(xLUstructGPU_t<Ftype>);
    
    // Per stream data
    // TODO: estimate based on ancestor size
    int_t maxBuffSize = sp_ienv_dist (8, options);
    int maxsup = sp_ienv_dist(3, options); // max. supernode size
    maxBuffSize = SUPERLU_MAX(maxsup * maxsup, maxBuffSize); // Sherry added 7/10/23
    
 #if 0   
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, totalNzvalSize); 
 #else 
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, SUPERLU_MAX(max_gemmCsize,totalNzvalSize)); /* Yang added 10/20/2023 */
 #endif
 
    int_t sym_v2_partner_stage_count = maxSymPartnerLvalCount;
    if (sym_v2_mode && Pr <= 1)
        sym_v2_partner_stage_count =
            SUPERLU_MAX(sym_v2_partner_stage_count, maxLvalCount);
    int_t sym_v2_raw_panel_count =
        (sym_v2_mode && symldl_v2_use_wpanel_cache(grid3d))
            ? maxLvalCount
            : 0;
    int_t lookahead_u_count = maxUvalCount;
    if (sym_v2_mode && Pr <= 1)
        lookahead_u_count = SUPERLU_MAX(lookahead_u_count, maxLvalCount);

    size_t dataPerStream =
        3 * sizeof(Ftype) * maxLvalCount +
        sizeof(Ftype) * (2 * maxUvalCount + lookahead_u_count) +
        2 * sizeof(int_t) * maxLidxCount +
        2 * sizeof(int_t) * maxUidxCount +
        sizeof(Ftype) * (2 * maxSymPartnerLvalCount +
                         sym_v2_partner_stage_count) +
        sizeof(int_t) * maxSymPartnerLidxCount +
        sizeof(Ftype) * maxSymPartnerLSendStageCount +
        sizeof(Ftype) * sym_v2_raw_panel_count +
        sizeof(Ftype) * (maxSymV2RowFragStageCount +
                         maxSymV2RowFragValRecvCount) +
        sizeof(int_t) * (maxSymV2RowFragIdxRecvCount +
                         maxSymV2RowFragValSendCount) +
        A_gpu.gemmBufferSize * sizeof(Ftype) +
        ldt * ldt * sizeof(Ftype);
    if (sym_v2_mode && superlu_sym_v2_workspace_arena_enabled())
    {
        dataPerStream = symldl_v2_stream_workspace_estimate<Ftype>(
            this, ldt, static_cast<size_t>(A_gpu.gemmBufferSize));
    }
    if (memReqData + 2 * dataPerStream > useableGPUMem)
    {
        printf("Not enough memory on GPU: available = %zu, required for 2 streams =%zu, exiting\n", useableGPUMem, memReqData + 2 * dataPerStream);
        exit(-1);
    }

    tRegion[0] = SuperLU_timer_() - tRegion[0];
#if ( PRNTlevel>=1 )    
    // print the time taken to estimate memory on GPU
    if (grid3d->iam == 0)
    {
        printf("GPU deviceCount=%d\n", deviceCount);
	printf("\t.. totalNzvalSize %ld, gemmBufferSize %ld\n",
	       (long) totalNzvalSize, (long) A_gpu.gemmBufferSize);
    }
#endif

    /*Memory for lapenlPanel Data*/
    tRegion[1] = SuperLU_timer_();

    int_t maxNumberOfStream = (useableGPUMem - memReqData) / dataPerStream;

    int numberOfStreams = SUPERLU_MIN(getNumLookAhead(options), maxNumberOfStream);
    numberOfStreams = SUPERLU_MIN(numberOfStreams, MAX_CUDA_STREAMS);
    int rNumberOfStreams;
    MPI_Allreduce(&numberOfStreams, &rNumberOfStreams, 1,
                  MPI_INT, MPI_MIN, grid3d->comm);
    A_gpu.numCudaStreams = rNumberOfStreams;
    symldl_v2_setup_raw_panel_ring(this, rNumberOfStreams);

#if ( PRNTlevel>=1 )    
    if (!grid3d->iam)
        printf("Using %d CUDA LookAhead streams\n", rNumberOfStreams);
    // size_t totalMemoryRequired = memReqData + numberOfStreams * dataPerStream;
#endif    

#if 0 /**** Old code ****/
    upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];
    lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];
    void *gpuBasePtr, *gpuCurrentPtr;
    cudaMalloc(&gpuBasePtr, totalMemoryRequired);
    gpuCurrentPtr = gpuBasePtr;

    A_gpu.xsup = (int_t *)gpuCurrentPtr;
    gpuCurrentPtr = (int_t *)gpuCurrentPtr + (nsupers + 1);
    cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + lPanelVec[i].totalSize();
        }
    }
    A_gpu.lPanelVec = (xlpanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>);
    cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>), cudaMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + uPanelVec[i].totalSize();
        }
    }
    A_gpu.uPanelVec = (xupanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>);
    cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
               CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>), cudaMemcpyHostToDevice);

    for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {

        cudaStreamCreate(&A_gpu.cuStreams[stream]);
        gpuErrchk(cudaEventCreateWithFlags(&A_gpu.panelReadyEvents[stream],
                                           cudaEventDisableTiming));
        if (sym_v2_mode && symldl_v2_use_wpanel_cache(grid3d))
            gpuErrchk(cudaEventCreateWithFlags(
                &A_gpu.symV2RawPanelReadyEvents[stream],
                cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(
            &A_gpu.symV2PartnerLPackReadyEvents[stream],
            cudaEventDisableTiming));
        gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[stream],
                                  A_gpu.cuStreams[stream]));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream],
            A_gpu.cuStreams[stream]));
        cublasCreate(&A_gpu.cuHandles[stream]);
        A_gpu.LvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        A_gpu.UvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
        A_gpu.LidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxLidxCount;
        A_gpu.UidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxUidxCount;

        A_gpu.gpuGemmBuffs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + A_gpu.gemmBufferSize;
        A_gpu.dFBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + ldt * ldt;

        /*lookAhead buffers and stream*/
        cublasCreate(&A_gpu.lookAheadLHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadLStream[stream]);
        A_gpu.lookAheadLGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        cublasCreate(&A_gpu.lookAheadUHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadUStream[stream]);
        A_gpu.lookAheadUGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
    }
    // cudaCheckError();
    // allocate
    dA_gpu = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr;

    cudaMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), cudaMemcpyHostToDevice);
    gpuCurrentPtr = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr + 1;

#else /* else of #if 0 ----> this is the current active code - Sherry */
    gpuErrchk(cudaMalloc(&A_gpu.xsup, (nsupers + 1) * sizeof(int_t)));
    gpuErrchk(cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice));

    double tLsend, tUsend;
#if 0
    tLsend = SuperLU_timer_();
    xupanelGPU_t<Ftype> *uPanelVec_GPU = copyUpanelsToGPU();
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = copyLpanelsToGPU();
    tUsend = SuperLU_timer_() - tUsend;
#else 
    xupanelGPU_t<Ftype> *uPanelVec_GPU =
        new xupanelGPU_t<Ftype>[gpu_u_panel_count];
    xlpanelGPU_t<Ftype> *lPanelVec_GPU =
        new xlpanelGPU_t<Ftype>[local_l_panel_count];
    tLsend = SuperLU_timer_();
    symldl_v2_copy_l_panels_to_gpu(this, lPanelVec_GPU,
                                   local_l_panel_count);
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    // cudaCheckError();
    for (i = 0; i < local_u_panel_count; ++i)
    {
        int_t k0 = sym_v2_mode ? symV2RowGid(i) : i * Pr + myrow;
        if (k0 < nsupers && isNodeInMyGrid[k0] == 1)
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU();
    }
    tUsend = SuperLU_timer_() - tUsend;
#endif
    tRegion[1] = SuperLU_timer_() - tRegion[1];

    gpuErrchk(cudaMalloc(&A_gpu.lPanelVec,
                         local_l_panel_count *
                             sizeof(xlpanelGPU_t<Ftype>)));
    gpuErrchk(cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               local_l_panel_count * sizeof(xlpanelGPU_t<Ftype>),
               cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&A_gpu.uPanelVec,
                         gpu_u_panel_count * sizeof(xupanelGPU_t<Ftype>)));
    gpuErrchk(cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
               gpu_u_panel_count * sizeof(xupanelGPU_t<Ftype>),
               cudaMemcpyHostToDevice));

    delete [] uPanelVec_GPU;
    delete [] lPanelVec_GPU;

    tRegion[2] = SuperLU_timer_();
    int dfactBufSize = superlu_gpu_getrf_workspace_size(
        ldt, !sym_v2_mode);
#if ( PRNTlevel >= 1 )    
    printf("Size of dfactBuf is %d\n", dfactBufSize);
#endif    
    tRegion[2] = SuperLU_timer_() - tRegion[2];
    
    tRegion[3] = SuperLU_timer_();

    double tcuMalloc=SuperLU_timer_();

    /* Sherry: where are these freed ?? */
    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        cudaStreamCreate(&A_gpu.cuStreams[stream]);
        gpuErrchk(cudaEventCreateWithFlags(&A_gpu.panelReadyEvents[stream],
                                           cudaEventDisableTiming));
        if (sym_v2_mode && symldl_v2_use_wpanel_cache(grid3d))
            gpuErrchk(cudaEventCreateWithFlags(
                &A_gpu.symV2RawPanelReadyEvents[stream],
                cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(
            &A_gpu.symV2PartnerLPackReadyEvents[stream],
            cudaEventDisableTiming));
        gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[stream],
                                  A_gpu.cuStreams[stream]));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream],
            A_gpu.cuStreams[stream]));
        SymV2GpuStreamWorkspaceSpec<Ftype> stream_spec =
            symldl_v2_make_stream_workspace_spec(this, dfactBufSize,
                                                 !sym_v2_mode);
        symldl_v2_setup_gpu_stream_workspace(this, stream, stream_spec);
    }
    
    /* Sherry: dfBufs[] changed to Ftype pointer **, max(batch, numCudaStreams) */
    int mxLeafNode = trf3Dpartition->mxLeafNode, mx_fsize = 0;

    /* Compute gemmCsize[] for batch operations 
       !!!!!!! WARNING: this only works for 1 MPI  !!!!!! */
    if ( options->batchCount > 0 ) {
	trf3Dpartition->gemmCsizes = int32Calloc_dist(mxLeafNode);
	int k, k0, k_st, k_end, offset, Csize;
	
	for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {  /* Loop through the Pz tree levels */
	    int treeId = trf3Dpartition->myTreeIdxs[ilvl];
	    sForest_t* sforest = trf3Dpartition->sForests[treeId];
	    if (sforest){
		int_t *perm_c_supno = sforest->nodeList ;
        mx_fsize = max((int_t)mx_fsize, sforest->nNodes);

		int maxTopoLevel = sforest->topoInfo.numLvl;/* number of levels at each outer-tree node */
		for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl) {
		    k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
		    k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];
		
		    for (k0 = k_st; k0 < k_end; ++k0) {
			offset = k0 - k_st;
			k = perm_c_supno[k0];
			Csize = lPanelVec[k].nzrows() * uPanelVec[k].nzcols();
			trf3Dpartition->gemmCsizes[offset] =
			    SUPERLU_MAX(trf3Dpartition->gemmCsizes[offset], Csize);
		    }
		}
	    }
	}
    }
    
    int num_dfbufs;  /* number of diagonal buffers */
    if ( options->batchCount > 0 ) { /* use batch code */
	num_dfbufs = mxLeafNode;
    } else { /* use pipelined code */
	// num_dfbufs = MAX_CUDA_STREAMS; // 
    num_dfbufs = A_gpu.numCudaStreams;
    }
    int num_gemmbufs = num_dfbufs;
#if ( PRNTlevel >= 1 )    
    printf(".. setLUstrut_GPU: num_dfbufs %d, num_gemmbufs %d\n", num_dfbufs, num_gemmbufs);
    fflush(stdout);
#endif

    A_gpu.dFBufs = (Ftype **) SUPERLU_MALLOC(num_dfbufs * sizeof(Ftype *));
    A_gpu.gpuGemmBuffs = (Ftype **) SUPERLU_MALLOC(num_gemmbufs * sizeof(Ftype *));
    
    int l;
    size_t sum_diag_size = 0;
    size_t sum_gemmC_size = 0;
    
    if ( options->batchCount > 0 ) { /* set up variable-size buffers for batch code */
	for (i = 0; i < num_dfbufs; ++i) {
	    l = trf3Dpartition->diagDims[i];
	    gpuErrchk(cudaMalloc(&(A_gpu.dFBufs[i]), l * l * sizeof(Ftype)));
	    //printf("\t diagDims[%d] %d\n", i, l);
	    gpuErrchk(cudaMalloc(&(A_gpu.gpuGemmBuffs[i]), trf3Dpartition->gemmCsizes[i] * sizeof(Ftype)));
	    sum_diag_size += static_cast<size_t>(l) * static_cast<size_t>(l);
	    sum_gemmC_size += static_cast<size_t>(trf3Dpartition->gemmCsizes[i]);
	}
    } else { /* uniform-size buffers */
        size_t dfbuf_elems =
            static_cast<size_t>(ldt) * static_cast<size_t>(ldt);
        symldl_v2_setup_gemm_workspace(this, num_dfbufs, dfbuf_elems,
                                       &sum_diag_size, &sum_gemmC_size);
    }
    
    // Wajih: Adding allocation for batched LU and SCU marshalled data
    // TODO: these are serialized workspaces, so the allocations can be shared
    
#if 0    
    A_gpu.marshall_data.setBatchSize(num_dfbufs);
    A_gpu.sc_marshall_data.setBatchSize(num_dfbufs);
#endif

    // TODO: where should these be freed?
    // Allocate GPU copy for the node list 
    gpuErrchk(cudaMalloc(&(A_gpu.dperm_c_supno), sizeof(int) * mx_fsize));
    // Allocate GPU copy of all the gemm buffer pointers and copy the host array to the GPU 
    gpuErrchk(cudaMalloc(&(A_gpu.dgpuGemmBuffs), sizeof(Ftype*) * num_gemmbufs));
    gpuErrchk(cudaMemcpy(A_gpu.dgpuGemmBuffs, A_gpu.gpuGemmBuffs, sizeof(Ftype*) * num_gemmbufs, cudaMemcpyHostToDevice));
    symldl_v2_setup_gpu_panel_index(this);

    tcuMalloc = SuperLU_timer_() - tcuMalloc;
#if ( PRNTlevel>=1 )
    printf("Time to allocate GPU memory: %g\n", tcuMalloc);
    printf("\t.. sum_diag_size %zu\t sum_gemmC_size %zu\n",
           sum_diag_size, sum_gemmC_size);
    fflush(stdout);
#endif

    double tcuStream=SuperLU_timer_();
    
    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        // cublasCreate(&A_gpu.cuHandles[stream]);
        superlu_gpu_create_cusolver_handle(&A_gpu.cuSolveHandles[stream],
                                           !sym_v2_mode);
    }
    tcuStream = SuperLU_timer_() - tcuStream;

    double tcuStreamCreate=SuperLU_timer_();
    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        cublasCreate(&A_gpu.cuHandles[stream]);
        /*lookAhead buffers and stream*/
        cublasCreate(&A_gpu.lookAheadLHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadLStream[stream]);
        cublasCreate(&A_gpu.lookAheadUHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadUStream[stream]);

    }
    tcuStreamCreate = SuperLU_timer_() - tcuStreamCreate;
    tRegion[3] = SuperLU_timer_() - tRegion[3];
    
#if ( PRNTlevel >= 1 )
    printf("Time to create cublas streams: %g\n", tcuStream);
    printf("Time to create CUDA streams: %g\n", tcuStreamCreate);
    printf("Time taken to estimate memory on GPU: %f\n", tRegion[0]);
    printf("TRegion L,U send: \t %g\n", tRegion[1]);
    printf("Time to send Lpanel=%g  and U panels =%g \n", tLsend, tUsend);
    printf("TRegion dfactBuf: \t %g\n", tRegion[2]);
    printf("TRegion stream: \t %g\n", tRegion[3]);
    fflush(stdout);
#endif

    // allocate
    gpuErrchk(cudaMalloc(&dA_gpu, sizeof(xLUstructGPU_t<Ftype>)));
    gpuErrchk(cudaMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), cudaMemcpyHostToDevice));

#endif /* match #if 0 ... #else ... */
    
    // cudaCheckError();
    
#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit setLUstruct_GPU()");
#endif
    return 0;
} /* setLUstruct_GPU */
