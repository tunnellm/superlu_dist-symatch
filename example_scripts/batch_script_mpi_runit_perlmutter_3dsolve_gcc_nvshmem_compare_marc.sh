#!/bin/bash
#
#modules:
module load PrgEnv-gnu
# module load gcc/11.2.0
module load cmake
module load cudatoolkit
# avoid bug in cray-libsci/21.08.1.2
# module load cray-libsci/22.11.1.2
module load cray-libsci
# module unload cray-libsci
# module use /global/common/software/nersc/pe/modulefiles/latest
module load nvshmem/2.11.0
ulimit -s unlimited


# Set gpu=1 for GPU nodes/factorizations or gpu=0 for CPU
# nodes/factorizations. An exported value overrides the default.

gpu=1
gpu=${gpu:-1}
if [[ $gpu != 0 && $gpu != 1 ]]; then
  echo "gpu must be either 0 or 1" >&2
  exit 1
fi

#MPI settings:
export SUPERLU_CUDA_AWARE_MPI=0
if ((gpu)); then
  export MPICH_GPU_SUPPORT_ENABLED=1
  export CRAY_ACCEL_TARGET=nvidia80
else
  export MPICH_GPU_SUPPORT_ENABLED=0
  unset CRAY_ACCEL_TARGET
fi
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
#SUPERLU settings:


export SUPERLU_LBS=GD  


configure_factorization() {
  local factor_mode=$1

  export SUPERLU_ACC_SOLVE=0
  export SUPERLU_CUDA_AWARE_MPI=0

  # Prevent LDLT-only settings from leaking into the LU runs.
  unset GPU3DV2_ASYNC_FACTOR
  unset GPU3DV2_PC_FRAGMENT_SCHUR
  unset GPU3DV2_PC_FRAGMENT_LDL_NATIVE
  unset GPU3DV2_PCFRAG_ASYNC_EXCHANGE
  unset GPU3DV2_PCFRAG_ASYNC_PIPELINE
  unset GPU3DV2_PINNED_STAGING
  unset GPU3DV2_PINNED_STAGING_POOL
  unset GPU3DV2_LOWER_ENVELOPE
  unset GPU3DV2_PANEL_ARENA
  unset GPU3DV2_WORKSPACE_ARENA
  unset GPU3DV2_CPU_SCHEDULER
  unset GPU3DV2_CPU_ASYNC_EXCHANGE

  case $factor_mode in
    gpu_ldlt)
      export SUPERLU_ACC_OFFLOAD=1
      export GPU3DVERSION=2
      export SUPERLU_CUDA_AWARE_MPI=0
      export GPU3DV2_ASYNC_FACTOR=1
      export GPU3DV2_PC_FRAGMENT_SCHUR=1
      export GPU3DV2_PC_FRAGMENT_LDL_NATIVE=1
      export GPU3DV2_PCFRAG_ASYNC_EXCHANGE=1
      export GPU3DV2_PCFRAG_ASYNC_PIPELINE=0
      export GPU3DV2_PINNED_STAGING=1
      export GPU3DV2_PINNED_STAGING_POOL=1
      export GPU3DV2_LOWER_ENVELOPE=1
      export GPU3DV2_PANEL_ARENA=1
      export GPU3DV2_WORKSPACE_ARENA=1
      rowperm=6 # MC80; enables symmetric factorization
      ;;
    gpu_old_lu)
      export SUPERLU_ACC_OFFLOAD=1
      export GPU3DVERSION=1
      export SUPERLU_CUDA_AWARE_MPI=1
      rowperm=1 # LargeDiag_MC64; ordinary LU
      ;;
    gpu_symmetric_lu)
      export SUPERLU_ACC_OFFLOAD=1
      export GPU3DVERSION=0
      export SUPERLU_CUDA_AWARE_MPI=0
      rowperm=6 # MC80; enables symmetric LU
      ;;
    cpu_ldlt)
      export SUPERLU_ACC_OFFLOAD=0
      export GPU3DVERSION=2
      export GPU3DV2_CPU_SCHEDULER=COMPLETION
      export GPU3DV2_CPU_ASYNC_EXCHANGE=1
      rowperm=6 # MC80; enables symmetric factorization
      ;;
    cpu_old_lu)
      export SUPERLU_ACC_OFFLOAD=0
      export GPU3DVERSION=0
      rowperm=1 # LargeDiag_MC64; ordinary LU
      ;;
    cpu_symmetric_lu)
      export SUPERLU_ACC_OFFLOAD=0
      export GPU3DVERSION=0
      rowperm=6 # MC80; enables symmetric LU
      ;;
    *)
      echo "Unknown factorization mode: $factor_mode" >&2
      return 1
      ;;
  esac
}



# ######## GPU LDLT
# export GPU3DVERSION=2
# export SUPERLU_ACC_OFFLOAD=1
# export SUPERLU_ACC_SOLVE=0
# export GPU3DV2_ASYNC_FACTOR=1
# export GPU3DV2_PC_FRAGMENT_SCHUR=1
# export GPU3DV2_PC_FRAGMENT_LDL_NATIVE=1
# export GPU3DV2_PCFRAG_ASYNC_EXCHANGE=1
# export GPU3DV2_PCFRAG_ASYNC_PIPELINE=0
# export SUPERLU_CUDA_AWARE_MPI=0
# export GPU3DV2_PINNED_STAGING=1
# export GPU3DV2_PINNED_STAGING_POOL=1
# export GPU3DV2_LOWER_ENVELOPE=1
# export GPU3DV2_PANEL_ARENA=1
# export GPU3DV2_WORKSPACE_ARENA=1
# rowperm=6 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80


# ######## GPU old LU 
# export GPU3DVERSION=1
# export SUPERLU_ACC_OFFLOAD=1
# export SUPERLU_ACC_SOLVE=0
# rowperm=1 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80


# ######## GPU symmetric LU
# export GPU3DVERSION=0
# export SUPERLU_ACC_OFFLOAD=1
# export SUPERLU_ACC_SOLVE=0
# rowperm=6 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80



# ######## CPU LDLT
# export GPU3DVERSION=2
# export SUPERLU_ACC_OFFLOAD=0
# export SUPERLU_ACC_SOLVE=0
# export GPU3DV2_CPU_SCHEDULER=COMPLETION
# export GPU3DV2_CPU_ASYNC_EXCHANGE=1
# rowperm=6 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80


# ######## CPU symmetric LU
# export GPU3DVERSION=0
# export SUPERLU_ACC_OFFLOAD=0
# export SUPERLU_ACC_SOLVE=0
# rowperm=6 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80


# ######## CPU old LU
# export GPU3DVERSION=0
# export SUPERLU_ACC_OFFLOAD=0
# export SUPERLU_ACC_SOLVE=0
# rowperm=1 ### 1: LargeDiag_MC64  4: SymMatch 6: MC80



export ANC25D=0
export NEW3DSOLVE=1    
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=$gpu # bind MPI ranks to GPUs only for GPU runs

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=256000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=10   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
nmpipergpu=1
if ((gpu)); then
  export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # 2: this can better saturate GPU
else
  unset SUPERLU_MPI_PROCESS_PER_GPU
fi
export SUPERLU_RANKORDER=Z

# ##NVSHMEM settings:
# # NVSHMEM_HOME=/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/nvshmem_src_2.8.0-3/build/
# export NVSHMEM_USE_GDRCOPY=1
# export NVSHMEM_MPI_SUPPORT=1
# export MPI_HOME=${MPICH_DIR}
# export NVSHMEM_LIBFABRIC_SUPPORT=1
# export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0
# export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
# export NVSHMEM_DISABLE_CUDA_VMM=1
# export FI_CXI_OPTIMIZED_MRS=false
# export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
# export NVSHMEM_BOOTSTRAP=MPI
# export NVSHMEM_REMOTE_TRANSPORT=libfabric

# #export NVSHMEM_DEBUG=TRACE
# #export NVSHMEM_DEBUG_SUBSYS=ALL
# #export NVSHMEM_DEBUG_FILE=nvdebug_success

if [[ $NERSC_HOST == edison ]]; then
  CORES_PER_NODE=24
  THREADS_PER_NODE=48
elif [[ $NERSC_HOST == cori ]]; then
  CORES_PER_NODE=32
  THREADS_PER_NODE=64
  # This does not take hyperthreading into account
elif [[ $NERSC_HOST == perlmutter ]]; then
  if ((gpu)); then
    CORES_PER_NODE=64
    THREADS_PER_NODE=128
    GPUS_PER_NODE=4
  else
    CORES_PER_NODE=128
    THREADS_PER_NODE=256
  fi
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi




if ((gpu)); then
  export SUPERLU_GPU_MEMORY_PROFILE=1
  factor_modes=(gpu_ldlt gpu_old_lu gpu_symmetric_lu)

  mats=(symmetric/Geo_1438.bin symmetric/StocF-1465.bin symmetric/nlpkkt80.bin symmetric/dielFilterV3real.mtx symmetric/Si41Ge41H72.mtx symmetric/pwtk.mtx symmetric/offshore.mtx symmetric/Spielman_k200_A_10.mtx symmetric/Cube_Coup_dt0.mtx)  
  nprows=(2 2 2 1 4 1 2 1 2)
  npcols=(1 1 1 2 1 2 1 2 1)
  npz=(8 8 8 8 4 8 8 8 8)  
  NTH=16
else
  factor_modes=(cpu_ldlt cpu_old_lu cpu_symmetric_lu)

  mats=(symmetric/Geo_1438.bin symmetric/StocF-1465.bin symmetric/nlpkkt80.bin symmetric/dielFilterV3real.mtx symmetric/Si41Ge41H72.mtx symmetric/pwtk.mtx symmetric/offshore.mtx symmetric/Spielman_k200_A_10.mtx symmetric/Cube_Coup_dt0.mtx)
  nprows=(8 2 8 2 4 2 4 1 8)
  npcols=(2 1 1 1 4 1 1 4 4)
  npz=(2 16 4 16 2 16 8 8 1)  
  NTH=16
fi

matrix_count=${#mats[@]}
if ((matrix_count != ${#nprows[@]} ||
     matrix_count != ${#npcols[@]} ||
     matrix_count != ${#npz[@]})); then
  echo "mats, nprows, npcols, and npz must have the same number of entries" >&2
  exit 1
fi


nrhs=(1)
NREP=1
batch=0 # whether to do batched test
tinyreplace=0 # whether to use tiny pivot replacement
it=0 # whether to use iterative refinement

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export SLURM_CPU_BIND=cores
export MPICH_MAX_THREAD_SAFETY=multiple
THREADS_PER_CORE=$((THREADS_PER_NODE / CORES_PER_NODE))
TH_PER_RANK=$((NTH * THREADS_PER_CORE))
if ((gpu)); then
  RANKS_PER_NODE=$GPUS_PER_NODE
else
  RANKS_PER_NODE=$((CORES_PER_NODE / NTH))
fi
if ((RANKS_PER_NODE < 1)); then
  echo "NTH=${NTH} is too large for CORES_PER_NODE=${CORES_PER_NODE}" >&2
  exit 1
fi

for ((mat_idx = 0; mat_idx < matrix_count; mat_idx++)); do
  MAT=${mats[mat_idx]}
  NROW=${nprows[mat_idx]}
  NCOL=${npcols[mat_idx]}
  NPZ=${npz[mat_idx]}

  NCORE_VAL_TOT=$((NROW * NCOL * NPZ))
  NODE_VAL=$(((NCORE_VAL_TOT + RANKS_PER_NODE - 1) / RANKS_PER_NODE))
  mkdir -p "$MAT"

  for ((rhs_idx = 0; rhs_idx < ${#nrhs[@]}; rhs_idx++)); do
    NRHS=${nrhs[rhs_idx]}

    for ((ii = 1; ii <= NREP; ii++)); do
      for factor_mode in "${factor_modes[@]}"; do
        configure_factorization "$factor_mode" || exit 1

        output_file="./${MAT}/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_${factor_mode}_gsolve_${SUPERLU_ACC_SOLVE}_rowperm${rowperm}_tinyreplace${tinyreplace}_it${it}_V${GPU3DVERSION}_nrhs${NRHS}_rep${ii}"

        echo "Running ${factor_mode}: MAT=${MAT}, grid=${NROW}x${NCOL}x${NPZ}, ranks=${NCORE_VAL_TOT}, nodes=${NODE_VAL}, NRHS=${NRHS}, repetition=${ii}"
        srun -n "$NCORE_VAL_TOT" -N "$NODE_VAL" -c "$TH_PER_RANK" \
          --cpu_bind=cores \
          ./EXAMPLE/pddrive3d-sym \
          -c "$NCOL" -r "$NROW" -d "$NPZ" -b "$batch" \
          -t "$tinyreplace" -i "$it" -p "$rowperm" -s "$NRHS" \
          "${CFS}/m2957/liuyangz/my_research/matrix/${MAT}" \
          | tee "$output_file"
      done
    done
  done
done

