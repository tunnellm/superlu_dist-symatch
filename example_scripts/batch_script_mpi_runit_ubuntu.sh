#!/bin/bash

module purge
module load gcc/9.1.0
module load openmpi/gcc-9.1.0/4.0.1
module load cmake/3.19.2

# ulimit -s unlimited
ulimit -c unlimited


export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=0
export ANC25D=0
export NEW3DSOLVE=1    
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_MPI_PROCESS_PER_GPU=1 # 2: this can better saturate GPU
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU


# nprows=(4 8 16)
# npcols=(1 1 1)
# npz=(64 32 16)
# nrhs=(1 50) 

nprows=(2 )
npcols=(2 )
npz=(1 )
nrhs=(1)

NTH=1
NREP=1
# NODE_VAL_TOT=1

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}
CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`

for ((s = 0; s < ${#nrhs[@]}; s++)); do
NRHS=${nrhs[s]}


# NODE_VAL=2
# NCORE_VAL_TOT=`expr $NODE_VAL_TOT \* $CORES_PER_NODE / $NTH`
batch=0 # whether to do batched test

OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`

export OMP_NUM_THREADS=$NTH
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=256
# export NREL=256
# for MAT in big.rua
# for MAT in g20.rua
# for MAT in s1_mat_0_253872.bin s2D9pt2048.rua
# for MAT in dielFilterV3real.bin
# for MAT in Geo_1438.bin s2D9pt2048.rua raefsky3.mtx rma10.mtx
# for MAT in Geo_1438.bin 
# for MAT in s1_mat_0_126936.bin
# for MAT in marcus_100000.dat 
# for MAT in marcus_500000.dat
# for MAT in s2D9pt2048.rua
# for MAT in s2D9pt1536.rua
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in temp_13k.mtx
# for MAT in matrix_ACTIVSg10k_AC_00.mtx


# for MAT in bcsstk17.mtx
# for MAT in boyd1.rb
# for MAT in mesh3e1.mtx
# for MAT in 662_bus.mtx

for MAT in matrix_ACTIVSg10k_AC_00.mtx 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx 
# for MAT in turon_m.mtx
do
mkdir -p $MAT
for ii in `seq 1 $NREP`
do	
export SUPERLU_ACC_SOLVE=0

rowperm=4  ### 1: LargeDiag_MC64  4: SymMatch
tinyreplace=1
it=1

# # srun -n $NCORE_VAL_TOT2D -N $NODE_VAL2D -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_gpu_${SUPERLU_ACC_OFFLOAD}
# export SUPERLU_ACC_OFFLOAD=0

mpirun --allow-run-as-root -n $CORE_VAL ./EXAMPLE/pddrive3d-sym -c $NCOL -r $NROW -d $NPZ -b $batch -t $tinyreplace -i $it -p $rowperm -s $NRHS /home/administrator/Desktop/Research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_gpu_${SUPERLU_ACC_OFFLOAD}_gsolve_${SUPERLU_ACC_SOLVE}_rowperm${rowperm}_tinyreplace${tinyreplace}_it${it}


done

done
done
done

