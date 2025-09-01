
export OMP_NUM_THREADS=1 

export SUPERLUROOT=$PWD/../
export LD_LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$DYLD_LIBRARY_PATH

export ParMETIS_DIR=$SUPERLUROOT/../parmetis-4.0.3/install/
export LD_LIBRARY_PATH=$ParMETIS_DIR/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ParMETIS_DIR/lib/:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$ParMETIS_DIR/lib/:$DYLD_LIBRARY_PATH

mat=matrix_ACTIVSg10k_AC_00.mtx
# mat=mesh3e1.mtx
# mat=ex5.mtx


rowperm=4  ### 1: LargeDiag_MC64  4: SymMatch
tinyreplace=1 ## whether to use tiny pivot replacement
it=1 # wether to use iterative refinement

$SUPERLUROOT/openmpi-5.0.6/bin/mpirun -n 1 ./EXAMPLE/pddrive-sym -r 1 -c 1 -t $tinyreplace -i $it -p $rowperm ../../$mat
$SUPERLUROOT/openmpi-5.0.6/bin/mpirun -n 1 ./EXAMPLE/pddrive3d-sym -r 1 -c 1 -d 1 -t $tinyreplace -i $it -p $rowperm ../../$mat