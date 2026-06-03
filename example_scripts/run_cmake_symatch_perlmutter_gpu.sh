#!/bin/bash

# nvshmem not included
# add any parameter to reconfigure (i.e., ./run_cmake.sh 1)

module load gcc-native/12.3
module load cudatoolkit/12.4
module load nccl
module load cmake
module load cray-libsci

if [ "$#" -eq 1 ]; then
    cmake .. \
	-DCMAKE_C_FLAGS="-O2 -std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_" \
	-DCMAKE_CXX_FLAGS="-O2 -std=c++14 -DMULTIPHASE -DPRINT -DNGPU=1 -DUSE_32BIT_GRAPH" \
	-DCMAKE_Fortran_FLAGS="-O2" \
	-DCMAKE_CXX_COMPILER=CC \
	-DCMAKE_C_COMPILER=cc \
	-DCMAKE_Fortran_COMPILER=ftn \
	-DXSDK_ENABLE_Fortran=ON \
	-DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
	-DTPL_ENABLE_LAPACKLIB=ON \
	-DBUILD_SHARED_LIBS=OFF \
	-DTPL_ENABLE_CUDALIB=ON \
	-DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include -ccbin=CC" \
	-DCMAKE_CUDA_ARCHITECTURES=80 \
	-DCMAKE_CUDA_STANDARD=14 \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_INSTALL_LIBDIR=./lib \
	-DCMAKE_BUILD_TYPE=Debug \
	-DTPL_BLAS_LIBRARIES=$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mp.so \
	-DTPL_LAPACK_LIBRARIES=$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mp.so \
	-DTPL_PARMETIS_INCLUDE_DIRS="$HOME/mso/code/lib/ParMETIS/parmetis/include;$HOME/mso/code/lib/METIS/metis/include;$HOME/mso/code/lib/GKlib/GKlib/include" \
	-DTPL_PARMETIS_LIBRARIES="$HOME/mso/code/lib/ParMETIS/parmetis/lib/libparmetis.a;$HOME/mso/code/lib/METIS/metis/lib/libmetis.a;$HOME/mso/code/lib/GKlib/GKlib/lib/libGKlib.a" \
	-DTPL_ENABLE_SYMATCHLIB=ON \
	-DTPL_SYMATCH_INCLUDE_DIRS="$HOME/mso/code/superlu_dist-symatch/matching/symatch/inc;$HOME/mso/code/superlu_dist-symatch/matching/symatch/util;$HOME/mso/code/superlu_dist-symatch/matching/lib/matching;${NCCL_HOME}/include;$HOME/mso/code/superlu_dist-symatch/matching/lib/sumac" \
	-DTPL_SYMATCH_LIBRARIES="$HOME/mso/code/superlu_dist-symatch/matching/lib/matching/lib/libsuitor.a;${NCCL_HOME}/lib/libnccl.so;$HOME/mso/code/superlu_dist-symatch/matching/lib/sumac/libsumac.a" \
        -DTPL_ENABLE_MC80=ON \
	-DTPL_MC80_INCLUDE_DIRS="$HOME/mso/code/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/include;$HOME/mso/code/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/build/lib" \
	-DTPL_MC80_LIBRARIES="$HOME/mso/code/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/build/lib/libhsl_mc80.a"
fi

# make pddrive -j16
# make pddrive3d -j16
make pddrive3d-sym -j16
# make pzdrive3d -j16
