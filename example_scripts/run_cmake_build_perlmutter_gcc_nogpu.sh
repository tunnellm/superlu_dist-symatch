# superlu_dist cmake build for Perlmutter CPU-only compute nodes
# updated 2023/04/01
module load PrgEnv-gnu
module unload cudatoolkit
module unload gpu
#module load PrgEnv-gnu
#module load gcc/11.2.0
module load cmake
module load cray-libsci
#module load cudatoolkit/11.7

# parmetis_dir=/global/cfs/cdirs/m3894/tpl/install/parmetis/parmetis-4.0.3/n9-gcc11.2.0
#parmetis_dir=/global/cfs/cdirs/m3894/tpl/install/parmetis/parmetis-4.0.3-64bit/n9-gcc11.2.0

cmake .. \
  -DCMAKE_C_FLAGS="-O2 -std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_" \
  -DCMAKE_CXX_FLAGS="-O2 -std=c++17 -DMULTIPHASE -DPRINT -DNGPU=1 -DUSE_32BIT_GRAPH" \
  -DCMAKE_Fortran_FLAGS="-O2" \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DTPL_ENABLE_CUDALIB=OFF \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_BLAS_LIBRARIES="$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mp.so" \
  -DTPL_LAPACK_LIBRARIES="$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mp.so" \
  -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/parmetis-4.0.3/include;/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/parmetis-4.0.3/metis/include" \
  -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DTPL_ENABLE_NVSHMEM=OFF \
  -DTPL_NVSHMEM_LIBRARIES="-L${CUDA_HOME}/lib64/stubs/ -lnvidia-ml -L/usr/lib64 -lgdrapi -lstdc++ -L/opt/cray/libfabric/1.22.0/lib64 -lfabric -L${NVSHMEM_HOME}/lib -lnvshmem" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DMPIEXEC_NUMPROC_FLAG=-n \
  -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
  -DMPIEXEC_MAX_NUMPROCS=16 \
  -DTPL_ENABLE_SYMATCHLIB=ON \
  -DTPL_ENABLE_SUMAC=OFF \
  -DTPL_ENABLE_MC80=ON \
  -DTPL_SYMATCH_INCLUDE_DIRS="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/symatch/inc;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/symatch/util;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/matching;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/sumac" \
  -DTPL_SYMATCH_LIBRARIES="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/matching/lib/libsuitor.a;${NCCL_HOME}/lib/libnccl.so;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/sumac/libsumac.a" \
  -DTPL_MC80_INCLUDE_DIRS="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/include" \
  -DTPL_MC80_LIBRARIES="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/build/lib/libhsl_mc80.a"

make pddrive -j16
make pddrive3d -j16
make pddrive3d-sym -j16
make pzdrive3d -j16



# -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/metis/include" \
# -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
# -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \

	
