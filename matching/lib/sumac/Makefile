NVCC:=nvcc
CC:=CC


SM=80
graph_ft_load=0
check=0
debug=0


NCCL_PATH=${NCCL_HOME}
CUDA_PATH=${CUDA_HOME}
NGPU=1

bit=32


ifeq ($(debug), 1)
	CUFLAGS = -g -G -Ddebug --std=c++11 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -fopenmp -DMULTIPHASE -DNGPU=${NGPU} -DPRINT
	CFLAGS = -g -Ddebug -std=c++11 -Wextra -Wall -DMULTIPHASE -DNGPU=1 -DPRINT
else 
	CUFLAGS = -O3 -Xptxas -O3 --std=c++11 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -O3 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	CFLAGS = -O3 -std=c++11 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	LDFLAGS = -I${NCCL_PATH}/include -I${CUDA_PATH}/include -L${NCCL_PATH}/lib -L${CUDA_PATH}/lib64 -lnccl -lcudart -lcuda
#LDFLAGS = -L${NCCL_PATH} -L${CUDA_PATH} -lnccl -lcudart -lcuda
endif

ifeq ($(check),1)
	CUFLAGS += -DCHECK
	CFLAGS  += -DCHECK
endif

ifeq ($(graph_ft_load),1)
	CUFLAGS += -DGRAPH_FT_LOAD=4
	CFLAGS  += -DGRAPH_FT_LOAD=4
else
	CUFLAGS += -DGRAPH_FT_LOAD=2
	CFLAGS  += -DGRAPH_FT_LOAD=2
endif

ifeq ($(bit),32)
	CUFLAGS += -DUSE_32BIT_GRAPH
	CFLAGS  += -DUSE_32BIT_GRAPH
else ifeq ($(bit),64)
	CUFLAGS += -DUSE_64BIT_GRAPH
	CFLAGS  += -DUSE_64BIT_GRAPH
else
	CUFLAGS += -DUSE_64BIT_GRAPH
	CFLAGS  += -DUSE_64BIT_GRAPH
endif

EXE:= ldgpu${NGPU}

all: ${EXE}

graph.o: graph.cpp graph.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

graph_gpu.o: graph_gpu.cpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
	${NVCC} ${CFLAGS} -c -I${CUDA_PATH}/include -I${NCCL_PATH}/include $< -o $@
#	${NVCC} ${CFLAGS} -c $< -o $@

main.o: main.cpp graph.hpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
#	${NVCC} ${CFLAGS} -I${CUDA_INC_PATH} -c $< -o $@
	${NVCC} ${CFLAGS} -I${CUDA_PATH}/include -c $< -o $@


graph_cuda.o: graph_cuda.cu graph_cuda.hpp graph.cpp graph.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

heap.o : heap.cpp heap.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

clustering.o: clustering.cpp clustering.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

ifeq ($(check),1)
graph_cpu.o: graph_cpu.cpp graph_cpu.hpp
	${NVCC} ${CFLAGS} -c $< -o $@
endif

ifeq ($(check),1)
${EXE}: graph_gpu.o graph_cuda.o graph.o main.o heap.o clustering.o graph_cpu.o
	${NVCC} ${CFLAGS} $^ -o $@
else
${EXE}: graph_gpu.o graph_cuda.o graph.o main.o heap.o clustering.o
	${NVCC} ${CFLAGS} $^ -o $@ ${LDFLAGS}
endif

lib: graph_gpu.o graph_cuda.o graph.o heap.o clustering.o
	ar rcs libsumac.a $^

clean:
	rm -f *.o ${EXE}
