CUDA_HOME=/usr/local/cuda
CUDACOMPILER=$(CUDA_HOME)/bin/nvcc
LINKERFLAGS= -l curand
NVCCFLAGS= $(LINKERFLAGS) -std=c++14 -I$(CUDA_HOME)/include -I src -o build/ising3d
SOURCE_FILES = src/kernel.cu src/traders.cu
all: ising3d

ising3d:
	$(CUDACOMPILER) $(SOURCE_FILES) $(NVCCFLAGS)

clean:
	rm *.o
