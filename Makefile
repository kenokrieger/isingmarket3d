CUDA_HOME=/usr/local/cuda
CUDACOMPILER=$(CUDA_HOME)/bin/nvcc
LINKERFLAGS= -l curand
NVCCFLAGS= $(LINKERFLAGS) -std=c++14 -I$(CUDA_HOME)/include -I src -o build/ising2d -arch=sm_61
SOURCE_FILES = src/kernel.cu src/traders.cu

all: ising2d

ising2d:
	$(CUDACOMPILER) $(SOURCE_FILES) $(NVCCFLAGS)

run:
	$(CUDACOMPILER) $(SOURCE_FILES) $(NVCCFLAGS) --run
