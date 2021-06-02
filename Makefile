CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LINKERFLAGS= -lcurand
NVCCFLAGS= $(LINKERFLAGS) -std=c++14 -I$(CUDA_HOME)/include -I src

all: ising3d

ising3d:
	$(CUDACC) src/kernel.cu src/traders.cu -o build/ising3d $(NVCCFLAGS)

clean:
	rm *.o
