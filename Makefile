CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LINKERFLAGS=-lcurand
NVCCFLAGS= -std=c++14 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I$(CUDA_HOME)/include -I src

all: ising3d

ising3d: %.o
	nvcc -o ising3d $(LINKERFLAGS)

%.o: %.cu:
	nvcc $(NVCCFLAGS) $<

clean:
	rm *.o
