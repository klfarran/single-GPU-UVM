# Makefile — build 
NVCC ?= nvcc
ARCH ?= native
CXXSTD ?= c++17
OPTFLAGS ?= -O2
LIBS      := -lcublas

all: microbench 

microbench: microbench.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)

clean:
	rm -f microbench
