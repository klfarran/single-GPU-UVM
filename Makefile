# Makefile — build 
NVCC ?= nvcc
ARCH ?= native
CXXSTD ?= c++17
OPTFLAGS ?= -O2
LIBS := -lcublas

TARGETS:= m1 m2 m3

all: clean $(TARGETS)

m1: m1.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)
m2: m2.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)
m3: m3.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS) 
