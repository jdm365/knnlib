## Makefile
.PHONY: gpu cpu run clean

## CPU Variables
CXX = clang++
CXX_FLAGS = -Wall -Wextra -std=c++17 -g
CXX_FLAGS += -Ofast -march=native
INCLUDE = -Iinclude/
LIBS = -lstdc++ -stdlib=libc++ -fopenmp -lgomp -pthread
BIN = bin/release/main_cpu

## GPU Variables
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_61 -std=c++17
NVCC_FLAGS += -Xcompiler -Wall,-Wextra,-Wno-deprecated-gpu-targets
NVCC_FLAGS += -Xcompiler -Ofast,-march=native
LIBS += -lcuda -lcublas -lcudart
BIN = bin/release/main_gpu


cpu:
	$(CXX) $(CXX_FLAGS) -o $(BIN) src/* $(INCLUDE) $(LIBS)

gpu:
	$(NVCC) $(NVCC_FLAGS) -o $(BIN) src/* $(INCLUDE) $(LIBS)

run: cpu
	./bin/release/main_cpu

clean:
	rm -rf bin/debug/*
	rm -rf bin/release/*
