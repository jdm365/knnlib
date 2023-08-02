## Makefile
.PHONY: gpu cpu run clean

## CPU Variables
CXX = clang++
CXX_FLAGS = -Wall -Wextra -std=c++17 -g
CXX_FLAGS += -Ofast -march=native
INCLUDE = -Iinclude/
LIBS = -lstdc++ -stdlib=libc++ -fopenmp -lgomp -lopenblas
BIN = bin/release/cpu

## GPU Variables
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17
NVCC_FLAGS += -Xcompiler -Wall,-Wextra,-Wno-deprecated-gpu-targets
CULIBS = -lcuda -lcublas -lcudart
CUBIN = bin/release/gpu

MODULE_NAME = knnlib
PYBIN = bin/release/$(MODULE_NAME)
PYBIND11_FLAGS = `python3-config --extension-suffix` python/* -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`

py:
	$(CXX) $(CXX_FLAGS) -o $(PYBIN)$(PYBIND11_FLAGS) src/* $(INCLUDE) $(LIBS)

cpu:
	$(CXX) $(CXX_FLAGS) -o $(BIN) src/* $(INCLUDE) $(LIBS)

gpu:
	$(NVCC) $(NVCC_FLAGS) -o $(CUBIN) cuda/src/* -Icuda/include $(CULIBS)

run: cpu
	./$(BIN)

run_gpu: gpu
	./$(CUBIN)

clean:
	rm -rf bin/debug/*
	rm -rf bin/release/*
