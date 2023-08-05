## Makefile
.PHONY: gpu cpu run clean

## CPU Variables
CXX = clang++
#CXX = g++
CXX_FLAGS = -Wall -Wextra -std=c++17 -g
CXX_FLAGS += -O3 -march=native -mtune=native -funroll-loops -fomit-frame-pointer
CXX_FLAGS += -ffast-math -fno-finite-math-only -fno-signed-zeros -fno-trapping-math
INCLUDE = -Iinclude/
LIBS = -lstdc++ -fopenmp -lgomp -lopenblas
CLANG_LIBS = -stdlib=libc++
ifeq ($(CXX),clang++)
	LIBS += $(CLANG_LIBS)
endif
BIN = bin/release/cpu

## GPU Variables
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17
NVCC_FLAGS += -Xcompiler -Wall,-Wextra,-Wno-deprecated-gpu-targets
CULIBS = -lcuda -lcublas -lcudart
CUBIN = bin/release/gpu

MODULE_NAME = knnlib
PYBIN = bin/python/knnlib/$(MODULE_NAME)
PYBIND11_FLAGS = `python3-config --extension-suffix` python/* -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`

py:
	$(CXX) $(CXX_FLAGS) -o $(PYBIN)$(PYBIND11_FLAGS) src/* $(INCLUDE) $(LIBS)
	cd bin/python && pip install .

cpu:
	$(CXX) $(CXX_FLAGS) -o $(BIN) src/* $(INCLUDE) $(LIBS) `python3 -m pybind11 --includes`

gpu:
	$(NVCC) $(NVCC_FLAGS) -o $(CUBIN) cuda/src/* -Icuda/include $(CULIBS)

run: cpu
	./$(BIN)

run_gpu: gpu
	./$(CUBIN)

clean:
	rm -rf bin/debug/*
	rm -rf bin/release/*
	rm -rf bin/python/knnlib/*.so
