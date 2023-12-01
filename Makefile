## Makefile
.PHONY: gpu cpu run clean

## CPU Variables
CXX = clang++
## CXX = g++
CXX_FLAGS = -Wall -Wextra -std=c++17 -g
CXX_FLAGS += -O3 -march=native -mtune=native -funroll-loops -fomit-frame-pointer
CXX_FLAGS += -Wno-unused-variable -Wno-unused-parameter
CXX_FLAGS += -ffast-math -fno-finite-math-only -fno-signed-zeros -fno-trapping-math
#CXX_FLAGS += -fsanitize=address -fsanitize=undefined -fno-sanitize-recover
INCLUDE = -Iinclude/
LIBS = -lstdc++ -lopenblas


OS := $(shell uname)
ifeq ($(OS),Darwin)
	BLAS_INCLUDE := $(shell brew --prefix openblas)/include 
	BLAS_LIB := $(shell brew --prefix openblas)/lib

	LIBS += -Xpreprocessor -fopenmp -lomp -L$(BLAS_LIB) -I$(BLAS_INCLUDE) -lopenblas
	DYNAMIC_LOOKUP = -undefined dynamic_lookup
else
	LIBS += -fopenmp -lgomp
	DYNAMIC_LOOKUP =
endif

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

PYTHON_BIN = python
PYBIND_FLAGS = $(shell $(PYTHON_BIN) -m pybind11 --includes)
PYTHON_CONFIG = $(PYTHON_BIN)-config
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags)

TARGET = bin/release/$(MODULE_NAME)
MODULE_NAME = knnlib
PYBIND_FLAGS = $(shell $(PYTHON_BIN) -m pybind11 --includes)
EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
PYBIN = bin/python/$(MODULE_NAME)/$(MODULE_NAME)$(EXTENSION_SUFFIX)

LD_FLAGS = $(LIBS) $(PYTHON_LDFLAGS)

cpu:
	$(CXX) $(CXX_FLAGS) -o $(BIN) src/* $(INCLUDE) $(LIBS) `python3 -m pybind11 --includes`

install:
	$(CXX) $(CXX_FLAGS) -o $(PYBIN) $(PYBIND_FLAGS) -shared -std=c++11 -fPIC $(DYNAMIC_LOOKUP) src/* python/*.cpp $(INCLUDE) $(LD_FLAGS)
	cd bin/python && python -m pip install .

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
