CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_ROOT := $(abspath $(dir $(NVCC_BIN))/..)
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump

SRC ?= mma_accept_queue_bench.cu
TARGET ?= mma_accept_queue_bench_cuda128_sm120a
ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

NVCC_FLAGS := -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)

SMOKE_ARGS ?= --samples 5 --warmup 1 --n 0,1,2,4,8,16 --modes mma,fma,empty
SWEEP_ARGS ?= --samples 101 --warmup 10 --n 0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,56,64 --modes mma,fma,empty

.PHONY: all sm120a sm90a saturation-sm120a saturation-sm90a clean run-smoke run-sweep sass ptx sass-sm90a sass-sm120a sass-saturation-sm90a sass-saturation-sm120a

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sm120a:
	$(MAKE) ARCH=sm_120a PTX_PROFILE=compute_120a TARGET=mma_accept_queue_bench_cuda128_sm120a all

sm90a:
	$(MAKE) ARCH=sm_90a PTX_PROFILE=compute_90a TARGET=mma_accept_queue_bench_cuda128_sm90a all

saturation-sm120a:
	$(MAKE) ARCH=sm_120a PTX_PROFILE=compute_120a SRC=mma_saturation_bench.cu TARGET=mma_saturation_bench_cuda128_sm120a all

saturation-sm90a:
	$(MAKE) ARCH=sm_90a PTX_PROFILE=compute_90a SRC=mma_saturation_bench.cu TARGET=mma_saturation_bench_cuda128_sm90a all

run-smoke: $(TARGET)
	./$(TARGET) $(SMOKE_ARGS)

run-sweep: $(TARGET)
	./$(TARGET) $(SWEEP_ARGS)

sass: $(TARGET)
	$(CUOBJDUMP) -sass $(TARGET)

ptx: $(TARGET)
	$(CUOBJDUMP) -ptx $(TARGET)

sass-sm90a: sm90a
	$(CUOBJDUMP) -sass mma_accept_queue_bench_cuda128_sm90a

sass-sm120a: sm120a
	$(CUOBJDUMP) -sass mma_accept_queue_bench_cuda128_sm120a

sass-saturation-sm90a: saturation-sm90a
	$(CUOBJDUMP) -sass mma_saturation_bench_cuda128_sm90a

sass-saturation-sm120a: saturation-sm120a
	$(CUOBJDUMP) -sass mma_saturation_bench_cuda128_sm120a

clean:
	rm -f mma_accept_queue_bench_cuda128_sm120a mma_accept_queue_bench_cuda128_sm90a
	rm -f mma_saturation_bench_cuda128_sm120a mma_saturation_bench_cuda128_sm90a
	rm -f *.ptx *.sass *.cubin
