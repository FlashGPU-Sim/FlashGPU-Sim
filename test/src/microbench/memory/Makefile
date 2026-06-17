CUDA_INSTALL_PATH ?= /home/wzr/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_ROOT := $(abspath $(dir $(NVCC_BIN))/..)
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump

SRC := global_load_bw_bench.cu
TARGET := global_load_bw_bench_cuda129

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

CONFIG_DIR ?= ../../../../configs/SM120_RTX5090
RUN_DIR ?= gem5_global_load_bw_smoke_cuda129

NVCC_FLAGS := -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)

.PHONY: all clean check-ptx gem5-dir gem5-smoke run-hw

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

gem5-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=256 --iters=1 --warmup=0 --tiles=1 --block-hot

run-hw: $(TARGET)
	./$(TARGET) --blocks=528 --threads=256 --iters=4096 --warmup=256 --tiles=528 --repeat=1 --block-hot

clean:
	rm -f $(TARGET)
	rm -f $(RUN_DIR)/$(TARGET)
	rm -f *.ptx *.ptxas
