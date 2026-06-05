CUDA_INSTALL_PATH ?= /home/wzr/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_ROOT := $(abspath $(dir $(NVCC_BIN))/..)
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump

SRC := tma_completion_latency_bench.cu
TARGET := tma_completion_latency_bench_shared_cuda129
DESC_SRC := tma_descriptor_setup_bench.cu
DESC_TARGET := tma_descriptor_setup_bench_cuda129

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

CONFIG_DIR ?= ../../../../configs/SM120_RTX5090
RUN_DIR ?= gem5_pair48_smoke_cuda129
DESC_RUN_DIR ?= gem5_descriptor_smoke_cuda129

NVCC_FLAGS := -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)

.PHONY: all clean check-ptx desc-check-ptx gem5-dir gem5-smoke desc-gem5-dir desc-gem5-smoke run-hw desc-run-hw

all: $(TARGET) $(DESC_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(DESC_TARGET): $(DESC_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

desc-check-ptx: $(DESC_TARGET)
	$(CUOBJDUMP) -lptx $(DESC_TARGET)

gem5-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --iters=1 --warmup=0 --tiles=64 --case=pair48 --issue=llama --stride

desc-gem5-dir: $(DESC_TARGET)
	mkdir -p $(DESC_RUN_DIR)
	rm -f $(DESC_RUN_DIR)/*.ptx $(DESC_RUN_DIR)/*.ptxas $(DESC_RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(DESC_TARGET) $(DESC_RUN_DIR)/
	cp $(CONFIG_DIR)/gpgpusim.config $(DESC_RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(DESC_RUN_DIR)/

desc-gem5-smoke: desc-gem5-dir
	cd $(DESC_RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(DESC_TARGET) --blocks=1 --iters=1 --warmup=0 --tiles=64 --case=full --stride

run-hw: $(TARGET)
	./$(TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=pair48 --issue=llama --stride

desc-run-hw: $(DESC_TARGET)
	./$(DESC_TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=full --stride

clean:
	rm -f $(TARGET) $(DESC_TARGET)
	rm -f $(RUN_DIR)/$(TARGET) $(DESC_RUN_DIR)/$(DESC_TARGET)
	rm -f *.ptx *.ptxas
