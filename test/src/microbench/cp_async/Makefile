CUDA_INSTALL_PATH ?= /home/wzr/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_ROOT := $(abspath $(dir $(NVCC_BIN))/..)
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump
NCU ?= ncu

SRC := cp_async_latency_bench.cu
TARGET ?= cp_async_latency_bench_cuda128
PTX_SRC := cp_async_ptx_bench.cu
PTX_TARGET ?= cp_async_ptx_bench_cuda128

ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

CONFIG_DIR ?= ../../../../configs/SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA2_REGALLOC
RUN_DIR ?= gem5_cp_async_fa2_smoke_cuda128

HW_BLOCKS ?= 132
HW_ITERS ?= 4096
HW_WARMUP ?= 256
HW_TILES ?= 4096

SMOKE_ARGS ?= --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=fa2-issue --fa2-stages=2 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot
ISSUE_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=issue --groups=16 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot
ISSUE_SAME_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=issue --groups=16 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot --same-address
COMPLETE_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=complete --groups=16 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot
DEPTH_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=depth --groups=32 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot
FA2_ISSUE_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=fa2-issue --fa2-stages=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=128 --block-hot
FA2_COMPLETE_ARGS ?= --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=fa2-complete --fa2-stages=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=128 --block-hot

NVCC_FLAGS := -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)

.PHONY: all ptx-bench clean check-ptx check-sass gem5-dir gem5-smoke gem5-smoke-suite run-hw-issue run-hw-issue-same run-hw-complete run-hw-depth run-hw-fa2-issue run-hw-fa2-complete profile-hw-issue profile-hw-issue-same profile-hw-depth profile-hw-fa2-issue profile-hw-fa2-complete sweep-hw-depth

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ptx-bench: $(PTX_TARGET)

$(PTX_TARGET): $(PTX_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

check-sass: $(TARGET)
	$(CUOBJDUMP) -sass $(TARGET)

gem5-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) $(SMOKE_ARGS)

gem5-smoke-suite: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=issue --groups=2 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot > sim_issue.log 2>&1
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=issue --groups=2 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot --same-address > sim_issue_same.log 2>&1
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=complete --groups=2 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot > sim_complete.log 2>&1
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=depth --groups=4 --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot > sim_depth.log 2>&1
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET) --blocks=1 --threads=128 --iters=1 --warmup=0 --tiles=4 --mode=fa2-complete --fa2-stages=2 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot > sim_fa2_complete.log 2>&1

run-hw-issue: $(TARGET)
	./$(TARGET) $(ISSUE_ARGS)

run-hw-issue-same: $(TARGET)
	./$(TARGET) $(ISSUE_SAME_ARGS)

run-hw-complete: $(TARGET)
	./$(TARGET) $(COMPLETE_ARGS)

run-hw-depth: $(TARGET)
	./$(TARGET) $(DEPTH_ARGS)

run-hw-fa2-issue: $(TARGET)
	./$(TARGET) $(FA2_ISSUE_ARGS)

run-hw-fa2-complete: $(TARGET)
	./$(TARGET) $(FA2_COMPLETE_ARGS)

profile-hw-issue: $(TARGET)
	$(NCU) --set full --target-processes all --force-overwrite -o ncu_cp_async_issue ./$(TARGET) $(ISSUE_ARGS)

profile-hw-issue-same: $(TARGET)
	$(NCU) --set full --target-processes all --force-overwrite -o ncu_cp_async_issue_same ./$(TARGET) $(ISSUE_SAME_ARGS)

profile-hw-depth: $(TARGET)
	$(NCU) --set full --target-processes all --force-overwrite -o ncu_cp_async_depth ./$(TARGET) $(DEPTH_ARGS)

profile-hw-fa2-issue: $(TARGET)
	$(NCU) --set full --target-processes all --force-overwrite -o ncu_cp_async_fa2_issue ./$(TARGET) $(FA2_ISSUE_ARGS)

profile-hw-fa2-complete: $(TARGET)
	$(NCU) --set full --target-processes all --force-overwrite -o ncu_cp_async_fa2_complete ./$(TARGET) $(FA2_COMPLETE_ARGS)

sweep-hw-depth: $(TARGET)
	for g in 1 2 4 8 12 16 24 32 48 64; do \
		./$(TARGET) --blocks=$(HW_BLOCKS) --threads=128 --iters=$(HW_ITERS) --warmup=$(HW_WARMUP) --tiles=$(HW_TILES) --mode=depth --groups=$$g --group-size=8 --active-warps=1 --active-lanes=32 --smem-slots=64 --global-slots=64 --block-hot; \
	done

clean:
	rm -f $(TARGET) $(PTX_TARGET)
	rm -f $(RUN_DIR)/$(TARGET)
	rm -f *.ptx *.ptxas
