SRC := tma_completion_latency_bench.cu
TARGET_NAME := tma_completion_latency_bench_shared_cuda129
DESC_SRC := tma_descriptor_setup_bench.cu
DESC_TARGET_NAME := tma_descriptor_setup_bench_cuda129

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
include ../common.mk
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/tma
TARGET ?= $(BIN_DIR)/$(TARGET_NAME)
DESC_TARGET ?= $(BIN_DIR)/$(DESC_TARGET_NAME)
TARGET_EXE := $(notdir $(TARGET))
DESC_TARGET_EXE := $(notdir $(DESC_TARGET))

CONFIG_DIR ?= ../../../../configs/SM120_RTX5090
RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/gem5_pair48_smoke_cuda129
DESC_RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/gem5_descriptor_smoke_cuda129
BW_BLOCKS ?= 132
BW_ITERS ?= 64
BW_WARMUP ?= 4
BW_TILES ?= 8192
BW_REPEAT ?= 1

.PHONY: all clean check-ptx desc-check-ptx gem5-dir gem5-smoke gem5-dram-bw desc-gem5-dir desc-gem5-smoke run-hw run-hw-dram-bw desc-run-hw

all: $(TARGET) $(DESC_TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(DESC_TARGET): $(DESC_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

desc-check-ptx: $(DESC_TARGET)
	$(CUOBJDUMP) -lptx $(DESC_TARGET)

gem5-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/$(TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET_EXE) --blocks=1 --iters=1 --warmup=0 --tiles=64 --case=pair48 --issue=llama --stride

gem5-dram-bw: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET_EXE) --blocks=$(BW_BLOCKS) --iters=$(BW_ITERS) --warmup=$(BW_WARMUP) --tiles=$(BW_TILES) --repeat=$(BW_REPEAT) --case=pair48 --issue=llama --stride

desc-gem5-dir: $(DESC_TARGET)
	mkdir -p $(DESC_RUN_DIR)
	rm -f $(DESC_RUN_DIR)/*.ptx $(DESC_RUN_DIR)/*.ptxas $(DESC_RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(DESC_TARGET) $(DESC_RUN_DIR)/$(DESC_TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(DESC_RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(DESC_RUN_DIR)/

desc-gem5-smoke: desc-gem5-dir
	cd $(DESC_RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(DESC_TARGET_EXE) --blocks=1 --iters=1 --warmup=0 --tiles=64 --case=full --stride

run-hw: $(TARGET)
	$(TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=pair48 --issue=llama --stride

run-hw-dram-bw: $(TARGET)
	$(TARGET) --blocks=$(BW_BLOCKS) --iters=$(BW_ITERS) --warmup=$(BW_WARMUP) --tiles=$(BW_TILES) --repeat=$(BW_REPEAT) --case=pair48 --issue=llama --stride

desc-run-hw: $(DESC_TARGET)
	$(DESC_TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=full --stride

clean:
	rm -f $(TARGET) $(DESC_TARGET)
	rm -f $(RUN_DIR)/$(TARGET_EXE) $(DESC_RUN_DIR)/$(DESC_TARGET_EXE)
	rm -f *.ptx *.ptxas
