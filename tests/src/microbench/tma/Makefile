SRC := tma_completion_latency_bench.cu
TARGET_NAME ?= tma_completion_latency_bench_$(ARCH)
DESC_SRC := tma_descriptor_setup_bench.cu
DESC_TARGET_NAME ?= tma_descriptor_setup_bench_$(ARCH)
FA3_M3_SRC := tma_fa3_m3_bench.cu
FA3_M3_TARGET_NAME ?= tma_fa3_m3_bench_$(ARCH)
THROUGHPUT_SRC := tma_throughput_bench.cu
THROUGHPUT_TARGET_NAME ?= tma_throughput_bench_$(ARCH)

ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
include ../common.mk
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/tma
TARGET ?= $(BIN_DIR)/$(TARGET_NAME)
DESC_TARGET ?= $(BIN_DIR)/$(DESC_TARGET_NAME)
FA3_M3_TARGET ?= $(BIN_DIR)/$(FA3_M3_TARGET_NAME)
THROUGHPUT_TARGET ?= $(BIN_DIR)/$(THROUGHPUT_TARGET_NAME)
TARGET_EXE := $(notdir $(TARGET))
DESC_TARGET_EXE := $(notdir $(DESC_TARGET))
FA3_M3_TARGET_EXE := $(notdir $(FA3_M3_TARGET))
THROUGHPUT_TARGET_EXE := $(notdir $(THROUGHPUT_TARGET))

CONFIG_DIR ?= ../../../../configs/SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA3_TMA128_OOB_READ
RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/gem5_pair48_smoke_$(ARCH)
DESC_RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/gem5_descriptor_smoke_$(ARCH)
FA3_M3_RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/fa3_m3_smoke_$(ARCH)
THROUGHPUT_CONFIG_DIR ?= ../../../../configs/SM100_B200
THROUGHPUT_RUN_DIR ?= $(STANDALONE_RUN_ROOT)/tma/b200_dram_peak_$(ARCH)
THROUGHPUT_CLOCK_DOMAINS := 1965:1965:1155:3996
THROUGHPUT_TMA_MAX_INFLIGHT ?= 2648
BW_BLOCKS ?= 132
BW_ITERS ?= 64
BW_WARMUP ?= 4
BW_TILES ?= 8192
BW_REPEAT ?= 1

.PHONY: all throughput clean check-ptx desc-check-ptx fa3-m3-check-ptx throughput-check-ptx gem5-dir gem5-smoke gem5-dram-bw desc-gem5-dir desc-gem5-smoke fa3-m3-gem5-dir fa3-m3-gem5-smoke throughput-sim-dir throughput-sim run-hw run-hw-dram-bw desc-run-hw fa3-m3-run-hw

all: $(TARGET) $(DESC_TARGET) $(FA3_M3_TARGET)

throughput: $(THROUGHPUT_TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(DESC_TARGET): $(DESC_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(FA3_M3_TARGET): $(FA3_M3_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

$(THROUGHPUT_TARGET): $(THROUGHPUT_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

desc-check-ptx: $(DESC_TARGET)
	$(CUOBJDUMP) -lptx $(DESC_TARGET)

fa3-m3-check-ptx: $(FA3_M3_TARGET)
	$(CUOBJDUMP) -lptx $(FA3_M3_TARGET)

throughput-check-ptx: $(THROUGHPUT_TARGET)
	$(CUOBJDUMP) -lptx $(THROUGHPUT_TARGET)

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

fa3-m3-gem5-dir: $(FA3_M3_TARGET)
	mkdir -p $(FA3_M3_RUN_DIR)
	rm -f $(FA3_M3_RUN_DIR)/*.ptx $(FA3_M3_RUN_DIR)/*.ptxas $(FA3_M3_RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(FA3_M3_TARGET) $(FA3_M3_RUN_DIR)/$(FA3_M3_TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(FA3_M3_RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(FA3_M3_RUN_DIR)/

fa3-m3-gem5-smoke: fa3-m3-gem5-dir
	cd $(FA3_M3_RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(FA3_M3_TARGET_EXE) --blocks=1 --iters=1 --warmup=0 --batch=1 --heads=1 --seqlen=512 --m-block=3 --sweep-hb=0 --csv=fa3_m3_smoke.csv

throughput-sim-dir: $(THROUGHPUT_TARGET)
	mkdir -p $(THROUGHPUT_RUN_DIR)
	rm -f $(THROUGHPUT_RUN_DIR)/*.ptx $(THROUGHPUT_RUN_DIR)/*.ptxas $(THROUGHPUT_RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(THROUGHPUT_TARGET) $(THROUGHPUT_RUN_DIR)/$(THROUGHPUT_TARGET_EXE)
	cp $(THROUGHPUT_CONFIG_DIR)/gpgpusim.config $(THROUGHPUT_RUN_DIR)/
	cp $(THROUGHPUT_CONFIG_DIR)/config_ampere_islip.icnt $(THROUGHPUT_RUN_DIR)/
	sed -i -E 's/^-gpgpu_clock_domains[[:space:]]+.*/-gpgpu_clock_domains $(THROUGHPUT_CLOCK_DOMAINS)/' $(THROUGHPUT_RUN_DIR)/gpgpusim.config
	sed -i -E 's/^-gpgpu_tma_max_inflight[[:space:]]+.*/-gpgpu_tma_max_inflight $(THROUGHPUT_TMA_MAX_INFLIGHT)/' $(THROUGHPUT_RUN_DIR)/gpgpusim.config

throughput-sim: throughput-sim-dir
	cd $(THROUGHPUT_RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(THROUGHPUT_TARGET_EXE)

run-hw: $(TARGET)
	$(TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=pair48 --issue=llama --stride

run-hw-dram-bw: $(TARGET)
	$(TARGET) --blocks=$(BW_BLOCKS) --iters=$(BW_ITERS) --warmup=$(BW_WARMUP) --tiles=$(BW_TILES) --repeat=$(BW_REPEAT) --case=pair48 --issue=llama --stride

desc-run-hw: $(DESC_TARGET)
	$(DESC_TARGET) --blocks=128 --iters=4096 --warmup=256 --tiles=4096 --case=full --stride

fa3-m3-run-hw: $(FA3_M3_TARGET)
	$(FA3_M3_TARGET) --blocks=128 --iters=8 --warmup=0 --batch=64 --heads=16 --seqlen=512 --head-dim=128 --block-m=128 --block-n=128 --m-block=3 --group-nblocks=2 --causal=1 --sweep-hb=1 --explicit-cluster=1 --load=kv --csv=fa3_m3_h16d128_causal_b64s512.csv

clean:
	rm -f $(TARGET) $(DESC_TARGET) $(FA3_M3_TARGET) $(THROUGHPUT_TARGET)
	rm -f $(RUN_DIR)/$(TARGET_EXE) $(DESC_RUN_DIR)/$(DESC_TARGET_EXE) $(FA3_M3_RUN_DIR)/$(FA3_M3_TARGET_EXE) $(THROUGHPUT_RUN_DIR)/$(THROUGHPUT_TARGET_EXE)
	rm -f *.ptx *.ptxas
