BW_SRC := global_load_bw_bench.cu
BW_TARGET_NAME ?= global_load_bw_bench_cuda129
LAT_SRC := l2_latency_bench.cu
LAT_TARGET_NAME ?= l2_latency_bench_cuda128_$(ARCH)
INTERLEAVE_SRC := l2_hbm_interleave_bench.cu
INTERLEAVE_TARGET_NAME ?= l2_hbm_interleave_bench_cuda128_$(ARCH)
L2PART_SRC := l2_partition_latency_probe.cu
L2PART_TARGET_NAME ?= l2_partition_latency_probe_cuda128_$(ARCH)

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
include ../common.mk
CUDA_DRIVER_LIBS ?= -lcuda
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/memory
BW_TARGET ?= $(BIN_DIR)/$(BW_TARGET_NAME)
LAT_TARGET ?= $(BIN_DIR)/$(LAT_TARGET_NAME)
INTERLEAVE_TARGET ?= $(BIN_DIR)/$(INTERLEAVE_TARGET_NAME)
L2PART_TARGET ?= $(BIN_DIR)/$(L2PART_TARGET_NAME)
TARGET ?= $(BW_TARGET)
TARGET_EXE := $(notdir $(BW_TARGET))

CONFIG_DIR ?= ../../../../configs/SM120_RTX5090
RUN_DIR ?= $(STANDALONE_RUN_ROOT)/memory/gem5_global_load_bw_smoke_cuda129

.PHONY: all clean check-ptx gem5-dir gem5-smoke run-hw interleave l2part run-interleave-quick run-l2part-quick
TARGETS := $(BW_TARGET) $(LAT_TARGET) $(INTERLEAVE_TARGET) $(L2PART_TARGET)

all: $(TARGETS)

$(BW_TARGET): $(BW_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LAT_TARGET): $(LAT_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(INTERLEAVE_TARGET): $(INTERLEAVE_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_DRIVER_LIBS)

$(L2PART_TARGET): $(L2PART_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

interleave: $(INTERLEAVE_TARGET)

l2part: $(L2PART_TARGET)

check-ptx: $(TARGETS)
	$(CUOBJDUMP) -lptx $(BW_TARGET)
	$(CUOBJDUMP) -lptx $(LAT_TARGET)
	$(CUOBJDUMP) -lptx $(INTERLEAVE_TARGET)
	$(CUOBJDUMP) -lptx $(L2PART_TARGET)

gem5-dir: $(BW_TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(BW_TARGET) $(RUN_DIR)/$(TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET_EXE) --blocks=1 --threads=256 --iters=1 --warmup=0 --tiles=1 --block-hot

run-hw: $(BW_TARGET)
	$(BW_TARGET) --blocks=528 --threads=256 --iters=4096 --warmup=256 --tiles=528 --repeat=1 --block-hot

run-l2-latency: $(LAT_TARGET)
	$(LAT_TARGET) --case=l2-hit --bytes=65536 --steps=4096 --samples=2048 --warmup-steps=65536 --cache-op=cg
	$(LAT_TARGET) --case=miss --bytes=268435456 --steps=8192 --samples=512 --warmup-steps=0 --cache-op=cg

run-interleave-quick: $(INTERLEAVE_TARGET)
	$(INTERLEAVE_TARGET) --op=ldg --pattern=stream --blocks=132 --threads=256 --iters=8 --warmup=0 --data-bytes=256M --tile-bytes=48K --event
	$(INTERLEAVE_TARGET) --op=cp_async --pattern=stream --blocks=132 --threads=256 --iters=8 --warmup=0 --data-bytes=256M --tile-bytes=48K --event

run-l2part-quick: $(L2PART_TARGET)
	$(L2PART_TARGET) --blocks=4 --threads=32 --repeats=32 --warmup=8 --bit-first=7 --bit-last=12 --dense-offsets=8 --data-bytes=4M --csv=/tmp/l2_partition_latency_probe_smoke.csv

clean:
	rm -f $(TARGETS)
	rm -f $(RUN_DIR)/$(TARGET_EXE)
	rm -f *.ptx *.ptxas
