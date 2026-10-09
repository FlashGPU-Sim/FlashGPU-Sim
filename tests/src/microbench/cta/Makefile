SRC := cta_replacement_bench.cu
TARGET_NAME ?= cta_replacement_bench_$(ARCH)

ARCH ?= sm_100a
PTX_PROFILE ?= compute_100a
include ../common.mk

BIN_DIR ?= $(STANDALONE_BIN_ROOT)/cta
TARGET ?= $(BIN_DIR)/$(TARGET_NAME)
TARGET_EXE := $(notdir $(TARGET))
CONFIG_DIR ?= ../../../../configs/SM100_B200
SIM_LATENCY ?= 1200
RUN_DIR ?= $(STANDALONE_RUN_ROOT)/cta/b200_replacement_$(SIM_LATENCY)_$(ARCH)
BLOCKS ?= 296
THREADS ?= 512
PAYLOAD_CYCLES ?= 2000
DYNAMIC_SMEM ?= 231424

.PHONY: all clean check-ptx sim-dir sim run-hw

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

sim-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/$(TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/
	sed -i -E 's/^-gpgpu_cta_replacement_latency[[:space:]]+.*/-gpgpu_cta_replacement_latency $(SIM_LATENCY)/' $(RUN_DIR)/gpgpusim.config

sim: sim-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET_EXE) \
		--blocks=$(BLOCKS) --threads=$(THREADS) \
		--payload-cycles=$(PAYLOAD_CYCLES) --dynamic-smem=$(DYNAMIC_SMEM) \
		--csv=cta_replacement.csv

run-hw: $(TARGET)
	$(TARGET) --blocks=$(BLOCKS) --threads=$(THREADS) \
		--payload-cycles=$(PAYLOAD_CYCLES) --dynamic-smem=$(DYNAMIC_SMEM) \
		--csv=cta_replacement.csv

clean:
	rm -f $(TARGET)
	rm -f $(RUN_DIR)/$(TARGET_EXE)
	rm -f *.ptx *.ptxas
