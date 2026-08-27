SRC := footprint_bench.cu
PARALLEL_SRC := parallel_pc_bench.cu
ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
include ../common.mk

BIN_DIR ?= $(STANDALONE_BIN_ROOT)/instruction_cache
RESULT_DIR ?= $(STANDALONE_RUN_ROOT)/instruction_cache
QUICK_STEPS ?= 64 352 1024 4096 5120 10240 11520
SWEEP_STEPS ?= 64 128 256 320 336 352 368 512 1024 2048 3584 3840 3968 4032 4096 4160 4352 4608 5120 8192 9216 9728 10240 10496 10624 10752 10880 11008 11264 11520 12288 16384
QUICK_TARGETS := $(addprefix $(BIN_DIR)/icc_gcc_footprint_,$(QUICK_STEPS))
SWEEP_TARGETS := $(addprefix $(BIN_DIR)/icc_gcc_footprint_,$(SWEEP_STEPS))
PARALLEL_TARGET := $(BIN_DIR)/icc_gcc_parallel_pc

.PHONY: all sweep parallel clean check test collect-cold collect-warm collect-skip collect-parallel

all: $(QUICK_TARGETS)

sweep: $(SWEEP_TARGETS)

parallel: $(PARALLEL_TARGET)

$(BIN_DIR)/icc_gcc_footprint_%: $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DFOOTPRINT_STEPS=$* -o $@ $<

$(PARALLEL_TARGET): $(PARALLEL_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DSTREAM_BODY_STEPS=16 -o $@ $<

check: all
	@for binary in $(QUICK_TARGETS); do \
		$$binary --repetitions=1 || exit $$?; \
	done

test:
	python3 -m unittest -v test_collect_ncu.py

collect-cold: sweep
	python3 collect_ncu.py --mode=cold --bin-dir=$(BIN_DIR) \
		--output=$(RESULT_DIR)/icc_gcc_cold.csv --ncu=$(NCU) \
		--cuobjdump=$(CUOBJDUMP)

collect-warm: sweep
	python3 collect_ncu.py --mode=warm --bin-dir=$(BIN_DIR) \
		--output=$(RESULT_DIR)/icc_gcc_warm.csv --ncu=$(NCU) \
		--cuobjdump=$(CUOBJDUMP)

collect-skip: sweep
	python3 collect_ncu.py --mode=skip --bin-dir=$(BIN_DIR) \
		--output=$(RESULT_DIR)/icc_gcc_skip.csv --ncu=$(NCU) \
		--cuobjdump=$(CUOBJDUMP)

collect-parallel: parallel
	python3 collect_parallel_ncu.py --binary=$(PARALLEL_TARGET) \
		--output=$(RESULT_DIR)/icc_parallel_pc.csv --ncu=$(NCU)

clean:
	rm -f $(SWEEP_TARGETS)
	rm -f $(PARALLEL_TARGET)
	rm -rf $(RESULT_DIR)
