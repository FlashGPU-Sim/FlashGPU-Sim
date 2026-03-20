# Makefile for GPT-2 Trace Tests
# Compiles Triton PTX kernels into fatbins and builds test harnesses.
# Data files are decompressed from gpt2_trace_data.tar.gz on first build.

NVCC ?= nvcc
ARCH = sm_120a
CUDA_FLAGS = -lcudart -lcuda

TRACE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PTX_DIR = $(TRACE_DIR)ptx
DATA_ARCHIVE = $(TRACE_DIR)data/gpt2_trace_data.tar.gz

# GPU config (set via run_tests.sh -c or default)
GPU_CONFIG ?= SM120_RTX5090_REDUCED
CONFIGS_DIR = $(TRACE_DIR)../../../configs

# Build output goes to test/build/trace/
BUILD_DIR ?= $(TRACE_DIR)../../build/trace
BIN_DIR = $(BUILD_DIR)/bin
DATA_DIR = $(BUILD_DIR)/bin/data

# Kernel PTX names (for fatbin compilation)
KERNELS = embedding_kernel gelu_kernel flash_attn_kernel layernorm_kernel \
          residual_add_kernel linear_kernel

# Test binary names (must match *_trace_test.cu filenames)
TEST_NAMES = embedding gelu flash_attn layernorm residual_add linear

# Targets
FATBINS = $(addprefix $(BIN_DIR)/, $(addsuffix .fatbin, $(KERNELS)))
TESTS = $(addprefix $(BIN_DIR)/, $(addsuffix _trace_test, $(TEST_NAMES)))

# Data files needed by data-driven tests (all except embedding)
DATA_STAMP = $(DATA_DIR)/.extracted

.PHONY: all clean run

CONFIG_STAMP = $(BIN_DIR)/.config_$(GPU_CONFIG)

all: $(FATBINS) $(TESTS) $(DATA_STAMP) $(CONFIG_STAMP)

# Create directories
$(BIN_DIR) $(DATA_DIR):
	mkdir -p $@

# Compile PTX to fatbin
$(BIN_DIR)/%.fatbin: $(PTX_DIR)/%.ptx | $(BIN_DIR)
	$(NVCC) -fatbin -arch=$(ARCH) -o $@ $<

# Compile test harnesses
$(BIN_DIR)/%_trace_test: $(TRACE_DIR)%_trace_test.cu $(TRACE_DIR)trace_test_utils.h | $(BIN_DIR)
	$(NVCC) -arch=$(ARCH) -I$(TRACE_DIR) -o $@ $< $(CUDA_FLAGS)

# Copy GPU config to bin directory
$(CONFIG_STAMP): $(CONFIGS_DIR)/$(GPU_CONFIG)/gpgpusim.config | $(BIN_DIR)
	cp $(CONFIGS_DIR)/$(GPU_CONFIG)/gpgpusim.config $(BIN_DIR)/
	cp $(CONFIGS_DIR)/$(GPU_CONFIG)/config_*.icnt $(BIN_DIR)/ 2>/dev/null || true
	touch $@

# Decompress trace data
$(DATA_STAMP): $(DATA_ARCHIVE) | $(DATA_DIR)
	tar -xzf $(DATA_ARCHIVE) -C $(DATA_DIR)
	touch $@

# Run all tests (from the bin directory so executables find their data)
run: all
	@echo "=== Running GPT-2 Trace Tests ==="
	@failed=0; \
	for test in $(TESTS); do \
		name=$$(basename $$test); \
		echo "--- $$name ---"; \
		(cd $(BIN_DIR) && ./$$name) || failed=$$((failed + 1)); \
		echo ""; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All trace tests PASSED"; \
	else \
		echo "$$failed trace test(s) FAILED"; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)
