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

# Targets
FATBINS = $(addprefix $(BIN_DIR)/, $(addsuffix .fatbin, $(KERNELS)))

# Two test binaries: embedding (CPU reference) + data_driven (all others)
EMBEDDING_TEST = $(BIN_DIR)/embedding_trace_test
DATA_DRIVEN_TEST = $(BIN_DIR)/data_driven_trace_test

# Data-driven test names (passed as argv to data_driven_trace_test)
DATA_DRIVEN_NAMES = gelu flash_attn layernorm residual_add linear

# Data files needed by data-driven tests
DATA_STAMP = $(DATA_DIR)/.extracted
CONFIG_STAMP = $(BIN_DIR)/.config_$(GPU_CONFIG)

.PHONY: all clean run

all: $(FATBINS) $(EMBEDDING_TEST) $(DATA_DRIVEN_TEST) $(DATA_STAMP) $(CONFIG_STAMP)

# Create directories
$(BIN_DIR) $(DATA_DIR):
	mkdir -p $@

# Compile PTX to fatbin
$(BIN_DIR)/%.fatbin: $(PTX_DIR)/%.ptx | $(BIN_DIR)
	$(NVCC) -fatbin -arch=$(ARCH) -o $@ $<

# Compile embedding test (CPU reference)
$(EMBEDDING_TEST): $(TRACE_DIR)embedding_trace_test.cu $(TRACE_DIR)trace_test_utils.h | $(BIN_DIR)
	$(NVCC) -arch=$(ARCH) -I$(TRACE_DIR) -o $@ $< $(CUDA_FLAGS)

# Compile unified data-driven test
$(DATA_DRIVEN_TEST): $(TRACE_DIR)data_driven_trace_test.cu $(TRACE_DIR)trace_test_utils.h | $(BIN_DIR)
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

# Run all tests
run: all
	@echo "=== Running GPT-2 Trace Tests ==="
	@failed=0; \
	echo "--- embedding_trace_test ---"; \
	(cd $(BIN_DIR) && ./embedding_trace_test) || failed=$$((failed + 1)); \
	echo ""; \
	for name in $(DATA_DRIVEN_NAMES); do \
		echo "--- data_driven_trace_test $$name ---"; \
		(cd $(BIN_DIR) && ./data_driven_trace_test $$name) || failed=$$((failed + 1)); \
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
