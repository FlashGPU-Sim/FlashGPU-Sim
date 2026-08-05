# Trace test runner metadata. The existing local Makefile remains responsible
# for generating and executing trace binaries.

TEST_GROUP_PROFILES_sm120_trace := gpt2
TEST_GROUP_BUILD_TARGET_sm120_trace_gpt2 := trace-sm120-gpt2
TEST_GROUP_BINARY_GROUP_sm120_trace_gpt2 := none
TEST_GROUP_EXECUTOR_sm120_trace_gpt2 := trace
TEST_GROUP_FILTER_sm120_trace_gpt2 := *

.PHONY: trace-sm120-gpt2
trace-sm120-gpt2:
	$(MAKE) -C $(TEST_SRC_DIR)/trace \
		ARCH=$(ARCH_NVCC_TARGET_sm120) \
		GPU_CONFIG=$(or $(GPU_CONFIG),$(ARCH_DEFAULT_CONFIG_sm120)) all
