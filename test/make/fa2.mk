# FlashAttention 2 split runners and analysis modes.

FA2_MK := $(lastword $(MAKEFILE_LIST))

HOPPER_FA2_DIR = $(HOPPER_SRC_DIR)/fa2
HOPPER_FA2_TEST_SOURCE = $(HOPPER_FA2_DIR)/fa2_fwd_fp16_test.cc
HOPPER_FA2_VARIANTS = h32d64_full h32d64_causal h16d128_full h16d128_causal
HOPPER_FA2_GROUPS = smoke small medium large

HOPPER_FA2_SMOKE_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/smoke_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_SMALL_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/small_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_MEDIUM_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/medium_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_LARGE_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/large_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_OBJECTS = $(HOPPER_FA2_SMOKE_OBJECTS) $(HOPPER_FA2_SMALL_OBJECTS) $(HOPPER_FA2_MEDIUM_OBJECTS) $(HOPPER_FA2_LARGE_OBJECTS)

HOPPER_FA2_BREAKDOWN_MODES = baseline skip_cp_async skip_mma skip_softmax fma_softmax only_mma only_cp_async only_softmax nothing
HOPPER_FA2_BREAKDOWN_OBJECTS = $(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(OBJ_DIR)/hopper/fa2/breakdown_$(mode)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_SCALING_MODES = baseline nothing only_cp_async only_softmax only_mma softmax_mma
HOPPER_FA2_SCALING_OBJECTS = $(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(OBJ_DIR)/hopper/fa2/scaling_$(mode)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_CONCURRENCY_MODES = only_cp_async only_cp_async_bhhalf qk_softmax_pv_only qk_pv_only
HOPPER_FA2_CONCURRENCY_OBJECTS = $(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(OBJ_DIR)/hopper/fa2/concurrency_$(mode)/fa2_fwd_fp16_test.cu.o)

HOPPER_FA2_COMMON_FLAGS = --ftemplate-backtrace-limit=0 -O3 \
                          --use_fast_math \
                          -lineinfo --expt-relaxed-constexpr \
                          --expt-extended-lambda \
                          -DNDEBUG \
                          -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
                          -I$(HOPPER_FA2_DIR) \
                          -I$(HOPPER_FA3_DIR)/flash-attention/csrc/flash_attn/src \
                          -I$(HOPPER_FA3_DIR)/flash-attention/csrc/cutlass/include
HOPPER_FA2_FLAGS = $(HOPPER_FA2_COMMON_FLAGS)

HOPPER_FA2_SMOKE_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_smoke_$(variant)_tests)
HOPPER_FA2_SMALL_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_small_$(variant)_tests)
HOPPER_FA2_MEDIUM_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_medium_$(variant)_tests)
HOPPER_FA2_LARGE_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_large_$(variant)_tests)
HOPPER_FA2_BREAKDOWN_TARGETS = $(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(BIN_DIR)/hopper/run_fa2_breakdown_$(mode)_tests)
HOPPER_FA2_SCALING_TARGETS = $(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(BIN_DIR)/hopper/run_fa2_scaling_$(mode)_tests)
HOPPER_FA2_CONCURRENCY_TARGETS = $(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(BIN_DIR)/hopper/run_fa2_concurrency_$(mode)_tests)

.PHONY: fa2-smoke fa2-small fa2-medium fa2-large \
fa2-breakdown fa2-scaling fa2-concurrency

fa2-smoke: setup-gtest $(HOPPER_FA2_SMOKE_TARGETS)

fa2-small: setup-gtest $(HOPPER_FA2_SMALL_TARGETS)

fa2-medium: setup-gtest $(HOPPER_FA2_MEDIUM_TARGETS)

fa2-large: setup-gtest $(HOPPER_FA2_LARGE_TARGETS)

fa2-breakdown: setup-gtest $(HOPPER_FA2_BREAKDOWN_TARGETS)

fa2-scaling: setup-gtest $(HOPPER_FA2_SCALING_TARGETS)

fa2-concurrency: setup-gtest $(HOPPER_FA2_CONCURRENCY_TARGETS)

define REGISTER_FA2_BUILD_TARGET
.PHONY: fa2-$(1)-$(2)
fa2-$(1)-$(2): setup-gtest $(BIN_DIR)/hopper/run_fa2_$(1)_$(2)_tests
endef
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,breakdown,$(mode))))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,scaling,$(mode))))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,concurrency,$(mode))))

# Each standard object instantiates one D/full-causal kernel family.
HOPPER_FA2_H32D64_FULL_FLAGS = -DFA2_PREFILL_ENABLE_H32D64_FULL
HOPPER_FA2_H32D64_CAUSAL_FLAGS = -DFA2_PREFILL_ENABLE_H32D64_CAUSAL
HOPPER_FA2_H16D128_FULL_FLAGS = -DFA2_PREFILL_ENABLE_H16D128_FULL
HOPPER_FA2_H16D128_CAUSAL_FLAGS = -DFA2_PREFILL_ENABLE_H16D128_CAUSAL

$(OBJ_DIR)/hopper/fa2/smoke_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)

$(HOPPER_FA2_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(FA2_MK) $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_BREAKDOWN_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_BREAKDOWN $(HOPPER_FA2_H16D128_FULL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/breakdown_baseline/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS)
$(OBJ_DIR)/hopper/fa2/breakdown_skip_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC
$(OBJ_DIR)/hopper/fa2/breakdown_skip_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_MMA
$(OBJ_DIR)/hopper/fa2/breakdown_skip_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_fma_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_FMA_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA
$(OBJ_DIR)/hopper/fa2/breakdown_nothing/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_SOFTMAX

$(HOPPER_FA2_BREAKDOWN_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(FA2_MK) $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_SCALING_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SCALING $(HOPPER_FA2_H16D128_FULL_FLAGS) $(HOPPER_FA2_H16D128_CAUSAL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/scaling_baseline/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS)
$(OBJ_DIR)/hopper/fa2/scaling_nothing/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_only_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS
$(OBJ_DIR)/hopper/fa2/scaling_only_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_softmax_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC

$(HOPPER_FA2_SCALING_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(FA2_MK) $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_CONCURRENCY_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_CONCURRENCY $(HOPPER_FA2_H16D128_FULL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/concurrency_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/concurrency_only_cp_async_bhhalf/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_PREFILL_CONCURRENCY_BH_HALF -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX -DFA2_FWD_CTA_PROFILE
$(OBJ_DIR)/hopper/fa2/concurrency_qk_softmax_pv_only/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC
$(OBJ_DIR)/hopper/fa2/concurrency_qk_pv_only/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX

$(HOPPER_FA2_CONCURRENCY_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(FA2_MK) $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(BIN_DIR)/hopper/run_fa2_%_tests: \
$(OBJ_DIR)/hopper/fa2/%/fa2_fwd_fp16_test.cu.o $(OBJ_DIR)/gtest_main.a \
$(TOP_MAKEFILE) $(FA2_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(FA2_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
