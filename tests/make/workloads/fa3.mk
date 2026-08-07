# FlashAttention 3 sources, analysis modes, preparation, and binaries.

FA3_MK := $(lastword $(MAKEFILE_LIST))

FA3_DIR = $(TEST_SRC_DIR)/fa3

# Runner profiles are properties of the FA3 build recipe. Architecture
# membership and the source inventory remain in arch/sm90.toml.
TEST_GROUP_PROFILES_sm90_fa3 := \
	smoke packgqa small medium large breakdown scaling concurrency
TEST_GROUP_BUILD_TARGET_sm90_fa3_smoke := fa3-standard
TEST_GROUP_BINARY_GROUP_sm90_fa3_smoke := fa3-standard
TEST_GROUP_EXECUTOR_sm90_fa3_smoke := gtest-single
TEST_GROUP_FILTER_sm90_fa3_smoke := Fa3PrefillFp16SmokeTest.*:Fa3PrefillFp16BackwardSmokeTest.*:Fa3FwdHdim128Fp16IntegrationTest.*
TEST_GROUP_BUILD_TARGET_sm90_fa3_packgqa := fa3-packgqa
TEST_GROUP_BINARY_GROUP_sm90_fa3_packgqa := fa3-packgqa
TEST_GROUP_EXECUTOR_sm90_fa3_packgqa := gtest-multi
TEST_GROUP_FILTER_sm90_fa3_packgqa := Fa3FwdPackGqaFp16IntegrationTest.*
TEST_GROUP_BUILD_TARGET_sm90_fa3_small := fa3-standard
TEST_GROUP_BINARY_GROUP_sm90_fa3_small := fa3-standard
TEST_GROUP_EXECUTOR_sm90_fa3_small := gtest-single
TEST_GROUP_FILTER_sm90_fa3_small := Fa3PrefillFp16SmallTest.*:Fa3PrefillFp16BackwardSmallTest.*
TEST_GROUP_BUILD_TARGET_sm90_fa3_medium := fa3-standard
TEST_GROUP_BINARY_GROUP_sm90_fa3_medium := fa3-standard
TEST_GROUP_EXECUTOR_sm90_fa3_medium := gtest-single
TEST_GROUP_FILTER_sm90_fa3_medium := Fa3PrefillFp16MediumTest.*:Fa3PrefillFp16BackwardMediumTest.*
TEST_GROUP_BUILD_TARGET_sm90_fa3_large := fa3-standard
TEST_GROUP_BINARY_GROUP_sm90_fa3_large := fa3-standard
TEST_GROUP_EXECUTOR_sm90_fa3_large := gtest-single
TEST_GROUP_FILTER_sm90_fa3_large := Fa3PrefillFp16IntegrationTest.*:Fa3PrefillFp16BackwardIntegrationTest.*
FA3_STANDARD_NAMES = \
	fa3_fwd_d64_noncausal_test \
	fa3_fwd_d64_causal_test \
	fa3_fwd_d128_noncausal_test \
	fa3_fwd_d128_causal_test \
	fa3_bwd_d64_noncausal_test \
	fa3_bwd_d64_causal_test \
	fa3_bwd_d128_noncausal_test \
	fa3_bwd_d128_causal_test
FA3_STANDARD_FWD_D64_NONCAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_fwd_d64_noncausal_test.cu.o
FA3_STANDARD_FWD_D64_CAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_fwd_d64_causal_test.cu.o
FA3_STANDARD_FWD_D128_NONCAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_fwd_d128_noncausal_test.cu.o
FA3_STANDARD_FWD_D128_CAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_fwd_d128_causal_test.cu.o
FA3_STANDARD_BWD_D64_NONCAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_bwd_d64_noncausal_test.cu.o
FA3_STANDARD_BWD_D64_CAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_bwd_d64_causal_test.cu.o
FA3_STANDARD_BWD_D128_NONCAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_bwd_d128_noncausal_test.cu.o
FA3_STANDARD_BWD_D128_CAUSAL_OBJECT = $(OBJ_DIR)/sm90/fa3/standard/fa3_bwd_d128_causal_test.cu.o
FA3_STANDARD_SOURCES = $(addprefix $(FA3_DIR)/,$(addsuffix .cu,$(FA3_STANDARD_NAMES)))
FA3_STANDARD_OBJECTS = $(addprefix $(OBJ_DIR)/sm90/fa3/standard/,$(addsuffix .cu.o,$(FA3_STANDARD_NAMES)))
FA3_PACKGQA_SOURCE = $(FA3_DIR)/fa3_fwd_packgqa_test.cu
FA3_PACKGQA_DEFAULT_OBJECT = $(OBJ_DIR)/sm90/fa3/packgqa/default/fa3_fwd_packgqa_test.cu.o
FA3_PACKGQA_NOINC_OBJECT = $(OBJ_DIR)/sm90/fa3/packgqa/noinc/fa3_fwd_packgqa_test.cu.o
FA3_PACKGQA_DEFAULT_TARGET = $(BIN_DIR)/sm90/fa3/packgqa_default_tests
FA3_PACKGQA_NOINC_TARGET = $(BIN_DIR)/sm90/fa3/packgqa_noinc_tests
FA3_PACKGQA_TARGETS = \
	$(FA3_PACKGQA_DEFAULT_TARGET) \
	$(FA3_PACKGQA_NOINC_TARGET)

FA3_MODES = \
	baseline softmax_only qk_only pv_only qk_pv_only \
	tma_only tma_only_m3_only tma_only_first_round_bh \
	tma_only_first_round_bh_m3_only baseline_noprofile \
	softmax_only_noprofile qk_pv_only_noprofile base_no_tma \
	softmax_only_no_tma qk_pv_only_no_tma base_no_tma_noprofile \
	softmax_only_no_tma_noprofile qk_pv_only_no_tma_noprofile \
	qk_pv_only_no_tma_timeline qk_pv_only_no_tma_reg_timeline \
	sync_only_no_tma sync_only_no_tma_noprofile \
	qk_pv_only_no_tma_extended qk_pv_only_no_tma_extended_noprofile \
	qk_pv_only_no_tma_extended_reg_timeline

TEST_GROUP_PROFILE_MODES_sm90_fa3_breakdown = $(FA3_MODES) all
TEST_GROUP_PROFILE_MODES_sm90_fa3_scaling = $(FA3_MODES) all
TEST_GROUP_PROFILE_MODES_sm90_fa3_concurrency = $(FA3_MODES) all

FA3_PROFILE_CASES_breakdown := H1D128FullB1S4096
FA3_PROFILE_CASES_scaling := \
	H1D128FullB1S128 H1D128FullB1S256 H1D128FullB1S512 \
	H1D128FullB1S1024 H1D128FullB1S2048 H1D128FullB1S4096 \
	H1D128FullB1S8192
FA3_PROFILE_CASES_concurrency := \
	H16D128FullB64S512 H4D128FullB64S512 H16D128FullB16S512 H1D128FullB1S512 \
	H16D128FullB32S1024 H4D128FullB32S1024 H16D128FullB8S1024 H1D128FullB1S1024 \
	H16D128FullB16S2048 H4D128FullB16S2048 H16D128FullB4S2048 H1D128FullB1S2048 \
	H16D128FullB8S4096 H4D128FullB8S4096 H16D128FullB2S4096 H1D128FullB1S4096 \
	H16D128FullB4S8192 H4D128FullB4S8192 H16D128FullB1S8192 H1D128FullB1S8192

define REGISTER_SM90_FA3_MODE
TEST_GROUP_BUILD_TARGET_sm90_fa3_$(1)_$(2) = fa3-mode-$(2)
TEST_GROUP_BINARY_GROUP_sm90_fa3_$(1)_$(2) = fa3-mode-$(2)
TEST_GROUP_EXECUTOR_sm90_fa3_$(1)_$(2) := fa3-profile
TEST_GROUP_FILTER_sm90_fa3_$(1)_$(2) := Fa3H1D128ProfileTest.SelectedD128FullCases
TEST_GROUP_CASES_sm90_fa3_$(1)_$(2) := $(FA3_PROFILE_CASES_$(1))
endef
FA3_MODE_OBJECT = $(OBJ_DIR)/sm90/fa3/modes/$(1)/fa3_fwd_h1d128_profile_test.cu.o
FA3_MODE_OBJECTS = $(foreach mode,$(FA3_MODES),$(call FA3_MODE_OBJECT,$(mode)))

FA3_PREPARE_SCRIPT = $(FA3_DIR)/prepare_flash_attention.sh
FA3_PATCHES = $(wildcard $(FA3_DIR)/patches/*.patch)
FA3_PREPARED_STAMP = $(FA3_DIR)/flash-attention/.gpgpusim-prepared
FA3_GENERATED_HEADERS = \
	$(FA3_DIR)/flash-attention/hopper/flash_profile.h \
	$(FA3_DIR)/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
FA3_PROFILE_HEADERS = \
	$(FA3_DIR)/fa3_fwd_hdim128_fp16_case.cuh \
	$(FA3_GENERATED_HEADERS)

FA3_COMMON_FLAGS = --ftemplate-backtrace-limit=0 -O3 \
                          --use_fast_math \
                          -lineinfo --expt-relaxed-constexpr \
                          --expt-extended-lambda \
                          -DCUTLASS_ENABLE_GDC_FOR_SM90 \
                          -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
                          -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
                          -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
                          -DNDEBUG \
                          -I$(FA3_DIR)/flash-attention/hopper \
                          -I$(FA3_DIR)/flash-attention/csrc/cutlass/include
FA3_EXTENDED_FLAGS = $(FA3_COMMON_FLAGS) -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED

FA3_STANDARD_TARGET = $(BIN_DIR)/sm90/fa3/standard_tests
FA3_MODE_TARGET = $(BIN_DIR)/sm90/fa3/$(1)_tests
FA3_MODE_TARGETS = $(foreach mode,$(FA3_MODES),$(call FA3_MODE_TARGET,$(mode)))

.PHONY: prepare-fa3-flash-attention fa3-standard fa3-packgqa fa3-modes

prepare-fa3-flash-attention: $(FA3_PREPARED_STAMP)

$(FA3_PREPARED_STAMP): $(FA3_PREPARE_SCRIPT) $(FA3_PATCHES)
	$(FA3_PREPARE_SCRIPT)
	@touch $@

$(FA3_GENERATED_HEADERS): $(FA3_PREPARED_STAMP)
	@test -f $@ || { echo "Prepared FlashAttention header is missing: $@" >&2; exit 1; }

fa3-standard: setup-gtest $(FA3_STANDARD_TARGET)

fa3-packgqa: setup-gtest $(FA3_PACKGQA_TARGETS)

# Analysis groups reuse these mode binaries with different case lists.
fa3-modes: setup-gtest $(FA3_MODE_TARGETS)

define REGISTER_FA3_BUILD_TARGET
.PHONY: fa3-mode-$(1)
fa3-mode-$(1): setup-gtest $(call FA3_MODE_TARGET,$(1))
endef
$(foreach mode,$(FA3_MODES),$(eval $(call REGISTER_FA3_BUILD_TARGET,$(mode))))

$(FA3_STANDARD_OBJECTS): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_EXTENDED_FLAGS)
$(OBJ_DIR)/sm90/fa3/standard/%.cu.o: $(FA3_DIR)/%.cu \
$(FA3_DIR)/fa3_fwd_hdim128_fp16_test.cu \
$(FA3_DIR)/fa3_reference.cuh $(TEST_HEADERS) \
$(TOP_MAKEFILE) $(FA3_MK) $(FA3_PREPARED_STAMP) | $(SM90_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(SM90_NVCCFLAGS) $(TEST_GROUP_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Each FA3 specialization is individually memory-heavy. Keep the chain serial
# even when the caller uses make -j so aggregate compiler RSS stays bounded.
$(FA3_STANDARD_FWD_D64_CAUSAL_OBJECT): | $(FA3_STANDARD_FWD_D64_NONCAUSAL_OBJECT)
$(FA3_STANDARD_FWD_D128_NONCAUSAL_OBJECT): | $(FA3_STANDARD_FWD_D64_CAUSAL_OBJECT)
$(FA3_STANDARD_FWD_D128_CAUSAL_OBJECT): | $(FA3_STANDARD_FWD_D128_NONCAUSAL_OBJECT)
$(FA3_STANDARD_BWD_D64_NONCAUSAL_OBJECT): | $(FA3_STANDARD_FWD_D128_CAUSAL_OBJECT)
$(FA3_STANDARD_BWD_D64_CAUSAL_OBJECT): | $(FA3_STANDARD_BWD_D64_NONCAUSAL_OBJECT)
$(FA3_STANDARD_BWD_D128_NONCAUSAL_OBJECT): | $(FA3_STANDARD_BWD_D64_CAUSAL_OBJECT)
$(FA3_STANDARD_BWD_D128_CAUSAL_OBJECT): | $(FA3_STANDARD_BWD_D128_NONCAUSAL_OBJECT)

$(FA3_PACKGQA_DEFAULT_OBJECT): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_EXTENDED_FLAGS)
$(FA3_PACKGQA_NOINC_OBJECT): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_EXTENDED_FLAGS) -DFLASH_FWD_PACKGQA_CPASYNC_NOINC
$(FA3_PACKGQA_DEFAULT_OBJECT) $(FA3_PACKGQA_NOINC_OBJECT): \
$(FA3_PACKGQA_SOURCE) $(FA3_DIR)/fa3_fwd_packgqa_case.cuh \
$(FA3_DIR)/fa3_fwd_hdim128_fp16_case.cuh $(TEST_HEADERS) \
$(TOP_MAKEFILE) $(FA3_MK) $(FA3_PREPARED_STAMP) | $(SM90_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(SM90_NVCCFLAGS) $(TEST_GROUP_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $(FA3_PACKGQA_SOURCE) -o $@

# The two variants instantiate the same memory-heavy FA3 specialization.
$(FA3_PACKGQA_NOINC_OBJECT): | $(FA3_PACKGQA_DEFAULT_OBJECT)

FA3_MODE_PROFILE_FLAGS = $(FA3_EXTENDED_FLAGS) -DFLASH_FWD_ENABLE_PROFILE_CLOCK -DFLASH_FWD_PROFILE_CLOCK_NON_ATOMIC -DFLASH_FWD_PROFILE_GLOBALTIMER -DFLASH_FWD_PROFILE_TASK_GLOBALTIMER
FA3_MODE_NOPROFILE_FLAGS = $(FA3_EXTENDED_FLAGS)
FA3_MODE_EXTENDED_PROFILE_FLAGS = $(FA3_EXTENDED_FLAGS) -DFLASH_FWD_ENABLE_PROFILE_CLOCK -DFLASH_FWD_PROFILE_CLOCK_NON_ATOMIC
FA3_MODE_EXTENDED_NOPROFILE_FLAGS = $(FA3_EXTENDED_FLAGS)
$(call FA3_MODE_OBJECT,baseline): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS)
$(call FA3_MODE_OBJECT,softmax_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call FA3_MODE_OBJECT,qk_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call FA3_MODE_OBJECT,pv_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_QK_WGMMA
$(call FA3_MODE_OBJECT,qk_pv_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call FA3_MODE_OBJECT,tma_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call FA3_MODE_OBJECT,tma_only_m3_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_M3_ONLY
$(call FA3_MODE_OBJECT,tma_only_first_round_bh): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_FIRST_BH_SECTION_ONLY
$(call FA3_MODE_OBJECT,tma_only_first_round_bh_m3_only): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_FIRST_BH_SECTION_ONLY -DFLASH_FWD_PROFILE_SCHED_M3_ONLY
$(call FA3_MODE_OBJECT,baseline_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS)
$(call FA3_MODE_OBJECT,softmax_only_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call FA3_MODE_OBJECT,qk_pv_only_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call FA3_MODE_OBJECT,base_no_tma): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,softmax_only_no_tma): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,base_no_tma_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,softmax_only_no_tma_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_timeline): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_TIMELINE
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_reg_timeline): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_REG_TIMELINE -DFLASH_FWD_PROFILE_REG_TIMELINE_ONLY
$(call FA3_MODE_OBJECT,sync_only_no_tma): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_SENS_SYNC_ONLY
$(call FA3_MODE_OBJECT,sync_only_no_tma_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_SENS_SYNC_ONLY
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended): SM90_NVCCFLAGS += $(WGMMA_PTX_NVCCFLAGS)
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_EXTENDED_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_noprofile): SM90_NVCCFLAGS += $(WGMMA_PTX_NVCCFLAGS)
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_noprofile): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_EXTENDED_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_reg_timeline): SM90_NVCCFLAGS += $(WGMMA_PTX_NVCCFLAGS)
$(call FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_reg_timeline): TEST_GROUP_EXTRA_NVCCFLAGS = $(FA3_MODE_EXTENDED_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_REG_TIMELINE -DFLASH_FWD_PROFILE_REG_TIMELINE_ONLY

$(FA3_MODE_OBJECTS): $(FA3_DIR)/fa3_fwd_h1d128_profile_test.cu \
$(TEST_HEADERS) $(FA3_PROFILE_HEADERS) $(TOP_MAKEFILE) $(FA3_MK) \
$(FA3_PREPARED_STAMP) | $(SM90_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(SM90_NVCCFLAGS) $(TEST_GROUP_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(FA3_STANDARD_TARGET): $(FA3_STANDARD_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(FA3_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(FA3_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(FA3_PACKGQA_DEFAULT_TARGET): $(FA3_PACKGQA_DEFAULT_OBJECT) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(FA3_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(FA3_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(FA3_PACKGQA_NOINC_TARGET): $(FA3_PACKGQA_NOINC_OBJECT) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(FA3_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(FA3_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(BIN_DIR)/sm90/fa3/%_tests: \
$(OBJ_DIR)/sm90/fa3/modes/%/fa3_fwd_h1d128_profile_test.cu.o \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(FA3_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(FA3_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
