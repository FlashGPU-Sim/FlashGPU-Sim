# Machine-readable suite/target/group registry and binary manifest.

# A binary group is an internal manifest used by the runner after the public
# suite/target/group selection has been resolved.
BINARY_GROUPS = \
	none \
	test-sm120-unit \
	test-sm120-integration \
	test-sm90-instructions \
	fa2-smoke \
	fa2-small \
	fa2-medium \
	fa2-large \
	fa2-breakdown \
	fa2-scaling \
	fa2-concurrency \
	fa3-standard \
	fa3-modes \
	microbench-sm120-mbarrier \
	microbench-sm120-mma \
	microbench-sm90-wgmma \
	$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),fa2-breakdown-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_SCALING_MODES),fa2-scaling-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),fa2-concurrency-$(mode)) \
	$(foreach mode,$(HOPPER_FA3_MODES),fa3-mode-$(mode))

BINARY_GROUP_BINARIES_none =
BINARY_GROUP_BINARIES_test-sm120-unit = $(SM120_UNIT_TARGET)
BINARY_GROUP_BINARIES_test-sm120-integration = $(SM120_INTEGRATION_TARGET)
BINARY_GROUP_BINARIES_test-sm90-instructions = $(SM90_INSTRUCTION_TARGETS)
BINARY_GROUP_BINARIES_fa2-smoke = $(HOPPER_FA2_SMOKE_TARGETS)
BINARY_GROUP_BINARIES_fa2-small = $(HOPPER_FA2_SMALL_TARGETS)
BINARY_GROUP_BINARIES_fa2-medium = $(HOPPER_FA2_MEDIUM_TARGETS)
BINARY_GROUP_BINARIES_fa2-large = $(HOPPER_FA2_LARGE_TARGETS)
BINARY_GROUP_BINARIES_fa2-breakdown = $(HOPPER_FA2_BREAKDOWN_TARGETS)
BINARY_GROUP_BINARIES_fa2-scaling = $(HOPPER_FA2_SCALING_TARGETS)
BINARY_GROUP_BINARIES_fa2-concurrency = $(HOPPER_FA2_CONCURRENCY_TARGETS)
BINARY_GROUP_BINARIES_fa3-standard = $(HOPPER_FA3_STANDARD_TARGET)
BINARY_GROUP_BINARIES_fa3-modes = $(HOPPER_FA3_MODE_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm120-mbarrier = $(MICROBENCH_SM120_MBAR_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm120-mma = $(MICROBENCH_SM120_MMA_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm90-wgmma = $(MICROBENCH_SM90_WGMMA_TARGETS)
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval BINARY_GROUP_BINARIES_fa2-breakdown-$(mode) = $(BIN_DIR)/hopper/run_fa2_breakdown_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval BINARY_GROUP_BINARIES_fa2-scaling-$(mode) = $(BIN_DIR)/hopper/run_fa2_scaling_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval BINARY_GROUP_BINARIES_fa2-concurrency-$(mode) = $(BIN_DIR)/hopper/run_fa2_concurrency_$(mode)_tests))
$(foreach mode,$(HOPPER_FA3_MODES),$(eval BINARY_GROUP_BINARIES_fa3-mode-$(mode) = $(call HOPPER_FA3_MODE_TARGET,$(mode))))

# Public runner hierarchy: suite -> architecture/workload target -> group.
TEST_SUITES = test analysis microbench trace

SUITE_TARGETS_test = sm120 sm90
SUITE_TARGETS_analysis = fa2 fa3
SUITE_TARGETS_microbench = sm120 sm90
SUITE_TARGETS_trace = sm120

SUITE_DEFAULT_TARGET_test = sm120
SUITE_DEFAULT_TARGET_analysis =
SUITE_DEFAULT_TARGET_microbench = sm120
SUITE_DEFAULT_TARGET_trace = sm120

define REGISTER_SUITE_TARGET
SUITE_TARGET_BUILD_GROUP_$(1)_$(2) = group-required
SUITE_TARGET_BINARY_GROUP_$(1)_$(2) = group-required
SUITE_TARGET_EXECUTOR_$(1)_$(2) = group-required
SUITE_TARGET_DEFAULT_CONFIG_$(1)_$(2) = $(3)
SUITE_TARGET_REQUIRED_CC_$(1)_$(2) = $(4)
SUITE_TARGET_CUDA_ARCH_$(1)_$(2) = $(5)
endef

$(eval $(call REGISTER_SUITE_TARGET,test,sm120,SM120_RTX5090,12.0,sm_120a))
$(eval $(call REGISTER_SUITE_TARGET,test,sm90,SM90_H100,9.0,sm_90a))
$(eval $(call REGISTER_SUITE_TARGET,analysis,fa2,SM90_H100,9.0,sm_90a))
$(eval $(call REGISTER_SUITE_TARGET,analysis,fa3,SM90_H100,9.0,sm_90a))
$(eval $(call REGISTER_SUITE_TARGET,microbench,sm120,SM120_RTX5090,12.0,sm_120a))
$(eval $(call REGISTER_SUITE_TARGET,microbench,sm90,SM90_H100,9.0,sm_90a))
$(eval $(call REGISTER_SUITE_TARGET,trace,sm120,SM120_RTX5090,12.0,sm_120a))

TARGET_GROUPS_test_sm120 = unit integration
TARGET_GROUPS_test_sm90 = instructions fa2-smoke fa3-smoke
TARGET_GROUPS_analysis_fa2 = small medium large breakdown scaling concurrency
TARGET_GROUPS_analysis_fa3 = small medium large breakdown scaling concurrency
TARGET_GROUPS_microbench_sm120 = mbarrier mma memory
TARGET_GROUPS_microbench_sm90 = cp-async mma tma wgmma
TARGET_GROUPS_trace_sm120 = gpt2

# SM120 correctness tests.
TARGET_GROUP_BUILD_GROUP_test_sm120_unit = test-sm120-unit
TARGET_GROUP_BINARY_GROUP_test_sm120_unit = test-sm120-unit
TARGET_GROUP_EXECUTOR_test_sm120_unit = gtest-single
TARGET_GROUP_FILTER_test_sm120_unit = *

TARGET_GROUP_BUILD_GROUP_test_sm120_integration = test-sm120-integration
TARGET_GROUP_BINARY_GROUP_test_sm120_integration = test-sm120-integration
TARGET_GROUP_EXECUTOR_test_sm120_integration = test
TARGET_GROUP_FILTER_test_sm120_integration = *

# SM90 instruction and smoke correctness tests.
TARGET_GROUP_BUILD_GROUP_test_sm90_instructions = test-sm90-instructions
TARGET_GROUP_BINARY_GROUP_test_sm90_instructions = test-sm90-instructions
TARGET_GROUP_EXECUTOR_test_sm90_instructions = gtest-single
TARGET_GROUP_FILTER_test_sm90_instructions = *

TARGET_GROUP_BUILD_GROUP_test_sm90_fa2-smoke = fa2-smoke
TARGET_GROUP_BINARY_GROUP_test_sm90_fa2-smoke = fa2-smoke
TARGET_GROUP_EXECUTOR_test_sm90_fa2-smoke = gtest-multi
TARGET_GROUP_FILTER_test_sm90_fa2-smoke = *

TARGET_GROUP_BUILD_GROUP_test_sm90_fa3-smoke = fa3-standard
TARGET_GROUP_BINARY_GROUP_test_sm90_fa3-smoke = fa3-standard
TARGET_GROUP_EXECUTOR_test_sm90_fa3-smoke = gtest-single
TARGET_GROUP_FILTER_test_sm90_fa3-smoke = Fa3PrefillFp16SmokeTest.*:Fa3PrefillFp16BackwardSmokeTest.*:Fa3FwdHdim128Fp16IntegrationTest.*

# FA2 functional sizes and compile-time analysis modes.
define REGISTER_FA2_STANDARD_GROUP
TARGET_GROUP_BUILD_GROUP_analysis_fa2_$(1) = fa2-$(1)
TARGET_GROUP_BINARY_GROUP_analysis_fa2_$(1) = fa2-$(1)
TARGET_GROUP_EXECUTOR_analysis_fa2_$(1) = gtest-multi
TARGET_GROUP_FILTER_analysis_fa2_$(1) = *
endef
$(foreach group,small medium large,$(eval $(call REGISTER_FA2_STANDARD_GROUP,$(group))))

TARGET_GROUP_MODES_analysis_fa2_breakdown = $(HOPPER_FA2_BREAKDOWN_MODES) all
TARGET_GROUP_MODES_analysis_fa2_scaling = $(HOPPER_FA2_SCALING_MODES) all
TARGET_GROUP_MODES_analysis_fa2_concurrency = $(HOPPER_FA2_CONCURRENCY_MODES) all

define REGISTER_FA2_MODE
TARGET_GROUP_BUILD_GROUP_analysis_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_BINARY_GROUP_analysis_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_EXECUTOR_analysis_fa2_$(1)_$(2) = gtest-multi
TARGET_GROUP_FILTER_analysis_fa2_$(1)_$(2) = *
endef
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval $(call REGISTER_FA2_MODE,breakdown,$(mode))))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval $(call REGISTER_FA2_MODE,scaling,$(mode))))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval $(call REGISTER_FA2_MODE,concurrency,$(mode))))
$(foreach group,breakdown scaling concurrency,$(eval $(call REGISTER_FA2_MODE,$(group),all)))
TARGET_GROUP_BUILD_GROUP_analysis_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BINARY_GROUP_analysis_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BUILD_GROUP_analysis_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BINARY_GROUP_analysis_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BUILD_GROUP_analysis_fa2_concurrency_all = fa2-concurrency
TARGET_GROUP_BINARY_GROUP_analysis_fa2_concurrency_all = fa2-concurrency

# FA3 functional sizes share the standard binary; analysis modes share the
# same mode binaries but select different runtime case lists.
TARGET_GROUP_BUILD_GROUP_analysis_fa3_small = fa3-standard
TARGET_GROUP_BINARY_GROUP_analysis_fa3_small = fa3-standard
TARGET_GROUP_EXECUTOR_analysis_fa3_small = gtest-single
TARGET_GROUP_FILTER_analysis_fa3_small = Fa3PrefillFp16SmallTest.*:Fa3PrefillFp16BackwardSmallTest.*
TARGET_GROUP_BUILD_GROUP_analysis_fa3_medium = fa3-standard
TARGET_GROUP_BINARY_GROUP_analysis_fa3_medium = fa3-standard
TARGET_GROUP_EXECUTOR_analysis_fa3_medium = gtest-single
TARGET_GROUP_FILTER_analysis_fa3_medium = Fa3PrefillFp16MediumTest.*:Fa3PrefillFp16BackwardMediumTest.*
TARGET_GROUP_BUILD_GROUP_analysis_fa3_large = fa3-standard
TARGET_GROUP_BINARY_GROUP_analysis_fa3_large = fa3-standard
TARGET_GROUP_EXECUTOR_analysis_fa3_large = gtest-single
TARGET_GROUP_FILTER_analysis_fa3_large = Fa3PrefillFp16IntegrationTest.*:Fa3PrefillFp16BackwardIntegrationTest.*

TARGET_GROUP_MODES_analysis_fa3_breakdown = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_analysis_fa3_scaling = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_analysis_fa3_concurrency = $(HOPPER_FA3_MODES) all

FA3_GROUP_CASES_breakdown = H1D128FullB1S4096
FA3_GROUP_CASES_scaling = \
	H1D128FullB1S128 H1D128FullB1S256 H1D128FullB1S512 \
	H1D128FullB1S1024 H1D128FullB1S2048 H1D128FullB1S4096 \
	H1D128FullB1S8192
FA3_GROUP_CASES_concurrency = \
	H16D128FullB64S512 H4D128FullB64S512 H16D128FullB16S512 H1D128FullB1S512 \
	H16D128FullB32S1024 H4D128FullB32S1024 H16D128FullB8S1024 H1D128FullB1S1024 \
	H16D128FullB16S2048 H4D128FullB16S2048 H16D128FullB4S2048 H1D128FullB1S2048 \
	H16D128FullB8S4096 H4D128FullB8S4096 H16D128FullB2S4096 H1D128FullB1S4096 \
	H16D128FullB4S8192 H4D128FullB4S8192 H16D128FullB1S8192 H1D128FullB1S8192

define REGISTER_FA3_MODE
TARGET_GROUP_BUILD_GROUP_analysis_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_BINARY_GROUP_analysis_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_EXECUTOR_analysis_fa3_$(1)_$(2) = fa3-profile
TARGET_GROUP_FILTER_analysis_fa3_$(1)_$(2) = Fa3H1D128ProfileTest.SelectedD128FullCases
TARGET_GROUP_CASES_analysis_fa3_$(1)_$(2) = $(FA3_GROUP_CASES_$(1))
endef
$(foreach group,breakdown scaling concurrency,$(foreach mode,$(HOPPER_FA3_MODES),$(eval $(call REGISTER_FA3_MODE,$(group),$(mode)))))
$(foreach group,breakdown scaling concurrency,$(eval $(call REGISTER_FA3_MODE,$(group),all)))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BUILD_GROUP_analysis_fa3_$(group)_all = fa3-modes))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BINARY_GROUP_analysis_fa3_$(group)_all = fa3-modes))

# Microbench groups with a gtest interface can run through the generic runner.
# Standalone calibration groups are build-only because they require bespoke
# performance arguments rather than a safe universal invocation.
TARGET_GROUP_BUILD_GROUP_microbench_sm120_mbarrier = microbench-sm120-mbarrier
TARGET_GROUP_BINARY_GROUP_microbench_sm120_mbarrier = microbench-sm120-mbarrier
TARGET_GROUP_EXECUTOR_microbench_sm120_mbarrier = gtest-multi
TARGET_GROUP_FILTER_microbench_sm120_mbarrier = *
TARGET_GROUP_BUILD_GROUP_microbench_sm120_mma = microbench-sm120-mma
TARGET_GROUP_BINARY_GROUP_microbench_sm120_mma = microbench-sm120-mma
TARGET_GROUP_EXECUTOR_microbench_sm120_mma = gtest-multi
TARGET_GROUP_FILTER_microbench_sm120_mma = *
TARGET_GROUP_BUILD_GROUP_microbench_sm120_memory = microbench-sm120-memory
TARGET_GROUP_BINARY_GROUP_microbench_sm120_memory = none
TARGET_GROUP_EXECUTOR_microbench_sm120_memory = build-only
TARGET_GROUP_FILTER_microbench_sm120_memory = *

TARGET_GROUP_BUILD_GROUP_microbench_sm90_cp-async = microbench-sm90-cp-async
TARGET_GROUP_BINARY_GROUP_microbench_sm90_cp-async = none
TARGET_GROUP_EXECUTOR_microbench_sm90_cp-async = build-only
TARGET_GROUP_FILTER_microbench_sm90_cp-async = *
TARGET_GROUP_BUILD_GROUP_microbench_sm90_mma = microbench-sm90-mma
TARGET_GROUP_BINARY_GROUP_microbench_sm90_mma = none
TARGET_GROUP_EXECUTOR_microbench_sm90_mma = build-only
TARGET_GROUP_FILTER_microbench_sm90_mma = *
TARGET_GROUP_BUILD_GROUP_microbench_sm90_tma = microbench-sm90-tma
TARGET_GROUP_BINARY_GROUP_microbench_sm90_tma = none
TARGET_GROUP_EXECUTOR_microbench_sm90_tma = build-only
TARGET_GROUP_FILTER_microbench_sm90_tma = *
TARGET_GROUP_BUILD_GROUP_microbench_sm90_wgmma = microbench-sm90-wgmma
TARGET_GROUP_BINARY_GROUP_microbench_sm90_wgmma = microbench-sm90-wgmma
TARGET_GROUP_EXECUTOR_microbench_sm90_wgmma = gtest-multi
TARGET_GROUP_FILTER_microbench_sm90_wgmma = *

TARGET_GROUP_BUILD_GROUP_trace_sm120_gpt2 = trace-sm120-gpt2
TARGET_GROUP_BINARY_GROUP_trace_sm120_gpt2 = none
TARGET_GROUP_EXECUTOR_trace_sm120_gpt2 = trace
TARGET_GROUP_FILTER_trace_sm120_gpt2 = *

.PHONY: list-binary-groups print-binary-group list-suites list-suite-targets \
print-suite-default-target print-suite-target-metadata list-target-groups \
list-target-group-modes print-target-group-metadata help

list-binary-groups:
	@printf '%s\n' $(BINARY_GROUPS)

print-binary-group:
	@if [ -z "$(BINARY_GROUP)" ]; then \
		echo "BINARY_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(BINARY_GROUP),$(BINARY_GROUPS))" ]; then \
		echo "Unknown BINARY_GROUP: $(BINARY_GROUP)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(BINARY_GROUP_BINARIES_$(BINARY_GROUP)); \
	fi

list-suites:
	@printf '%s\n' $(TEST_SUITES)

list-suite-targets:
	@if [ -z "$(SUITE)" ]; then \
		echo "SUITE is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(SUITE_TARGETS_$(SUITE)); \
	fi

print-suite-default-target:
	@if [ -z "$(SUITE)" ]; then \
		echo "SUITE is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	else \
		printf '%s\n' "$(SUITE_DEFAULT_TARGET_$(SUITE))"; \
	fi

print-suite-target-metadata:
	@if [ -z "$(SUITE)" ] || [ -z "$(SUITE_TARGET)" ]; then \
		echo "SUITE and SUITE_TARGET are required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE_TARGET),$(SUITE_TARGETS_$(SUITE)))" ]; then \
		echo "Unknown target for $(SUITE): $(SUITE_TARGET)" >&2; \
		exit 2; \
	else \
		printf '%s|%s|%s|%s|%s|%s\n' \
			"$(SUITE_TARGET_BUILD_GROUP_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_BINARY_GROUP_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_EXECUTOR_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_DEFAULT_CONFIG_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_REQUIRED_CC_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_CUDA_ARCH_$(SUITE)_$(SUITE_TARGET))"; \
	fi

list-target-groups:
	@if [ -z "$(SUITE)" ] || [ -z "$(SUITE_TARGET)" ]; then \
		echo "SUITE and SUITE_TARGET are required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE_TARGET),$(SUITE_TARGETS_$(SUITE)))" ]; then \
		echo "Unknown target for $(SUITE): $(SUITE_TARGET)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)); \
	fi

list-target-group-modes:
	@if [ -z "$(TARGET_GROUP)" ]; then \
		echo "TARGET_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TARGET_GROUP),$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)))" ]; then \
		echo "Unknown group for $(SUITE)/$(SUITE_TARGET): $(TARGET_GROUP)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)); \
	fi

print-target-group-metadata:
	@if [ -z "$(TARGET_GROUP)" ]; then \
		echo "TARGET_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TARGET_GROUP),$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)))" ]; then \
		echo "Unknown group for $(SUITE)/$(SUITE_TARGET): $(TARGET_GROUP)" >&2; \
		exit 2; \
	elif [ -n "$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP))" ] && [ -z "$(TARGET_MODE)" ]; then \
		echo "Group $(TARGET_GROUP) requires TARGET_MODE" >&2; \
		exit 2; \
	elif [ -z "$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP))" ] && [ -n "$(TARGET_MODE)" ]; then \
		echo "Group $(TARGET_GROUP) does not accept TARGET_MODE" >&2; \
		exit 2; \
	elif [ -n "$(TARGET_MODE)" ] && [ -z "$(filter $(TARGET_MODE),$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)))" ]; then \
		echo "Unknown mode for $(SUITE)/$(SUITE_TARGET)/$(TARGET_GROUP): $(TARGET_MODE)" >&2; \
		exit 2; \
	else \
		printf '%s|%s|%s|%s|%s\n' \
			"$(TARGET_GROUP_BUILD_GROUP_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_BINARY_GROUP_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_EXECUTOR_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_FILTER_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_CASES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))"; \
	fi

help:
	@echo "Supported runner hierarchy"
	@echo "=========================="
	@echo "  test:      sm120/{unit,integration}; sm90/{instructions,fa2-smoke,fa3-smoke}"
	@echo "  analysis:  fa2|fa3/{small,medium,large,breakdown,scaling,concurrency}"
	@echo "  microbench: sm120/{mbarrier,mma,memory}; sm90/{cp-async,mma,tma,wgmma}"
	@echo "  trace:     sm120/gpt2"
	@echo ""
	@echo "Use './run_tests.sh help' for the supported CLI."
