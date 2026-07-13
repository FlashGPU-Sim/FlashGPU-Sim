# Machine-readable binary, suite/target, and FA group/mode registry.

TEST_GROUPS = \
	test \
	microbench-default \
	microbench-sm90 \
	dev \
	test-sm90 \
	fa2-smoke \
	fa2-small \
	fa2-medium \
	fa2-large \
	fa2-breakdown \
	fa2-scaling \
	fa2-concurrency \
	fa3 \
	fa3-single-tile \
	fa3-h1d128-profile \
	fa3-modes \
	$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),fa2-breakdown-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_SCALING_MODES),fa2-scaling-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),fa2-concurrency-$(mode)) \
	$(foreach mode,$(HOPPER_FA3_MODES),fa3-mode-$(mode))

TEST_GROUP_BINARIES_test = $(MAIN_TEST_TARGET)
TEST_GROUP_BINARIES_microbench-default = $(MICROBENCH_DEFAULT_TARGETS)
TEST_GROUP_BINARIES_microbench-sm90 = $(MICROBENCH_SM90_TARGETS)
TEST_GROUP_BINARIES_dev = $(DEV_TEST_TARGET)
TEST_GROUP_BINARIES_test-sm90 = $(SM90_TEST_TARGETS)
TEST_GROUP_BINARIES_fa2-smoke = $(HOPPER_FA2_SMOKE_TARGETS)
TEST_GROUP_BINARIES_fa2-small = $(HOPPER_FA2_SMALL_TARGETS)
TEST_GROUP_BINARIES_fa2-medium = $(HOPPER_FA2_MEDIUM_TARGETS)
TEST_GROUP_BINARIES_fa2-large = $(HOPPER_FA2_LARGE_TARGETS)
TEST_GROUP_BINARIES_fa2-breakdown = $(HOPPER_FA2_BREAKDOWN_TARGETS)
TEST_GROUP_BINARIES_fa2-scaling = $(HOPPER_FA2_SCALING_TARGETS)
TEST_GROUP_BINARIES_fa2-concurrency = $(HOPPER_FA2_CONCURRENCY_TARGETS)
TEST_GROUP_BINARIES_fa3 = $(HOPPER_FA3_EXTENDED_TARGET)
TEST_GROUP_BINARIES_fa3-single-tile = $(HOPPER_FA3_SINGLE_TILE_TARGET)
TEST_GROUP_BINARIES_fa3-h1d128-profile = $(HOPPER_FA3_H1D128_PROFILE_TARGET)
TEST_GROUP_BINARIES_fa3-modes = $(HOPPER_FA3_MODE_TARGETS)
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval TEST_GROUP_BINARIES_fa2-breakdown-$(mode) = $(BIN_DIR)/hopper/run_fa2_breakdown_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval TEST_GROUP_BINARIES_fa2-scaling-$(mode) = $(BIN_DIR)/hopper/run_fa2_scaling_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval TEST_GROUP_BINARIES_fa2-concurrency-$(mode) = $(BIN_DIR)/hopper/run_fa2_concurrency_$(mode)_tests))
$(foreach mode,$(HOPPER_FA3_MODES),$(eval TEST_GROUP_BINARIES_fa3-mode-$(mode) = $(call HOPPER_FA3_MODE_TARGET,$(mode))))

# One target is an independently buildable and runnable architecture domain.
TEST_SUITES = test microbench dev trace

SUITE_TARGETS_test = default sm90 fa2 fa3
SUITE_TARGETS_microbench = default sm90
SUITE_TARGETS_dev = default
SUITE_TARGETS_trace = default

SUITE_DEFAULT_TARGET_test = default
SUITE_DEFAULT_TARGET_microbench = default
SUITE_DEFAULT_TARGET_dev = default
SUITE_DEFAULT_TARGET_trace = default

SUITE_TARGET_BUILD_GROUP_test_default = test
SUITE_TARGET_BINARY_GROUP_test_default = test
SUITE_TARGET_EXECUTOR_test_default = test
SUITE_TARGET_DEFAULT_CONFIG_test_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_test_default = 12.0
SUITE_TARGET_CUDA_ARCH_test_default = sm_120a

SUITE_TARGET_BUILD_GROUP_test_sm90 = test-sm90
SUITE_TARGET_BINARY_GROUP_test_sm90 = test-sm90
SUITE_TARGET_EXECUTOR_test_sm90 = gtest-single
SUITE_TARGET_DEFAULT_CONFIG_test_sm90 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_sm90 = 9.0
SUITE_TARGET_CUDA_ARCH_test_sm90 = sm_90a

SUITE_TARGET_BUILD_GROUP_test_fa2 = group-required
SUITE_TARGET_BINARY_GROUP_test_fa2 = group-required
SUITE_TARGET_EXECUTOR_test_fa2 = group-required
SUITE_TARGET_DEFAULT_CONFIG_test_fa2 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_fa2 = 9.0
SUITE_TARGET_CUDA_ARCH_test_fa2 = sm_90a

SUITE_TARGET_BUILD_GROUP_test_fa3 = group-required
SUITE_TARGET_BINARY_GROUP_test_fa3 = group-required
SUITE_TARGET_EXECUTOR_test_fa3 = group-required
SUITE_TARGET_DEFAULT_CONFIG_test_fa3 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_fa3 = 9.0
SUITE_TARGET_CUDA_ARCH_test_fa3 = sm_90a

TARGET_GROUPS_test_fa2 = smoke small medium large breakdown scaling concurrency
TARGET_GROUPS_test_fa3 = smoke small medium large breakdown scaling concurrency

TARGET_GROUP_MODES_test_fa2_breakdown = $(HOPPER_FA2_BREAKDOWN_MODES) all
TARGET_GROUP_MODES_test_fa2_scaling = $(HOPPER_FA2_SCALING_MODES) all
TARGET_GROUP_MODES_test_fa2_concurrency = $(HOPPER_FA2_CONCURRENCY_MODES) all
TARGET_GROUP_MODES_test_fa3_breakdown = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_test_fa3_scaling = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_test_fa3_concurrency = $(HOPPER_FA3_MODES) all

TARGET_GROUP_BUILD_GROUP_test_fa2_smoke = fa2-smoke
TARGET_GROUP_BINARY_GROUP_test_fa2_smoke = fa2-smoke
TARGET_GROUP_EXECUTOR_test_fa2_smoke = gtest-multi
TARGET_GROUP_FILTER_test_fa2_smoke = *
TARGET_GROUP_BUILD_GROUP_test_fa2_small = fa2-small
TARGET_GROUP_BINARY_GROUP_test_fa2_small = fa2-small
TARGET_GROUP_EXECUTOR_test_fa2_small = gtest-multi
TARGET_GROUP_FILTER_test_fa2_small = *
TARGET_GROUP_BUILD_GROUP_test_fa2_medium = fa2-medium
TARGET_GROUP_BINARY_GROUP_test_fa2_medium = fa2-medium
TARGET_GROUP_EXECUTOR_test_fa2_medium = gtest-multi
TARGET_GROUP_FILTER_test_fa2_medium = *
TARGET_GROUP_BUILD_GROUP_test_fa2_large = fa2-large
TARGET_GROUP_BINARY_GROUP_test_fa2_large = fa2-large
TARGET_GROUP_EXECUTOR_test_fa2_large = gtest-multi
TARGET_GROUP_FILTER_test_fa2_large = *

define REGISTER_FA2_MODE
TARGET_GROUP_BUILD_GROUP_test_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_BINARY_GROUP_test_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_EXECUTOR_test_fa2_$(1)_$(2) = gtest-multi
TARGET_GROUP_FILTER_test_fa2_$(1)_$(2) = *
endef
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval $(call REGISTER_FA2_MODE,breakdown,$(mode))))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval $(call REGISTER_FA2_MODE,scaling,$(mode))))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval $(call REGISTER_FA2_MODE,concurrency,$(mode))))
$(eval $(call REGISTER_FA2_MODE,breakdown,all))
$(eval $(call REGISTER_FA2_MODE,scaling,all))
$(eval $(call REGISTER_FA2_MODE,concurrency,all))
TARGET_GROUP_BUILD_GROUP_test_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BINARY_GROUP_test_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BUILD_GROUP_test_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BINARY_GROUP_test_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BUILD_GROUP_test_fa2_concurrency_all = fa2-concurrency
TARGET_GROUP_BINARY_GROUP_test_fa2_concurrency_all = fa2-concurrency

TARGET_GROUP_BUILD_GROUP_test_fa3_smoke = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_smoke = fa3
TARGET_GROUP_EXECUTOR_test_fa3_smoke = gtest-single
TARGET_GROUP_FILTER_test_fa3_smoke = Fa3PrefillFp16SmokeTest.*:Fa3PrefillFp16BackwardSmokeTest.*:Fa3FwdHdim128Fp16IntegrationTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_small = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_small = fa3
TARGET_GROUP_EXECUTOR_test_fa3_small = gtest-single
TARGET_GROUP_FILTER_test_fa3_small = Fa3PrefillFp16SmallTest.*:Fa3PrefillFp16BackwardSmallTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_medium = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_medium = fa3
TARGET_GROUP_EXECUTOR_test_fa3_medium = gtest-single
TARGET_GROUP_FILTER_test_fa3_medium = Fa3PrefillFp16MediumTest.*:Fa3PrefillFp16BackwardMediumTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_large = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_large = fa3
TARGET_GROUP_EXECUTOR_test_fa3_large = gtest-single
TARGET_GROUP_FILTER_test_fa3_large = Fa3PrefillFp16IntegrationTest.*:Fa3PrefillFp16BackwardIntegrationTest.*

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
TARGET_GROUP_BUILD_GROUP_test_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_BINARY_GROUP_test_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_EXECUTOR_test_fa3_$(1)_$(2) = fa3-profile
TARGET_GROUP_FILTER_test_fa3_$(1)_$(2) = Fa3H1D128ProfileTest.SelectedD128FullCases
TARGET_GROUP_CASES_test_fa3_$(1)_$(2) = $(FA3_GROUP_CASES_$(1))
endef
$(foreach group,breakdown scaling concurrency,$(foreach mode,$(HOPPER_FA3_MODES),$(eval $(call REGISTER_FA3_MODE,$(group),$(mode)))))
$(foreach group,breakdown scaling concurrency,$(eval $(call REGISTER_FA3_MODE,$(group),all)))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BUILD_GROUP_test_fa3_$(group)_all = fa3-modes))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BINARY_GROUP_test_fa3_$(group)_all = fa3-modes))

SUITE_TARGET_BUILD_GROUP_microbench_default = microbench-default
SUITE_TARGET_BINARY_GROUP_microbench_default = microbench-default
SUITE_TARGET_EXECUTOR_microbench_default = gtest-multi
SUITE_TARGET_DEFAULT_CONFIG_microbench_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_microbench_default = 12.0
SUITE_TARGET_CUDA_ARCH_microbench_default = sm_120a

SUITE_TARGET_BUILD_GROUP_microbench_sm90 = microbench-sm90
SUITE_TARGET_BINARY_GROUP_microbench_sm90 = microbench-sm90
SUITE_TARGET_EXECUTOR_microbench_sm90 = gtest-multi
SUITE_TARGET_DEFAULT_CONFIG_microbench_sm90 = SM90_H100
SUITE_TARGET_REQUIRED_CC_microbench_sm90 = 9.0
SUITE_TARGET_CUDA_ARCH_microbench_sm90 = sm_90a

SUITE_TARGET_BUILD_GROUP_dev_default = dev
SUITE_TARGET_BINARY_GROUP_dev_default = dev
SUITE_TARGET_EXECUTOR_dev_default = dev
SUITE_TARGET_DEFAULT_CONFIG_dev_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_dev_default = 12.0
SUITE_TARGET_CUDA_ARCH_dev_default = sm_120a

SUITE_TARGET_BUILD_GROUP_trace_default = trace
SUITE_TARGET_BINARY_GROUP_trace_default = none
SUITE_TARGET_EXECUTOR_trace_default = trace
SUITE_TARGET_DEFAULT_CONFIG_trace_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_trace_default = 12.0
SUITE_TARGET_CUDA_ARCH_trace_default = sm_120a

.PHONY: list-test-groups print-test-binaries list-suites list-suite-targets \
print-suite-default-target print-suite-target-metadata list-target-groups \
list-target-group-modes print-target-group-metadata help

list-test-groups:
	@printf '%s\n' $(TEST_GROUPS)

print-test-binaries:
	@if [ -z "$(TEST_GROUP)" ]; then \
		echo "TEST_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TEST_GROUP),$(TEST_GROUPS))" ]; then \
		echo "Unknown TEST_GROUP: $(TEST_GROUP)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TEST_GROUP_BINARIES_$(TEST_GROUP)); \
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
	elif [ -z "$(filter $(SUITE_TARGET),$(SUITE_TARGETS_$(SUITE)))" ]; then \
		echo "Unknown target for $(SUITE): $(SUITE_TARGET)" >&2; \
		exit 2; \
	elif [ -z "$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET))" ]; then \
		echo "$(SUITE)/$(SUITE_TARGET) does not define groups" >&2; \
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
	@echo "Internal test build graph"
	@echo "========================="
	@echo "Use './run_tests.sh help' for the supported test CLI."
	@echo "The suite build targets in this Makefile are internal implementation details."
	@echo ""
	@echo "Standalone calibration targets:"
	@echo "  standalone-bench - Build standalone CUDA calibration microbenchmarks"
	@echo "  cp-async-bench - Build standalone cp.async calibration binaries"
	@echo "  mma-standalone-bench - Build standalone warp-MMA calibration binaries"
	@echo "  prepare-fa3-flash-attention - Prepare local flash-attention headers for FA2/FA3"
	@echo ""
	@echo "Registry inspection:"
	@echo "  list-test-groups - List groups in the binary manifest"
	@echo "  print-test-binaries TEST_GROUP=<group> - Print one group's binaries"
	@echo "  list-suites - List user-facing suites"
	@echo "  list-suite-targets SUITE=<suite> - List one suite's targets"
