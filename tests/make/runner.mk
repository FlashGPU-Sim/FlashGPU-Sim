# Machine-readable build metadata used by the test runner.

# FA3 uses the same mode binaries for multiple runtime case sets.
$(foreach profile,breakdown scaling concurrency,\
  $(foreach mode,$(FA3_MODES),\
    $(eval $(call REGISTER_SM90_FA3_MODE,$(profile),$(mode)))))

define REGISTER_FA3_ALL_MODE
TEST_GROUP_BUILD_TARGET_sm90_fa3_$(1)_all := fa3-modes
TEST_GROUP_BINARY_GROUP_sm90_fa3_$(1)_all := fa3-modes
TEST_GROUP_EXECUTOR_sm90_fa3_$(1)_all := build-only
TEST_GROUP_FILTER_sm90_fa3_$(1)_all := *
endef
$(foreach profile,breakdown scaling concurrency,$(eval $(call REGISTER_FA3_ALL_MODE,$(profile))))

# Binary sets are an internal execution detail. They are deliberately separate
# from the public architecture/test-group hierarchy.
BINARY_GROUPS += \
	none \
	$(FA2_BINARY_GROUPS) \
	fa3-standard fa3-packgqa fa3-modes \
	microbench-sm120-mbarrier microbench-sm120-mma \
	microbench-sm90-wgmma \
	$(foreach mode,$(FA3_MODES),fa3-mode-$(mode))

BINARY_GROUP_BINARIES_none =
BINARY_GROUP_BINARIES_fa3-standard = $(FA3_STANDARD_TARGET)
BINARY_GROUP_BINARIES_fa3-packgqa = $(FA3_PACKGQA_TARGETS)
BINARY_GROUP_BINARIES_fa3-modes = $(FA3_MODE_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm120-mbarrier = $(MICROBENCH_SM120_MBAR_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm120-mma = $(MICROBENCH_SM120_MMA_TARGETS)
BINARY_GROUP_BINARIES_microbench-sm90-wgmma = $(MICROBENCH_SM90_WGMMA_TARGETS)
$(foreach mode,$(FA3_MODES),$(eval BINARY_GROUP_BINARIES_fa3-mode-$(mode) = $(call FA3_MODE_TARGET,$(mode))))

BINARY_GROUPS := $(sort $(BINARY_GROUPS))

ACTIVE_PROFILES = $(TEST_GROUP_PROFILES_$(ARCH)_$(TEST_GROUP))
ACTIVE_MODES = $(TEST_GROUP_PROFILE_MODES_$(ARCH)_$(TEST_GROUP)_$(PROFILE))
SELECTION_KEY = $(ARCH)_$(TEST_GROUP)$(if $(PROFILE),_$(PROFILE))$(if $(MODE),_$(MODE))
SELECTION_BINARY_GROUP = $(TEST_GROUP_BINARY_GROUP_$(SELECTION_KEY))

.PHONY: validate-architecture validate-test-group validate-selection \
list-architectures list-test-groups list-test-group-profiles \
list-test-group-modes print-architecture-metadata \
print-test-group-metadata print-test-group-binaries print-test-group-sources \
list-binary-groups print-binary-group help

validate-architecture:
	@if [ -z "$(ARCH)" ]; then \
		echo "ARCH is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(ARCH),$(ARCHITECTURES))" ]; then \
		echo "Unknown architecture: $(ARCH)" >&2; \
		exit 2; \
	fi

validate-test-group: validate-architecture
	@if [ -z "$(TEST_GROUP)" ]; then \
		echo "TEST_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TEST_GROUP),$(ARCH_TEST_GROUPS_$(ARCH)))" ]; then \
		echo "Unsupported test group for $(ARCH): $(TEST_GROUP)" >&2; \
		exit 2; \
	fi

validate-selection: validate-test-group
	@if [ -n "$(ACTIVE_PROFILES)" ] && [ -z "$(PROFILE)" ]; then \
		echo "Test group $(ARCH)/$(TEST_GROUP) requires PROFILE" >&2; \
		exit 2; \
	elif [ -z "$(ACTIVE_PROFILES)" ] && [ -n "$(PROFILE)" ]; then \
		echo "Test group $(ARCH)/$(TEST_GROUP) does not accept PROFILE" >&2; \
		exit 2; \
	elif [ -n "$(PROFILE)" ] && [ -z "$(filter $(PROFILE),$(ACTIVE_PROFILES))" ]; then \
		echo "Unknown profile for $(ARCH)/$(TEST_GROUP): $(PROFILE)" >&2; \
		exit 2; \
	elif [ -n "$(ACTIVE_MODES)" ] && [ -z "$(MODE)" ]; then \
		echo "Profile $(ARCH)/$(TEST_GROUP)/$(PROFILE) requires MODE" >&2; \
		exit 2; \
	elif [ -z "$(ACTIVE_MODES)" ] && [ -n "$(MODE)" ]; then \
		echo "Selection $(ARCH)/$(TEST_GROUP)/$(PROFILE) does not accept MODE" >&2; \
		exit 2; \
	elif [ -n "$(MODE)" ] && [ -z "$(filter $(MODE),$(ACTIVE_MODES))" ]; then \
		echo "Unknown mode for $(ARCH)/$(TEST_GROUP)/$(PROFILE): $(MODE)" >&2; \
		exit 2; \
	elif [ -z "$(TEST_GROUP_BUILD_TARGET_$(SELECTION_KEY))" ] || \
	     [ -z "$(TEST_GROUP_BINARY_GROUP_$(SELECTION_KEY))" ] || \
	     [ -z "$(TEST_GROUP_EXECUTOR_$(SELECTION_KEY))" ]; then \
		echo "Incomplete manifest selection: $(SELECTION_KEY)" >&2; \
		exit 2; \
	fi

list-architectures:
	@printf '%s\n' $(ARCHITECTURES)

list-test-groups: validate-architecture
	@printf '%s\n' $(ARCH_TEST_GROUPS_$(ARCH))

list-test-group-profiles: validate-test-group
	@printf '%s\n' $(ACTIVE_PROFILES)

list-test-group-modes: validate-test-group
	@if [ -z "$(PROFILE)" ]; then \
		echo "PROFILE is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(PROFILE),$(ACTIVE_PROFILES))" ]; then \
		echo "Unknown profile for $(ARCH)/$(TEST_GROUP): $(PROFILE)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(ACTIVE_MODES); \
	fi

print-architecture-metadata: validate-architecture
	@printf '%s|%s\n' \
		"$(ARCH_DEFAULT_CONFIG_$(ARCH))" \
		"$(ARCH_NVCC_TARGET_$(ARCH))"

print-test-group-metadata: validate-selection
	@printf '%s|%s|%s|%s|%s\n' \
		"$(TEST_GROUP_BUILD_TARGET_$(SELECTION_KEY))" \
		"$(SELECTION_BINARY_GROUP)" \
		"$(TEST_GROUP_EXECUTOR_$(SELECTION_KEY))" \
		"$(or $(TEST_GROUP_FILTER_$(SELECTION_KEY)),*)" \
		"$(TEST_GROUP_CASES_$(SELECTION_KEY))"

print-test-group-binaries: validate-selection
	@printf '%s\n' $(BINARY_GROUP_BINARIES_$(SELECTION_BINARY_GROUP))

print-test-group-sources: validate-test-group
	@printf '%s\n' $(TEST_GROUP_SOURCES_$(ARCH)_$(TEST_GROUP))

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

help:
	@echo "Architecture/test-group hierarchy"
	@echo "================================="
	@$(foreach arch,$(ARCHITECTURES),echo "  $(arch): $(ARCH_TEST_GROUPS_$(arch))";)
	@echo ""
	@echo "Use './run_tests.py help' for the public CLI."
