# Shared compile and link rules for ordinary GoogleTest test groups.

BUILD_MK := $(lastword $(MAKEFILE_LIST))

# Workload fragments claim groups that have custom build graphs. Every
# remaining first path component in arch/*.toml is an ordinary GoogleTest
# binary group.
$(foreach arch,$(ARCHITECTURES),\
  $(eval STANDARD_TEST_GROUPS_$(arch) := \
    $(filter-out $(WORKLOAD_MANAGED_TEST_GROUPS_$(arch)),\
      $(ARCH_TEST_GROUPS_$(arch)))))

# Workload fragments loaded before this file may add group-specific link inputs
# and configuration prerequisites through TEST_GROUP_EXTRA_{OBJECTS,PREREQUISITES}.
define TEST_SOURCE_OBJECTS
$(patsubst $(TEST_SRC_DIR)/%.cu,$(OBJ_DIR)/$(1)/%.cu.o,$(filter %.cu,$(2))) \
$(patsubst $(TEST_SRC_DIR)/%.cc,$(OBJ_DIR)/$(1)/%.cc.o,$(filter %.cc,$(2)))
endef

define REGISTER_ARCH_COMPILE_RULES
$(OBJ_DIR)/$(1)/%.cu.o: $(TEST_SRC_DIR)/%.cu $(TEST_HEADERS) \
$(TOP_MAKEFILE) $(BUILD_MK) arch/$(1).toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $$(dir $$@)
	$$(NVCC) $$(NVCC_COMPILE_FLAGS) $$(INCLUDES) $$(GPGPUSIM_FLAGS) \
		-c $$< -o $$@

$(OBJ_DIR)/$(1)/%.cc.o: $(TEST_SRC_DIR)/%.cc $(TEST_HEADERS) $(TOP_MAKEFILE) \
$(BUILD_MK) arch/$(1).toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) $$(INCLUDES) $$(GPGPUSIM_FLAGS) -c $$< -o $$@
endef

$(foreach arch,$(ARCHITECTURES),$(eval $(call REGISTER_ARCH_COMPILE_RULES,$(arch))))

define REGISTER_STANDARD_TEST_GROUP
TEST_GROUP_OBJECTS_$(1)_$(2) := $$(call TEST_SOURCE_OBJECTS,$(1),$$(TEST_GROUP_SOURCES_$(1)_$(2)))
TEST_GROUP_TARGET_$(1)_$(2) := $$(BIN_DIR)/$(1)/$(2)_tests
TEST_GROUP_EFFECTIVE_NVCCFLAGS_$(1)_$(2) = $$(BASE_NVCCFLAGS) \
  $$(if $$(filter sm120,$(1)),$$(ARCH_NVCC_CODEGEN_$(1)),\
    -arch=$$(ARCH_NVCC_TARGET_$(1))) $$(TEST_GROUP_EXTRA_FLAGS_$(1)_$(2))

$$(TEST_GROUP_OBJECTS_$(1)_$(2)): NVCC_COMPILE_FLAGS = $$(TEST_GROUP_EFFECTIVE_NVCCFLAGS_$(1)_$(2))

.PHONY: build-$(1)-$(2)
build-$(1)-$(2): setup-gtest $$(TEST_GROUP_TARGET_$(1)_$(2))

$$(TEST_GROUP_TARGET_$(1)_$(2)): $$(TEST_GROUP_OBJECTS_$(1)_$(2)) \
$$(TEST_GROUP_EXTRA_OBJECTS_$(1)_$(2)) $$(OBJ_DIR)/gtest_main.a \
$$(TEST_GROUP_EXTRA_PREREQUISITES_$(1)_$(2)) \
$$(TOP_MAKEFILE) $$(BUILD_MK) arch/$(1).toml $$(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) $$(TEST_GROUP_OBJECTS_$(1)_$(2)) \
		$$(TEST_GROUP_EXTRA_OBJECTS_$(1)_$(2)) $$(OBJ_DIR)/gtest_main.a \
		-o $$@ -lpthread $$(CUDA_LIBS)

TEST_GROUP_BUILD_TARGET_$(1)_$(2) := build-$(1)-$(2)
TEST_GROUP_BINARY_GROUP_$(1)_$(2) := $(1)-$(2)
TEST_GROUP_EXECUTOR_$(1)_$(2) := gtest-single
TEST_GROUP_FILTER_$(1)_$(2) ?= *
BINARY_GROUPS += $(1)-$(2)
BINARY_GROUP_BINARIES_$(1)-$(2) := $$(TEST_GROUP_TARGET_$(1)_$(2))
endef

$(foreach arch,$(ARCHITECTURES),\
  $(foreach test_group,$(STANDARD_TEST_GROUPS_$(arch)),\
    $(eval $(call REGISTER_STANDARD_TEST_GROUP,$(arch),$(test_group)))))
