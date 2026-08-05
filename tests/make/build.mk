# Shared compile and link rules for ordinary GoogleTest test groups.

BUILD_MK := $(lastword $(MAKEFILE_LIST))
SPECIAL_TEST_GROUPS := fa2 fa3 microbench trace

# Every non-special first path component in arch/*.toml is one ordinary
# GoogleTest binary group.
$(foreach arch,$(ARCHITECTURES),\
  $(eval STANDARD_TEST_GROUPS_$(arch) := \
    $(filter-out $(SPECIAL_TEST_GROUPS),$(ARCH_TEST_GROUPS_$(arch)))))

# WGMMA emits forward-compatible PTX in addition to the manifest-owned target.
TEST_GROUP_EXTRA_FLAGS_sm90_wgmma := $(WGMMA_EXTRA_NVCCFLAGS)

TEST_GROUP_EXTRA_OBJECTS_sm120_unit := \
	$(OBJ_DIR)/sm120/support/bulk_group.cu.o \
	$(OBJ_DIR)/sm120/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm120/support/mshr-table.cu.o

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
  -arch=$$(ARCH_NVCC_TARGET_$(1)) $$(TEST_GROUP_EXTRA_FLAGS_$(1)_$(2))

$$(TEST_GROUP_OBJECTS_$(1)_$(2)): NVCC_COMPILE_FLAGS = $$(TEST_GROUP_EFFECTIVE_NVCCFLAGS_$(1)_$(2))

.PHONY: build-$(1)-$(2)
build-$(1)-$(2): setup-gtest $$(TEST_GROUP_TARGET_$(1)_$(2))

$$(TEST_GROUP_TARGET_$(1)_$(2)): $$(TEST_GROUP_OBJECTS_$(1)_$(2)) \
$$(TEST_GROUP_EXTRA_OBJECTS_$(1)_$(2)) $$(OBJ_DIR)/gtest_main.a \
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

# Host-side simulator objects required only by the SM120 unit test group.
$(OBJ_DIR)/sm120/support/bulk_group.cu.o: $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc \
$(SRC_DIR)/gpgpu-sim/flash/bulk_group.h $(TOP_MAKEFILE) $(BUILD_MK)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/local_interconnect.cc.o: \
$(SRC_DIR)/gpgpu-sim/local_interconnect.cc \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h $(TOP_MAKEFILE) $(BUILD_MK)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/mshr-table.cu.o: $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(BUILD_MK)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@
