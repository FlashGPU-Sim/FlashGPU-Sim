# Blackwell PTX-ISA regression tests.

BLACKWELL_MK := $(lastword $(MAKEFILE_LIST))
WORKLOAD_MANAGED_TEST_GROUPS_sm120 += blackwell

TEST_GROUP_BUILD_TARGET_sm120_blackwell := build-sm120-blackwell
TEST_GROUP_BINARY_GROUP_sm120_blackwell := sm120-blackwell
TEST_GROUP_EXECUTOR_sm120_blackwell := gtest-single
TEST_GROUP_FILTER_sm120_blackwell := *

BLACKWELL_SM120_OBJECTS := $(patsubst $(TEST_SRC_DIR)/%.cu,\
	$(OBJ_DIR)/sm120/%.cu.o,$(TEST_GROUP_SOURCES_sm120_blackwell))
BLACKWELL_SM120_BINARY := $(BIN_DIR)/sm120/blackwell_tests.bin
BLACKWELL_SM120_LAUNCHER := $(BIN_DIR)/sm120/blackwell_tests
BLACKWELL_SM120_LAUNCHER_SOURCE := \
	$(TEST_SRC_DIR)/blackwell/run_vector_ldst_v8_test.sh
BLACKWELL_SM120_PTX_SOURCE := $(TEST_SRC_DIR)/blackwell/vector_ldst_v8_test.ptx
BLACKWELL_SM120_PTX := $(BIN_DIR)/sm120/vector_ldst_v8_test.ptx
BLACKWELL_SM120_PTXINFO_SOURCE := \
	$(TEST_SRC_DIR)/blackwell/vector_ldst_v8_test.ptxinfo
BLACKWELL_SM120_PTXINFO := $(BIN_DIR)/sm120/vector_ldst_v8_test.ptxinfo

BINARY_GROUPS += sm120-blackwell
BINARY_GROUP_BINARIES_sm120-blackwell := $(BLACKWELL_SM120_LAUNCHER)

$(BLACKWELL_SM120_OBJECTS): NVCC_COMPILE_FLAGS = $(BASE_NVCCFLAGS) \
	-arch=$(ARCH_NVCC_TARGET_sm120)

.PHONY: build-sm120-blackwell
build-sm120-blackwell: setup-gtest $(BLACKWELL_SM120_LAUNCHER)

$(BLACKWELL_SM120_BINARY): $(BLACKWELL_SM120_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(BLACKWELL_MK) arch/sm120.toml \
$(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(BLACKWELL_SM120_OBJECTS) $(OBJ_DIR)/gtest_main.a \
		-o $@ -lpthread $(CUDA_LIBS)

$(BLACKWELL_SM120_PTX): $(BLACKWELL_SM120_PTX_SOURCE) $(BLACKWELL_MK)
	@mkdir -p $(dir $@)
	cp $< $@

$(BLACKWELL_SM120_PTXINFO): $(BLACKWELL_SM120_PTXINFO_SOURCE) $(BLACKWELL_MK)
	@mkdir -p $(dir $@)
	cp $< $@

$(BLACKWELL_SM120_LAUNCHER): $(BLACKWELL_SM120_BINARY) \
$(BLACKWELL_SM120_PTX) $(BLACKWELL_SM120_PTXINFO) \
$(BLACKWELL_SM120_LAUNCHER_SOURCE) $(BLACKWELL_MK)
	@mkdir -p $(dir $@)
	cp $(BLACKWELL_SM120_LAUNCHER_SOURCE) $@
	chmod +x $@
