# SM120 unit/integration tests and GoogleTest support.

CORE_MK := $(lastword $(MAKEFILE_LIST))

UNIT_TEST_SOURCES = $(wildcard $(TEST_SRC_DIR)/unit/*_test.cc)
INTEGRATION_TEST_SOURCES = $(shell find $(TEST_SRC_DIR)/integration -name '*_test.cc' 2>/dev/null)
UNIT_TEST_OBJECTS = $(UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
INTEGRATION_TEST_OBJECTS = $(INTEGRATION_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)

SM120_UNIT_TARGET = $(BIN_DIR)/sm120/run_unit_tests
SM120_INTEGRATION_TARGET = $(BIN_DIR)/sm120/run_integration_tests

UNIT_OBJ_DIR = $(OBJ_DIR)/unit
INTEGRATION_OBJ_DIR = $(OBJ_DIR)/integration
INTEGRATION_MMA_OBJ_DIR = $(OBJ_DIR)/integration/mma
FLASH_OBJ_DIR = $(OBJ_DIR)/flash

# bulk_group_test exercises this implementation directly.
FLASH_SOURCES = $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc
FLASH_OBJECTS = $(OBJ_DIR)/flash/bulk_group.cu.o

.PHONY: test-sm120-unit test-sm120-integration setup-gtest clean clean-all debug

test-sm120-unit: setup-gtest $(SM120_UNIT_TARGET)

test-sm120-integration: setup-gtest $(SM120_INTEGRATION_TARGET)

setup-gtest:
	@if [ ! -d "$(GTEST_DIR)" ]; then \
		echo "Downloading Google Test..."; \
		git clone https://github.com/google/googletest.git $(GTEST_CLONE_DIR); \
		cd $(GTEST_DIR) && git checkout release-1.12.1; \
	else \
		echo "Google Test already exists."; \
	fi

$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(UNIT_OBJ_DIR) $(INTEGRATION_OBJ_DIR) \
$(INTEGRATION_MMA_OBJ_DIR) $(HOPPER_OBJ_DIR) $(FLASH_OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)/gtest-all.o: $(GTEST_SRCS_) $(TOP_MAKEFILE) $(CORE_MK) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o: $(GTEST_SRCS_) $(TOP_MAKEFILE) $(CORE_MK) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest_main.cc -o $@

$(OBJ_DIR)/gtest.a: $(OBJ_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest_main.a: $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/flash/%.cu.o: $(SRC_DIR)/gpgpu-sim/flash/%.cc \
$(TOP_MAKEFILE) $(CORE_MK) | $(FLASH_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/unit/%.cu.o: $(TEST_SRC_DIR)/unit/%.cc $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(CORE_MK) | $(UNIT_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/integration/%.cu.o: $(TEST_SRC_DIR)/integration/%.cc \
$(CUH_HEADERS) $(TOP_MAKEFILE) $(CORE_MK)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM120_UNIT_TARGET): $(UNIT_TEST_OBJECTS) $(FLASH_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(CORE_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(CORE_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(SM120_INTEGRATION_TARGET): $(INTEGRATION_TEST_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(CORE_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(CORE_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean-all: clean
	rm -rf $(GTEST_DIR)
	@echo "Cleaned everything including Google Test"

debug:
	@echo "UNIT_TEST_SOURCES: $(UNIT_TEST_SOURCES)"
	@echo "INTEGRATION_TEST_SOURCES: $(INTEGRATION_TEST_SOURCES)"
	@echo "SM120_UNIT_TARGET: $(SM120_UNIT_TARGET)"
	@echo "SM120_INTEGRATION_TARGET: $(SM120_INTEGRATION_TARGET)"
	@echo "SM90_INSTRUCTION_SOURCES: $(SM90_INSTRUCTION_SOURCES)"
	@echo "SM90_INSTRUCTION_TARGET: $(SM90_INSTRUCTION_TARGET)"
	@echo "CUH_HEADERS: $(CUH_HEADERS)"
