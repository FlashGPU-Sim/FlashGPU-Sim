# Default unit/integration tests, GoogleTest support, and standalone dev tests.

CORE_MK := $(lastword $(MAKEFILE_LIST))

UNIT_TEST_SOURCES = $(wildcard $(TEST_SRC_DIR)/unit/*_test.cc)
INTEGRATION_TEST_SOURCES = $(shell find $(TEST_SRC_DIR)/integration -name '*_test.cc' 2>/dev/null)
TEST_SOURCES = $(UNIT_TEST_SOURCES) $(INTEGRATION_TEST_SOURCES)
TEST_OBJECTS = $(UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o) \
               $(INTEGRATION_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
TEST_TARGETS = $(TEST_SOURCES:$(TEST_SRC_DIR)/%_test.cc=$(BIN_DIR)/%)

STANDALONE_SOURCES = $(wildcard $(TEST_SRC_DIR)/standalone/*_test.cc)
STANDALONE_OBJECTS = $(STANDALONE_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)

MAIN_TEST_TARGET = $(BIN_DIR)/run_all_tests
DEV_TEST_TARGET = $(BIN_DIR)/run_dev_tests

UNIT_OBJ_DIR = $(OBJ_DIR)/unit
INTEGRATION_OBJ_DIR = $(OBJ_DIR)/integration
INTEGRATION_MMA_OBJ_DIR = $(OBJ_DIR)/integration/mma
STANDALONE_OBJ_DIR = $(OBJ_DIR)/standalone
FLASH_OBJ_DIR = $(OBJ_DIR)/flash

# Only bulk_group is linked into the default verification runner.
FLASH_SOURCES = $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc
FLASH_OBJECTS = $(OBJ_DIR)/flash/bulk_group.cu.o

.PHONY: test dev setup-gtest clean clean-all debug

# Unit + integration tests used by run_tests.sh and CI.
test: setup-gtest $(MAIN_TEST_TARGET)

# Standalone dev tests remain a separate binary.
dev: setup-gtest $(DEV_TEST_TARGET)

setup-gtest:
	@if [ ! -d "$(GTEST_DIR)" ]; then \
		echo "Downloading Google Test..."; \
		git clone https://github.com/google/googletest.git $(GTEST_CLONE_DIR); \
		cd $(GTEST_DIR) && git checkout release-1.12.1; \
	else \
		echo "Google Test already exists."; \
	fi

$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(UNIT_OBJ_DIR) $(INTEGRATION_OBJ_DIR) \
$(INTEGRATION_MMA_OBJ_DIR) $(STANDALONE_OBJ_DIR) $(HOPPER_OBJ_DIR) \
$(FLASH_OBJ_DIR):
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

$(OBJ_DIR)/standalone/%.cu.o: $(TEST_SRC_DIR)/standalone/%.cc \
$(CUH_HEADERS) $(TOP_MAKEFILE) $(CORE_MK) | $(STANDALONE_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(BIN_DIR)/%_test: $(OBJ_DIR)/%_test.o $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread

$(MAIN_TEST_TARGET): $(TEST_OBJECTS) $(FLASH_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(CORE_MK) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(CORE_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(DEV_TEST_TARGET): $(STANDALONE_OBJECTS) $(OBJ_DIR)/gtest_main.a \
$(TOP_MAKEFILE) $(CORE_MK) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(CORE_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean-all: clean
	rm -rf $(GTEST_DIR)
	@echo "Cleaned everything including Google Test"

debug:
	@echo "TEST_SOURCES: $(TEST_SOURCES)"
	@echo "TEST_OBJECTS: $(TEST_OBJECTS)"
	@echo "TEST_TARGETS: $(TEST_TARGETS)"
	@echo "SM90_TEST_SOURCES: $(SM90_TEST_SOURCES)"
	@echo "SM90_TEST_TARGETS: $(SM90_TEST_TARGETS)"
	@echo "MAIN_TEST_TARGET: $(MAIN_TEST_TARGET)"
	@echo "SM90_TEST_TARGET: $(SM90_TEST_TARGET)"
	@echo "CUH_HEADERS: $(CUH_HEADERS)"
