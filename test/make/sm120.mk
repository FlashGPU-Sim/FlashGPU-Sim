# SM120 unit and integration test binaries.

SM120_MK := $(lastword $(MAKEFILE_LIST))

SM120_UNIT_TEST_SOURCES = $(filter-out $(SM100_UNIT_TEST_SOURCES),\
                          $(wildcard $(TEST_SRC_DIR)/unit/*_test.cc))
INTEGRATION_TEST_SOURCES = $(shell find $(TEST_SRC_DIR)/integration -name '*_test.cc' 2>/dev/null)
SM120_UNIT_TEST_OBJECTS = $(SM120_UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
INTEGRATION_TEST_OBJECTS = $(INTEGRATION_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)

SM120_UNIT_TARGET = $(BIN_DIR)/sm120/run_unit_tests
SM120_INTEGRATION_TARGET = $(BIN_DIR)/sm120/run_integration_tests

UNIT_OBJ_DIR = $(OBJ_DIR)/unit
INTEGRATION_OBJ_DIR = $(OBJ_DIR)/integration
INTEGRATION_MMA_OBJ_DIR = $(OBJ_DIR)/integration/mma
FLASH_OBJ_DIR = $(OBJ_DIR)/flash
GPGPUSIM_OBJ_DIR = $(OBJ_DIR)/gpgpu-sim

# bulk_group_test exercises this implementation directly.
SM120_FLASH_OBJECTS = $(OBJ_DIR)/flash/bulk_group.cu.o
LOCAL_INTERCONNECT_OBJECT = $(OBJ_DIR)/unit/local_interconnect.cc.o
MSHR_TABLE_OBJECT = $(GPGPUSIM_OBJ_DIR)/mshr-table.cu.o

.PHONY: test-sm120-unit test-sm120-integration

test-sm120-unit: setup-gtest $(SM120_UNIT_TARGET)

test-sm120-integration: setup-gtest $(SM120_INTEGRATION_TARGET)

$(UNIT_OBJ_DIR) $(INTEGRATION_OBJ_DIR) $(INTEGRATION_MMA_OBJ_DIR) \
$(FLASH_OBJ_DIR) $(GPGPUSIM_OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)/flash/%.cu.o: $(SRC_DIR)/gpgpu-sim/flash/%.cc \
$(TOP_MAKEFILE) $(SM120_MK) | $(FLASH_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/unit/%.cu.o: $(TEST_SRC_DIR)/unit/%.cc $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(SM120_MK) | $(UNIT_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(LOCAL_INTERCONNECT_OBJECT): $(SRC_DIR)/gpgpu-sim/local_interconnect.cc \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h $(TOP_MAKEFILE) $(SM120_MK) | $(UNIT_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(MSHR_TABLE_OBJECT): $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(SM120_MK) | $(GPGPUSIM_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/integration/%.cu.o: $(TEST_SRC_DIR)/integration/%.cc \
$(CUH_HEADERS) $(TOP_MAKEFILE) $(SM120_MK)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM120_UNIT_TARGET): $(SM120_UNIT_TEST_OBJECTS) $(SM120_FLASH_OBJECTS) \
$(LOCAL_INTERCONNECT_OBJECT) $(MSHR_TABLE_OBJECT) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(SM120_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(SM120_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(SM120_INTEGRATION_TARGET): $(INTEGRATION_TEST_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(SM120_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(SM120_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
