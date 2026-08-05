# SM100/B200 FA4 and TCGen05 unit tests.

SM100_MK := $(lastword $(MAKEFILE_LIST))

SM100_UNIT_TEST_SOURCES = \
	$(TEST_SRC_DIR)/unit/fa4_opaque_tensormap_test.cc \
	$(TEST_SRC_DIR)/unit/tcgen05_tmem_test.cc

SM100_OBJ_DIR = $(OBJ_DIR)/sm100
SM100_UNIT_OBJ_DIR = $(SM100_OBJ_DIR)/unit
SM100_FLASH_OBJ_DIR = $(SM100_OBJ_DIR)/flash

SM100_UNIT_TEST_OBJECTS = \
	$(SM100_UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/unit/%.cc=$(SM100_UNIT_OBJ_DIR)/%.cu.o)
SM100_FLASH_OBJECTS = \
	$(SM100_FLASH_OBJ_DIR)/tcgen05/descriptor.cu.o \
	$(SM100_FLASH_OBJ_DIR)/tcgen05/mma.cu.o \
	$(SM100_FLASH_OBJ_DIR)/tcgen05/tmem.cu.o \
	$(SM100_FLASH_OBJ_DIR)/tcgen05/timing.cu.o

SM100_UNIT_TARGET = $(BIN_DIR)/sm100/run_unit_tests

.PHONY: test-sm100-unit

# Keep direct Makefile use architecture-correct. A command-line CUDA_ARCH
# (including the value supplied by run_tests.sh) still takes precedence.
test-sm100-unit $(SM100_UNIT_TARGET): CUDA_ARCH = sm_100a

test-sm100-unit: setup-gtest $(SM100_UNIT_TARGET)

$(SM100_OBJ_DIR) $(SM100_UNIT_OBJ_DIR) $(SM100_FLASH_OBJ_DIR):
	mkdir -p $@

$(SM100_UNIT_OBJ_DIR)/%.cu.o: $(TEST_SRC_DIR)/unit/%.cc $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(SM100_MK) | $(SM100_UNIT_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM100_FLASH_OBJ_DIR)/%.cu.o: $(SRC_DIR)/gpgpu-sim/flash/%.cc \
$(TOP_MAKEFILE) $(SM100_MK) | $(SM100_FLASH_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM100_UNIT_TARGET): $(SM100_UNIT_TEST_OBJECTS) $(SM100_FLASH_OBJECTS) \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(SM100_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(SM100_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
