# Unit-test-specific simulator support objects.

UNIT_MK := $(lastword $(MAKEFILE_LIST))

TEST_GROUP_EXTRA_OBJECTS_sm120_unit := \
	$(OBJ_DIR)/sm120/support/bulk_group.cu.o \
	$(OBJ_DIR)/sm120/support/tma_reduction.cu.o \
	$(OBJ_DIR)/sm120/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm120/support/mshr-table.cu.o

# Relink the unit binary when its support-object configuration changes.
TEST_GROUP_EXTRA_PREREQUISITES_sm120_unit := $(UNIT_MK)

$(OBJ_DIR)/sm120/support/bulk_group.cu.o: $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc \
$(SRC_DIR)/gpgpu-sim/flash/bulk_group.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/tma_reduction.cu.o: \
$(SRC_DIR)/gpgpu-sim/flash/tma_reduction.cc \
$(SRC_DIR)/gpgpu-sim/flash/tma_reduction.h \
$(SRC_DIR)/gpgpu-sim/flash/tensormap.h \
$(SRC_DIR)/cuda-sim/half.h $(SRC_DIR)/cuda-sim/half.hpp \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/local_interconnect.cc.o: \
$(SRC_DIR)/gpgpu-sim/local_interconnect.cc \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/mshr-table.cu.o: $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@
