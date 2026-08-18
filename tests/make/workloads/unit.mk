# Unit-test-specific simulator support objects.

UNIT_MK := $(lastword $(MAKEFILE_LIST))

TEST_GROUP_EXTRA_OBJECTS_sm100_unit := \
	$(OBJ_DIR)/sm100/support/addrdec.cc.o \
	$(OBJ_DIR)/sm100/support/hashing.cc.o \
	$(OBJ_DIR)/sm100/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm100/support/mshr-table.cc.o \
	$(OBJ_DIR)/sm100/support/option_parser.cc.o \
	$(OBJ_DIR)/sm100/support/tcgen05/descriptor.cu.o \
	$(OBJ_DIR)/sm100/support/tcgen05/mma.cu.o \
	$(OBJ_DIR)/sm100/support/tcgen05/tmem.cu.o

TEST_GROUP_EXTRA_OBJECTS_sm120_unit := \
	$(OBJ_DIR)/sm120/support/bulk_group.cu.o \
	$(OBJ_DIR)/sm120/support/tma_reduction.cu.o \
	$(OBJ_DIR)/sm120/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm120/support/mshr-table.cu.o

# Relink the unit binary when its support-object configuration changes.
TEST_GROUP_EXTRA_PREREQUISITES_sm100_unit := $(UNIT_MK)
TEST_GROUP_EXTRA_PREREQUISITES_sm120_unit := $(UNIT_MK)

# These host tests include simulator headers outside TEST_HEADERS. Keep their
# object layouts in lockstep with the production types they exercise so an
# incremental build cannot link stale test objects.
$(OBJ_DIR)/sm100/unit/local_interconnect_test.cc.o \
$(OBJ_DIR)/sm120/unit/local_interconnect_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h

$(OBJ_DIR)/sm100/unit/memory_transport_test.cc.o \
$(OBJ_DIR)/sm120/unit/memory_transport_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/shader.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h

$(OBJ_DIR)/sm100/unit/rop_delay_output_test.cc.o \
$(OBJ_DIR)/sm120/unit/rop_delay_output_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(SRC_DIR)/gpgpu-sim/l2cache.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h

$(OBJ_DIR)/sm100/unit/l2_bandwidth_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(SRC_DIR)/gpgpu-sim/l2cache.h

$(OBJ_DIR)/sm100/unit/l2_multi_issue_ports_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(SRC_DIR)/gpgpu-sim/l2cache.h \
$(SRC_DIR)/gpgpu-sim/mem_fetch.h

# Host-side simulator objects required by the SM100 unit test group.
$(OBJ_DIR)/sm100/support/tcgen05/%.cu.o: \
$(SRC_DIR)/gpgpu-sim/flash/tcgen05/%.cc \
$(wildcard $(SRC_DIR)/gpgpu-sim/flash/tcgen05/*.h) \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm100.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm100) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm100/support/addrdec.cc.o: $(SRC_DIR)/gpgpu-sim/addrdec.cc \
$(SRC_DIR)/gpgpu-sim/addrdec.h $(SRC_DIR)/gpgpu-sim/hashing.h \
$(SRC_DIR)/option_parser.h $(TOP_MAKEFILE) $(UNIT_MK) arch/sm100.toml \
$(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm100/support/hashing.cc.o: $(SRC_DIR)/gpgpu-sim/hashing.cc \
$(SRC_DIR)/gpgpu-sim/hashing.h $(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm100.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm100/support/mshr-table.cc.o: \
$(SRC_DIR)/gpgpu-sim/mshr-table.cc $(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm100.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm100/support/option_parser.cc.o: $(SRC_DIR)/option_parser.cc \
$(SRC_DIR)/option_parser.h $(TOP_MAKEFILE) $(UNIT_MK) arch/sm100.toml \
$(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm100/support/local_interconnect.cc.o: \
$(SRC_DIR)/gpgpu-sim/local_interconnect.cc \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm100.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Host-side simulator objects required by the SM120 unit test group.
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
$(SRC_DIR)/gpgpu-sim/local_interconnect.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/mshr-table.cu.o: $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@
