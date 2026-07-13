# SM90-only instruction verification tests.

SM90_MK := $(lastword $(MAKEFILE_LIST))

SM90_TEST_SOURCES = $(HOPPER_SRC_DIR)/named_barrier_test.cc \
                    $(sort $(wildcard $(HOPPER_SRC_DIR)/wgmma/*_test.cc))
SM90_TEST_CUH_HEADERS = $(shell find $(TEST_COMMON_DIR) $(SRC_DIR) -name '*.cuh' 2>/dev/null) \
                        $(wildcard $(HOPPER_SRC_DIR)/wgmma/*.cuh)
SM90_TEST_OBJECTS = $(SM90_TEST_SOURCES:$(HOPPER_SRC_DIR)/%.cc=$(OBJ_DIR)/hopper/%.cu.o)
SM90_TEST_TARGET = $(BIN_DIR)/sm90/run_sm90_tests
ifneq ($(strip $(SM90_TEST_SOURCES)),)
SM90_TEST_TARGETS = $(SM90_TEST_TARGET)
endif

.PHONY: test-sm90

test-sm90: setup-gtest $(SM90_TEST_TARGETS)

# These objects do not depend on FA2/FA3 or a FlashAttention checkout.
$(OBJ_DIR)/hopper/%.cu.o: $(HOPPER_SRC_DIR)/%.cc \
$(SM90_TEST_CUH_HEADERS) $(TOP_MAKEFILE) $(SM90_MK) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM90_TEST_TARGET): $(SM90_TEST_OBJECTS) $(OBJ_DIR)/gtest_main.a \
$(TOP_MAKEFILE) $(SM90_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(SM90_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
