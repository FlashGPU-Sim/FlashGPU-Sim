# Non-FlashAttention SM90 verification tests.

SM90_MK := $(lastword $(MAKEFILE_LIST))

SM90_INSTRUCTION_SOURCES = $(HOPPER_SRC_DIR)/named_barrier_test.cc \
                           $(sort $(wildcard $(HOPPER_SRC_DIR)/wgmma/*_test.cc))
SM90_INSTRUCTION_CUH_HEADERS = $(shell find $(TEST_COMMON_DIR) $(SRC_DIR) -name '*.cuh' 2>/dev/null) \
                               $(wildcard $(HOPPER_SRC_DIR)/wgmma/*.cuh)
SM90_INSTRUCTION_OBJECTS = $(SM90_INSTRUCTION_SOURCES:$(HOPPER_SRC_DIR)/%.cc=$(OBJ_DIR)/hopper/%.cu.o)
SM90_INSTRUCTION_TARGET = $(BIN_DIR)/sm90/run_instruction_tests
ifneq ($(strip $(SM90_INSTRUCTION_SOURCES)),)
SM90_INSTRUCTION_TARGETS = $(SM90_INSTRUCTION_TARGET)
endif

.PHONY: test-sm90-instructions

test-sm90-instructions: setup-gtest $(SM90_INSTRUCTION_TARGETS)

# These objects do not depend on FA2/FA3 or a FlashAttention checkout.
$(OBJ_DIR)/hopper/%.cu.o: $(HOPPER_SRC_DIR)/%.cc \
$(SM90_INSTRUCTION_CUH_HEADERS) $(TOP_MAKEFILE) $(SM90_MK) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(SM90_INSTRUCTION_TARGET): $(SM90_INSTRUCTION_OBJECTS) $(OBJ_DIR)/gtest_main.a \
$(TOP_MAKEFILE) $(SM90_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(SM90_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
