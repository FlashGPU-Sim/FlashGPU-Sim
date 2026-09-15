# Unit-test-specific simulator support objects.

UNIT_MK := $(lastword $(MAKEFILE_LIST))
SASS_UNIT_SASSIR_TOOL := \
	$(SRC_DIR)/gpgpu-sim/flash/sass/tools/nvdisasm_to_sassir.py
SASS_UNIT_SASSIR_DIR := $(BUILD_DIR)/generated/sass-unit
SASS_UNIT_GEMM_SASSIR := $(SASS_UNIT_SASSIR_DIR)/sass_gemm_sm120.sassir
SASS_UNIT_MMA_GEMM_SASSIR := \
	$(SASS_UNIT_SASSIR_DIR)/sass_mma_gemm_sm120.sassir
SASS_UNIT_TRITON_GEMM_SASSIR := \
	$(SASS_UNIT_SASSIR_DIR)/triton_gemm_sm120.sassir
SASS_UNIT_TRITON_ATTENTION_SASSIR := \
	$(SASS_UNIT_SASSIR_DIR)/triton_attention_sm120.sassir
SASS_UNIT_LLAMA_QKV_SASSIR := \
	$(SASS_UNIT_SASSIR_DIR)/llama_qkv_sm120.sassir
SASS_UNIT_LLAMA_RESIDUAL_SASSIR := \
	$(SASS_UNIT_SASSIR_DIR)/llama_residual_sm120.sassir
SASS_UNIT_TRITON_GEMM_CUBIN := \
	$(TEST_SRC_DIR)/../ci/perf/traces/SM120_RTX5090/gemm-m4096-n128-k4096/kernel_tma_gemm_launch1_kernel.cubin
SASS_UNIT_TRITON_ATTENTION_CUBIN := \
	$(TEST_SRC_DIR)/../ci/perf/traces/SM120_RTX5090/llama3-prefill-b2-s128/_llama3_gqa_attn_fwd_qkv_tiled_launch3_kernel.cubin
SASS_UNIT_LLAMA_QKV_CUBIN := \
	$(TEST_SRC_DIR)/../ci/perf/traces/SM120_RTX5090/llama3-prefill-b2-s128/_llama3_layer_matmul_tile_launch2_kernel.cubin
SASS_UNIT_LLAMA_RESIDUAL_CUBIN := \
	$(TEST_SRC_DIR)/../ci/perf/traces/SM120_RTX5090/llama3-decode-b256-kv128/_llama3_layer_matmul_residual_tile_launch4_kernel.cubin
SASS_UNIT_SASSIRS := \
	$(SASS_UNIT_GEMM_SASSIR) \
	$(SASS_UNIT_MMA_GEMM_SASSIR) \
	$(SASS_UNIT_TRITON_GEMM_SASSIR) \
	$(SASS_UNIT_TRITON_ATTENTION_SASSIR) \
	$(SASS_UNIT_LLAMA_QKV_SASSIR) \
	$(SASS_UNIT_LLAMA_RESIDUAL_SASSIR)

TEST_GROUP_EXTRA_OBJECTS_sm120_unit := \
	$(OBJ_DIR)/sm120/support/bulk_group.cu.o \
	$(OBJ_DIR)/sm120/support/tma_store_source.cc.o \
	$(OBJ_DIR)/sm120/support/tma_reduction.cu.o \
	$(OBJ_DIR)/sm120/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm120/support/mshr-table.cu.o \
	$(OBJ_DIR)/sm120/support/sass_frontend.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_common.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_control.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_data_movement.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_float.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_integer.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_mbarrier.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_memory.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_mma.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_tma.cc.o \
	$(OBJ_DIR)/sm120/support/sass_functional_wgmma.cc.o \
	$(OBJ_DIR)/sm120/support/sass_tensor_map.cc.o \
	$(OBJ_DIR)/sm120/support/sass_timing_projection.cc.o \
	$(OBJ_DIR)/sm120/support/sass_sassir_decoder.cc.o \
	$(OBJ_DIR)/sm120/support/sass_operand_parser.cc.o \
	$(OBJ_DIR)/sm120/support/sm90_control.cc.o \
	$(OBJ_DIR)/sm120/support/sm120_control.cc.o

# Relink the unit binary when its support-object configuration changes.
$(TEST_GROUP_EXTRA_OBJECTS_sm120_unit): $(SRC_DIR)/gpgpu-sim/flash/panic.h

TEST_GROUP_EXTRA_PREREQUISITES_sm120_unit := \
	$(UNIT_MK) $(SASS_UNIT_SASSIRS)

define GENERATE_SASS_UNIT_SASSIR
$(1): $(2) $(SASS_UNIT_SASSIR_TOOL)
	@mkdir -p $$(dir $$@)
	python3 $(SASS_UNIT_SASSIR_TOOL) \
		--nvdisasm "$(CUDA_INSTALL_PATH)/bin/nvdisasm" \
		--cuobjdump "$(CUDA_INSTALL_PATH)/bin/cuobjdump" \
		$$< '$(strip $(3))' -o $$@.tmp
	mv $$@.tmp $$@
endef

$(eval $(call GENERATE_SASS_UNIT_SASSIR,$(SASS_UNIT_GEMM_SASSIR),\
$(TEST_SRC_DIR)/unit/fixtures/sass_gemm_sm120.cubin,sass_gemm_2x2x4))
$(eval $(call GENERATE_SASS_UNIT_SASSIR,$(SASS_UNIT_MMA_GEMM_SASSIR),\
$(TEST_SRC_DIR)/unit/fixtures/sass_mma_gemm_sm120.cubin,\
sass_mma_gemm_m16n8k8))
$(eval $(call GENERATE_SASS_UNIT_SASSIR,$(SASS_UNIT_TRITON_GEMM_SASSIR),\
$(SASS_UNIT_TRITON_GEMM_CUBIN),kernel_tma_gemm))
$(eval $(call GENERATE_SASS_UNIT_SASSIR,\
$(SASS_UNIT_TRITON_ATTENTION_SASSIR),\
$(SASS_UNIT_TRITON_ATTENTION_CUBIN),\
_llama3_gqa_attn_fwd_qkv_tiled))
$(eval $(call GENERATE_SASS_UNIT_SASSIR,\
$(SASS_UNIT_LLAMA_QKV_SASSIR),\
$(SASS_UNIT_LLAMA_QKV_CUBIN),\
_llama3_layer_matmul_tile))
$(eval $(call GENERATE_SASS_UNIT_SASSIR,\
$(SASS_UNIT_LLAMA_RESIDUAL_SASSIR),\
$(SASS_UNIT_LLAMA_RESIDUAL_CUBIN),\
_llama3_layer_matmul_residual_tile))

$(OBJ_DIR)/sm120/support/bulk_group.cu.o: $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc \
$(SRC_DIR)/gpgpu-sim/flash/bulk_group.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) $(call ARCH_NVCCFLAGS,sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/tma_store_source.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/tma_store_source.cc \
$(SRC_DIR)/gpgpu-sim/flash/tma_store_source.h \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/tma_reduction.cu.o: \
$(SRC_DIR)/gpgpu-sim/flash/tma_reduction.cc \
$(SRC_DIR)/gpgpu-sim/flash/tma_reduction.h \
$(SRC_DIR)/gpgpu-sim/flash/tensormap.h \
$(SRC_DIR)/cuda-sim/half.h $(SRC_DIR)/cuda-sim/half.hpp \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) $(call ARCH_NVCCFLAGS,sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/local_interconnect.cc.o: \
$(SRC_DIR)/gpgpu-sim/local_interconnect.cc \
$(SRC_DIR)/gpgpu-sim/local_interconnect.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/mshr-table.cu.o: $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) $(call ARCH_NVCCFLAGS,sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_frontend.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/decoder.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_functional.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/functional.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/functional.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/internal.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/runtime/tensor_map.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_functional_%.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/%.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/internal.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/functional/functional.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/wgmma_instruction.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/runtime/tensor_map.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_tensor_map.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/runtime/tensor_map.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/runtime/tensor_map.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_timing_projection.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/timing/timing_projection.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/timing/timing_projection.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/wgmma_instruction.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h \
$(SRC_DIR)/abstract_hardware_model.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

SASS_UNIT_TEST_OBJECTS := $(patsubst $(TEST_SRC_DIR)/%.cc,$(OBJ_DIR)/sm120/%.cc.o,\
  $(wildcard $(TEST_SRC_DIR)/unit/sass/*/*_test.cc))

$(SASS_UNIT_TEST_OBJECTS): \
$(SRC_DIR)/gpgpu-sim/flash/async_proxy_timing.h \
$(SRC_DIR)/gpgpu-sim/flash/cta_barrier_timing.h \
$(SRC_DIR)/gpgpu-sim/flash/instruction_dependency_tracker.h \
$(SRC_DIR)/gpgpu-sim/flash/tensor_core_admission_timing.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/frontend.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/timing/timing_projection.h

$(SASS_UNIT_TEST_OBJECTS): CXXFLAGS += \
	-DSASS_GENERATED_SASSIR_DIR=\"$(abspath $(SASS_UNIT_SASSIR_DIR))\"

$(OBJ_DIR)/sm120/support/sass_sassir_decoder.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sassir_decoder.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sassir_decoder.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/decoder.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/operand_parser.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm90_control.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm120_control.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sass_operand_parser.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/operand_parser.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/operand_parser.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sm120_control.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm120_control.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/control_fields.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm120_control.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/sm90_control.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm90_control.cc \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/control_fields.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/decode/sm90_control.h \
$(SRC_DIR)/gpgpu-sim/flash/sass/ir.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@
