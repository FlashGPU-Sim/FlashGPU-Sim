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

$(OBJ_DIR)/sm90/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm100/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm120/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm100/unit/ptx_scheduler_test.cc.o: ../libcuda/gpgpu_context.h

$(OBJ_DIR)/sm100/unit/trace_config_test.cc.o: \
  $(SRC_DIR)/trace.h $(SRC_DIR)/trace_streams.tup

$(OBJ_DIR)/sm100/unit/scoreboard_forwarding_test.cc.o: \
  $(SRC_DIR)/gpgpu-sim/scoreboard.h $(SRC_DIR)/abstract_hardware_model.h

TEST_GROUP_EXTRA_OBJECTS_sm100_unit := \
	$(OBJ_DIR)/sm100/support/addrdec.cc.o \
	$(OBJ_DIR)/sm100/support/hashing.cc.o \
	$(OBJ_DIR)/sm100/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm100/support/mshr-table.cc.o \
	$(OBJ_DIR)/sm100/support/option_parser.cc.o \
	$(OBJ_DIR)/sm100/support/tcgen05/descriptor.cu.o \
	$(OBJ_DIR)/sm100/support/tcgen05/mma.cu.o \
	$(OBJ_DIR)/sm100/support/tcgen05/timing.cu.o \
	$(OBJ_DIR)/sm100/support/tcgen05/tmem.cu.o

TEST_GROUP_EXTRA_OBJECTS_sm120_unit := \
	$(OBJ_DIR)/sm120/support/instruction-stream-buffer.cc.o \
	$(OBJ_DIR)/sm120/support/bulk_group.cu.o \
	$(OBJ_DIR)/sm120/support/tma_store_source.cc.o \
	$(OBJ_DIR)/sm120/support/tma_reduction.cu.o \
	$(OBJ_DIR)/sm120/support/local_interconnect.cc.o \
	$(OBJ_DIR)/sm120/support/mshr-table.cu.o \
	$(OBJ_DIR)/sm120/support/gpu_topology.cu.o \
	$(OBJ_DIR)/sm120/support/option_parser.cc.o \
	$(OBJ_DIR)/sm120/support/dsm_fabric.cc.o \
	$(OBJ_DIR)/sm120/support/dsm_endpoint.cc.o \
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

TEST_GROUP_EXTRA_PREREQUISITES_sm100_unit := $(UNIT_MK)

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

$(OBJ_DIR)/sm120/unit/operand_collector_allocation_test.cc.o: \
  INCLUDES += -I.. -I../libcuda -I$(UNIT_SIM_BUILD_DIR)/cuda-sim
$(OBJ_DIR)/sm120/unit/operand_collector_allocation_test.cc.o: \
  $(SRC_DIR)/gpgpu-sim/shader.h $(SRC_DIR)/abstract_hardware_model.h \
  $(UNIT_MK)

$(OBJ_DIR)/sm100/unit/rop_delay_output_test.cc.o \
$(OBJ_DIR)/sm120/unit/rop_delay_output_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(SRC_DIR)/gpgpu-sim/l2cache.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h

$(OBJ_DIR)/sm100/unit/l2_bandwidth_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h \
$(SRC_DIR)/gpgpu-sim/l2cache.h

$(OBJ_DIR)/sm100/unit/dram_bandwidth_test.cc.o: \
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
$(SRC_DIR)/gpgpu-sim/local_interconnect.h \
$(SRC_DIR)/gpgpu-sim/mem_transport_budget.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT) $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/mshr-table.cu.o: $(SRC_DIR)/gpgpu-sim/mshr-table.cc \
$(SRC_DIR)/gpgpu-sim/gpu-cache.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_SASSIR_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) $(call ARCH_NVCCFLAGS,sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

# Register-view and guided-reorder regressions call the production PTX loader.
UNIT_SIM_LIB_DIR := $(abspath ../lib/$(GPGPUSIM_CONFIG))
UNIT_SIM_BUILD_DIR := $(abspath ../build/$(GPGPUSIM_CONFIG))
$(BIN_DIR)/sm90/unit_tests $(BIN_DIR)/sm100/unit_tests $(BIN_DIR)/sm120/unit_tests: \
  CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda \
    $(UNIT_SIM_LIB_DIR)/libcudart.so -Wl,-rpath,$(UNIT_SIM_LIB_DIR)
$(BIN_DIR)/sm90/unit_tests $(BIN_DIR)/sm100/unit_tests $(BIN_DIR)/sm120/unit_tests: \
  $(UNIT_SIM_LIB_DIR)/libcudart.so $(UNIT_MK)
$(OBJ_DIR)/sm90/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm100/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm120/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm90/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm100/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm120/unit/register_view_reload_test.cc.o: \
  INCLUDES += -I.. -I../libcuda -I$(UNIT_SIM_BUILD_DIR)/cuda-sim
$(OBJ_DIR)/sm90/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm100/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm120/unit/ptx_scheduler_test.cc.o \
$(OBJ_DIR)/sm90/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm100/unit/register_view_reload_test.cc.o \
$(OBJ_DIR)/sm120/unit/register_view_reload_test.cc.o: \
  $(UNIT_MK) $(wildcard ../libcuda/*.h) $(wildcard ../src/cuda-sim/*.h) \
  $(wildcard ../src/gpgpu-sim/*.h) ../src/abstract_hardware_model.h \
  $(UNIT_SIM_BUILD_DIR)/cuda-sim/ptx.tab.h

$(OBJ_DIR)/sm90/unit/mbarrier_retry_timing_test.cc.o \
$(OBJ_DIR)/sm100/unit/mbarrier_retry_timing_test.cc.o \
$(OBJ_DIR)/sm120/unit/mbarrier_retry_timing_test.cc.o: \
  $(SRC_DIR)/gpgpu-sim/flash/mbarrier.h

# Instruction-front-end host tests and their production support object.
$(OBJ_DIR)/sm120/unit/instruction_address_mapping_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/instruction_cache/address_mapping.h

$(OBJ_DIR)/sm120/unit/instruction_stream_buffer_test.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/instruction_cache/stream_buffer.h

$(OBJ_DIR)/sm120/support/instruction-stream-buffer.cc.o: \
$(SRC_DIR)/gpgpu-sim/flash/instruction_cache/stream_buffer.cc \
$(SRC_DIR)/gpgpu-sim/flash/instruction_cache/stream_buffer.h \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/gpu_topology.cu.o: $(SRC_DIR)/gpgpu-sim/gpu_topology.cc \
$(SRC_DIR)/gpgpu-sim/gpu_topology.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) $(INCLUDES) \
		$(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/option_parser.cc.o: $(SRC_DIR)/option_parser.cc \
$(SRC_DIR)/option_parser.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/dsm_fabric.cc.o: $(SRC_DIR)/gpgpu-sim/dsm_fabric.cc \
$(SRC_DIR)/gpgpu-sim/dsm_fabric.h $(SRC_DIR)/gpgpu-sim/transport.h \
$(SRC_DIR)/gpgpu-sim/gpu_topology.h $(TOP_MAKEFILE) $(UNIT_MK) \
arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm120/support/dsm_endpoint.cc.o: $(SRC_DIR)/gpgpu-sim/dsm_endpoint.cc \
$(SRC_DIR)/gpgpu-sim/dsm_endpoint.h $(SRC_DIR)/gpgpu-sim/dsm_fabric.h \
$(TOP_MAKEFILE) $(UNIT_MK) arch/sm120.toml $(ARCH_MANIFEST_SCRIPT)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

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

$(OBJ_DIR)/sm100/unit/kernel_info_frontend_test.cc.o \
$(OBJ_DIR)/sm120/unit/kernel_info_frontend_test.cc.o: \
  INCLUDES += -I.. -I../libcuda -I$(UNIT_SIM_BUILD_DIR)/cuda-sim
$(OBJ_DIR)/sm100/unit/kernel_info_frontend_test.cc.o \
$(OBJ_DIR)/sm120/unit/kernel_info_frontend_test.cc.o: \
  ../src/kernel_info.h ../libcuda/gpgpu_context.h $(UNIT_MK)

$(OBJ_DIR)/sm100/unit/kernel_info_frontend_test.cc.o \
$(OBJ_DIR)/sm120/unit/kernel_info_frontend_test.cc.o: \
  GPGPUSIM_FLAGS += -DFLASH_GPGPU_SIM_OMP

# Real-core regressions run separately from CUDA-bearing component tests:
# their subprocesses own a small GPU and the interconnect globals.
$(BIN_DIR)/sm120/core_tests: \
  CUDA_LIBS = $(UNIT_SIM_LIB_DIR)/libcudart.so -Wl,-rpath,$(UNIT_SIM_LIB_DIR)
$(BIN_DIR)/sm120/core_tests: $(UNIT_SIM_LIB_DIR)/libcudart.so $(UNIT_MK)
$(BIN_DIR)/sm120/core_tests: \
  GPGPUSIM_FLAGS += -DFLASH_GPGPU_SIM -DFLASH_GPGPU_SIM_OMP
$(OBJ_DIR)/sm120/core/predicated_atomic_test.cc.o \
$(OBJ_DIR)/sm120/core/sass_frontend_lifecycle_test.cc.o: \
  INCLUDES += -I.. -I../libcuda -I$(UNIT_SIM_BUILD_DIR)/cuda-sim
$(OBJ_DIR)/sm120/core/predicated_atomic_test.cc.o \
$(OBJ_DIR)/sm120/core/sass_frontend_lifecycle_test.cc.o: \
  $(wildcard ../libcuda/*.h) $(wildcard ../src/gpgpu-sim/*.h) \
  ../src/abstract_hardware_model.h $(UNIT_MK)

# SASS tests embed kernel/warp_state by value. Recompile their callers whenever
# the frontend ABI changes, not only when a test-local header changes.
SASS_HOST_HEADERS := $(wildcard $(SRC_DIR)/gpgpu-sim/flash/sass/*.h) \
  $(wildcard $(SRC_DIR)/gpgpu-sim/flash/sass/*/*.h)
$(patsubst $(TEST_SRC_DIR)/%.cc,$(OBJ_DIR)/sm120/%.cc.o,\
  $(wildcard $(TEST_SRC_DIR)/unit/sass/*/*.cc)) \
$(OBJ_DIR)/sm120/core/sass_frontend_lifecycle_test.cc.o: $(SASS_HOST_HEADERS)
