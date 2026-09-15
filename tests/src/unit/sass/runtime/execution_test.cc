#include "../test_support.h"

namespace {

TEST(SassFunctionalFloatTest, ExecutesNonFtzUniformAddAndMultiply) {
  kernel image;
  image.name = "non_ftz_uniform_float";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UFMUL", "UR2,UR0,UR1"),
      official_instruction(0x10, "UFADD", "UR3,UR2,UR0"),
      official_instruction(0x20, "EXIT"),
  };
  frontend target;
  target.add_kernel(std::move(image));
  register_sm120_functional_semantics(target);
  execution_context context;
  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 0x3fc00000u);  // 1.5f
  state.write_uniform_register(1, 0x40000000u);  // 2.0f

  ASSERT_EQ(target.step("non_ftz_uniform_float", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(2), 0x40400000u);  // 3.0f

  ASSERT_EQ(target.step("non_ftz_uniform_float", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(3), 0x40900000u);  // 4.5f
}

TEST(SassWarpStateTest, PreservesArchitecturalZeroAndTrueRegisters) {
  warp_state state;
  state.write_register(3, 7, 42);
  state.write_register(3, 255, 99);
  state.write_uniform_register(9, 17);
  state.write_uniform_register(255, 99);
  state.write_predicate(3, 2, true);
  state.write_predicate(3, 7, false);
  state.write_uniform_predicate(4, true);
  state.write_uniform_predicate(7, false);

  EXPECT_EQ(state.read_register(3, 7), 42u);
  EXPECT_EQ(state.read_register(3, 255), 0u);
  EXPECT_EQ(state.read_uniform_register(9), 17u);
  EXPECT_EQ(state.read_uniform_register(255), 0u);
  EXPECT_TRUE(state.read_predicate(3, 2));
  EXPECT_TRUE(state.read_predicate(3, 7));
  EXPECT_TRUE(state.read_uniform_predicate(4));
  EXPECT_TRUE(state.read_uniform_predicate(7));
}

TEST(SassFunctionalTest, ExecutesRealTritonGemmThroughSassFrontend) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/triton_gemm_sm120.sassir";
  sassir_decoder decoder;
  frontend sass;
  sass.load(decoder, sassir, "kernel_tma_gemm");
  register_sm120_functional_semantics(sass);

  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0, 1024, false);
  memory.map(memory_space::kGlobal, 0, 512 * 1024, true);
  memory.map(memory_space::kShared, 0, 64 * 1024, true);
  const uint32_t one = 1;
  const uint32_t stack_pointer = 0x1000;
  const uint32_t dimension = 128;
  const uint64_t a_pointer = 0x10000;
  const uint64_t b_pointer = 0x20000;
  const uint64_t c_pointer = 0x30000;
  const uint64_t dimensions = dimension | (uint64_t{dimension} << 32);
  const uint64_t descriptor_scratch = 0x40000;
  memory.initialize(memory_space::kConstant, 0x370, one);
  memory.initialize(memory_space::kConstant, 0x374, one);
  memory.initialize(memory_space::kConstant, 0x37c, stack_pointer);
  memory.initialize(memory_space::kConstant, 0x380, a_pointer);
  memory.initialize(memory_space::kConstant, 0x388, b_pointer);
  memory.initialize(memory_space::kConstant, 0x390, c_pointer);
  memory.initialize(memory_space::kConstant, 0x398, dimensions);
  memory.initialize(memory_space::kConstant, 0x3a0, dimension);
  memory.initialize(memory_space::kConstant, 0x3a8, descriptor_scratch);
  constexpr uint16_t kOutputSentinel = 0x3555;
  const auto b_value = [](uint32_t column, uint32_t k) {
    return static_cast<int>((column * 17 + k * 7) % 31) - 15;
  };
  for (uint32_t row = 0; row < dimension; ++row) {
    for (uint32_t k = 0; k < dimension; ++k) {
      const int a_value = k == row % 64 || k == 64 + row % 64 ? 1 : 0;
      memory.initialize(memory_space::kGlobal,
                        a_pointer + (row * dimension + k) * sizeof(uint16_t),
                        exact_integer_half(a_value));
      memory.initialize(memory_space::kGlobal,
                        b_pointer + (row * dimension + k) * sizeof(uint16_t),
                        exact_integer_half(b_value(row, k)));
      memory.initialize(memory_space::kGlobal,
                        c_pointer + (row * dimension + k) * sizeof(uint16_t),
                        kOutputSentinel);
    }
  }

  cta_executor cta(sass, "kernel_tma_gemm", 4, &memory);

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(cta.instructions_executed(), 16384u)
        << "dynamic CTA did not reach a frontier, warp=" << cta.last_warp_id()
        << " pc=0x" << std::hex << cta.warp(cta.last_warp_id()).pc;
  } while (result.status == step_status::kAdvanced);

  EXPECT_EQ(result.status, step_status::kExited);
  EXPECT_EQ(cta.last_warp_id(), 0u);
  EXPECT_EQ(cta.instructions_executed(), 2787u);
  EXPECT_TRUE(result.detail.empty());
  for (unsigned warp = 0; warp < 4; ++warp)
    EXPECT_EQ(cta.warp(warp).pc, 0x3980u) << "warp " << warp;
  EXPECT_FALSE(cta.cta_state().mbarrier_initialized(0xc400));
  EXPECT_FALSE(cta.cta_state().mbarrier_initialized(0xc408));
  EXPECT_FALSE(cta.cta_state().mbarrier_initialized(0xc410));

  for (uint32_t row = 0; row < dimension; ++row) {
    for (uint32_t column = 0; column < dimension; ++column) {
      const uint16_t expected =
          row < 64 && column < 64
              ? exact_integer_half(b_value(column, row) +
                                   b_value(column, row + 64))
              : kOutputSentinel;
      EXPECT_EQ(memory.inspect<uint16_t>(
                    memory_space::kGlobal,
                    c_pointer + (row * dimension + column) * sizeof(uint16_t)),
                expected)
          << "row=" << row << " column=" << column;
    }
  }
}

TEST(SassFunctionalTest, KeepsLocalStackStatePrivateToEachThread) {
  kernel image;
  image.name = "local_stack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R1,0x20"),
      official_instruction(0x10, "S2R", "R0,SR_TID.X"),
      official_instruction(0x20, "STL", "[R1],R0"),
      official_instruction(0x30, "MOV", "R2,RZ"),
      official_instruction(0x40, "LDL.LU", "R2[R1]"),
      official_instruction(0x50, "MOV", "R4,0x89abcdef"),
      official_instruction(0x60, "MOV", "R5,0x1234567"),
      official_instruction(0x70, "STL.64", "[R1+0x8],R4"),
      official_instruction(0x80, "MOV", "R6,RZ"),
      official_instruction(0x90, "MOV", "R7,RZ"),
      official_instruction(0xa0, "LDL.LU.64", "R6[R1+0x8]"),
      official_instruction(0xb0, "LDL.64", "R8[R1+0x8]"),
      official_instruction(0xc0, "LDL", "R3[R1]"),
      official_instruction(0xd0, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kLocal, 0, 64 * 0x100, true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "local_stack", 2, &memory);
  for (unsigned warp = 0; warp < cta.warp_count(); ++warp)
    cta.context(warp).local_memory_thread_stride = 0x100;

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned warp = 0; warp < cta.warp_count(); ++warp)
    for (unsigned lane = 0; lane < 32; ++lane) {
      EXPECT_EQ(cta.warp(warp).read_register(lane, 2), warp * 32 + lane);
      EXPECT_EQ(cta.warp(warp).read_register(lane, 3), warp * 32 + lane);
      EXPECT_EQ(cta.warp(warp).read_register(lane, 6), 0x89abcdefu);
      EXPECT_EQ(cta.warp(warp).read_register(lane, 7), 0x01234567u);
      EXPECT_EQ(cta.warp(warp).read_register(lane, 8), 0x89abcdefu);
      EXPECT_EQ(cta.warp(warp).read_register(lane, 9), 0x01234567u);
    }
  EXPECT_EQ(memory.local_writes, 128u);
  EXPECT_EQ(memory.local_reads, 256u);
}

TEST(SassExecutionContextTest, RecoversArchitecturalLocalMemoryOffset) {
  execution_context context;
  context.local_memory_base = 0x100000;
  context.local_memory_thread_stride = 0x10000;
  context.thread_linear_id[7] = 39;

  uint64_t offset = 0;
  const uint64_t thread_base = 0x100000 + uint64_t{39} * 0x10000;
  EXPECT_TRUE(
      context.decode_local_memory_offset(7, thread_base + 0x8034, offset));
  EXPECT_EQ(offset, 0x8034u);
  EXPECT_FALSE(context.decode_local_memory_offset(7, thread_base - 1, offset));
  EXPECT_FALSE(context.decode_local_memory_offset(
      7, thread_base + context.local_memory_thread_stride, offset));
}

TEST(SassFunctionalTest, ExecutesRealTritonAttentionFunctionalGate) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/triton_attention_sm120.sassir";
  sassir_decoder decoder;
  frontend sass;
  sass.load(decoder, sassir, "_llama3_gqa_attn_fwd_qkv_tiled");
  register_sm120_functional_semantics(sass);

  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0, 1024, false);
  memory.map(memory_space::kGlobal, 0x100000, 8 * 1024 * 1024, true);
  memory.map(memory_space::kShared, 0, 64 * 1024, true);
  memory.map(memory_space::kLocal, 0, 9 * 1024 * 1024, true);
  const uint32_t grid_x = 2;
  const uint32_t grid_y = 64;
  const uint32_t stack_pointer = 0x6000;
  const uint64_t qkv_pointer = 0x100000;
  const uint64_t output_pointer = 0x400000;
  const float sm_scale = 0.08838834764831843f;
  const uint64_t descriptor_scratch = 0x700000;
  const uint64_t profile_scratch = 0;
  memory.initialize(memory_space::kConstant, 0x370, grid_x);
  memory.initialize(memory_space::kConstant, 0x374, grid_y);
  memory.initialize(memory_space::kConstant, 0x37c, stack_pointer);
  memory.initialize(memory_space::kConstant, 0x380, qkv_pointer);
  memory.initialize(memory_space::kConstant, 0x388, output_pointer);
  memory.initialize(memory_space::kConstant, 0x390, sm_scale);
  memory.initialize(memory_space::kConstant, 0x398, descriptor_scratch);
  memory.initialize(memory_space::kConstant, 0x3a0, profile_scratch);

  constexpr uint32_t kTotalHeads = 48;
  constexpr uint32_t kQueryHead = 0;
  constexpr uint32_t kKeyHead = 32;
  constexpr uint32_t kValueHead = 40;
  constexpr uint32_t kTileRows = 64;
  constexpr uint32_t kHeadDim = 128;
  constexpr uint16_t kOutputSentinel = 0x7bff;
  const auto qkv_address = [&](uint32_t tile, uint32_t head, uint32_t row,
                               uint32_t dimension) {
    const uint64_t element =
        (((static_cast<uint64_t>(tile) * kTotalHeads + head) * kTileRows +
          row) *
             kHeadDim +
         dimension);
    return qkv_pointer + element * sizeof(uint16_t);
  };
  for (uint32_t tile = 0; tile < 2; ++tile) {
    for (uint32_t row = 0; row < kTileRows; ++row) {
      for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        const int value =
            static_cast<int>((tile * kTileRows + row + dimension * 3) % 5) - 2;
        memory.initialize(memory_space::kGlobal,
                          qkv_address(tile, kValueHead, row, dimension),
                          exact_integer_half(value));
      }
    }
  }
  for (uint32_t row = 0; row < kTileRows; ++row) {
    for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
      const int q_value = (row * 3 + dimension * 5) % 17 < 2 ? 1 : 0;
      const int k_value = (row * 7 + dimension * 3) % 19 < 3 ? 1 : 0;
      memory.initialize(memory_space::kGlobal,
                        qkv_address(0, kQueryHead, row, dimension),
                        exact_integer_half(q_value));
      memory.initialize(memory_space::kGlobal,
                        qkv_address(0, kKeyHead, row, dimension),
                        exact_integer_half(k_value));
    }
  }
  for (uint32_t row = 0; row < kTileRows; ++row)
    for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension)
      memory.initialize(
          memory_space::kGlobal,
          output_pointer + (static_cast<uint64_t>(row) * kHeadDim + dimension) *
                               sizeof(uint16_t),
          kOutputSentinel);

  cta_executor cta(sass, "_llama3_gqa_attn_fwd_qkv_tiled", 4, &memory);
  for (unsigned warp = 0; warp < cta.warp_count(); ++warp)
    cta.context(warp).local_memory_thread_stride = 0x10000;
  step_result result;
  do {
    result = cta.step();
    ASSERT_LT(cta.instructions_executed(), 1000000u);
  } while (result.status == step_status::kAdvanced);

  EXPECT_EQ(result.status, step_status::kExited) << result.detail;
  EXPECT_EQ(cta.instructions_executed(), 13003u);
  for (unsigned warp = 0; warp < cta.warp_count(); ++warp)
    EXPECT_EQ(cta.warp(warp).pc, 0xe1e0u) << "warp " << warp;

  float maximum_error = 0.0f;
  unsigned mismatches = 0;
  for (uint32_t row = 0; row < kTileRows; ++row) {
    std::array<float, kTileRows> scores{};
    float maximum_score = -std::numeric_limits<float>::infinity();
    for (uint32_t key = 0; key <= row; ++key) {
      float dot = 0.0f;
      for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        const int q_value = (row * 3 + dimension * 5) % 17 < 2 ? 1 : 0;
        const int k_value = (key * 7 + dimension * 3) % 19 < 3 ? 1 : 0;
        dot += q_value * k_value;
      }
      scores[key] = dot * sm_scale;
      maximum_score = std::max(maximum_score, scores[key]);
    }
    float denominator = 0.0f;
    for (uint32_t key = 0; key <= row; ++key) {
      scores[key] = std::exp(scores[key] - maximum_score);
      denominator += scores[key];
    }
    for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
      float numerator = 0.0f;
      for (uint32_t key = 0; key <= row; ++key) {
        const int value = static_cast<int>((key + dimension * 3) % 5) - 2;
        numerator += scores[key] * value;
      }
      const float expected = numerator / denominator;
      const uint16_t bits = memory.inspect<uint16_t>(
          memory_space::kGlobal,
          output_pointer + (static_cast<uint64_t>(row) * kHeadDim + dimension) *
                               sizeof(uint16_t));
      const float actual = half_to_float(bits);
      const float error = std::fabs(actual - expected);
      maximum_error = std::max(maximum_error, error);
      if (!std::isfinite(actual) || bits == kOutputSentinel || error > 0.02f)
        ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << "maximum absolute error=" << maximum_error;
  EXPECT_LE(maximum_error, 0.02f);
}

TEST(SassFrontendTest, FetchesByDynamicPcAndDispatchesSemantics) {
  kernel image;
  image.name = "kernel";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  instruction nop;
  nop.pc = 0;
  nop.opcode = "NOP";
  nop.decoded = true;
  image.instructions.push_back(nop);

  frontend sass;
  sass.add_kernel(std::move(image));
  sass.register_semantics("NOP", [](const instruction &inst, warp_state &state,
                                    execution_context &) {
    state.write_register(0, 1, state.read_register(0, 1) + 1);
    state.pc = inst.pc + 16;
    return step_result{step_status::kAdvanced, {}};
  });

  warp_state state;
  execution_context context;
  EXPECT_EQ(sass.step("kernel", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.pc, 16u);
  EXPECT_EQ(state.read_register(0, 1), 1u);
  EXPECT_EQ(context.instructions_executed, 1u);
  EXPECT_EQ(sass.step("kernel", state, context).status,
            step_status::kMissingPc);
}

TEST(SassFunctionalTest, ExecutesRealSm120CubinGemmWithoutPtxFallback) {
  constexpr uint64_t kA = 0x10000000;
  constexpr uint64_t kB = 0x20000000;
  constexpr uint64_t kC = 0x30000000;
  constexpr uint64_t kParameterA = 0x380;
  constexpr uint64_t kParameterB = 0x388;
  constexpr uint64_t kParameterC = 0x390;

  const std::array<float, 8> a = {1.0f,  2.0f, 3.0f, 4.0f,
                                  -1.0f, 0.5f, 2.0f, -3.0f};
  const std::array<float, 8> b = {2.0f,  -1.0f, 0.5f, 3.0f,
                                  -2.0f, 4.0f,  1.0f, 2.0f};

  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0, 1024, false);
  memory.map(memory_space::kGlobal, kA, sizeof(a), false);
  memory.map(memory_space::kGlobal, kB, sizeof(b), false);
  memory.map(memory_space::kGlobal, kC, 4 * sizeof(float), true);
  memory.initialize(memory_space::kConstant, kParameterA, kA);
  memory.initialize(memory_space::kConstant, kParameterB, kB);
  memory.initialize(memory_space::kConstant, kParameterC, kC);
  for (size_t i = 0; i < a.size(); ++i)
    memory.initialize(memory_space::kGlobal, kA + i * sizeof(float), a[i]);
  for (size_t i = 0; i < b.size(); ++i)
    memory.initialize(memory_space::kGlobal, kB + i * sizeof(float), b[i]);

  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/sass_gemm_sm120.sassir";
  sassir_decoder decoder;
  frontend sass;
  sass.load(decoder, sassir, "sass_gemm_2x2x4");
  register_sm120_functional_semantics(sass);

  const kernel *image = sass.find_kernel("sass_gemm_2x2x4");
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->decoder_name, "nvdisasm");
  EXPECT_FALSE(image->decoder_version.empty());
  ASSERT_EQ(image->instructions.size(), 56u);
  size_t official_attributes = 0;
  for (const instruction &inst : image->instructions)
    if (!inst.decoded)
      ADD_FAILURE() << "undecoded cubin instruction at pc 0x" << std::hex
                    << inst.pc;
    else
      official_attributes += inst.attributes.size();
  EXPECT_EQ(official_attributes, 2u);

  warp_state state;
  state.active_mask = 1;
  execution_context context{&memory};
  step_result result;
  do {
    result = sass.step("sass_gemm_2x2x4", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 100u);
  } while (result.status != step_status::kExited);

  std::array<float, 4> reference{};
  for (unsigned m = 0; m < 2; ++m) {
    for (unsigned n = 0; n < 2; ++n) {
      for (unsigned k = 0; k < 4; ++k)
        reference[m * 2 + n] =
            std::fma(a[m * 4 + k], b[k * 2 + n], reference[m * 2 + n]);
    }
  }
  for (size_t i = 0; i < reference.size(); ++i)
    EXPECT_FLOAT_EQ(
        memory.inspect<float>(memory_space::kGlobal, kC + i * sizeof(float)),
        reference[i]);

  EXPECT_EQ(state.pc, 0x290u);
  EXPECT_EQ(state.active_mask, 0u);
  EXPECT_EQ(context.instructions_executed, 42u);
  EXPECT_EQ(memory.constant_reads, 5u);
  EXPECT_EQ(memory.global_reads, 16u);
  EXPECT_EQ(memory.global_writes, 4u);
}

TEST(SassFunctionalTest, ExecutesRealSm120TensorCoreGemmWithoutPtxFallback) {
  constexpr uint64_t kA = 0x40000000;
  constexpr uint64_t kB = 0x50000000;
  constexpr uint64_t kD = 0x60000000;
  constexpr uint64_t kParameterA = 0x380;
  constexpr uint64_t kParameterB = 0x388;
  constexpr uint64_t kParameterD = 0x390;
  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0, 1024, false);
  memory.map(memory_space::kGlobal, kA, 16 * 8 * sizeof(uint16_t), false);
  memory.map(memory_space::kGlobal, kB, 8 * 8 * sizeof(uint16_t), false);
  memory.map(memory_space::kGlobal, kD, 16 * 8 * sizeof(float), true);
  memory.initialize(memory_space::kConstant, kParameterA, kA);
  memory.initialize(memory_space::kConstant, kParameterB, kB);
  memory.initialize(memory_space::kConstant, kParameterD, kD);
  std::array<float, 16 * 8> a{};
  std::array<float, 8 * 8> b{};
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned k = 0; k < 8; ++k) {
      const int value = static_cast<int>((row * 3 + k * 2) % 7) - 3;
      a[row * 8 + k] = static_cast<float>(value);
      memory.initialize(memory_space::kGlobal,
                        kA + (row * 8 + k) * sizeof(uint16_t),
                        exact_integer_half(value));
    }
  }
  // B is stored column-major as b[column][k], matching the MMA fragment ABI.
  for (unsigned column = 0; column < 8; ++column) {
    for (unsigned k = 0; k < 8; ++k) {
      const int value = static_cast<int>((k * 2 + column * 3) % 5) - 2;
      b[column * 8 + k] = static_cast<float>(value);
      memory.initialize(memory_space::kGlobal,
                        kB + (column * 8 + k) * sizeof(uint16_t),
                        exact_integer_half(value));
    }
  }

  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/sass_mma_gemm_sm120.sassir";
  sassir_decoder decoder;
  frontend sass;
  sass.load(decoder, sassir, "sass_mma_gemm_m16n8k8");
  register_sm120_functional_semantics(sass);

  const kernel *image = sass.find_kernel("sass_mma_gemm_m16n8k8");
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->decoder_name, "nvdisasm");
  EXPECT_FALSE(image->decoder_schema.empty());
  ASSERT_EQ(image->instructions.size(), 40u);
  for (const instruction &inst : image->instructions)
    ASSERT_TRUE(inst.decoded)
        << "undecoded cubin instruction at pc 0x" << std::hex << inst.pc;

  warp_state state;
  execution_context context;
  context.memory = &memory;
  for (unsigned lane = 0; lane < 32; ++lane) context.thread_idx_x[lane] = lane;
  step_result result;
  do {
    result = sass.step("sass_mma_gemm_m16n8k8", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 100u);
  } while (result.status != step_status::kExited);

  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned column = 0; column < 8; ++column) {
      float reference = 0.0f;
      for (unsigned k = 0; k < 8; ++k)
        reference = std::fma(a[row * 8 + k], b[column * 8 + k], reference);
      const unsigned output = row * 8 + column;
      EXPECT_FLOAT_EQ(memory.inspect<float>(memory_space::kGlobal,
                                            kD + output * sizeof(float)),
                      reference)
          << "mismatch at output element " << output;
    }
  }

  EXPECT_EQ(state.pc, 0x1b0u);
  EXPECT_EQ(state.active_mask, 0u);
  EXPECT_EQ(context.instructions_executed, 28u);
  EXPECT_EQ(memory.constant_reads, 5u);
  EXPECT_EQ(memory.global_reads, 192u);
  EXPECT_EQ(memory.global_writes, 128u);
}

}  // namespace
