#include "../test_support.h"

namespace {

TEST(SassFunctionalTest, ExecutesGpuStrongGlobalLoad) {
  kernel image;
  image.name = "gpu_strong_global_load";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "LDG.E.STRONG.GPU", "R4,desc[UR8][R2.64]"),
      official_instruction(0x10, "EXIT"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t address = 0x200000120ull;
  memory.map(memory_space::kGlobal, address, sizeof(uint32_t), true);
  memory.initialize<uint32_t>(memory_space::kGlobal, address, 0x12345678u);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, static_cast<uint32_t>(address));
  state.write_register(0, 3, static_cast<uint32_t>(address >> 32));
  execution_context context{&memory};

  ASSERT_EQ(sass.step("gpu_strong_global_load", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x12345678u);
  EXPECT_EQ(memory.global_reads, 1u);
}

TEST(SassFunctionalTest, ReadsFlatSharedWindowHighAddress) {
  kernel image;
  image.name = "shared_window_high";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "S2UR", "UR4,SR_SWINHI"),
      official_instruction(0x10, "UMOV.64", "UR6,0x4"),
      official_instruction(0x20, "UMOV.64", "UR8,0xffffffffffffffc0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  ASSERT_EQ(sass.step("shared_window_high", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(4), 0u);
  ASSERT_EQ(sass.step("shared_window_high", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(6), 4u);
  EXPECT_EQ(state.read_uniform_register(7), 0u);
  ASSERT_EQ(sass.step("shared_window_high", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(8), 0xffffffc0u);
  EXPECT_EQ(state.read_uniform_register(9), 0xffffffffu);
}

TEST(SassFunctionalTest, LoadsDynamicStaticConstantBankAddress) {
  kernel image;
  image.name = "static_constant_bank";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR4,0x4"),
      official_instruction(0x10, "LDCU", "UR6,c[0x2][UR4]"),
      official_instruction(0x20, "MOV", "R0,0x4"),
      official_instruction(0x30, "LDC", "R8,c[0x2][R0+0x4]"),
      official_instruction(0x40, "EXIT"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t bank_two = uint64_t{2} << 32;
  memory.map(memory_space::kConstant, bank_two, 12, false);
  memory.initialize(memory_space::kConstant, bank_two + 4,
                    uint32_t{0x12345678});
  memory.initialize(memory_space::kConstant, bank_two + 8,
                    uint32_t{0x89abcdef});
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("static_constant_bank", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_uniform_register(6), 0x12345678u);
  EXPECT_EQ(state.read_register(0, 8), 0x89abcdefu);
  EXPECT_EQ(memory.constant_reads, 2u);
}

TEST(SassFunctionalTest, ExecutesGenericSharedAndGlobalAccesses) {
  kernel image;
  image.name = "generic_shared";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ST.E", "desc[UR4][R2.64],R5"),
      official_instruction(0x10, "LD.E", "R6,desc[UR4][R2.64]"),
      official_instruction(0x20, "ST.E.128", "desc[UR8][R10.64],R12"),
      official_instruction(0x30, "LD.E.128", "R16,desc[UR8][R10.64]"),
      official_instruction(0x40, "LD.E.U16.STRONG.SYS",
                           "R20,desc[UR8][R10.64]"),
      official_instruction(0x50, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint32_t), true);
  memory.map(memory_space::kGlobal, 0x1000, 4 * sizeof(uint32_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x100);
  state.write_register(0, 3, 0);
  state.write_register(0, 5, 0x12345678);
  state.write_uniform_register(8, 1);
  state.write_register(0, 10, 0x1000);
  state.write_register(0, 11, 0);
  for (unsigned word = 0; word < 4; ++word)
    state.write_register(0, 12 + word, 0xabc00000 + word);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("generic_shared", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 6), 0x12345678u);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x100),
            0x12345678u);
  for (unsigned word = 0; word < 4; ++word) {
    EXPECT_EQ(state.read_register(0, 16 + word), 0xabc00000u + word);
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal,
                                       0x1000 + word * sizeof(uint32_t)),
              0xabc00000u + word);
  }
  EXPECT_EQ(state.read_register(0, 20), 0u);
}

TEST(SassFunctionalTest, AggregatesPopcAtomicIncrementBySharedAddress) {
  kernel image;
  image.name = "atoms_popc_increment";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ATOMS.POPC.INC.32", "RZ[R0+URZ]", "@P0"),
      official_instruction(0x10, "ATOMS.ADD", "RZ[R0+URZ],R1", "@P0"),
      official_instruction(0x20, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, 2 * sizeof(uint32_t), true);
  memory.initialize<uint32_t>(memory_space::kShared, 0x100, 10);
  memory.initialize<uint32_t>(memory_space::kShared, 0x104, 20);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0xf;
  state.write_register(0, 0, 0x100);
  state.write_register(1, 0, 0x100);
  state.write_register(2, 0, 0x104);
  state.write_register(3, 0, 0x104);
  state.write_register(0, 1, 3);
  state.write_register(1, 1, 4);
  state.write_register(2, 1, 5);
  state.write_register(3, 1, 6);
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, true);
  state.write_predicate(2, 0, true);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("atoms_popc_increment", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x100), 19u);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x104), 26u);
}

TEST(SassFunctionalTest, ReturnsSharedAtomicOldValueWithoutGlobalMemoryAccess) {
  constexpr unsigned zero_register = 255;
  for (const auto arch : {architecture::kSm90, architecture::kSm120}) {
    for (const unsigned destination : {0u, 1u, 2u, zero_register}) {
      SCOPED_TRACE(destination);
      kernel image;
      image.name = "shared_atomic_add";
      image.arch = arch;
      image.instruction_bytes = 16;
      const std::string reg = destination == zero_register
                                  ? "RZ" : "R" + std::to_string(destination);
      image.instructions = {official_instruction(
          0, "ATOMS.ADD", reg + "[R0+URZ],R1", "@P0")};
      mapped_functional_memory memory;
      memory.map(memory_space::kShared, 0x100, sizeof(uint32_t), true);
      memory.initialize<uint32_t>(memory_space::kShared, 0x100, 0xfffffffeu);
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm120_functional_semantics(sass);
      warp_state state;
      state.active_mask = 7;
      for (unsigned lane = 0; lane < 3; ++lane) {
        state.write_register(lane, 0, 0x100);
        state.write_register(lane, 1, 3 + lane);
        state.write_register(lane, 2, 99);
        state.write_predicate(lane, 0, lane < 2);
      }
      const auto inactive = state.read_register(2, destination);
      execution_context context{&memory};
      const auto result = sass.step("shared_atomic_add", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x100), 5u);
      if (destination != zero_register) {
        EXPECT_EQ(state.read_register(0, destination), 0xfffffffeu);
        EXPECT_EQ(state.read_register(1, destination), 1u);
      }
      EXPECT_EQ(state.read_register(2, destination), inactive);
      EXPECT_EQ(memory.global_reads, 0u);
      for (unsigned lane = 0; lane < 2; ++lane) {
        ASSERT_EQ(context.memory_accesses[lane].size(), 1u);
        EXPECT_EQ(context.memory_accesses[lane][0].space, memory_space::kShared);
        EXPECT_EQ(context.memory_accesses[lane][0].address, 0x100u);
        EXPECT_FALSE(context.memory_accesses[lane][0].write);
      }
      EXPECT_TRUE(context.memory_accesses[2].empty());
    }
  }
}

TEST(SassFunctionalTest, CopiesUniformBulkTileFromSharedToGlobal) {
  kernel image;
  image.name = "uniform_bulk_copy";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR4,0x4"),
      official_instruction(0x10, "UMOV", "UR5,0x200"),
      official_instruction(0x20, "UMOV", "UR6,0x1000"),
      official_instruction(0x30, "UMOV", "UR7,URZ"),
      official_instruction(0x40, "UBLKCP.G.S", "[UR6][UR5],UR4"),
      official_instruction(0x50, "MEMBAR.ALL.GPU"),
      official_instruction(0x60, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x200, 64, false);
  memory.map(memory_space::kGlobal, 0x1000, 64, true);
  for (uint32_t index = 0; index < 16; ++index)
    memory.initialize(memory_space::kShared, 0x200 + index * sizeof(uint32_t),
                      0x100u + index);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  execution_context context{&memory};
  flash_gpgpu_sim::sass::functional_tma_effect timing_effect;
  step_result result;
  do {
    result = sass.step("uniform_bulk_copy", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    if (context.tma_effects[0].valid) timing_effect = context.tma_effects[0];
  } while (result.status != step_status::kExited);

  for (uint32_t index = 0; index < 16; ++index)
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal,
                                       0x1000 + index * sizeof(uint32_t)),
              0x100u + index);
  EXPECT_EQ(memory.shared_reads, 1u);
  EXPECT_EQ(memory.global_writes, 1u);
  EXPECT_TRUE(timing_effect.valid);
  EXPECT_EQ(timing_effect.destination_address, 0x1000u);
  EXPECT_EQ(timing_effect.source_address, 0x200u);
  EXPECT_EQ(timing_effect.size_bytes, 64u);
}

TEST(SassFunctionalTest, ReducesUniformBulkFloatTileIntoGlobal) {
  kernel image;
  image.name = "uniform_bulk_reduce";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR4,0x1"),
      official_instruction(0x10, "UMOV", "UR5,0x200"),
      official_instruction(0x20, "UMOV", "UR6,0x1000"),
      official_instruction(0x30, "UMOV", "UR7,URZ"),
      official_instruction(0x40, "UMOV", "UR8,URZ"),
      official_instruction(0x50, "UMOV", "UR9,0x10000000"),
      official_instruction(0x60, "UBLKRED.G.S.ADD.F32.RN",
                           "[UR6][UR5],UR4,desc[UR8]"),
      official_instruction(0x70, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x200, 4 * sizeof(float), false);
  memory.map(memory_space::kGlobal, 0x1000, 4 * sizeof(float), true);
  const std::array<float, 4> source{{1.0f, -2.0f, 0.5f, 4.0f}};
  const std::array<float, 4> initial{{10.0f, 20.0f, -0.5f, 6.0f}};
  const std::array<float, 4> expected{{11.0f, 18.0f, 0.0f, 10.0f}};
  memory.initialize(memory_space::kShared, 0x200, source);
  memory.initialize(memory_space::kGlobal, 0x1000, initial);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  execution_context context{&memory};
  flash_gpgpu_sim::sass::functional_tma_effect timing_effect;
  step_result result;
  do {
    result = sass.step("uniform_bulk_reduce", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    if (context.tma_effects[0].valid) timing_effect = context.tma_effects[0];
  } while (result.status != step_status::kExited);

  for (size_t index = 0; index < expected.size(); ++index)
    EXPECT_EQ(memory.inspect<float>(memory_space::kGlobal,
                                    0x1000 + index * sizeof(float)),
              expected[index]);
  EXPECT_TRUE(timing_effect.valid);
  EXPECT_EQ(timing_effect.destination_address, 0x1000u);
  EXPECT_EQ(timing_effect.source_address, 0x200u);
  EXPECT_EQ(timing_effect.size_bytes, 16u);

  interleavable_functional_memory concurrent_memory;
  concurrent_memory.storage = memory;
  concurrent_memory.storage.initialize(memory_space::kGlobal, 0x1000, initial);
  std::array<std::thread, 4> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      warp_state local;
      local.active_mask = 1;
      local.write_uniform_register(4, 1);
      local.write_uniform_register(5, 0x200);
      local.write_uniform_register(6, 0x1000);
      local.write_uniform_register(9, 0x10000000);
      execution_context local_context{&concurrent_memory};
      for (unsigned i = 0; i < 128; ++i) {
        local.pc = 0x60;
        ASSERT_EQ(sass.step("uniform_bulk_reduce", local, local_context).status,
                  step_status::kAdvanced);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  for (unsigned i = 0; i < source.size(); ++i)
    EXPECT_EQ(concurrent_memory.storage.inspect<float>(
                  memory_space::kGlobal, 0x1000 + i * sizeof(float)),
              initial[i] + 512 * source[i]);
}

TEST(SassFunctionalTest, WrapsSharedAddressArithmeticAt32Bits) {
  kernel image;
  image.name = "shared_address_wrap";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "STS", "[R0+UR14+-0x40],R2"),
      official_instruction(0x10, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, sizeof(uint32_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0x180);
  state.write_register(0, 2, 0x12345678);
  state.write_uniform_register(14, 0xfffffec0u);
  execution_context context{&memory};

  const step_result result = sass.step("shared_address_wrap", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0), 0x12345678u);
}

TEST(SassFunctionalTest, PreservesNarrowConstantAndGlobalMemoryWidths) {
  kernel image;
  image.name = "u16_memory";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "LDC.U8", "R4,c[0x0][0x380]"),
      official_instruction(0x10, "LDC.U16", "R5,c[0x0][0x380]"),
      official_instruction(0x20, "STG.E.U16", "desc[UR4][R2.64],R5"),
      official_instruction(0x30, "LDG.E.U16", "R6,desc[UR4][R2.64+0x2]"),
      official_instruction(0x40, "LDG.E.U8", "R7,desc[UR4][R2.64+0x1]"),
      official_instruction(0x50, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0x380, sizeof(uint16_t), false);
  memory.map(memory_space::kGlobal, 0x1000, sizeof(uint32_t), true);
  memory.initialize(memory_space::kConstant, 0x380, uint16_t{0xabcd});
  memory.initialize(memory_space::kGlobal, 0x1000, uint32_t{0x11223344});

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x1000);
  state.write_register(0, 3, 0);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("u16_memory", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 4), 0xcdu);
  EXPECT_EQ(state.read_register(0, 5), 0xabcdu);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, 0x1000),
            0x1122abcdu);
  EXPECT_EQ(state.read_register(0, 6), 0x1122u);
  EXPECT_EQ(state.read_register(0, 7), 0xabu);
}

TEST(SassFunctionalTest, Preserves64BitGlobalMemoryWidths) {
  kernel image;
  image.name = "global_store_64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "STG.E.64", "desc[UR4][R2.64+0x4],R6"),
      official_instruction(0x10, "LDG.E.64", "R8,desc[UR4][R2.64+0x4]"),
      official_instruction(0x20, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1000, 16, true);
  memory.initialize(memory_space::kGlobal, 0x1000,
                    uint64_t{0x1111222233334444ull});
  memory.initialize(memory_space::kGlobal, 0x1008,
                    uint64_t{0x5555666677778888ull});

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x1000);
  state.write_register(0, 3, 0);
  state.write_register(0, 6, 0x89abcdef);
  state.write_register(0, 7, 0x01234567);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("global_store_64", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, 0x1000),
            0x33334444u);
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, 0x1004),
            0x0123456789abcdefull);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, 0x100c),
            0x55556666u);
  EXPECT_EQ(state.read_register(0, 8), 0x89abcdefu);
  EXPECT_EQ(state.read_register(0, 9), 0x01234567u);
}

TEST(SassFunctionalTest, LoadsAndZeroStores128BitGlobalVectors) {
  kernel image;
  image.name = "global_load_128";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "LDG.E.128.CONSTANT",
                           "R4,desc[UR8][R2.64+0x10]"),
      official_instruction(0x10, "LDG.E.128", "R8,desc[UR8][R2.64+0x20]"),
      official_instruction(0x20, "STG.E.128", "desc[UR8][R2.64+0x30],RZ"),
      official_instruction(0x30, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1010, 48, true);
  const std::array<uint32_t, 4> expected{0x01234567u, 0x89abcdefu, 0x13579bdfu,
                                         0x2468ace0u};
  const std::array<uint32_t, 4> ordinary_expected{0xdeadbeefu, 0xc001d00du,
                                                  0x76543210u, 0xfedcba98u};
  memory.initialize(memory_space::kGlobal, 0x1010, expected);
  memory.initialize(memory_space::kGlobal, 0x1020, ordinary_expected);
  memory.initialize(memory_space::kGlobal, 0x1030, ordinary_expected);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x1000);
  state.write_register(0, 3, 0);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("global_load_128", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned word = 0; word < expected.size(); ++word)
    EXPECT_EQ(state.read_register(0, 4 + word), expected[word]);
  for (unsigned word = 0; word < ordinary_expected.size(); ++word)
    EXPECT_EQ(state.read_register(0, 8 + word), ordinary_expected[word]);
  for (unsigned word = 0; word < ordinary_expected.size(); ++word)
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal,
                                       0x1030 + word * sizeof(uint32_t)),
              0u);
}

TEST(SassFunctionalTest, ExecutesPackedHalfConstantAndU8SharedAccesses) {
  kernel image;
  image.name = "half2_u8_shared";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HFMA2",
                           "R3,-RZ,RZ,0,5.36441802978515625e-06"),
      official_instruction(0x10, "STS.U8", "[R4],R3"),
      official_instruction(0x20, "LDS.U8", "R5[R4]"),
      official_instruction(0x30, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x400, sizeof(uint32_t), true);
  memory.initialize(memory_space::kShared, 0x400, uint32_t{0x11223344});

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 4, 0x400);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("half2_u8_shared", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 3), 0x5au);
  EXPECT_EQ(state.read_register(0, 5), 0x5au);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x400),
            0x1122335au);
}

TEST(SassFunctionalTest, PreservesWideSharedLoadAndStoreWidths) {
  kernel image;
  image.name = "wide_shared";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "STS.128", "[R2],R4"),
      official_instruction(0x10, "LDS.128", "R8[R2]"),
      official_instruction(0x20, "STS.64", "[R3],R8"),
      official_instruction(0x30, "LDS.64", "R12[R3]"),
      official_instruction(0x40, "STS.128", "[R14],RZ"),
      official_instruction(0x50, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, 0x40, true);
  memory.initialize(memory_space::kShared, 0x130,
                    uint64_t{0xffffffffffffffffu});
  memory.initialize(memory_space::kShared, 0x138,
                    uint64_t{0xffffffffffffffffu});

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x100);
  state.write_register(0, 3, 0x120);
  state.write_register(0, 14, 0x130);
  for (unsigned word = 0; word < 4; ++word)
    state.write_register(0, 4 + word, 0x11111111u * (word + 1));
  execution_context context{&memory};

  step_result result = sass.step("wide_shared", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  ASSERT_EQ(context.memory_accesses[0].size(), 1u);
  EXPECT_EQ(context.memory_accesses[0][0].address, 0x100u);
  EXPECT_EQ(context.memory_accesses[0][0].bytes, 16u);
  EXPECT_TRUE(context.memory_accesses[0][0].write);

  result = sass.step("wide_shared", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  ASSERT_EQ(context.memory_accesses[0].size(), 1u);
  EXPECT_EQ(context.memory_accesses[0][0].address, 0x100u);
  EXPECT_EQ(context.memory_accesses[0][0].bytes, 16u);
  EXPECT_FALSE(context.memory_accesses[0][0].write);

  do {
    result = sass.step("wide_shared", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned word = 0; word < 4; ++word)
    EXPECT_EQ(state.read_register(0, 8 + word), 0x11111111u * (word + 1));
  EXPECT_EQ(state.read_register(0, 12), 0x11111111u);
  EXPECT_EQ(state.read_register(0, 13), 0x22222222u);
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x130), 0u);
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x138), 0u);
}

TEST(SassFunctionalTest, ExecutesLdGstsCopyEnableAndSourceSizeLowering) {
  kernel image;
  image.name = "ldgsts_source_size";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "LDGSTS.E.BYPASS.128",
                           "[R2],desc[UR4][R4.64]"),
      official_instruction(0x10, "LDGSTS.E.BYPASS.128.ZFILL",
                           "[R3],desc[UR4][R6.64+0xc],P0"),
      official_instruction(0x20, "LDGSTS.E.BYPASS.128",
                           "[R8],desc[UR4][R4.64],!PT"),
      official_instruction(0x30, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1000, 16, false);
  memory.map(memory_space::kShared, 0x200, 0x40, true);
  for (unsigned word = 0; word < 4; ++word)
    memory.initialize(memory_space::kGlobal, 0x1000 + word * 4,
                      uint32_t{0x11111111u * (word + 1)});

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x200);
  state.write_register(0, 3, 0x210);
  state.write_register(0, 4, 0x1000);
  state.write_register(0, 5, 0);
  state.write_register(0, 6, 0x1000);
  state.write_register(0, 7, 0);
  state.write_register(0, 8, 0x220);
  state.write_predicate(0, 0, true);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("ldgsts_source_size", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned word = 0; word < 4; ++word)
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x200 + word * 4),
              0x11111111u * (word + 1));
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x210),
            0x11111111u);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x214), 0u);
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x220), 0u);
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x228), 0u);
}

TEST(SassFunctionalTest, ExecutesUniform64BitAddAndSharedStore) {
  kernel image;
  image.name = "uniform_64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR4,0x100"),
      official_instruction(0x10, "UMOV", "UR5,0x1"),
      official_instruction(0x20, "UMOV", "UR6,0x20"),
      official_instruction(0x30, "UMOV", "UR7,0x2"),
      official_instruction(0x40, "UIADD3.64", "UR8,UPT,UPT,UR4,UR6,URZ"),
      official_instruction(0x50, "MOV.64", "R10,UR8"),
      official_instruction(0x60, "UMOV", "UR12,0x180"),
      official_instruction(0x70, "STS.64", "[UR12],R10"),
      official_instruction(0x80, "MOV", "R12,0xabcd1234"),
      official_instruction(0x90, "STS.U16", "[UR12+0x8],R12"),
      official_instruction(0xa0, "LDS.U16", "R13[UR12+0x8]"),
      official_instruction(0xb0, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x180, 16, true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("uniform_64", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  constexpr uint64_t expected = 0x0000000300000120ull;
  EXPECT_EQ(state.read_uniform_register(8), 0x120u);
  EXPECT_EQ(state.read_uniform_register(9), 0x3u);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 10), 0x120u);
    EXPECT_EQ(state.read_register(lane, 11), 0x3u);
    EXPECT_EQ(state.read_register(lane, 13), 0x1234u);
  }
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x180), expected);
  EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kShared, 0x188), 0x1234u);
  EXPECT_EQ(memory.shared_writes, 64u);
  EXPECT_EQ(memory.shared_reads, 32u);
}

TEST(SassFunctionalTest, ExecutesDiscardedGlobalExchangeAndFunctionalFences) {
  kernel image;
  image.name = "atomic_exchange";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "S2R", "R4,SR_LANEID"),
      official_instruction(0x10, "ATOMG.E.EXCH.STRONG.GPU", "PT,RZ[R2+UR6],R4"),
      official_instruction(0x20, "DEPBAR", "{5,4,3,2,1,0}"),
      official_instruction(0x30, "CCTL.E.C.LDCU.IV.DEEP", "[UR4]"),
      official_instruction(0x40, "UTMACCTL.IV", "[UR4]"),
      official_instruction(0x50, "UTMACCTL.PF", "[UR4]"),
      official_instruction(0x60, "UTMACMDFLUSH"),
      official_instruction(0x70, "DEPBAR.LE", "SB0,0x0"),
      official_instruction(0x80, "FENCE.VIEW.ASYNC.S"),
      official_instruction(0x90, "MEMBAR.ALL.CTA"),
      official_instruction(0xa0, "MEMBAR.SC.CTA"),
      official_instruction(0xb0, "CCTL.IVALL"),
      official_instruction(0xc0, "ACQBULK"),
      official_instruction(0xd0, "EXIT"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t address = 0x200000120ull;
  memory.map(memory_space::kGlobal, address, sizeof(uint32_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 2, 0x100);
    state.write_register(lane, 3, 0x1);
  }
  state.write_uniform_register(6, 0x20);
  state.write_uniform_register(7, 0x1);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("atomic_exchange", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 31u);
  EXPECT_EQ(memory.global_reads, 32u);
  EXPECT_EQ(memory.global_writes, 32u);
}

TEST(SassFunctionalTest, ExecutesDescriptorAddressedDiscardedGlobalExchange) {
  kernel image;
  image.name = "descriptor_atomic_exchange";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ATOMG.E.EXCH.STRONG.GPU",
                           "PT,RZ,desc[UR8][R10.64],R3"),
      official_instruction(0x10, "EXIT"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t address = 0x200000120ull;
  memory.map(memory_space::kGlobal, address, sizeof(uint32_t), true);
  memory.initialize<uint32_t>(memory_space::kGlobal, address, 7u);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x1;
  state.write_register(0, 3, 42u);
  state.write_register(0, 10, static_cast<uint32_t>(address));
  state.write_register(0, 11, static_cast<uint32_t>(address >> 32));
  execution_context context{&memory};

  ASSERT_EQ(sass.step("descriptor_atomic_exchange", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 42u);
  EXPECT_EQ(memory.global_reads, 1u);
  EXPECT_EQ(memory.global_writes, 1u);
}

TEST(SassFunctionalTest, ExecutesGlobalAtomicAddWithReturnedOldValue) {
  kernel image;
  image.name = "atomic_add";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "S2R", "R6,SR_LTMASK"),
      official_instruction(0x10, "ATOMG.E.ADD.STRONG.GPU",
                           "PT,R2,desc[UR22][R2.64],R5"),
      official_instruction(0x20, "EXIT"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t address = 0x200000120ull;
  memory.map(memory_space::kGlobal, address, sizeof(uint32_t), true);
  memory.initialize<uint32_t>(memory_space::kGlobal, address, 10u);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 2, static_cast<uint32_t>(address));
    state.write_register(lane, 3, static_cast<uint32_t>(address >> 32));
  }
  state.write_register(0, 5, 5u);
  state.write_register(1, 5, 7u);
  execution_context context{&memory};

  ASSERT_EQ(sass.step("atomic_add", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 6), 0u);
  EXPECT_EQ(state.read_register(1, 6), 1u);
  const step_result result = sass.step("atomic_add", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.read_register(0, 2), 10u);
  EXPECT_EQ(state.read_register(1, 2), 15u);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 22u);

  // Separate read/write calls can each be race-free while the simulated RMW
  // still loses updates. Yield between them to exercise concurrent SMs.
  interleavable_functional_memory concurrent_memory;
  concurrent_memory.storage.map(memory_space::kGlobal, address, 4, true);
  std::array<std::thread, 4> workers;
  std::array<uint64_t, 4> old_sums{};
  for (unsigned worker = 0; worker < workers.size(); ++worker) {
    workers[worker] = std::thread([&, worker] {
      warp_state local;
      local.active_mask = 1;
      local.write_register(0, 5, 1u);
      execution_context local_context{&concurrent_memory};
      for (unsigned i = 0; i < 128; ++i) {
        local.pc = 0x10;
        local.write_register(0, 2, static_cast<uint32_t>(address));
        local.write_register(0, 3, static_cast<uint32_t>(address >> 32));
        ASSERT_EQ(sass.step("atomic_add", local, local_context).status,
                  step_status::kAdvanced);
        old_sums[worker] += local.read_register(0, 2);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(concurrent_memory.storage.inspect<uint32_t>(memory_space::kGlobal,
                                                       address), 512u);
  EXPECT_EQ(old_sums[0] + old_sums[1] + old_sums[2] + old_sums[3],
            uint64_t{512} * 511 / 2);
}

}  // namespace
