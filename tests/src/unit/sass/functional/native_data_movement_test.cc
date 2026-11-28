#include "../test_support.h"
namespace {
frontend image_for(std::vector<instruction> instructions) {
  kernel image;
  image.name = "test";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = std::move(instructions);
  frontend f;
  f.add_kernel(std::move(image));
  register_sm100_functional_semantics(f);
  return f;
}
TEST(SassNativeDataMovement, DirectLoadWithUniformOffsetUsesOneAddressPair) {
  for (const char *operands : {"R4[R2.64+UR10+0x80]", "R4[R2+UR10+0x80]"}) {
    auto f =
        image_for({official_instruction(0, "LDG.E.LTC128B", operands, "@P0")});
    mapped_functional_memory memory;
    constexpr uint64_t base = 0x100000100ull;
    constexpr uint64_t offset = 0xfffffff0ull;
    const auto address = base + offset + 0x80;
    memory.map(memory_space::kGlobal, address, 8, true);
    memory.initialize<uint32_t>(memory_space::kGlobal, address, 42);
    memory.initialize<uint32_t>(memory_space::kGlobal, address + 4, 73);
    warp_state state;
    state.active_mask = 7;
    state.write_uniform_register(10, offset);
    state.write_uniform_register(11, 0x12345678);  // Must not form a UR pair.
    for (unsigned lane = 0; lane < 3; ++lane) {
      state.write_register(lane, 2, uint32_t(base) + 4 * lane);
      state.write_register(lane, 3, base >> 32);
      state.write_register(lane, 4, 999);
      state.write_predicate(lane, 0, lane != 2);
    }
    execution_context context{&memory};
    const auto result = f.step("test", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.read_register(0, 4), 42u);
    EXPECT_EQ(state.read_register(1, 4), 73u);
    EXPECT_EQ(state.read_register(2, 4), 999u);
    EXPECT_EQ(context.memory_accesses[0][0].address, address);
    EXPECT_EQ(context.memory_accesses[1][0].address, address + 4);
    EXPECT_TRUE(context.memory_accesses[2].empty());
  }
}
TEST(SassNativeDataMovement, DirectUniformOffsetTimingKeepsThreeSources) {
  auto inst = official_instruction(0, "LDG.E.LTC128B", "R4[R2.64+UR10+0x80]");
  ASSERT_TRUE(flash_gpgpu_sim::sass::normalize_ldg_operands(inst));
  const timing_instruction timing(
      official_instruction(0, "LDG.E.LTC128B", "R4[R2.64+UR10+0x80]"));
  EXPECT_EQ(timing.incount, 3u);
  EXPECT_EQ(timing.outcount, 1u);
  EXPECT_EQ(timing.data_size, 4u);
  ASSERT_EQ(inst.operands.size(), 2u);
  const auto &terms = inst.operands[1].address_groups[0].terms;
  EXPECT_TRUE(terms[0].wide);
  EXPECT_FALSE(terms[1].wide);
}
TEST(SassNativeDataMovement, DirectUniformOffsetRejectsUnqualifiedAddresses) {
  for (const char *operands :
       {"R4[UR10]", "R4[R2.64+R6.64+UR10]", "R4[R2.64-UR10]",
        "R4[R2.64+UR10+UR12]", "R4[R2.64+UR256]"}) {
    auto inst = official_instruction(0, "LDG.E.LTC128B", operands);
    EXPECT_FALSE(inst.operands_structured &&
                 flash_gpgpu_sim::sass::normalize_ldg_operands(inst))
        << operands;
    EXPECT_EQ(inst.operands.size(), 1u);
  }
}
TEST(SassNativeDataMovement, DirectStoreUsesImplicitAddressPairAndLaneGuards) {
  for (const char *address : {"[R130]", "[R130.64]", "[R130+UR10+0x80]"}) {
    auto f = image_for({official_instruction(
        0, "STG.E", std::string(address) + ",R65", "@P1")});
    mapped_functional_memory memory;
    constexpr uint64_t base = 0x100000100ull;
    const bool offset = std::string(address).find("UR") != std::string::npos;
    const auto location = base + (offset ? 0xfffffff0ull + 0x80 : 0);
    memory.map(memory_space::kGlobal, location, 12, true);
    warp_state state;
    state.active_mask = 7;
    state.write_uniform_register(10, 0xfffffff0u);
    state.write_uniform_register(11, 0x12345678);
    for (unsigned lane = 0; lane < 3; ++lane) {
      state.write_register(lane, 130, uint32_t(base) + 4 * lane);
      state.write_register(lane, 131, base >> 32);
      state.write_register(lane, 65, 42 + lane);
      state.write_predicate(lane, 1, lane != 2);
      memory.initialize<uint32_t>(memory_space::kGlobal, location + lane * 4,
                                  999);
    }
    execution_context context{&memory};
    const auto result = f.step("test", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    for (unsigned lane = 0; lane < 3; ++lane)
      EXPECT_EQ(
          memory.inspect<uint32_t>(memory_space::kGlobal, location + lane * 4),
          lane == 2 ? 999u : 42 + lane);
    EXPECT_EQ(context.memory_accesses[0][0].address, location);
    EXPECT_EQ(context.memory_accesses[1][0].address, location + 4);
    EXPECT_TRUE(context.memory_accesses[2].empty());
  }
}
TEST(SassNativeDataMovement, DirectStoreTimingReadsHighAddressWord) {
  const timing_instruction timing(
      official_instruction(0, "STG.E", "[R130],R65"));
  EXPECT_EQ(timing.incount, 3u);
  EXPECT_EQ(timing.outcount, 0u);
  EXPECT_EQ(timing.data_size, 4u);
  std::vector<unsigned> regs;
  for (unsigned i = 0; i < timing.incount; ++i)
    regs.push_back(timing.arch_reg.src[i]);
  for (unsigned r : {65u, 130u, 131u})
    EXPECT_NE(std::find(regs.begin(), regs.end(), r + 1), regs.end()) << r;
}
TEST(SassNativeDataMovement, DirectStoreRejectsUnqualifiedAddresses) {
  for (const char *address : {"[UR10]", "[R2.64+R6.64]", "[R2-UR10]",
                              "[R2+UR10+UR12]", "[R254]", "R4[R2]"}) {
    auto f = image_for(
        {official_instruction(0, "STG.E", std::string(address) + ",R8")});
    mapped_functional_memory memory;
    memory.map(memory_space::kGlobal, 0x100, 8, true);
    memory.initialize<uint32_t>(memory_space::kGlobal, 0x100, 777);
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 2, 0x100);
    execution_context context{&memory};
    EXPECT_EQ(f.step("test", state, context).status, step_status::kUnsupported)
        << address;
    EXPECT_TRUE(context.memory_accesses[0].empty());
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, 0x100), 777u);
  }
}
TEST(SassNativeDataMovement, UniformSpillFillRoundtripAndLaneGuards) {
  auto f = image_for({official_instruction(0, "MOV.SPILL", "R5,UR6"),
                      official_instruction(16, "R2UR.FILL", "UR6,R5")});
  for (unsigned mask : {0xffffffffu, 1u, 1u << 7, 1u << 31})
    for (unsigned seed : {0u, 1u, 0xffffffffu, 0xa55a0ff0u}) {
      warp_state s;
      s.active_mask = mask;
      s.write_uniform_register(6, seed);
      for (unsigned lane = 0; lane < 32; ++lane)
        s.write_register(lane, 5, 0x1000 + lane);
      execution_context c;
      ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(s.read_register(lane, 5),
                  mask & (1u << lane) ? seed : 0x1000 + lane);
      s.write_uniform_register(6, ~seed);
      ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
      EXPECT_EQ(s.read_uniform_register(6), seed);
    }
}
TEST(SassNativeDataMovement, RejectsUnqualifiedSpillFillWithoutMutation) {
  for (const char *opcode : {"MOV.SPILL", "R2UR.FILL"}) {
    const bool fill = std::string(opcode) == "R2UR.FILL";
    auto f = image_for(
        {official_instruction(0, opcode, fill ? "UR6,R5" : "R5,UR6")});
    warp_state s;
    s.active_mask = 3;
    s.write_uniform_register(6, 123);
    s.write_register(0, 5, 42);
    execution_context c;
    EXPECT_EQ(f.step("test", s, c).status, step_status::kUnsupported);
    EXPECT_EQ(s.pc, 0u);
    EXPECT_EQ(s.read_uniform_register(6), 123u);
    EXPECT_EQ(s.read_register(0, 5), 42u);
  }
  auto f = image_for({official_instruction(0, "R2UR.FILL", "UR6,R5")});
  warp_state s;
  s.write_uniform_register(6, 123);
  s.write_register(31, 5, 1);
  execution_context c;
  EXPECT_EQ(f.step("test", s, c).status, step_status::kUnsupported);
  EXPECT_EQ(s.read_uniform_register(6), 123u);
}
TEST(SassNativeDataMovement, WideGlobalRoundtripUsesTwoNoncontiguousGroups) {
  for (const char *store : {"STG.E.ENL2.256", "STG.E.NA.EFL2.256"}) {
    const auto load = official_instruction(0, "LDG.E.ENL2.256",
                                           "R20,R4,desc[UR8][R2.64]", "@P0");
    const auto save =
        official_instruction(16, store, "desc[UR8][R2.64+0x40],R4,R20", "@P0");
    auto f = image_for({load, save});
    mapped_functional_memory m;
    m.map(memory_space::kGlobal, 0x100, 128, true);
    for (unsigned word = 0; word < 32; ++word)
      m.initialize<uint32_t>(memory_space::kGlobal, 0x100 + 4 * word,
                             100 + word);
    warp_state s;
    s.active_mask = 3;
    for (unsigned lane = 0; lane < 2; ++lane) {
      s.write_register(lane, 2, 0x100);
      s.write_predicate(lane, 0, lane == 0);
      for (unsigned i = 0; i < 4; ++i) {
        s.write_register(lane, 4 + i, 999);
        s.write_register(lane, 20 + i, 999);
      }
    }
    execution_context c{&m};
    ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
    for (unsigned i = 0; i < 4; ++i) {
      EXPECT_EQ(s.read_register(0, 4 + i), 100 + i);
      EXPECT_EQ(s.read_register(0, 20 + i), 104 + i);
      EXPECT_EQ(s.read_register(1, 4 + i), 999u);
      EXPECT_EQ(s.read_register(1, 20 + i), 999u);
    }
    ASSERT_EQ(c.memory_accesses[0].size(), 1u);
    EXPECT_EQ(c.memory_accesses[0][0].bytes, 32u);
    ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
    ASSERT_EQ(c.memory_accesses[0].size(), 1u);
    EXPECT_EQ(c.memory_accesses[0][0].bytes, 32u);
    EXPECT_TRUE(c.memory_accesses[1].empty());
    for (unsigned word = 0; word < 32; ++word)
      EXPECT_EQ(m.inspect<uint32_t>(memory_space::kGlobal, 0x100 + 4 * word),
                word >= 16 && word < 24 ? 100 + word - 16 : 100 + word);
    const timing_instruction timing(save);
    EXPECT_EQ(timing.data_size, 32u);
    std::vector<unsigned> regs;
    for (unsigned i = 0; i < timing.get_register_file_source_count(); ++i)
      regs.push_back(timing.get_register_file_source(i).reg);
    for (unsigned r : {4u, 5u, 6u, 7u, 20u, 21u, 22u, 23u})
      EXPECT_NE(std::find(regs.begin(), regs.end(), r), regs.end()) << r;
    EXPECT_EQ(std::count(regs.begin(), regs.end(), 8u), 0);
  }
}
TEST(SassNativeDataMovement, SignedByteLoadsSignExtend) {
  for (unsigned byte : {0u, 127u, 128u, 255u}) {
    auto f =
        image_for({official_instruction(0, "LDG.E.S8", "R4,desc[UR8][R2.64]")});
    mapped_functional_memory m;
    m.map(memory_space::kGlobal, 0x100, 1, true);
    m.initialize<uint8_t>(memory_space::kGlobal, 0x100, byte);
    warp_state s;
    s.active_mask = 1;
    s.write_register(0, 2, 0x100);
    execution_context c{&m};
    ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(s.read_register(0, 4), byte < 128 ? byte : byte | 0xffffff00u);
    ASSERT_EQ(c.memory_accesses[0].size(), 1u);
    EXPECT_EQ(c.memory_accesses[0][0].bytes, 1u);
    const timing_instruction timing(
        official_instruction(0, "LDG.E.S8", "R4,desc[UR8][R2.64]"));
    EXPECT_EQ(timing.data_size, 1u);
  }
}
TEST(SassNativeDataMovement, Generic64AndNoAllocate128StoreWidths) {
  for (const bool shared : {false, true}) {
    auto f =
        image_for({official_instruction(0, "ST.E.64", "desc[UR4][R2.64],R8")});
    mapped_functional_memory m;
    auto space = shared ? memory_space::kShared : memory_space::kGlobal;
    m.map(space, 0x100, 16, true);
    warp_state s;
    s.active_mask = 1;
    s.write_uniform_register(4, shared ? 0 : 1);
    s.write_register(0, 2, 0x100);
    s.write_register(0, 8, 0x12345678);
    s.write_register(0, 9, 0xabcdef01);
    m.initialize<uint32_t>(space, 0x108, 777);
    execution_context c{&m};
    ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(m.inspect<uint64_t>(space, 0x100), 0xabcdef0112345678ull);
    EXPECT_EQ(m.inspect<uint32_t>(space, 0x108), 777u);
    EXPECT_EQ(c.memory_accesses[0][0].bytes, 8u);
  }
  auto f = image_for(
      {official_instruction(0, "STG.E.NA.128", "desc[UR4][R2.64],R8")});
  mapped_functional_memory m;
  m.map(memory_space::kGlobal, 0x100, 20, true);
  warp_state s;
  s.active_mask = 1;
  s.write_register(0, 2, 0x100);
  for (unsigned i = 0; i < 4; ++i) s.write_register(0, 8 + i, 100 + i);
  m.initialize<uint32_t>(memory_space::kGlobal, 0x110, 777);
  execution_context c{&m};
  ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
  for (unsigned i = 0; i < 4; ++i)
    EXPECT_EQ(m.inspect<uint32_t>(memory_space::kGlobal, 0x100 + 4 * i),
              100 + i);
  EXPECT_EQ(m.inspect<uint32_t>(memory_space::kGlobal, 0x110), 777u);
  EXPECT_EQ(c.memory_accesses[0][0].bytes, 16u);
}
TEST(SassNativeDataMovement, CacheLineHintPreservesLoadValueAndWidth) {
  auto f = image_for(
      {official_instruction(0, "LDG.E.LTC128B", "R4,desc[UR8][R2.64]")});
  mapped_functional_memory m;
  m.map(memory_space::kGlobal, 0x100, 4, true);
  m.initialize<uint32_t>(memory_space::kGlobal, 0x100, 0xabcdeff0);
  warp_state s;
  s.active_mask = 1;
  s.write_register(0, 2, 0x100);
  execution_context c{&m};
  ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
  EXPECT_EQ(s.read_register(0, 4), 0xabcdeff0u);
  EXPECT_EQ(c.memory_accesses[0][0].bytes, 4u);
}
TEST(SassNativeDataMovement, SpillFillRejectsGuardsAndOperandModifiers) {
  for (const auto &inst :
       {official_instruction(0, "MOV.SPILL", "R5,UR6", "@P0"),
        official_instruction(0, "MOV.SPILL", "R5,-UR6"),
        official_instruction(0, "R2UR.FILL", "UR6,R5", "@P0"),
        official_instruction(0, "R2UR.FILL", "UR6,R5.64")}) {
    auto f = image_for({inst});
    warp_state s;
    s.write_uniform_register(6, 123);
    s.write_register(0, 5, 456);
    s.write_predicate(0, 0, true);
    execution_context c;
    EXPECT_EQ(f.step("test", s, c).status, step_status::kUnsupported);
    EXPECT_EQ(s.pc, 0u);
    EXPECT_EQ(s.read_uniform_register(6), 123u);
    EXPECT_EQ(s.read_register(0, 5), 456u);
  }
}
TEST(SassNativeDataMovement, WideStoresZeroGroupsAndRejectInvalidAccesses) {
  for (const char *groups : {"RZ,R4", "R4,RZ"}) {
    auto f = image_for({official_instruction(
        0, "STG.E.ENL2.256", std::string("desc[UR8][R2.64],") + groups)});
    mapped_functional_memory m;
    m.map(memory_space::kGlobal, 0x100, 32, true);
    warp_state s;
    s.active_mask = 1;
    s.write_register(0, 2, 0x100);
    for (unsigned i = 0; i < 4; ++i) s.write_register(0, 4 + i, 100 + i);
    execution_context c{&m};
    ASSERT_EQ(f.step("test", s, c).status, step_status::kAdvanced);
    for (unsigned i = 0; i < 8; ++i) {
      const bool zero = (std::string(groups) == "RZ,R4") ? i < 4 : i >= 4;
      EXPECT_EQ(m.inspect<uint32_t>(memory_space::kGlobal, 0x100 + 4 * i),
                zero ? 0u : 100 + i % 4);
    }
  }
  for (const auto &operands :
       {"desc[UR8][R2.64],R252,R4", "desc[UR8][R2.64+0x4],R4,R20",
        "desc[UR8][R2.64],R4,R20,0xf3"}) {
    auto f = image_for({official_instruction(0, "STG.E.ENL2.256", operands)});
    mapped_functional_memory m;
    m.map(memory_space::kGlobal, 0x100, 64, true);
    m.initialize<uint32_t>(memory_space::kGlobal, 0x100, 777);
    warp_state s;
    s.active_mask = 1;
    s.write_register(0, 2, 0x100);
    execution_context c{&m};
    EXPECT_EQ(f.step("test", s, c).status, step_status::kUnsupported);
    EXPECT_TRUE(c.memory_accesses[0].empty());
    EXPECT_EQ(m.inspect<uint32_t>(memory_space::kGlobal, 0x100), 777u);
  }
}
}  // namespace
