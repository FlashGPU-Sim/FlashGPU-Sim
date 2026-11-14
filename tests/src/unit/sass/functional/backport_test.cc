#include "../test_support.h"

namespace {
TEST(SassIntegerBoundaryTest,
     LowLeaCarryIsAddCarryAfterShiftIncludingNegatedZero) {
  using namespace flash_gpgpu_sim::sass;
  for (bool uniform : {false, true})
    for (bool negated : {false, true}) {
      frontend sass;
      register_sm120_functional_semantics(sass);
      kernel image;
      image.name = "lea_low_carry";
      image.arch = architecture::kSm120;
      image.instruction_bytes = kInstructionBytes;
      for (unsigned shift = 0; shift < 32; ++shift) {
        instruction inst;
        inst.pc = shift * kInstructionBytes;
        inst.decoded = true;
        inst.opcode = uniform ? "ULEA" : "LEA";
        inst.operand_text = (uniform ? "UR0,UP0," : "R0,P0,") +
                            std::string(negated ? "-" : "") +
                            (uniform ? "UR1,UR2," : "R1,R2,") +
                            std::to_string(shift);
        parse_instruction_operands(inst);
        image.instructions.push_back(inst);
      }
      sass.add_kernel(std::move(image));
      for (uint32_t input : {0u, 1u, 0x80000000u, 0xffffffffu})
        for (uint32_t addend : {0u, 128u, 0xffffffffu}) {
          warp_state state;
          state.active_mask = 1;
          execution_context context;
          for (unsigned shift = 0; shift < 32; ++shift) {
            SCOPED_TRACE(::testing::Message()
                         << uniform << "," << negated << "," << input << ","
                         << addend << "," << shift);
            if (uniform) {
              state.write_uniform_register(1, input);
              state.write_uniform_register(2, addend);
            } else {
              state.write_register(0, 1, input);
              state.write_register(0, 2, addend);
            }
            const uint32_t shifted = input << shift;
            const uint64_t sum =
                uint64_t(addend) +
                (negated ? (uint64_t{1} << 32) - shifted : uint64_t(shifted));
            auto r = sass.step("lea_low_carry", state, context);
            ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
            EXPECT_EQ(uniform ? state.read_uniform_register(0)
                              : state.read_register(0, 0),
                      uint32_t(sum));
            EXPECT_EQ(uniform ? state.read_uniform_predicate(0)
                              : state.read_predicate(0, 0),
                      bool(sum >> 32));
          }
        }
    }
}

TEST(SassIntegerBoundaryTest, MhcGridBarrierSubtractionPreservesHighWord) {
  using namespace flash_gpgpu_sim::sass;
  frontend sass;
  register_sm120_functional_semantics(sass);
  kernel image;
  image.name = "mhc_grid_sub";
  image.arch = architecture::kSm120;
  image.instruction_bytes = kInstructionBytes;
  const std::pair<const char *, const char *> ops[] = {
      {"SHF.L.U64.HI", "R9,R4,0x7,R5"},
      {"LEA", "R6,P0,-R4,R6,0x7"},
      {"IADD3.X", "R7,PT,PT,R7,~R9,RZ,P0,!PT"}};
  for (auto op : ops) {
    instruction inst;
    inst.pc = image.instructions.size() * kInstructionBytes;
    inst.decoded = true;
    inst.opcode = op.first;
    inst.operand_text = op.second;
    parse_instruction_operands(inst);
    image.instructions.push_back(inst);
  }
  sass.add_kernel(std::move(image));
  for (uint64_t grid : {0ull, 1ull, 0x2000000ull, 0x100000001ull})
    for (uint64_t count : {0ull, 1ull, 15ull, 16ull}) {
      // Same value produced by SplitBarrier::encode_state(grid,count).
      const uint64_t barrier = (grid + 1) * 128 + count;
      warp_state state;
      state.active_mask = 1;
      execution_context context;
      state.write_register(0, 4, uint32_t(grid));
      state.write_register(0, 5, uint32_t(grid >> 32));
      state.write_register(0, 6, uint32_t(barrier));
      state.write_register(0, 7, uint32_t(barrier >> 32));
      for (unsigned i = 0; i < 3; ++i) {
        auto r = sass.step("mhc_grid_sub", state, context);
        ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
      }
      const uint64_t result = uint64_t(state.read_register(0, 6)) |
                              (uint64_t(state.read_register(0, 7)) << 32);
      EXPECT_EQ(result, 128 + count) << grid << "," << count;
    }
}

TEST(SassFunctionalTest, UniformSignedExtendedGeUsesSignedHighAndLowPredicate) {
  kernel image;
  image.name = "uniform_signed_ge_extended";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "UISETP.GE.AND.EX", "UP0,UPT,UR0,UR1,UPT,UP1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (int high : {-1, 0, 1}) for (bool low_ge : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_uniform_register(0, uint32_t(high)); state.write_uniform_register(1, 0);
    state.write_uniform_predicate(1, low_ge);
    execution_context context;
    ASSERT_EQ(sass.step("uniform_signed_ge_extended", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_uniform_predicate(0), high > 0 || (high == 0 && low_ge));
  }
}

TEST(SassFunctionalTest, ComparesSigned64PairsWithUniformSources) {
  kernel image;
  image.name = "isetp_s64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.GE.S64.AND", "P0,PT,R0,R2,PT"),
      official_instruction(0x10, "ISETP.GE.S64.AND", "P1,PT,R0,UR4,PT"),
      official_instruction(0x20, "ISETP.GE.S64.AND", "P2,PT,R0,UR4,!PT"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  // Check upper-word ordering, negative versus positive, signed extrema and
  // equality; the uniform operand is 2^32, so reading UR4 alone is wrong.
  const std::array<int64_t, 8> a = {
      int64_t{1} << 32, (int64_t{1} << 32) - 1, -1, 0,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(), -(int64_t{1} << 32), -1};
  const std::array<int64_t, 8> b = {
      (int64_t{1} << 32) - 1, int64_t{1} << 32, 0, -1,
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::min(), -1, -1};
  warp_state state;
  state.active_mask = 0xff;
  state.write_uniform_register(4, 0);
  state.write_uniform_register(5, 1);
  for (unsigned lane = 0; lane < a.size(); ++lane) {
    state.write_register(lane, 0, static_cast<uint64_t>(a[lane]));
    state.write_register(lane, 1, static_cast<uint64_t>(a[lane]) >> 32);
    state.write_register(lane, 2, static_cast<uint64_t>(b[lane]));
    state.write_register(lane, 3, static_cast<uint64_t>(b[lane]) >> 32);
    state.write_predicate(lane, 2, true);
  }
  execution_context context;
  for (unsigned step = 0; step < 4; ++step) {
    const auto result = sass.step("isetp_s64", state, context);
    ASSERT_EQ(result.status, step == 3 ? step_status::kExited
                                      : step_status::kAdvanced) << result.detail;
  }
  for (unsigned lane = 0; lane < a.size(); ++lane) {
    EXPECT_EQ(state.read_predicate(lane, 0), a[lane] >= b[lane]) << lane;
    EXPECT_EQ(state.read_predicate(lane, 1), a[lane] >= (int64_t{1} << 32)) << lane;
    EXPECT_FALSE(state.read_predicate(lane, 2)) << lane;
  }
}

TEST(SassFunctionalTest, ComparesUnsigned64EqualityAndCombinesPredicate) {
  kernel image;
  image.name = "u64_eq";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.EQ.U64.AND", "P0,PT,R4,UR4,PT"),
      official_instruction(16, "ISETP.EQ.U64.OR", "P1,PT,R4,RZ,!P2"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 7;
  state.write_uniform_register(4, 0);
  state.write_uniform_register(5, 1);
  for (unsigned lane = 0; lane < 3; ++lane) {
    state.write_register(lane, 4, 0);
    state.write_register(lane, 5, lane);
    state.write_predicate(lane, 2, lane != 2);
  }
  execution_context context;
  ASSERT_EQ(sass.step("u64_eq", state, context).status, step_status::kAdvanced);
  ASSERT_EQ(sass.step("u64_eq", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 3; ++lane) {
    EXPECT_EQ(state.read_predicate(lane, 0), lane == 1);
    EXPECT_EQ(state.read_predicate(lane, 1), lane == 0 || lane == 2);
  }
}

TEST(SassTimingProjectionTest, WideIntegerCompareReadsBothSourceWords) {
  for (const auto *opcode : {"ISETP.GE.S64.AND", "ISETP.GE.U64.AND"}) {
    const timing_instruction inst(
        official_instruction(0, opcode, "P0,PT,R10,UR4,PT"));
    EXPECT_EQ(inst.outcount, 1u);
    ASSERT_EQ(inst.incount, 4u);
    EXPECT_EQ(inst.arch_reg.src[0], 11);
    EXPECT_EQ(inst.arch_reg.src[1], 12);
    ASSERT_EQ(inst.get_register_file_source_count(), 2u);
    EXPECT_EQ(inst.get_register_file_source(0).reg, 10u);
    EXPECT_EQ(inst.get_register_file_source(1).reg, 11u);
    const timing_instruction zero(
        official_instruction(0, opcode, "P0,PT,RZ,URZ,PT"));
    EXPECT_EQ(zero.incount, 0u);
    EXPECT_EQ(zero.get_register_file_source_count(), 0u);
  }
  const timing_instruction uniform(official_instruction(
      0, "UISETP.GE.U64.AND", "UP0,UPT,UR4,UR6,UPT"));
  EXPECT_EQ(uniform.incount, 4u);
}

TEST(SassFunctionalTest,
     DeferredBarrierPreservesGlobalCounterUntilPeerArrives) {
  kernel image;
  image.name = "deferred_global_cleanup";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BAR.SYNC.DEFER_BLOCKING", "0x1,0x40"),
      official_instruction(16, "MOV", "R4,0x2a"),
      official_instruction(32, "STG.E", "desc[UR8][R2.64],RZ", "@P0"),
  };
  mapped_functional_memory memory;
  constexpr uint64_t address = 0x200000120ull;
  memory.map(memory_space::kGlobal, address, 4, true);
  memory.initialize<uint32_t>(memory_space::kGlobal, address, 63);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_execution_state cta(2);
  warp_state cleanup, peer;
  for (unsigned lane = 0; lane < 32; ++lane) {
    cleanup.write_register(lane, 2, uint32_t(address));
    cleanup.write_register(lane, 3, uint32_t(address >> 32));
    cleanup.write_predicate(lane, 0, lane == 0);
  }
  execution_context context{&memory};
  context.cta = &cta;
  context.warp_id = 0;
  ASSERT_EQ(sass.step("deferred_global_cleanup", cleanup, context).status,
            step_status::kAdvanced);
  // Register-only work may proceed, but the cleanup store must wait.
  ASSERT_EQ(sass.step("deferred_global_cleanup", cleanup, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(cleanup.read_register(0, 4), 42u);
  EXPECT_EQ(sass.step("deferred_global_cleanup", cleanup, context).status,
            step_status::kBlocked);
  EXPECT_EQ(cleanup.pc, 32u);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 63u);
  context.warp_id = 1;
  ASSERT_EQ(sass.step("deferred_global_cleanup", peer, context).status,
            step_status::kAdvanced);
  context.warp_id = 0;
  ASSERT_EQ(sass.step("deferred_global_cleanup", cleanup, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 0u);
}

TEST(SassFunctionalTest, DirectedFmaRetainsTinyResidualAndHandlesOverflowAndZero) {
  const unsigned inputs[][3]={{0x3f800000,0x3f800000,0x8d800000},
    {1,0x3f000000,0},{0x80000001,0x3f000000,0x80000000},
    {0x3f800000,0x3f800000,0xbf800000},{0x7f7fffff,0x40000000,0},
    {0,0x3f800000,0},{0x80000000,0x3f800000,0x80000000}};
  const unsigned expected[][3]={{0x3f7fffff,0x3f800000,0x3f7fffff},
    {0,1,0},{0x80000001,0x80000000,0x80000000},
    {0x80000000,0,0},{0x7f7fffff,0x7f800000,0x7f7fffff},
    {0,0,0},{0x80000000,0x80000000,0x80000000}};
  unsigned mode=0;
  for(const char* opcode:{"FFMA.RM","FFMA.RP","FFMA.RZ"}){
    kernel image;image.name="directed";image.arch=architecture::kSm120;image.instruction_bytes=16;
    image.instructions={official_instruction(0,opcode,"R0,R0,R1,R2")};
    frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
    for(unsigned i=0;i<7;++i){warp_state state;execution_context context;
      for(unsigned lane=0;lane<32;++lane)for(unsigned j=0;j<3;++j)state.write_register(lane,j,inputs[i][j]);
      ASSERT_EQ(sass.step("directed",state,context).status,step_status::kAdvanced);
      for(unsigned lane=0;lane<32;++lane)EXPECT_EQ(state.read_register(lane,0),expected[i][mode])<<opcode<<" case="<<i;
    }
    ++mode;
  }
}

TEST(SassFunctionalTest, TmaLoadAndStoreHonorNonzeroSwizzleBasePhase) {
  for (unsigned swizzle : {32u, 64u, 128u}) {
    kernel image;
    image.name = "base_phase";
    image.arch = architecture::kSm120;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "UTMALDG.2D", "[UR4][UR12],desc[URZ]"),
        official_instruction(16, "UTMASTG.2D", "[UR20][UR14]")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    cta_execution_state cta(1);
    mapped_functional_memory memory;
    memory.map(memory_space::kGlobal, 0x20000, 512, false);
    memory.map(memory_space::kGlobal, 0x21000, 512, true);
    const unsigned base = 0x400 + swizzle * 4;
    memory.map(memory_space::kShared, base, 512, true);
    // Distinguish every 16B atom, including the two halves of each 128B row.
    for (unsigned i = 0; i < 512; ++i)
      memory.initialize<uint8_t>(memory_space::kGlobal, 0x20000 + i, i / 16);
    functional_tensor_map_2d d;
    d.global_address = 0x20000; d.element_bytes = 1;
    d.global_dim = {{swizzle, 512 / swizzle}}; d.box_dim = d.global_dim;
    d.row_stride_bytes = swizzle; d.swizzle_bytes = swizzle; d.oob_zero = true;
    ASSERT_TRUE(cta.register_tensor_map_2d(0x1000, d));
    d.global_address = 0x21000;
    ASSERT_TRUE(cta.register_tensor_map_2d(0x1100, d));
    ASSERT_TRUE(cta.initialize_mbarrier(0x80, 1));
    ASSERT_TRUE(cta.arrive_mbarrier(0x80, 1, 512));
    warp_state state;
    state.write_uniform_register(4, base);
    state.write_uniform_register(5, 0x80);
    state.write_uniform_register(12, 0x1000);
    state.write_uniform_register(14, 0x1100);
    state.write_uniform_register(20, base);
    execution_context context{&memory}; context.cta = &cta;
    auto result = sass.step("base_phase", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    // Independent PTX address-bit XOR, based on absolute shared addresses.
    for (unsigned i = 0; i < 512; ++i) {
      const unsigned absolute = base + i;
      const unsigned physical = absolute ^ ((absolute >> 3) & (swizzle - 16));
      EXPECT_EQ(memory.inspect<uint8_t>(memory_space::kShared, physical), i / 16);
    }
    result = sass.step("base_phase", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    for (unsigned i = 0; i < 512; ++i)
      EXPECT_EQ(memory.inspect<uint8_t>(memory_space::kGlobal, 0x21000 + i), i / 16);
  }
}
}

TEST(SassFunctionalTest, LastDeferredArriverRetainsTimingHandoff) {
  kernel image;
  image.name = "last_deferred";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "BAR.SYNC.DEFER_BLOCKING", "0x1,0x40")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_execution_state cta(2);
  execution_context context;
  context.cta = &cta;
  for (unsigned warp = 0; warp < 2; ++warp) {
    warp_state state;
    context.warp_id = warp;
    ASSERT_EQ(sass.step("last_deferred", state, context).status, step_status::kAdvanced);
    EXPECT_TRUE(state.deferred_cta_barrier_pending[1]);
    EXPECT_EQ(state.deferred_cta_barrier_generation[1], 0u);
  }
  EXPECT_EQ(cta.generation(1), 1u);
}
