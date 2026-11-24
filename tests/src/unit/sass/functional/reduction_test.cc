#include "../test_support.h"
namespace {
TEST(SassReduction, AddReusesAtomicRmwWithoutAnOutputRegister) {
  for (const char *opcode :
       {"REDG.E.ADD.STRONG.GPU", "REDG.E.ADD.S32.STRONG.GPU"}) {
    kernel image;
    image.name = "red";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, opcode, "desc[UR8][R2.64],R5")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    mapped_functional_memory memory;
    constexpr uint64_t address = 0x200000100ull;
    memory.map(memory_space::kGlobal, address, 4, true);
    memory.initialize<uint32_t>(memory_space::kGlobal, address, 10);
    execution_context context{&memory};
    warp_state state;
    state.active_mask = 3;
    for (unsigned lane = 0; lane < 2; ++lane) {
      state.write_register(lane, 2, uint32_t(address));
      state.write_register(lane, 3, address >> 32);
      state.write_register(lane, 5, lane ? 7u : 0xfffffffbu);
    }
    const auto r = sass.step("red", state, context);
    ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, address), 12u);
    EXPECT_EQ(state.read_register(0, 2), uint32_t(address));
    ASSERT_EQ(context.memory_accesses[0].size(), 1u);
    EXPECT_FALSE(context.memory_accesses[0][0].write);
    const timing_instruction timing(
        official_instruction(0, opcode, "desc[UR8][R2.64],R5"));
    EXPECT_EQ(timing.outcount, 0u);
    EXPECT_EQ(timing.data_size, 4u);
    EXPECT_EQ(timing.op, LOAD_OP);
    EXPECT_EQ(std::vector<int>(timing.arch_reg.src,
                               timing.arch_reg.src + timing.incount),
              (std::vector<int>{265, 3, 4, 6}));
  }
}
}  // namespace
