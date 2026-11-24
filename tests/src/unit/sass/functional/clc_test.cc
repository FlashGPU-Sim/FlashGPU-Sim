#include "../test_support.h"
namespace {
TEST(ClcAdapter, ClcRequiresServiceAndOnlyRecordsAsynchronousEffect) {
  for (bool broadcast : {false, true}) {
    kernel image;
    image.name = "clc_effect";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(
        0, broadcast ? "UGETNEXTWORKID.BROADCAST" : "UGETNEXTWORKID.SELFCAST",
        "[UR4][UR5]")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    mapped_functional_memory memory;
    memory.map(memory_space::kShared, 0x100, 16, true);
    cta_execution_state cta(1);
    ASSERT_TRUE(cta.initialize_mbarrier(0x80, 1));
    ASSERT_TRUE(cta.arrive_mbarrier(0x80, 1, 16));
    execution_context context{&memory};
    context.cta = &cta;
    warp_state state;
    state.write_uniform_register(4, 0x100);
    state.write_uniform_register(5, 0x80);
    EXPECT_EQ(sass.step("clc_effect", state, context).status,
              step_status::kUnsupported);
    EXPECT_EQ(state.pc, 0u);
    EXPECT_FALSE(context.clc_effect.valid);
    context.clc_service_available = true;
    ASSERT_EQ(sass.step("clc_effect", state, context).status,
              step_status::kAdvanced);
    ASSERT_TRUE(context.clc_effect.valid);
    EXPECT_EQ(context.clc_effect.broadcast, broadcast);
    EXPECT_EQ(context.clc_effect.response_address, 0x100u);
    EXPECT_EQ(context.clc_effect.barrier_address, 0x80u);
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x108), 0u);
    EXPECT_EQ(cta.mbarrier_pending_transaction_bytes(0x80), 16u);
    EXPECT_EQ(cta.mbarrier_phase(0x80), 0u);
  }
}

}  // namespace

namespace {
TEST(ClcAdapter, NonzeroRankImportsLocalBarrierWithTaggedTimingOwner) {
  kernel image;
  image.name = "clc_peer";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UGETNEXTWORKID.SELFCAST", "[UR4][UR5]")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, 512, true);
  const uint64_t encoded = (uint64_t(0xfffff << 11) << 32) | (0xfffffu << 1);
  ASSERT_TRUE(
      memory.write(memory_space::kShared, 128, &encoded, sizeof(encoded)));
  cta_execution_state cta(1);
  execution_context context{&memory};
  context.cta = &cta;
  context.cga_cta_id = 1;
  context.clc_service_available = true;
  warp_state state;
  state.write_uniform_register(4, (1u << 24) | 256);
  state.write_uniform_register(5, (1u << 24) | 128);
  ASSERT_EQ(sass.step("clc_peer", state, context).status,
            step_status::kAdvanced);
  EXPECT_TRUE(cta.mbarrier_initialized(128));
  EXPECT_EQ(context.clc_effect.barrier_address, 128u);
  ASSERT_EQ(context.imported_mbarriers.size(), 1u);
  EXPECT_EQ(context.imported_mbarriers[0].address, (1u << 24) | 128);
}
}  // namespace

namespace {
TEST(ClcAdapter, MaskedOffRequestDoesNotReadUninitializedPeerOperands) {
  kernel image;
  image.name = "clc_masked";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  auto inst = official_instruction(0, "UGETNEXTWORKID.BROADCAST", "[UR4][UR5]");
  inst.has_guard = true;
  inst.guard.kind = operand_kind::kUniformPredicate;
  inst.guard.index = 0;
  image.instructions = {inst};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  warp_state state;
  state.write_uniform_predicate(0, false);
  execution_context context;
  context.cga_cta_id = 1;
  ASSERT_EQ(sass.step("clc_masked", state, context).status,
            step_status::kAdvanced);
  EXPECT_FALSE(context.clc_effect.valid);
  EXPECT_EQ(state.pc, 16u);
}
}  // namespace
