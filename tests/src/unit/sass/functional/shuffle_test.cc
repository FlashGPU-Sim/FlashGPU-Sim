#include "../test_support.h"

namespace {
frontend shuffle_image(const char *mode, const char *guard = "") {
  kernel image;
  image.name = "shuffle";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(
      0, std::string("SHFL.") + mode, "P0,R0,R0,R1,R2", guard)};
  frontend f;
  f.add_kernel(std::move(image));
  register_sm100_functional_semantics(f);
  return f;
}
TEST(SassShuffle, DirectionsClampsAndAliasedSources) {
  for (const char *mode : {"UP", "DOWN", "BFLY", "IDX"}) {
    const bool up = std::string(mode) == "UP";
    for (unsigned configuration : {0u, 31u, 0x1800u, 0x181fu, 0x1802u})
      for (unsigned selector : {0u, 1u, 2u, 31u, 32u, 63u}) {
        auto f = shuffle_image(mode);
        warp_state s;
        for (unsigned lane = 0; lane < 32; ++lane) {
          s.write_register(lane, 0, 100 + lane);
          s.write_register(lane, 1, selector);
          s.write_register(lane, 2, configuration);
        }
        execution_context c;
        ASSERT_EQ(f.step("shuffle", s, c).status, step_status::kAdvanced);
        for (int lane = 0; lane < 32; ++lane) {
          const int segment = (configuration >> 8) & 31;
          const int bound = (lane & segment) | ((configuration & 31) & ~segment);
          int origin = lane;
          const int delta = selector % 32;
          if (up) origin = lane - delta;
          else if (std::string(mode) == "DOWN") origin = lane + delta;
          else if (std::string(mode) == "BFLY") origin = lane ^ delta;
          else origin = (lane & segment) | (delta & ~segment);
          const bool valid = up ? origin >= bound : origin <= bound;
          EXPECT_EQ(s.read_register(lane, 0), 100u + (valid ? origin : lane));
          EXPECT_EQ(s.read_predicate(lane, 0), valid);
        }
      }
  }
}
TEST(SassShuffle, GuardAliasingAndInactiveDestinations) {
  auto f = shuffle_image("UP", "@P0");
  warp_state s;
  s.active_mask = 0xffff;
  for (unsigned lane = 0; lane < 32; ++lane) {
    s.write_register(lane, 0, 100 + lane);
    s.write_register(lane, 1, 2);
    s.write_register(lane, 2, 0);
    s.write_predicate(lane, 0, lane % 2);
  }
  execution_context c;
  ASSERT_EQ(f.step("shuffle", s, c).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    const bool runs = lane < 16 && lane % 2;
    EXPECT_EQ(s.read_register(lane, 0), 100 + lane - (runs && lane > 1 ? 2 : 0));
    EXPECT_EQ(s.read_predicate(lane, 0), runs ? lane > 1 : bool(lane % 2));
  }
}
TEST(SassShuffle, TimingRetainsDataAndPredicateOutputs) {
  for (const char *opcode : {"SHFL.UP", "SHFL.DOWN"}) {
    const timing_instruction inst(official_instruction(0, opcode, "P0,R4,R8,R12,R13"));
    EXPECT_EQ(inst.outcount, 2u);
    EXPECT_EQ(inst.incount, 3u);
    const timing_instruction reference(official_instruction(0, "SHFL.BFLY", "P0,R4,R8,R12,R13"));
    EXPECT_EQ(inst.op, reference.op);
    EXPECT_EQ(inst.latency, reference.latency);
    EXPECT_EQ(inst.initiation_interval, reference.initiation_interval);
  }
}
}  // namespace
