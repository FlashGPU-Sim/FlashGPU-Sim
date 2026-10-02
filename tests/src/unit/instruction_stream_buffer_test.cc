#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/flash/instruction_cache/stream_buffer.h"

namespace {

using flash_gpgpu_sim::instruction_demand_state;
using flash_gpgpu_sim::instruction_stream_buffer;
using flash_gpgpu_sim::instruction_stream_buffer_config;

instruction_stream_buffer make_buffer(unsigned streams = 1, unsigned depth = 4,
                                      unsigned issue_width = 2) {
  instruction_stream_buffer_config config;
  config.line_size = 128;
  config.streams = streams;
  config.depth = depth;
  config.issue_width = issue_width;
  return instruction_stream_buffer(config);
}

TEST(InstructionStreamBufferTest, DepthBoundsLogicalAheadWindow) {
  auto buffer = make_buffer(/*streams=*/1, /*depth=*/4, /*issue_width=*/2);
  EXPECT_EQ(buffer.observe_demand(7, 0x1000, true),
            instruction_demand_state::kUntracked);
  EXPECT_EQ(buffer.pending_entries(), 4u);

  const auto first = buffer.issue();
  const auto second = buffer.issue();
  const auto third = buffer.issue();
  ASSERT_EQ(first.size(), 2u);
  ASSERT_EQ(second.size(), 2u);
  EXPECT_TRUE(third.empty());
  EXPECT_EQ(first[0].address, 0x1080u);
  EXPECT_EQ(second[1].address, 0x1200u);
  EXPECT_EQ(buffer.pending_entries(), 4u);
}

TEST(InstructionStreamBufferTest, DemandAdvanceSlidesWindowByDistance) {
  auto buffer = make_buffer(/*streams=*/1, /*depth=*/4, /*issue_width=*/4);
  buffer.observe_demand(3, 0x2000, true);
  const auto requests = buffer.issue();
  ASSERT_EQ(requests.size(), 4u);
  buffer.fill(requests[0]);
  EXPECT_EQ(buffer.stats().resident, 1u);

  EXPECT_EQ(buffer.observe_demand(3, 0x2080, false),
            instruction_demand_state::kReady);
  EXPECT_EQ(buffer.pending_entries(), 4u);
  const auto next = buffer.issue();
  ASSERT_EQ(next.size(), 1u);
  EXPECT_EQ(next[0].address, 0x2280u);
}

TEST(InstructionStreamBufferTest, ReplacedStreamRejectsOldFillGeneration) {
  auto buffer = make_buffer(/*streams=*/1, /*depth=*/2, /*issue_width=*/1);
  buffer.observe_demand(1, 0x1000, true);
  const auto old_request = buffer.issue().front();

  buffer.observe_demand(2, 0x8000, true);
  ASSERT_FALSE(buffer.current(old_request));
  buffer.fill(old_request);

  EXPECT_EQ(buffer.stats().streams_replaced, 1u);
  EXPECT_EQ(buffer.stats().stale_fills, 1u);
  EXPECT_EQ(buffer.pending_entries(), 2u);
}

TEST(InstructionStreamBufferTest, DemandOvertakesInflightWithoutReowningFill) {
  auto buffer = make_buffer(/*streams=*/1, /*depth=*/2, /*issue_width=*/1);
  buffer.observe_demand(5, 0x1000, true);
  const auto request = buffer.issue().front();

  EXPECT_EQ(buffer.observe_demand(5, request.address, true),
            instruction_demand_state::kInflight);
  EXPECT_EQ(buffer.pending_entries(), 2u);
  EXPECT_FALSE(buffer.current(request));

  buffer.fill(request);
  EXPECT_EQ(buffer.stats().late, 1u);
  EXPECT_EQ(buffer.stats().stale_fills, 1u);
  EXPECT_EQ(buffer.pending_entries(), 2u);
}

TEST(InstructionStreamBufferTest, CancelMakesInflightFillNonOwning) {
  auto buffer = make_buffer(/*streams=*/1, /*depth=*/2, /*issue_width=*/1);
  buffer.observe_demand(9, 0x4000, true);
  const auto request = buffer.issue().front();
  buffer.cancel_context(9);

  EXPECT_EQ(buffer.pending_entries(), 0u);
  EXPECT_FALSE(buffer.current(request));
  buffer.fill(request);
  EXPECT_EQ(buffer.stats().stale_fills, 1u);
}

TEST(InstructionStreamBufferTest, IssueWidthIsGlobalAcrossStreams) {
  auto buffer = make_buffer(/*streams=*/2, /*depth=*/4, /*issue_width=*/1);
  buffer.observe_demand(1, 0x1000, true);
  buffer.observe_demand(2, 0x8000, true);

  EXPECT_EQ(buffer.issue().size(), 1u);
  EXPECT_EQ(buffer.issue().size(), 1u);
  EXPECT_EQ(buffer.stats().requests_issued, 2u);
}

TEST(InstructionStreamBufferTest, GccPreloadWindowStaysLaunchRelative) {
  instruction_stream_buffer_config config;
  config.line_size = 128;
  config.streams = 1;
  config.depth = 4;
  config.issue_width = 4;
  config.gcc_preload_lines = 4;
  instruction_stream_buffer buffer(config);

  buffer.observe_demand(7, 0x1000, true);
  const auto initial = buffer.issue();
  ASSERT_EQ(initial.size(), 4u);
  EXPECT_TRUE(buffer.gcc_preload_contains(initial[0]));
  EXPECT_TRUE(buffer.gcc_preload_contains(initial[2]));
  EXPECT_FALSE(buffer.gcc_preload_contains(initial[3]));

  buffer.observe_demand(7, 0x8000, true);
  const auto replacement = buffer.issue();
  ASSERT_GE(replacement.size(), 1u);
  EXPECT_FALSE(buffer.gcc_preload_contains(replacement.front()));
}

}  // namespace
