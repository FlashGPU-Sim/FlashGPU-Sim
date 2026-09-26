#include <gtest/gtest.h>

#include <deque>
#include <string>
#include <vector>

#include "../../../src/gpgpu-sim/shader.h"

namespace {

TEST(MemoryTransportTest, ChargesDataSectorsAndControlSlots) {
  EXPECT_EQ(memory_transport_data_sectors(READ_REQUEST, SECTOR_SIZE), 1u);
  EXPECT_EQ(memory_transport_data_sectors(READ_REQUEST, 2 * SECTOR_SIZE), 2u);
  EXPECT_EQ(memory_transport_data_sectors(READ_REPLY, 4 * SECTOR_SIZE), 4u);
  EXPECT_EQ(memory_transport_data_sectors(WRITE_REQUEST, 4 * SECTOR_SIZE), 4u);
  EXPECT_EQ(memory_transport_data_sectors(WRITE_ACK, 4 * SECTOR_SIZE), 0u);
  EXPECT_EQ(memory_transport_service_slots(0), 1u);
}

TEST(MemoryTransportTest, OversizedHeadPacketMakesProgressWithCredit) {
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  for (unsigned tick = 0; tick < 3; ++tick) {
    budget.begin_tick(1);
    EXPECT_FALSE(budget.can_accept(4));
    budget.note_width_limited(4);
    budget.end_tick(&stats);
  }
  budget.begin_tick(1);
  ASSERT_TRUE(budget.can_accept(4));
  budget.consume(4);
  stats.record_accept(4);
  budget.end_tick(&stats);
  EXPECT_EQ(stats.accepted_data_sectors, 4u);
  EXPECT_EQ(stats.width_limited_ticks, 3u);
  EXPECT_EQ(budget.carried_credit(), 0u);
}

TEST(MemoryTransportTest, IdleAndBackpressureDoNotBankCredit) {
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  budget.begin_tick(2);
  budget.end_tick(&stats);
  EXPECT_EQ(budget.carried_credit(), 0u);

  budget.begin_tick(2);
  budget.note_width_limited(4);
  budget.note_downstream_full();
  budget.end_tick(&stats);
  EXPECT_EQ(budget.carried_credit(), 0u);
  EXPECT_EQ(stats.width_limited_ticks, 1u);
  EXPECT_EQ(stats.downstream_full_ticks, 1u);
}

TEST(MemoryTransportTest, ResidualBudgetDoesNotBecomeBurstCredit) {
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  budget.begin_tick(4);
  budget.consume(3);
  budget.note_width_limited(2);
  budget.end_tick(&stats);
  EXPECT_EQ(budget.carried_credit(), 0u);

  budget.begin_tick(4);
  EXPECT_EQ(budget.remaining_slots(), 4u);
  budget.end_tick(&stats);
}

TEST(MemoryTransportTest,
     OrdinaryResponseRetirementSustainsFourPastStagingDepth) {
  struct SyntheticInstruction {
    unsigned id;
  };

  memory_transport_response_retirement_queue<unsigned, SyntheticInstruction>
      retirement;
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;

  // Four dynamic loads are interleaved for eight ticks.  This is 32 replies,
  // well past the old four-packet staging depth, while still producing only
  // one RF-level completion per instruction.
  for (unsigned instruction = 0; instruction < 4; ++instruction) {
    retirement.expect_responses(instruction, /*count=*/8,
                                {instruction});
  }
  for (unsigned tick = 0; tick < 8; ++tick) {
    budget.begin_tick(/*width=*/4);
    for (unsigned instruction = 0; instruction < 4; ++instruction) {
      ASSERT_TRUE(budget.can_accept(/*data_sectors=*/1));
      const bool completed = retirement.retire_response(instruction);
      EXPECT_EQ(completed, tick == 7);
      budget.consume(/*data_sectors=*/1);
      stats.record_accept(/*data_sectors=*/1);
    }
    budget.end_tick(&stats);
  }

  EXPECT_EQ(retirement.retired_response_count(), 32u);
  EXPECT_EQ(retirement.pending_instruction_count(), 0u);
  EXPECT_EQ(retirement.completion_count(), 4u);
  EXPECT_EQ(stats.accepted_data_sectors, 32u);
  EXPECT_EQ(stats.service_ticks, 8u);
  EXPECT_EQ(stats.max_service_slots_per_tick, 4u);
  EXPECT_EQ(stats.downstream_full_ticks, 0u);

  // Instruction/RF writeback remains a separate one-at-a-time consumer.
  for (unsigned instruction = 0; instruction < 4; ++instruction) {
    ASSERT_TRUE(retirement.completion_ready());
    EXPECT_EQ(retirement.next_completion().id, instruction);
    retirement.pop_completion();
  }
  EXPECT_FALSE(retirement.completion_ready());
}

unsigned DrainOneSectorPackets(unsigned width, unsigned count) {
  if (width == 0) return count == 0 ? 0 : 1;
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  budget.begin_tick(width);
  unsigned accepted = 0;
  while (accepted < count && budget.can_accept(1)) {
    budget.consume(1);
    stats.record_accept(1);
    ++accepted;
  }
  if (accepted < count) budget.note_width_limited(1);
  budget.end_tick(&stats);
  return accepted;
}

TEST(MemoryTransportTest, SectorBudgetHasOneTwoFourSensitivity) {
  EXPECT_EQ(DrainOneSectorPackets(0, 8), 1u);
  for (unsigned width : {1u, 2u, 4u}) {
    SCOPED_TRACE(width);
    EXPECT_EQ(DrainOneSectorPackets(width, 8), width);
  }
}

enum SyntheticConsumer { ORDINARY, TMA, CP_ASYNC };

struct SyntheticResponse {
  SyntheticConsumer consumer;
  unsigned sectors;
};

std::vector<unsigned> DispatchSynthetic(
    unsigned width, const std::vector<SyntheticResponse> &responses) {
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  std::vector<unsigned> accepted(3, 0);
  budget.begin_tick(width);
  for (const SyntheticResponse &response : responses) {
    if (!budget.can_accept(response.sectors)) {
      budget.note_width_limited(response.sectors);
      break;
    }
    budget.consume(response.sectors);
    stats.record_accept(response.sectors);
    accepted[response.consumer] += response.sectors;
  }
  budget.end_tick(&stats);
  EXPECT_LE(stats.accepted_data_sectors, width);
  return accepted;
}

TEST(MemoryTransportTest, MixedConsumersShareOneDispatchBudget) {
  const std::vector<SyntheticResponse> responses = {
      {ORDINARY, 1}, {TMA, 1}, {CP_ASYNC, 1}, {ORDINARY, 1}, {TMA, 1}};
  const std::vector<unsigned> accepted = DispatchSynthetic(4, responses);
  EXPECT_EQ(accepted[ORDINARY], 2u);
  EXPECT_EQ(accepted[TMA], 1u);
  EXPECT_EQ(accepted[CP_ASYNC], 1u);
}

TEST(MemoryTransportTest, EachConsumerCanOwnTheFullResponseBudget) {
  for (SyntheticConsumer consumer : {ORDINARY, TMA, CP_ASYNC}) {
    std::vector<SyntheticResponse> responses(4, {consumer, 1});
    const std::vector<unsigned> accepted = DispatchSynthetic(4, responses);
    EXPECT_EQ(accepted[consumer], 4u);
  }
}

TEST(MemoryTransportTest, LdstRequestWidthsPushOneTwoOrFourPerTick) {
  for (unsigned width : {1u, 2u, 4u}) {
    SCOPED_TRACE(width);
    std::deque<unsigned> children(6, 1);
    std::vector<unsigned> push_ticks;
    memory_transport_service_budget budget;
    memory_transport_service_stats stats;

    budget.begin_tick(width);
    const ldst_request_issue_result result =
        memory_transport_issue_ldst_sector_children(
            &budget, &stats, [&]() { return children.size(); },
            []() { return false; },
            [&]() -> unsigned {
              push_ticks.push_back(/*tick=*/0);
              const unsigned sectors = children.front();
              children.pop_front();
              return sectors;
            });
    budget.end_tick(&stats);

    EXPECT_EQ(result.reason, LDST_REQUEST_WIDTH_LIMITED);
    EXPECT_EQ(result.issued, width);
    EXPECT_EQ(push_ticks.size(), width);
    for (unsigned tick : push_ticks) EXPECT_EQ(tick, 0u);
    EXPECT_EQ(children.size(), 6u - width);
    EXPECT_EQ(stats.accepted_data_sectors, width);
    EXPECT_EQ(stats.width_limited_ticks, 1u);
    EXPECT_EQ(stats.downstream_full_ticks, 0u);
    EXPECT_EQ(stats.service_ticks, 1u);
    EXPECT_EQ(stats.max_service_slots_per_tick, width);
  }
}

TEST(MemoryTransportTest, LdstRequestBackpressurePreservesChildren) {
  std::deque<unsigned> children(4, 1);
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  unsigned pushes = 0;

  budget.begin_tick(/*width=*/4);
  ldst_request_issue_result result =
      memory_transport_issue_ldst_sector_children(
          &budget, &stats, [&]() { return children.size(); },
          [&]() { return pushes == 2; },
          [&]() -> unsigned {
            ++pushes;
            children.pop_front();
            return 1;
          });
  budget.end_tick(&stats);

  EXPECT_EQ(result.reason, LDST_REQUEST_DOWNSTREAM_FULL);
  EXPECT_EQ(result.issued, 2u);
  EXPECT_EQ(pushes, 2u);
  EXPECT_EQ(children.size(), 2u);
  EXPECT_EQ(stats.accepted_data_sectors + children.size(), 4u);
  EXPECT_EQ(stats.downstream_full_ticks, 1u);
  EXPECT_EQ(stats.width_limited_ticks, 0u);

  budget.begin_tick(/*width=*/4);
  result = memory_transport_issue_ldst_sector_children(
      &budget, &stats, [&]() { return children.size(); },
      []() { return false; },
      [&]() -> unsigned {
        ++pushes;
        children.pop_front();
        return 1;
      });
  budget.end_tick(&stats);

  EXPECT_EQ(result.reason, LDST_REQUEST_DRAINED);
  EXPECT_EQ(result.issued, 2u);
  EXPECT_TRUE(children.empty());
  EXPECT_EQ(pushes, 4u);
  EXPECT_EQ(stats.accepted_data_sectors, 4u);
  EXPECT_EQ(stats.service_ticks, 2u);
  EXPECT_EQ(stats.max_service_slots_per_tick, 2u);
}

TEST(MemoryTransportTest,
     LdstRequestInitializesOneCompletionForAllChildren) {
  struct SyntheticInstruction {
    unsigned id;
  };

  std::deque<unsigned> children(4, 1);
  memory_transport_response_retirement_queue<unsigned, SyntheticInstruction>
      retirement;
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  unsigned initialization_count = 0;
  const unsigned instruction_id = 17;

  budget.begin_tick(/*width=*/4);
  const ldst_request_issue_result issue_result =
      memory_transport_issue_ldst_sector_children(
          &budget, &stats, [&]() { return children.size(); },
          []() { return false; },
          [&]() -> unsigned {
            if (!retirement.has_pending_responses(instruction_id)) {
              retirement.expect_responses(instruction_id, children.size(),
                                          {instruction_id});
              ++initialization_count;
            }
            children.pop_front();
            return 1;
          });
  budget.end_tick(&stats);

  EXPECT_EQ(issue_result.reason, LDST_REQUEST_DRAINED);
  EXPECT_EQ(initialization_count, 1u);
  ASSERT_TRUE(retirement.has_pending_responses(instruction_id));
  EXPECT_EQ(retirement.pending_responses(instruction_id), 4u);

  for (unsigned response = 0; response < 3; ++response) {
    EXPECT_FALSE(retirement.retire_response(instruction_id));
    EXPECT_FALSE(retirement.completion_ready());
  }
  EXPECT_TRUE(retirement.retire_response(instruction_id));
  ASSERT_TRUE(retirement.completion_ready());
  EXPECT_EQ(retirement.completion_count(), 1u);
  EXPECT_EQ(retirement.next_completion().id, instruction_id);
  retirement.pop_completion();
  EXPECT_FALSE(retirement.completion_ready());
  EXPECT_EQ(retirement.pending_instruction_count(), 0u);
  EXPECT_EQ(retirement.retired_response_count(), 4u);
}

TEST(MemoryTransportTest,
     LdstRequestSideEffectsRunOncePerAcceptedStoreOrAtomicChild) {
  enum ChildKind { STORE_CHILD, ATOMIC_CHILD };
  struct Child {
    ChildKind kind;
    unsigned atomic_lanes;
  };

  std::deque<Child> children = {
      {STORE_CHILD, 0}, {ATOMIC_CHILD, 3},
      {STORE_CHILD, 0}, {ATOMIC_CHILD, 5}};
  memory_transport_service_budget budget;
  memory_transport_service_stats stats;
  unsigned pushes = 0;
  unsigned store_requests = 0;
  unsigned atomic_response_lanes = 0;

  while (!children.empty()) {
    budget.begin_tick(/*width=*/2);
    const ldst_request_issue_result result =
        memory_transport_issue_ldst_sector_children(
            &budget, &stats, [&]() { return children.size(); },
            []() { return false; },
            [&]() -> unsigned {
              const Child child = children.front();
              children.pop_front();
              ++pushes;
              if (child.kind == STORE_CHILD) {
                ++store_requests;
              } else {
                atomic_response_lanes += child.atomic_lanes;
              }
              return 1;
            });
    budget.end_tick(&stats);
    if (children.empty())
      EXPECT_EQ(result.reason, LDST_REQUEST_DRAINED);
    else
      EXPECT_EQ(result.reason, LDST_REQUEST_WIDTH_LIMITED);
  }

  EXPECT_EQ(pushes, 4u);
  EXPECT_EQ(store_requests, 2u);
  EXPECT_EQ(atomic_response_lanes, 8u);
  EXPECT_EQ(stats.accepted_data_sectors, pushes);
  EXPECT_EQ(stats.service_ticks, 2u);
  EXPECT_EQ(stats.max_service_slots_per_tick, 2u);
}

}  // namespace
