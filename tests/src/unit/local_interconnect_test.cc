#include <gtest/gtest.h>

#include "../../../src/abstract_hardware_model.h"
#include "../../../src/gpgpu-sim/local_interconnect.h"

const char *mem_access_type_str(enum mem_access_type) { return "TEST"; }

namespace {

inct_config NumericConfig(Arbiteration_type arbiter, unsigned input_width,
                          unsigned output_width) {
  inct_config config{};
  config.in_buffer_limit = 64;
  config.out_buffer_limit = 64;
  config.subnets = 2;
  config.arbiter_algo = arbiter;
  config.grant_cycles = 1;
  config.use_voq = 1;
  config.request_input_sectors_per_cycle = input_width;
  config.request_output_sectors_per_cycle = output_width;
  config.reply_input_sectors_per_cycle = input_width;
  config.reply_output_sectors_per_cycle = output_width;
  return config;
}

unsigned PopCount(xbar_router *router, unsigned output) {
  unsigned count = 0;
  while (router->Pop(output) != nullptr) ++count;
  return count;
}

TEST(LocalInterconnectTest, VoqGrantsEachInputAtMostOncePerCycle) {
  inct_config config{};
  config.in_buffer_limit = 8;
  config.out_buffer_limit = 8;
  config.subnets = 2;
  config.arbiter_algo = iSLIP;
  config.grant_cycles = 1;
  config.use_voq = 1;

  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int packet_to_output_1 = 1;
  int packet_to_output_2 = 2;
  router.Push(/*input_deviceID=*/0, /*output_deviceID=*/1,
              &packet_to_output_1, /*size=*/1);
  router.Push(/*input_deviceID=*/0, /*output_deviceID=*/2,
              &packet_to_output_2, /*size=*/1);

  router.Advance();

  void *first_output = router.Pop(1);
  void *second_output = router.Pop(2);
  EXPECT_NE(first_output == nullptr, second_output == nullptr);
  EXPECT_EQ(router.input_grants[0], 1u);

  router.Advance();

  if (first_output == nullptr) first_output = router.Pop(1);
  if (second_output == nullptr) second_output = router.Pop(2);
  EXPECT_EQ(first_output, &packet_to_output_1);
  EXPECT_EQ(second_output, &packet_to_output_2);
  EXPECT_EQ(router.input_grants[0], 2u);
}

TEST(LocalInterconnectTest, VoqAllowsConfiguredRequestMultiGrant) {
  inct_config config{};
  config.in_buffer_limit = 8;
  config.out_buffer_limit = 8;
  config.subnets = 2;
  config.arbiter_algo = iSLIP;
  config.grant_cycles = 1;
  config.use_voq = 1;
  config.multi_grant_request = 1;

  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int packet_to_output_1 = 1;
  int packet_to_output_2 = 2;
  router.Push(/*input_deviceID=*/0, /*output_deviceID=*/1,
              &packet_to_output_1, /*size=*/1);
  router.Push(/*input_deviceID=*/0, /*output_deviceID=*/2,
              &packet_to_output_2, /*size=*/1);

  router.Advance();

  EXPECT_EQ(router.Pop(1), &packet_to_output_1);
  EXPECT_EQ(router.Pop(2), &packet_to_output_2);
  EXPECT_EQ(router.input_grants[0], 2u);
}

TEST(LocalInterconnectTest, VoqAllowsConfiguredReplyMultiGrant) {
  inct_config config{};
  config.in_buffer_limit = 8;
  config.out_buffer_limit = 8;
  config.subnets = 2;
  config.arbiter_algo = iSLIP;
  config.grant_cycles = 1;
  config.use_voq = 1;
  config.multi_grant_reply = 1;

  xbar_router router(/*router_id=*/1, REPLY_NET, /*n_shader=*/2, /*n_mem=*/1,
                     config);
  int packet_to_output_0 = 0;
  int packet_to_output_1 = 1;
  router.Push(/*input_deviceID=*/2, /*output_deviceID=*/0,
              &packet_to_output_0, /*size=*/1);
  router.Push(/*input_deviceID=*/2, /*output_deviceID=*/1,
              &packet_to_output_1, /*size=*/1);

  router.Advance();

  EXPECT_EQ(router.Pop(0), &packet_to_output_0);
  EXPECT_EQ(router.Pop(1), &packet_to_output_1);
  EXPECT_EQ(router.input_grants[2], 2u);
}

TEST(LocalInterconnectTest, ExplicitWidthsAreSectorSensitive) {
  for (unsigned width : {1u, 2u, 4u}) {
    inct_config config = NumericConfig(iSLIP, width, width);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                       config);
    int packets[4] = {};
    for (unsigned i = 0; i < 4; ++i)
      router.Push(0, 1, &packets[i], /*size=*/1, /*data_sectors=*/1);

    router.Advance();
    EXPECT_EQ(PopCount(&router, 1), width);
  }
}

TEST(LocalInterconnectTest, InputAndOutputWidthsLimitIndependently) {
  {
    inct_config config = NumericConfig(iSLIP, 2, 4);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                       config);
    int packets[4] = {};
    for (unsigned i = 0; i < 4; ++i) router.Push(0, 1, &packets[i], 1, 1);
    router.Advance();
    EXPECT_EQ(PopCount(&router, 1), 2u);
  }
  {
    inct_config config = NumericConfig(iSLIP, 4, 2);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/4, /*n_mem=*/1,
                       config);
    int packets[4] = {};
    for (unsigned input = 0; input < 4; ++input)
      router.Push(input, 4, &packets[input], 1, 1);
    router.Advance();
    EXPECT_EQ(PopCount(&router, 4), 2u);
  }
}

TEST(LocalInterconnectTest, ZeroInputWidthPreservesNaiveRROneGrant) {
  inct_config config = NumericConfig(NAIVE_RR, /*input_width=*/0,
                                     /*output_width=*/4);
  config.multi_grant_request = 1;
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int packets[2] = {};
  router.Push(0, 1, &packets[0], 1, 1);
  router.Push(0, 2, &packets[1], 1, 1);

  router.Advance();
  EXPECT_EQ(PopCount(&router, 1) + PopCount(&router, 2), 1u);
  EXPECT_EQ(router.input_service_stats[0].service_ticks, 1u);
}

TEST(LocalInterconnectTest, ZeroInputWidthPreservesISLIPMultiGrant) {
  inct_config config = NumericConfig(iSLIP, /*input_width=*/0,
                                     /*output_width=*/4);
  config.multi_grant_request = 1;
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int packets[2] = {};
  router.Push(0, 1, &packets[0], 1, 1);
  router.Push(0, 2, &packets[1], 1, 1);

  router.Advance();
  EXPECT_EQ(router.Pop(1), &packets[0]);
  EXPECT_EQ(router.Pop(2), &packets[1]);
}

TEST(LocalInterconnectTest, ZeroOutputWidthPreservesOneGrantPerOutput) {
  inct_config config = NumericConfig(iSLIP, /*input_width=*/4,
                                     /*output_width=*/0);
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/2, /*n_mem=*/1,
                     config);
  int packets[2] = {};
  router.Push(0, 2, &packets[0], 1, 1);
  router.Push(1, 2, &packets[1], 1, 1);

  router.Advance();

  EXPECT_EQ(PopCount(&router, 2), 1u);
  EXPECT_EQ(router.output_service_stats[2].service_ticks, 1u);
}

TEST(LocalInterconnectTest, RequestAndReplyUseTheirOwnWidths) {
  inct_config config = NumericConfig(iSLIP, 4, 4);
  config.reply_input_sectors_per_cycle = 2;
  config.reply_output_sectors_per_cycle = 2;
  xbar_router request(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                      config);
  xbar_router reply(/*router_id=*/1, REPLY_NET, /*n_shader=*/1, /*n_mem=*/1,
                    config);
  int request_packets[4] = {};
  int reply_packets[4] = {};
  for (unsigned i = 0; i < 4; ++i) {
    request.Push(0, 1, &request_packets[i], 1, 1);
    reply.Push(1, 0, &reply_packets[i], 1, 1);
  }

  request.Advance();
  reply.Advance();
  EXPECT_EQ(PopCount(&request, 1), 4u);
  EXPECT_EQ(PopCount(&reply, 0), 2u);
}

TEST(LocalInterconnectTest, MultipleInputsCanGrantOneOutput) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, 4, 4);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/4, /*n_mem=*/1,
                       config);
    int packets[4] = {};
    for (unsigned input = 0; input < 4; ++input)
      router.Push(input, 4, &packets[input], /*size=*/1,
                  /*data_sectors=*/1);

    router.Advance();
    EXPECT_EQ(PopCount(&router, 4), 4u);
    for (unsigned input = 0; input < 4; ++input)
      EXPECT_EQ(router.input_grants[input], 1u);
  }
}

TEST(LocalInterconnectTest, OneInputCanGrantMultipleOutputs) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, 4, 4);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/4,
                       config);
    int packets[4] = {};
    for (unsigned output = 1; output <= 4; ++output)
      router.Push(0, output, &packets[output - 1], /*size=*/1,
                  /*data_sectors=*/1);

    router.Advance();
    for (unsigned output = 1; output <= 4; ++output)
      EXPECT_EQ(router.Pop(output), &packets[output - 1]);
    EXPECT_EQ(router.input_grants[0], 4u);
  }
}

TEST(LocalInterconnectTest, MixedDestinationsUseBothBudgets) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, 2, 2);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/3, /*n_mem=*/2,
                       config);
    int packets[4] = {};
    router.Push(0, 3, &packets[0], 1, 1);
    router.Push(0, 4, &packets[1], 1, 1);
    router.Push(1, 3, &packets[2], 1, 1);
    router.Push(2, 4, &packets[3], 1, 1);

    router.Advance();
    EXPECT_EQ(PopCount(&router, 3), 2u);
    EXPECT_EQ(PopCount(&router, 4), 2u);
    EXPECT_EQ(router.input_grants[0], 2u);
    EXPECT_EQ(router.input_grants[1], 1u);
    EXPECT_EQ(router.input_grants[2], 1u);
  }
}

TEST(LocalInterconnectTest, OversizedPacketAccumulatesOnlyWidthCredit) {
  inct_config config = NumericConfig(iSLIP, 1, 1);
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                     config);
  int packet = 0;
  router.Push(0, 1, &packet, /*size=*/1, /*data_sectors=*/4);

  for (unsigned tick = 0; tick < 3; ++tick) {
    router.Advance();
    EXPECT_EQ(router.Pop(1), nullptr);
  }
  router.Advance();
  EXPECT_EQ(router.Pop(1), &packet);
  EXPECT_EQ(router.input_service_stats[0].accepted_data_sectors, 4u);
  EXPECT_EQ(router.input_service_stats[0].width_limited_ticks, 3u);
}

TEST(LocalInterconnectTest, OversizedPacketProgressesWithUnequalWidths) {
  inct_config config = NumericConfig(iSLIP, 1, 3);
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                     config);
  int packet = 0;
  router.Push(0, 1, &packet, /*size=*/1, /*data_sectors=*/4);

  for (unsigned tick = 0; tick < 3; ++tick) {
    router.Advance();
    EXPECT_EQ(router.Pop(1), nullptr);
  }
  router.Advance();
  EXPECT_EQ(router.Pop(1), &packet);
}

TEST(LocalInterconnectTest, SmallPacketsCannotStealOversizedOutputCredit) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, /*input_width=*/4,
                                       /*output_width=*/1);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/2, /*n_mem=*/1,
                       config);
    int oversized = 0;
    int small[8] = {};
    router.Push(0, 2, &oversized, /*size=*/1, /*data_sectors=*/4);
    for (unsigned i = 0; i < 8; ++i)
      router.Push(1, 2, &small[i], /*size=*/1, /*data_sectors=*/1);

    bool saw_oversized = false;
    for (unsigned tick = 0; tick < 4; ++tick) {
      router.Advance();
      for (void *packet = router.Pop(2); packet != nullptr;
           packet = router.Pop(2)) {
        if (packet == &oversized) saw_oversized = true;
      }
    }
    EXPECT_TRUE(saw_oversized);
  }
}

TEST(LocalInterconnectTest, SmallPacketsCannotStealOversizedInputCredit) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, /*input_width=*/1,
                                       /*output_width=*/4);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                       config);
    int oversized = 0;
    int small[8] = {};
    router.Push(0, 1, &oversized, /*size=*/1, /*data_sectors=*/4);
    for (unsigned i = 0; i < 8; ++i)
      router.Push(0, 2, &small[i], /*size=*/1, /*data_sectors=*/1);

    bool saw_oversized = false;
    for (unsigned tick = 0; tick < 4; ++tick) {
      router.Advance();
      for (unsigned output = 1; output <= 2; ++output) {
        for (void *packet = router.Pop(output); packet != nullptr;
             packet = router.Pop(output)) {
          if (packet == &oversized) saw_oversized = true;
        }
      }
    }
    EXPECT_TRUE(saw_oversized);
  }
}

TEST(LocalInterconnectTest,
     OutputCreditWaitDoesNotBlockAnotherVoqOnTheSameInput) {
  inct_config config = NumericConfig(NAIVE_RR, /*input_width=*/4,
                                     /*output_width=*/1);
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int oversized = 0;
  int independent = 1;
  router.Push(0, 1, &oversized, /*size=*/1, /*data_sectors=*/4);
  router.Push(0, 2, &independent, /*size=*/1, /*data_sectors=*/1);

  router.Advance();

  EXPECT_EQ(router.Pop(1), nullptr);
  EXPECT_EQ(router.Pop(2), &independent);
}

TEST(LocalInterconnectTest, ControlPacketUsesAServiceSlot) {
  inct_config config = NumericConfig(iSLIP, 2, 2);
  xbar_router router(/*router_id=*/1, REPLY_NET, /*n_shader=*/1, /*n_mem=*/1,
                     config);
  int acknowledgements[3] = {};
  for (unsigned i = 0; i < 3; ++i)
    router.Push(/*input_deviceID=*/1, /*output_deviceID=*/0,
                &acknowledgements[i], /*size=*/1, /*data_sectors=*/0);

  router.Advance();
  EXPECT_EQ(PopCount(&router, 0), 2u);
  EXPECT_EQ(router.input_service_stats[1].accepted_data_sectors, 0u);
  EXPECT_EQ(router.input_service_stats[1].accepted_control_packets, 2u);
  router.Advance();
  EXPECT_EQ(router.Pop(0), &acknowledgements[2]);
}

TEST(LocalInterconnectTest, OutputBackpressurePreservesQueuedPacket) {
  inct_config config = NumericConfig(iSLIP, 4, 4);
  config.out_buffer_limit = 1;
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                     config);
  int first = 1;
  int second = 2;
  router.Push(0, 1, &first, /*size=*/1, /*data_sectors=*/1);
  router.Push(0, 1, &second, /*size=*/1, /*data_sectors=*/1);

  router.Advance();
  EXPECT_EQ(router.Pop(1), &first);
  EXPECT_GE(router.output_service_stats[1].downstream_full_ticks, 1u);
  router.Advance();
  EXPECT_EQ(router.Pop(1), &second);
}

TEST(LocalInterconnectTest, VoqControlsNumericHeadOfLineBlocking) {
  for (unsigned use_voq : {0u, 1u}) {
    inct_config config = NumericConfig(iSLIP, 4, 4);
    config.use_voq = use_voq;
    config.out_buffer_limit = 1;
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                       config);
    int blocker = 0;
    int blocked_head = 1;
    int independent_tail = 2;

    router.Push(0, 1, &blocker, /*size=*/1, /*data_sectors=*/1);
    router.Advance();
    ASSERT_EQ(router.Top(1), &blocker);
    router.Push(0, 1, &blocked_head, /*size=*/1, /*data_sectors=*/1);
    router.Push(0, 2, &independent_tail, /*size=*/1, /*data_sectors=*/1);

    router.Advance();

    EXPECT_EQ(router.Pop(2),
              use_voq ? static_cast<void *>(&independent_tail) : nullptr);
    EXPECT_EQ(router.Top(1), &blocker);
  }
}

TEST(LocalInterconnectTest,
     FullUnrelatedVoqDoesNotPreventOversizedPacketCredit) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, 1, 1);
    config.out_buffer_limit = 1;
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                       config);
    int blocker = 0;
    int oversized = 1;
    int blocked = 2;

    router.Push(0, 2, &blocker, /*size=*/1, /*data_sectors=*/1);
    router.Advance();
    ASSERT_EQ(router.Top(2), &blocker);

    router.Push(0, 1, &oversized, /*size=*/1, /*data_sectors=*/4);
    router.Push(0, 2, &blocked, /*size=*/1, /*data_sectors=*/1);
    for (unsigned tick = 0; tick < 3; ++tick) {
      router.Advance();
      EXPECT_EQ(router.Pop(1), nullptr);
    }
    router.Advance();
    EXPECT_EQ(router.Pop(1), &oversized);
    EXPECT_EQ(router.Top(2), &blocker);
  }
}

TEST(LocalInterconnectTest, LegacyStatsRecordPerTickService) {
  inct_config config = NumericConfig(iSLIP, 0, 0);
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/1,
                     config);
  int packets[2] = {};
  router.Push(0, 1, &packets[0], /*size=*/1, /*data_sectors=*/2);
  router.Push(0, 1, &packets[1], /*size=*/1, /*data_sectors=*/0);

  router.Advance();
  EXPECT_EQ(router.Pop(1), &packets[0]);
  router.Advance();
  EXPECT_EQ(router.Pop(1), &packets[1]);

  EXPECT_EQ(router.input_service_stats[0].service_ticks, 2u);
  EXPECT_EQ(router.input_service_stats[0].max_service_slots_per_tick, 2u);
  EXPECT_EQ(router.output_service_stats[1].service_ticks, 2u);
  EXPECT_EQ(router.output_service_stats[1].max_service_slots_per_tick, 2u);
}

TEST(LocalInterconnectTest, LegacyStatsAttributeInputAndOutputLimits) {
  inct_config config = NumericConfig(iSLIP, 0, 0);
  config.use_voq = 1;
  xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/1, /*n_mem=*/2,
                     config);
  int packets[2] = {};
  router.Push(0, 1, &packets[0], 1, 1);
  router.Push(0, 2, &packets[1], 1, 1);

  router.Advance();

  EXPECT_EQ(router.input_service_stats[0].width_limited_ticks, 1u);
  EXPECT_EQ(router.output_service_stats[1].width_limited_ticks, 0u);
  EXPECT_EQ(router.output_service_stats[2].width_limited_ticks, 0u);
}

TEST(LocalInterconnectTest, ArbitrationRotatesFairlyAcrossInputs) {
  for (Arbiteration_type arbiter : {NAIVE_RR, iSLIP}) {
    inct_config config = NumericConfig(arbiter, 1, 1);
    xbar_router router(/*router_id=*/0, REQ_NET, /*n_shader=*/3, /*n_mem=*/1,
                       config);
    int packets[6] = {};
    for (unsigned input = 0; input < 3; ++input) {
      router.Push(input, 3, &packets[2 * input], 1, 1);
      router.Push(input, 3, &packets[2 * input + 1], 1, 1);
    }
    for (unsigned tick = 0; tick < 3; ++tick) {
      router.Advance();
      ASSERT_NE(router.Pop(3), nullptr);
    }
    for (unsigned input = 0; input < 3; ++input)
      EXPECT_EQ(router.input_grants[input], 1u);
  }
}

}  // namespace
