#include <gtest/gtest.h>

#include "../../../src/abstract_hardware_model.h"
#include "../../../src/gpgpu-sim/local_interconnect.h"

const char *mem_access_type_str(enum mem_access_type) { return "TEST"; }

namespace {

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

}  // namespace
