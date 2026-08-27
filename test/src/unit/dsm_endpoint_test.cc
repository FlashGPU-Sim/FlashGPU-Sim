// Unit tests for shipped dsm_endpoint_protocol_t + dsm_fabric_t.
// Topology matches SM90_H200_REDUCED_CLUSTER16x2 (2 GPCs × 16 SMs).

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "../../../src/gpgpu-sim/dsm_endpoint.h"
#include "../../../src/gpgpu-sim/dsm_fabric.h"
#include "../../../src/gpgpu-sim/gpu_topology.h"

namespace {

gpu_topology_t make_16x2() {
  gpu_topology_t t;
  t.build(2, 16, 3);
  return t;
}

struct EpEnv {
  gpu_topology_t topo;
  dsm_fabric_t fab;
  dsm_endpoint_protocol_t ep;
  unsigned long long now = 0;

  EpEnv(const dsm_fabric_config_t &fcfg, const dsm_endpoint_config_t &ecfg)
      : topo(make_16x2()), fab(topo, 0, fcfg), ep(&fab, ecfg) {}

  bool step_until_idle(unsigned cap = 100000) {
    unsigned n = 0;
    while (ep.busy() && n++ < cap) ep.cycle(now++);
    return !ep.busy();
  }
};

dsm_fabric_config_t small_buf() {
  dsm_fabric_config_t c;
  c.request_vc_flits = 8;
  c.response_vc_flits = 8;
  c.ejection_vc_flits = 8;
  return c;
}

}  // namespace

TEST(DsmEndpoint, ReducedCluster16x2Packing) {
  gpu_topology_t topo;
  topo.build(2, 16, 3);
  EXPECT_EQ(topo.num_gpcs(), 2u);
  EXPECT_EQ(topo.num_sms_in_gpc(0), 16u);
  EXPECT_EQ(topo.num_sms_in_gpc(1), 16u);
  EXPECT_EQ(topo.num_sms(), 32u);
  EXPECT_EQ(topo.cpcs_per_gpc(), 3u);
  EXPECT_FALSE(topo.slot_is_enabled(0, 2, 4));
  EXPECT_FALSE(topo.slot_is_enabled(0, 2, 5));
  dsm_fabric_t fab(topo, 0, dsm_fabric_config_t{});
  EXPECT_EQ(fab.num_sms(), 16u);
}

TEST(DsmEndpoint, HeavyStoresDrainNoDeadlock) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 4;
  ecfg.ack_timeout_cycles = 64;
  EpEnv e(small_buf(), ecfg);
  unsigned issued = 0;
  unsigned guard = 0;
  while (issued < 48 && guard++ < 100000) {
    if (e.ep.can_store(0, 1, 128)) {
      e.ep.issue_store(0, 1, 128, (uint64_t)issued * 128);
      issued++;
    }
    e.ep.cycle(e.now++);
  }
  ASSERT_EQ(issued, 48u);
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_FALSE(e.ep.busy());
  EXPECT_EQ(e.ep.outstanding(0), 0u);
  EXPECT_LT(e.ep.stats().ack_packets, e.ep.stats().store_packets);
  EXPECT_EQ(e.ep.stats().sram_store_bytes, 48u * 128u);
}

TEST(DsmEndpoint, AckPacketsFewerThanStores) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 4;
  ecfg.ack_timeout_cycles = 100000;
  EpEnv e(small_buf(), ecfg);
  unsigned issued = 0;
  unsigned guard = 0;
  while (issued < 32 && guard++ < 100000) {
    if (e.ep.can_store(0, 1, 128)) {
      e.ep.issue_store(0, 1, 128, issued);
      issued++;
    }
    e.ep.cycle(e.now++);
  }
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_LT(e.ep.stats().ack_packets, e.ep.stats().store_packets);
  EXPECT_GE(e.ep.coalescing_ratio(), 1.0);
}

TEST(DsmEndpoint, SymmetricStoresReverseFlitsFarFewer) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 4;
  ecfg.ack_timeout_cycles = 100000;
  EpEnv e(small_buf(), ecfg);
  unsigned n0 = 0, n1 = 0;
  unsigned guard = 0;
  while ((n0 < 32 || n1 < 32) && guard++ < 100000) {
    if (n0 < 32 && e.ep.can_store(0, 1, 128)) {
      e.ep.issue_store(0, 1, 128, n0);
      n0++;
    }
    if (n1 < 32 && e.ep.can_store(1, 0, 128)) {
      e.ep.issue_store(1, 0, 128, n1);
      n1++;
    }
    e.ep.cycle(e.now++);
  }
  ASSERT_TRUE(e.step_until_idle());
  const unsigned data_flits =
      (n0 + n1) *
      dsm_payload_flits(dsm_packet_class_t::write_data, 128,
                        e.fab.flit_payload_bytes());
  const unsigned ack_flits = (unsigned)e.ep.stats().ack_packets;
  EXPECT_LT(ack_flits, e.ep.stats().store_packets);
  EXPECT_LT(ack_flits * 4u, data_flits);
}

TEST(DsmEndpoint, RemoteLoadPairsCommandAndData) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  ASSERT_TRUE(e.ep.issue_load(0, 1, 128, 0));
  ASSERT_TRUE(e.ep.issue_load(0, 1, 64, 256));
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_EQ(e.ep.stats().load_commands, 2u);
  EXPECT_EQ(e.ep.stats().load_data_packets, 2u);
  EXPECT_EQ(e.ep.stats().sram_load_bytes, 128u + 64u);
  EXPECT_EQ(e.ep.outstanding(0), 0u);
}

TEST(DsmEndpoint, OutstandingFullDoesNotTouchVcCredits) {
  dsm_endpoint_config_t ecfg;
  ecfg.max_outstanding_per_sm = 2;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  ASSERT_TRUE(e.ep.issue_store(0, 1, 128, 0));
  ASSERT_TRUE(e.ep.issue_store(0, 1, 128, 128));
  EXPECT_EQ(e.ep.outstanding(0), 2u);
  EXPECT_FALSE(e.ep.can_store(0, 1, 32));
  EXPECT_FALSE(e.ep.can_load(0, 1, 32));
  const unsigned req_c = e.fab.credit_remaining(1, dsm_vc_t::request);
  const unsigned rsp_c = e.fab.credit_remaining(1, dsm_vc_t::response);
  EXPECT_FALSE(e.ep.can_store(0, 2, 32));
  EXPECT_EQ(e.fab.credit_remaining(1, dsm_vc_t::request), req_c);
  EXPECT_EQ(e.fab.credit_remaining(1, dsm_vc_t::response), rsp_c);
  EXPECT_EQ(e.fab.credit_remaining(2, dsm_vc_t::request),
            e.fab.credit_remaining(2, dsm_vc_t::request));
}

TEST(DsmEndpoint, AckFromOneTargetDoesNotRetireSiblingStore) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 100;
  ecfg.ack_timeout_cycles = 100000;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  ASSERT_TRUE(e.ep.issue_store(0, 1, 128, 0));
  ASSERT_TRUE(e.ep.issue_store(0, 2, 32, 32));
  EXPECT_EQ(e.ep.outstanding(0), 2u);
  unsigned guard = 0;
  while (e.ep.outstanding(0) > 1 && guard++ < 10000) e.ep.cycle(e.now++);
  ASSERT_EQ(e.ep.outstanding(0), 1u);
  const auto &tbl = e.ep.tx_table(0);
  ASSERT_EQ(tbl.size(), 1u);
  EXPECT_EQ(tbl.begin()->second.requester, 0u);
  EXPECT_EQ(tbl.begin()->second.target, 1u);
  ASSERT_TRUE(e.step_until_idle());
}

TEST(DsmEndpoint, ThresholdFlushProducesWriteAck) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 4;
  ecfg.ack_timeout_cycles = 100000;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  for (unsigned i = 0; i < 4; i++)
    ASSERT_TRUE(e.ep.issue_store(0, 1, 128, i));
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_GE(e.ep.stats().threshold_flushes, 1u);
  EXPECT_GE(e.ep.stats().ack_packets, 1u);
  EXPECT_EQ(e.ep.stats().ack_completions, 4u);
}

TEST(DsmEndpoint, TimeoutFlushProducesWriteAck) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 100;
  ecfg.ack_timeout_cycles = 8;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  for (unsigned i = 0; i < 8; i++)
    ASSERT_TRUE(e.ep.issue_store(0, 1, 128, i));
  unsigned guard = 0;
  while (e.ep.stats().timeout_flushes == 0 && guard++ < 10000)
    e.ep.cycle(e.now++);
  EXPECT_GE(e.ep.stats().timeout_flushes, 1u);
  ASSERT_TRUE(e.step_until_idle());
}

TEST(DsmEndpoint, IdleResponseFlushProducesWriteAck) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 100;
  ecfg.ack_timeout_cycles = 100000;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  ASSERT_TRUE(e.ep.issue_store(0, 1, 128, 0));
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_GE(e.ep.stats().idle_flushes, 1u);
  EXPECT_EQ(e.ep.stats().timeout_flushes, 0u);
  EXPECT_EQ(e.ep.stats().ack_packets, 1u);
}

TEST(DsmEndpoint, DumpShowsOutstandingDebtRatioTimeout) {
  dsm_endpoint_config_t ecfg;
  ecfg.ack_coalesce_threshold = 4;
  ecfg.ack_timeout_cycles = 8;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  for (unsigned i = 0; i < 8; i++)
    ASSERT_TRUE(e.ep.issue_store(0, 1, 128, i));
  while (e.ep.stats().timeout_flushes == 0 && e.now < 10000)
    e.ep.cycle(e.now++);
  ASSERT_TRUE(e.step_until_idle());
  char *buf = nullptr;
  size_t sz = 0;
  FILE *fp = open_memstream(&buf, &sz);
  ASSERT_NE(fp, nullptr);
  e.ep.display_state(fp);
  fclose(fp);
  std::string s(buf, sz);
  free(buf);
  EXPECT_NE(s.find("outstanding="), std::string::npos);
  EXPECT_NE(s.find("ack_debt="), std::string::npos);
  EXPECT_NE(s.find("coalescing_ratio="), std::string::npos);
  EXPECT_NE(s.find("timeout_flushes="), std::string::npos);
  printf("%s", s.c_str());
}
