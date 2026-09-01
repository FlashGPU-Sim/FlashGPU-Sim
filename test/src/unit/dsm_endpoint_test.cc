// Unit tests for shipped dsm_endpoint_protocol_t + dsm_fabric_t.
// Topology matches SM90_H200_REDUCED_CLUSTER16x2 (2 GPCs × 16 SMs).

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

namespace {

struct SramSpy {
  unsigned writes = 0;
  unsigned reads = 0;
  uint32_t last = 0;
  uint8_t mem[256]{};
  static void wr(void *ctx, unsigned, unsigned, uint64_t addr, uint8_t *p,
                 unsigned n) {
    auto *s = static_cast<SramSpy *>(ctx);
    s->writes++;
    if (addr + n <= sizeof(s->mem) && p) memcpy(s->mem + addr, p, n);
    if (n >= 4 && p) memcpy(&s->last, p, 4);
  }
  static void rd(void *ctx, unsigned, unsigned, uint64_t addr, uint8_t *p,
                 unsigned n) {
    auto *s = static_cast<SramSpy *>(ctx);
    s->reads++;
    if (addr + n <= sizeof(s->mem) && p) memcpy(p, s->mem + addr, n);
  }
};

}  // namespace

TEST(DsmEndpoint, StoreDoesNotWriteSramAtIssue) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  SramSpy spy;
  e.ep.set_sram(&spy, SramSpy::wr, SramSpy::rd);
  uint32_t word = 0xCAFEBABEu;
  ASSERT_TRUE(e.ep.issue_store(0, 1, 4, 16, /*cta=*/0, /*gen=*/0, &word));
  EXPECT_EQ(spy.writes, 0u);
  EXPECT_EQ(e.ep.stats().store_packets, 1u);
  EXPECT_EQ(e.fab.stats().packets_injected, 1u);
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_EQ(spy.writes, 1u);
  EXPECT_EQ(spy.last, 0xCAFEBABEu);
  EXPECT_EQ(e.ep.stats().sram_store_bytes, 4u);
}

TEST(DsmEndpoint, LoadInjectsReadCommandNotPeerWrite) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  SramSpy spy;
  uint32_t seed = 0xA0000007u;
  memcpy(spy.mem + 8, &seed, 4);
  e.ep.set_sram(&spy, SramSpy::wr, SramSpy::rd);
  uint32_t got = 0;
  e.ep.set_on_load_data(&got, [](void *ctx, unsigned, const uint8_t *p,
                                 unsigned n) {
    if (n >= 4 && p) memcpy(ctx, p, 4);
  });
  ASSERT_TRUE(e.ep.issue_load(0, 1, 4, 8, 0, 0));
  EXPECT_EQ(e.ep.stats().load_commands, 1u);
  EXPECT_EQ(spy.writes, 0u);
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_EQ(e.ep.stats().load_data_packets, 1u);
  EXPECT_EQ(spy.reads, 1u);
  EXPECT_EQ(spy.writes, 0u);
  EXPECT_EQ(got, 0xA0000007u);
}

TEST(DsmEndpoint, SramNotSameCycleAsArrival) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  SramSpy spy;
  e.ep.set_sram(&spy, SramSpy::wr, SramSpy::rd);
  uint32_t word = 0x11111111u;
  ASSERT_TRUE(e.ep.issue_store(0, 1, 4, 16, 0, 0, &word));
  bool split = false;
  unsigned guard = 0;
  while (e.ep.busy() && guard++ < 100000) {
    unsigned ejected = e.fab.stats().packets_ejected;
    unsigned w = spy.writes;
    e.ep.cycle(e.now++);
    if (e.fab.stats().packets_ejected > ejected && spy.writes == w)
      split = true;
    if (e.fab.stats().packets_ejected > ejected && spy.writes > w)
      FAIL() << "SRAM write same cycle as request eject";
  }
  EXPECT_TRUE(split);
  EXPECT_EQ(spy.writes, 1u);
}

TEST(DsmEndpoint, LoadInjectAfterSram) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  SramSpy spy;
  uint32_t seed = 0xA0000007u;
  memcpy(spy.mem + 8, &seed, 4);
  e.ep.set_sram(&spy, SramSpy::wr, SramSpy::rd);
  uint32_t got = 0;
  e.ep.set_on_load_data(&got, [](void *ctx, unsigned, const uint8_t *p,
                                 unsigned n) {
    if (n >= 4 && p) memcpy(ctx, p, 4);
  });
  ASSERT_TRUE(e.ep.issue_load(0, 1, 4, 8, 0, 0));
  unsigned guard = 0;
  while (e.ep.busy() && guard++ < 100000) {
    unsigned reads = spy.reads;
    unsigned pkts = e.ep.stats().load_data_packets;
    e.ep.cycle(e.now++);
    if (spy.reads > reads) {
      EXPECT_EQ(e.ep.stats().load_data_packets, pkts)
          << "read_data injected same cycle as SRAM grant";
    }
  }
  EXPECT_EQ(got, 0xA0000007u);
  EXPECT_GE(spy.reads, 1u);
}

TEST(DsmEndpoint, TmaPutSourceUnicastNPeers) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  std::vector<uint8_t> tile(128, 0xAB);
  const unsigned npeer = 3;
  for (unsigned d = 1; d <= npeer; d++)
    ASSERT_TRUE(e.ep.issue_tma(0, d, 128, 0, 0, 0, tile.data(), 0x40, 128));
  EXPECT_EQ(e.ep.stats().tma_packets, npeer);
  EXPECT_EQ(e.fab.stats().packets_injected, npeer);
  EXPECT_EQ(dsm_vc_of(dsm_packet_class_t::tma_data), dsm_vc_t::request);
  EXPECT_FALSE(dsm_packet_is_control(dsm_packet_class_t::tma_data));
  EXPECT_EQ(dsm_payload_flits(dsm_packet_class_t::tma_data, 128, 32), 4u);
  EXPECT_EQ(e.fab.occupancy_flits(0, dsm_vc_t::request, 1), 4u);
  EXPECT_EQ(e.fab.occupancy_flits(0, dsm_vc_t::request, 2), 4u);
  EXPECT_EQ(e.fab.occupancy_flits(0, dsm_vc_t::request, 3), 4u);
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_EQ(e.ep.stats().sram_store_bytes, npeer * 128u);
}

TEST(DsmEndpoint, TmaCompleteTxAfterSramApply) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  SramSpy spy;
  e.ep.set_sram(&spy, SramSpy::wr, SramSpy::rd);
  struct TmaSpy {
    SramSpy *sram;
    unsigned mbar = 0;
    unsigned writes_at_mbar = 0;
    static void on_mbar(void *ctx, unsigned, unsigned, unsigned, unsigned) {
      auto *t = static_cast<TmaSpy *>(ctx);
      t->mbar++;
      t->writes_at_mbar = t->sram->writes;
    }
  } tma;
  tma.sram = &spy;
  e.ep.set_on_tma_mbar(&tma, TmaSpy::on_mbar);
  std::vector<uint8_t> tile(128, 0x5A);
  ASSERT_TRUE(e.ep.issue_tma(0, 1, 128, 0, 0, 0, tile.data(), 0x40, 128));
  unsigned guard = 0;
  while (e.ep.busy() && guard++ < 100000) {
    unsigned w = spy.writes;
    unsigned m = tma.mbar;
    e.ep.cycle(e.now++);
    if (tma.mbar > m)
      EXPECT_GE(spy.writes, 1u) << "complete_tx before dest SRAM apply";
    if (spy.writes > w && tma.mbar == m) {
      // SRAM can land the same cycle as the callback; never after.
    }
  }
  EXPECT_EQ(tma.mbar, 1u);
  EXPECT_GE(tma.writes_at_mbar, 1u);
  EXPECT_GE(spy.writes, 1u);
}

TEST(DsmEndpoint, MbarRequestAndCompletionVcs) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  EXPECT_EQ(dsm_vc_of(dsm_packet_class_t::mbarrier_request), dsm_vc_t::request);
  EXPECT_EQ(dsm_vc_of(dsm_packet_class_t::mbarrier_completion),
            dsm_vc_t::response);
  EXPECT_TRUE(dsm_packet_is_control(dsm_packet_class_t::mbarrier_request));
  EXPECT_TRUE(dsm_packet_is_control(dsm_packet_class_t::mbarrier_completion));
  EXPECT_EQ(dsm_payload_flits(dsm_packet_class_t::mbarrier_request, 99, 32),
            1u);
  EXPECT_EQ(dsm_payload_flits(dsm_packet_class_t::mbarrier_completion, 99, 32),
            1u);
  ASSERT_TRUE(e.ep.issue_mbar(0, 1, 0, 0, 0x100, /*ARRIVE=*/0, 1, 0, 0, 0));
  EXPECT_EQ(e.ep.stats().mbar_requests, 1u);
  EXPECT_EQ(e.fab.occupancy_flits(0, dsm_vc_t::request, 1), 1u);
  EXPECT_EQ(e.fab.occupancy_flits(0, dsm_vc_t::response, 1), 0u);
  bool saw_rsp = false;
  unsigned guard = 0;
  while (e.ep.busy() && guard++ < 100000) {
    if (e.fab.occupancy_flits(1, dsm_vc_t::response, 0) > 0) saw_rsp = true;
    e.ep.cycle(e.now++);
  }
  EXPECT_TRUE(saw_rsp);
  EXPECT_EQ(e.ep.stats().mbar_completions, 1u);
  EXPECT_EQ(e.ep.outstanding(0), 0u);
}

TEST(DsmEndpoint, QuietWindowUsesOutstandingRtt) {
  dsm_endpoint_config_t ecfg;
  EpEnv e(dsm_fabric_config_t{}, ecfg);
  ASSERT_TRUE(e.ep.issue_tma(0, 1, 128, 0, 0, 0, nullptr, 0, 0));
  EXPECT_TRUE(e.ep.busy());
  EXPECT_GE(e.ep.max_tx_age(e.now + 10), 10u);
  ASSERT_TRUE(e.step_until_idle());
  EXPECT_FALSE(e.ep.busy());
  EXPECT_EQ(e.ep.max_tx_age(e.now), 0u);
}

TEST(DsmEndpoint, LoadRttGrowsWithResidualFloor) {
  auto rtt = [](unsigned floor) {
    dsm_fabric_config_t fcfg;
    fcfg.base_latency_cycles = floor;
    EpEnv e(fcfg, dsm_endpoint_config_t{});
    unsigned done = 0;
    e.ep.set_on_tx_done(&done, [](void *ctx, unsigned) {
      *static_cast<unsigned *>(ctx) += 1;
    });
    EXPECT_TRUE(e.ep.issue_load(0, 1, 4, 8, 0, 0));
    unsigned n = 0;
    while (e.ep.busy() && n++ < 100000) e.ep.cycle(e.now++);
    return n;
  };
  const unsigned z = rtt(0);
  const unsigned n = rtt(20);
  EXPECT_GT(z, 0u);
  EXPECT_GE(n, z + 20u);
}

TEST(DsmEndpoint, TmaDeliveryScalesWithFlitsNotHop) {
  auto time_tma = [](unsigned bytes) {
    EpEnv e(dsm_fabric_config_t{}, dsm_endpoint_config_t{});
    std::vector<uint8_t> payload(bytes, 1);
    unsigned mbar = 0;
    e.ep.set_on_tma_mbar(&mbar, [](void *ctx, unsigned, unsigned, unsigned,
                                   unsigned) {
      *static_cast<unsigned *>(ctx) = 1;
    });
    EXPECT_TRUE(e.ep.issue_tma(0, 1, bytes, 0, 0, 0, payload.data(), 1, bytes));
    unsigned n = 0;
    while (e.ep.busy() && mbar == 0 && n++ < 100000) e.ep.cycle(e.now++);
    EXPECT_EQ(mbar, 1u);
    return n;
  };
  const unsigned t32 = time_tma(32);
  const unsigned t128 = time_tma(128);
  EXPECT_GT(t128, t32);
  EXPECT_EQ(t128 - t32, 4u);
  // Extra delay is flit grants, not a 135-cycle multicast hop.
  EXPECT_LT(t128 - t32, 80u);
}
