// Unit tests for shipped dsm_fabric_t. Links src/gpgpu-sim/dsm_fabric.cc
// — no local reimplementation of the grant engine.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../../../src/gpgpu-sim/dsm_fabric.h"
#include "../../../src/gpgpu-sim/gpu_topology.h"

namespace {

std::unique_ptr<dsm_packet_t> mk_pkt(unsigned src, unsigned dst,
                                     dsm_packet_class_t cls, unsigned bytes,
                                     uint64_t addr = 0) {
  auto p = std::make_unique<dsm_packet_t>();
  p->network_src_sm_id = src;
  p->network_dst_sm_id = dst;
  p->packet_class = cls;
  p->vc = dsm_vc_of(cls);
  p->payload_bytes = bytes;
  p->payload_address = addr;
  return p;
}

void drain(dsm_fabric_t &f) {
  for (unsigned d = 0; d < f.num_sms(); d++) {
    while (f.top(d, dsm_vc_t::request)) f.pop(d, dsm_vc_t::request);
    while (f.top(d, dsm_vc_t::response)) f.pop(d, dsm_vc_t::response);
  }
}

void fill_src(dsm_fabric_t &f, unsigned src, unsigned dst,
              dsm_packet_class_t cls, unsigned bytes, uint64_t *addr) {
  const unsigned flits =
      dsm_payload_flits(cls, bytes, f.flit_payload_bytes());
  while (f.can_inject(src, dsm_vc_of(cls), dst, flits)) {
    f.inject(mk_pkt(src, dst, cls, bytes, *addr));
    *addr += f.flit_payload_bytes();
  }
}

dsm_fabric_t make_fabric(unsigned n_sms, unsigned cpcs = 1,
                         dsm_fabric_config_t cfg = {}) {
  gpu_topology_t topo;
  topo.build(/*num_gpcs=*/1, n_sms, cpcs);
  return dsm_fabric_t(topo, 0, cfg);
}

struct RateSample {
  double bytes_per_cycle;
  unsigned long long flits;
};

RateSample measure_payload(dsm_fabric_t &f, unsigned warmup, unsigned window,
                           const std::vector<std::pair<unsigned, unsigned>>
                               &pairs,
                           dsm_packet_class_t cls = dsm_packet_class_t::write_data) {
  uint64_t addr = 1;
  unsigned long long now = 0;
  const unsigned bytes = 128;
  auto step = [&]() {
    for (auto pr : pairs) fill_src(f, pr.first, pr.second, cls, bytes, &addr);
    f.cycle(now++);
    drain(f);
  };
  for (unsigned i = 0; i < warmup; i++) step();
  const auto s0 = f.stats();
  for (unsigned i = 0; i < window; i++) step();
  const auto s1 = f.stats();
  RateSample r;
  r.bytes_per_cycle =
      (double)(s1.payload_bytes_granted - s0.payload_bytes_granted) /
      (double)window;
  r.flits = s1.flits_granted - s0.flits_granted;
  return r;
}

std::set<unsigned> eligible_set(const dsm_fabric_t &f, unsigned long long cyc) {
  std::set<unsigned> s;
  for (unsigned sm = 0; sm < f.num_sms(); sm++) {
    if (f.sm_eligible(sm, cyc)) s.insert(sm);
  }
  return s;
}

}  // namespace

TEST(DsmFabric, OneSmIdleNeighborsCap) {
  dsm_fabric_t f = make_fabric(6, 1);
  const unsigned warmup = 30;
  const unsigned window = 300;  // multiple of shaper period 3
  const RateSample r =
      measure_payload(f, warmup, window, {{0, 1}});
  const double expect = (2.0 / 3.0) * (double)f.flit_payload_bytes();
  EXPECT_NEAR(r.bytes_per_cycle, expect, 0.05);
}

TEST(DsmFabric, IdleNeighborDoesNotRaiseRate) {
  dsm_fabric_t f = make_fabric(6, 1);
  const RateSample r = measure_payload(f, 30, 300, {{0, 1}});
  const double expect = (2.0 / 3.0) * (double)f.flit_payload_bytes();
  EXPECT_NEAR(r.bytes_per_cycle, expect, 0.05);
  EXPECT_LE(r.bytes_per_cycle, expect + 0.05);
}

TEST(DsmFabric, SameDirRequestResponseShareCap) {
  dsm_fabric_t f = make_fabric(6, 1);
  uint64_t addr = 1;
  unsigned long long now = 0;
  const unsigned warmup = 30, window = 300;
  auto step = [&]() {
    fill_src(f, 1, 0, dsm_packet_class_t::write_data, 128, &addr);
    fill_src(f, 1, 0, dsm_packet_class_t::read_data, 128, &addr);
    f.cycle(now++);
    drain(f);
  };
  for (unsigned i = 0; i < warmup; i++) step();
  const auto s0 = f.stats();
  for (unsigned i = 0; i < window; i++) step();
  const auto s1 = f.stats();
  const double rate =
      (double)(s1.payload_bytes_granted - s0.payload_bytes_granted) / window;
  const double expect = (2.0 / 3.0) * (double)f.flit_payload_bytes();
  EXPECT_NEAR(rate, expect, 0.05);
  EXPECT_GT(s1.flits_request - s0.flits_request, 0u);
  EXPECT_GT(s1.flits_response - s0.flits_response, 0u);
}

TEST(DsmFabric, OppositeDirsExceedOneDirCap) {
  dsm_fabric_t f = make_fabric(6, 1);
  const RateSample r = measure_payload(f, 30, 300, {{0, 1}, {1, 0}});
  const double one_dir = (2.0 / 3.0) * (double)f.flit_payload_bytes();
  EXPECT_GT(r.bytes_per_cycle, one_dir + 1.0);
}

TEST(DsmFabric, ReadCommandIsOneReverseFlit) {
  dsm_fabric_t f = make_fabric(4, 1);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::read_command, 0, 0));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), 1u);
  unsigned long long now = 0;
  while (!f.top(1, dsm_vc_t::request) && now < 16) f.cycle(now++);
  ASSERT_NE(f.top(1, dsm_vc_t::request), nullptr);
  auto p = f.pop(1, dsm_vc_t::request);
  EXPECT_EQ(p->packet_class, dsm_packet_class_t::read_command);
  EXPECT_EQ(p->total_flits, 1u);
  EXPECT_EQ(p->remaining_flits, 0u);
  EXPECT_EQ(p->vc, dsm_vc_t::request);
  EXPECT_EQ(f.stats().flits_granted, 1u);
  EXPECT_EQ(f.stats().payload_bytes_granted, 0u);
}

TEST(DsmFabric, RequestFullDoesNotConsumeResponseCredits) {
  dsm_fabric_config_t cfg;
  cfg.request_vc_flits = 4;
  cfg.response_vc_flits = 8;
  cfg.ejection_vc_flits = 8;
  dsm_fabric_t f = make_fabric(4, 1, cfg);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 128, 0));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), 4u);
  EXPECT_FALSE(f.can_inject(0, dsm_vc_t::request, 1, 1));
  EXPECT_EQ(f.credit_remaining(1, dsm_vc_t::response), 8u);
  EXPECT_EQ(f.credit_remaining(1, dsm_vc_t::request), 8u);
  EXPECT_TRUE(f.can_inject(0, dsm_vc_t::response, 1, 1));
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::read_data, 32, 32));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::response, 1), 1u);
  EXPECT_EQ(f.credit_remaining(1, dsm_vc_t::response), 8u);
  EXPECT_FALSE(f.can_inject(0, dsm_vc_t::request, 1, 1));
}

TEST(DsmFabric, DestVoqBlockedDoesNotHolSibling) {
  dsm_fabric_config_t cfg;
  cfg.request_vc_flits = 2;
  dsm_fabric_t f = make_fabric(4, 1, cfg);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 64, 0));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), 2u);
  EXPECT_FALSE(f.can_inject(0, dsm_vc_t::request, 1, 1));
  EXPECT_TRUE(f.can_inject(0, dsm_vc_t::request, 2, 1));
  f.inject(mk_pkt(0, 2, dsm_packet_class_t::write_data, 32, 32));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 2), 1u);
  unsigned long long now = 0;
  bool saw_dst2 = false;
  while (now < 24 && !saw_dst2) {
    f.cycle(now++);
    if (f.top(2, dsm_vc_t::request)) {
      auto p = f.pop(2, dsm_vc_t::request);
      EXPECT_EQ(p->network_dst_sm_id, 2u);
      saw_dst2 = true;
    }
    if (f.top(1, dsm_vc_t::request)) f.pop(1, dsm_vc_t::request);
  }
  EXPECT_TRUE(saw_dst2);
}

TEST(DsmFabric, Data128BNeedsFourGrants) {
  dsm_fabric_t f = make_fabric(4, 1);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 128, 0));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), 4u);
  unsigned long long now = 0;
  unsigned long long last_grants = 0;
  while (!f.top(1, dsm_vc_t::request) && now < 32) {
    f.cycle(now++);
    const unsigned long long g = f.stats().flits_granted;
    if (g < 4) EXPECT_EQ(f.top(1, dsm_vc_t::request), nullptr);
    last_grants = g;
  }
  ASSERT_NE(f.top(1, dsm_vc_t::request), nullptr);
  EXPECT_EQ(last_grants, 4u);
  auto p = f.pop(1, dsm_vc_t::request);
  EXPECT_EQ(p->total_flits, 4u);
  EXPECT_EQ(p->payload_bytes, 128u);
}

TEST(DsmFabric, PayloadLargerThanVcDepthCompletes) {
  dsm_fabric_config_t cfg;
  cfg.request_vc_flits = 4;
  cfg.ejection_vc_flits = 4;
  dsm_fabric_t f = make_fabric(4, 1, cfg);
  const unsigned flits =
      dsm_payload_flits(dsm_packet_class_t::tma_data, 288,
                        f.flit_payload_bytes());
  ASSERT_EQ(flits, 9u);
  ASSERT_TRUE(f.can_inject(0, dsm_vc_t::request, 1, flits));
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::tma_data, 288, 0));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), 4u);

  unsigned long long now = 0;
  while (!f.top(1, dsm_vc_t::request) && now < 64) f.cycle(now++);
  ASSERT_NE(f.top(1, dsm_vc_t::request), nullptr);
  auto pkt = f.pop(1, dsm_vc_t::request);
  EXPECT_EQ(pkt->total_flits, 9u);
  EXPECT_EQ(f.credit_remaining(1, dsm_vc_t::request), 4u);
}

TEST(DsmFabric, MulticastGroupBranchesOnePhysicalStream) {
  dsm_fabric_t f = make_fabric(4, 1);
  for (unsigned dst = 1; dst < 4; dst++) {
    auto packet = mk_pkt(0, dst, dsm_packet_class_t::tma_data, 128, 0);
    packet->multicast_group = 7;
    f.inject(std::move(packet));
  }

  unsigned long long now = 0;
  while ((!f.top(1, dsm_vc_t::request) ||
          !f.top(2, dsm_vc_t::request) ||
          !f.top(3, dsm_vc_t::request)) &&
         now < 64)
    f.cycle(now++);

  ASSERT_NE(f.top(1, dsm_vc_t::request), nullptr);
  ASSERT_NE(f.top(2, dsm_vc_t::request), nullptr);
  ASSERT_NE(f.top(3, dsm_vc_t::request), nullptr);
  EXPECT_EQ(f.stats().flits_granted, 4u);
  const unsigned long long tail_cycle =
      f.top(1, dsm_vc_t::request)->tail_arrival_cycle;
  for (unsigned dst = 1; dst < 4; dst++) {
    auto packet = f.pop(dst, dsm_vc_t::request);
    EXPECT_EQ(packet->tail_arrival_cycle, tail_cycle);
    EXPECT_EQ(f.credit_remaining(dst, dsm_vc_t::request), 64u);
  }
}

TEST(DsmFabric, CanInjectFalseRefusesInject) {
  dsm_fabric_config_t cfg;
  cfg.request_vc_flits = 4;
  dsm_fabric_t f = make_fabric(4, 1, cfg);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 128, 0));
  EXPECT_FALSE(f.can_inject(0, dsm_vc_t::request, 1, 1));
  const unsigned occ = f.occupancy_flits(0, dsm_vc_t::request, 1);
  EXPECT_EQ(occ, 4u);
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 32, 99));
  EXPECT_EQ(f.occupancy_flits(0, dsm_vc_t::request, 1), occ);
  EXPECT_EQ(f.stats().stall_inject, 1u);
  EXPECT_EQ(f.stats().packets_injected, 1u);
}

TEST(DsmFabric, GxPlanesOneReducesRoutesVsTwo) {
  dsm_fabric_config_t cfg1;
  cfg1.gx_planes = 1;
  dsm_fabric_config_t cfg2;
  cfg2.gx_planes = 2;
  dsm_fabric_t f1 = make_fabric(6, 1, cfg1);
  dsm_fabric_t f2 = make_fabric(6, 1, cfg2);
  EXPECT_EQ(f1.num_routes(), 1u * f1.lanes_per_cpc());
  EXPECT_EQ(f2.num_routes(), 2u * f2.lanes_per_cpc());
  EXPECT_LT(f1.num_routes(), f2.num_routes());

  // Same address, SM0 and SM4, dest 1: hash (addr/32+src+dst*3) collides
  // on 4 routes (both → 3) and splits on 8 routes (3 vs 7).
  const dsm_route_t r0 = f1.hash_route(0, 0, 1, 1);
  const dsm_route_t r4 = f1.hash_route(0, 4, 1, 1);
  EXPECT_EQ(r0.gx_plane * f1.lanes_per_cpc() + r0.lane,
            r4.gx_plane * f1.lanes_per_cpc() + r4.lane);
  const dsm_route_t s0 = f2.hash_route(0, 0, 1, 1);
  const dsm_route_t s4 = f2.hash_route(0, 4, 1, 1);
  EXPECT_NE(s0.gx_plane * f2.lanes_per_cpc() + s0.lane,
            s4.gx_plane * f2.lanes_per_cpc() + s4.lane);

  auto fill_camp = [](dsm_fabric_t &f, unsigned src) {
    const unsigned flits =
        dsm_payload_flits(dsm_packet_class_t::write_data, 128,
                          f.flit_payload_bytes());
    while (f.can_inject(src, dsm_vc_t::request, 1, flits)) {
      auto p = mk_pkt(src, 1, dsm_packet_class_t::write_data, 128, 0);
      p->packet_id = 1;
      f.inject(std::move(p));
    }
  };
  auto pump = [&](dsm_fabric_t &f) {
    unsigned long long now = 0;
    const unsigned warmup = 30, window = 300;
    auto step = [&]() {
      fill_camp(f, 0);
      fill_camp(f, 4);
      f.cycle(now++);
      drain(f);
    };
    for (unsigned i = 0; i < warmup; i++) step();
    const auto a = f.stats();
    for (unsigned i = 0; i < window; i++) step();
    const auto b = f.stats();
    return b.flits_granted - a.flits_granted;
  };
  EXPECT_LT(pump(f1), pump(f2));
}

TEST(DsmFabric, PgdSlotNeverSends) {
  gpu_topology_t topo;
  topo.build(/*num_gpcs=*/1, /*num_sms_per_gpc=*/16, /*cpcs_per_gpc=*/3);
  EXPECT_FALSE(topo.slot_is_enabled(0, 2, 4));
  EXPECT_FALSE(topo.slot_is_enabled(0, 2, 5));
  dsm_fabric_config_t cfg;
  cfg.shaper = "fixed_tdm";
  dsm_fabric_t f(topo, 0, cfg);
  EXPECT_EQ(f.num_sms(), 16u);
  EXPECT_FALSE(f.can_inject(16, dsm_vc_t::request, 1));
  EXPECT_FALSE(f.sm_eligible(16, 0));
  EXPECT_FALSE(f.sm_eligible(17, 1));
  // Phase 1 slots {2,3,4,5}: CPC2 locals 14,15 live, 16,17 PG'd.
  std::set<unsigned> e1 = eligible_set(f, 1);
  EXPECT_TRUE(e1.count(14));
  EXPECT_TRUE(e1.count(15));
  EXPECT_FALSE(e1.count(16));
  EXPECT_FALSE(e1.count(17));
  unsigned cpc2 = 0;
  for (unsigned sm : e1) {
    if (topo.locate_sm(sm).cpc_id == 2) cpc2++;
  }
  EXPECT_EQ(cpc2, 2u);

  auto bad = mk_pkt(16, 0, dsm_packet_class_t::write_data, 32, 0);
  f.inject(std::move(bad));
  EXPECT_EQ(f.stats().packets_injected, 0u);
}

TEST(DsmFabric, SkipModAndFixedTdmCoEligibleDiffer) {
  dsm_fabric_config_t skip;
  skip.shaper = "skip_mod";
  skip.shaper_index = "cpc_slot";
  dsm_fabric_config_t tdm;
  tdm.shaper = "fixed_tdm";
  dsm_fabric_t f_skip = make_fabric(6, 1, skip);
  dsm_fabric_t f_tdm = make_fabric(6, 1, tdm);
  bool differ = false;
  for (unsigned c = 0; c < 3; c++) {
    auto a = eligible_set(f_skip, c);
    auto b = eligible_set(f_tdm, c);
    EXPECT_EQ(a.size(), 4u);
    EXPECT_EQ(b.size(), 4u);
    if (a != b) differ = true;
  }
  EXPECT_TRUE(differ);
  const RateSample rs = measure_payload(f_skip, 30, 300, {{0, 1}});
  const RateSample rt = measure_payload(f_tdm, 30, 300, {{0, 1}});
  const double expect =
      (2.0 / 3.0) * (double)f_skip.flit_payload_bytes();
  EXPECT_NEAR(rs.bytes_per_cycle, expect, 0.05);
  EXPECT_NEAR(rt.bytes_per_cycle, expect, 0.05);
}

TEST(DsmFabric, HardRateCapAveragesTwoThirds) {
  dsm_fabric_config_t cfg;
  cfg.shaper = "hard_rate_cap";
  dsm_fabric_t f = make_fabric(6, 1, cfg);
  const RateSample r = measure_payload(f, 30, 300, {{0, 1}});
  const double expect = (2.0 / 3.0) * (double)f.flit_payload_bytes();
  EXPECT_NEAR(r.bytes_per_cycle, expect, 0.05);
}

TEST(DsmFabric, TwoFabricsIsolated) {
  gpu_topology_t topo;
  topo.build(2, 4, 1);
  dsm_fabric_config_t cfg;
  dsm_fabric_t a(topo, 0, cfg);
  dsm_fabric_t b(topo, 1, cfg);
  a.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 128, 0));
  EXPECT_TRUE(a.busy());
  EXPECT_FALSE(b.busy());
  EXPECT_EQ(b.occupancy_flits(0, dsm_vc_t::request, 1), 0u);
  EXPECT_EQ(b.stats().packets_injected, 0u);
}

TEST(DsmFabric, DisplayStateShowsUsedAndWasted) {
  dsm_fabric_t f = make_fabric(6, 1);
  measure_payload(f, 12, 36, {{0, 1}});
  char *buf = nullptr;
  size_t sz = 0;
  FILE *fp = open_memstream(&buf, &sz);
  ASSERT_NE(fp, nullptr);
  f.display_state(fp);
  fclose(fp);
  std::string s(buf, sz);
  free(buf);
  EXPECT_NE(s.find("eligibility used="), std::string::npos);
  EXPECT_NE(s.find("wasted="), std::string::npos);
  EXPECT_GT(f.stats().eligibility_used, 0u);
  EXPECT_GT(f.stats().eligibility_wasted, 0u);
  printf("%s", s.c_str());
}

unsigned long long inject_until_visible(dsm_fabric_t &f, unsigned bytes) {
  f.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, bytes, 0));
  unsigned long long now = 0;
  while (!f.top(1, dsm_vc_t::request) && now < 256) f.cycle(now++);
  return now;
}

TEST(DsmFabric, ResidualFloorDelaysTailNotOccupancy) {
  dsm_fabric_config_t z;
  z.base_latency_cycles = 0;
  dsm_fabric_t f0 = make_fabric(4, 1, z);
  f0.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 32, 0));
  EXPECT_EQ(f0.occupancy_flits(0, dsm_vc_t::request, 1), 1u);
  unsigned long long now = 0;
  while (!f0.top(1, dsm_vc_t::request) && now < 64) f0.cycle(now++);
  ASSERT_NE(f0.top(1, dsm_vc_t::request), nullptr);
  const unsigned long long t0 = now;
  EXPECT_EQ(f0.stats().flits_granted, 1u);

  dsm_fabric_config_t ncfg;
  ncfg.base_latency_cycles = 20;
  dsm_fabric_t f1 = make_fabric(4, 1, ncfg);
  f1.inject(mk_pkt(0, 1, dsm_packet_class_t::write_data, 32, 1));
  EXPECT_EQ(f1.occupancy_flits(0, dsm_vc_t::request, 1), 1u);
  now = 0;
  while (!f1.top(1, dsm_vc_t::request) && now < 128) f1.cycle(now++);
  ASSERT_NE(f1.top(1, dsm_vc_t::request), nullptr);
  // Floor is max(tail, injected+20). Inject is cycle 0, so visible at 20
  // (loop leaves now one past the granting cycle). Grants still match.
  EXPECT_GT(now, t0);
  EXPECT_GE(now, 20u);
  EXPECT_EQ(f1.stats().flits_granted, f0.stats().flits_granted);
  EXPECT_EQ(f1.occupancy_flits(0, dsm_vc_t::request, 1), 0u);
}

TEST(DsmFabric, ResidualIsFloorNotAddedToBulk) {
  dsm_fabric_config_t z;
  z.base_latency_cycles = 0;
  dsm_fabric_t f0 = make_fabric(4, 1, z);
  const unsigned long long t0 = inject_until_visible(f0, 128);
  ASSERT_NE(f0.top(1, dsm_vc_t::request), nullptr);
  EXPECT_GE(f0.stats().flits_granted, 4u);

  dsm_fabric_config_t ncfg;
  ncfg.base_latency_cycles = 2;
  dsm_fabric_t f1 = make_fabric(4, 1, ncfg);
  const unsigned long long t1 = inject_until_visible(f1, 128);
  ASSERT_NE(f1.top(1, dsm_vc_t::request), nullptr);
  EXPECT_EQ(f1.stats().flits_granted, f0.stats().flits_granted);
  EXPECT_LE(t1, t0 + 1);
}
