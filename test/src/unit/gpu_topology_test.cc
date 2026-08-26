// Unit tests for shipped gpu_topology_t. Links src/gpgpu-sim/gpu_topology.cc
// and src/option_parser.cc — no local reimplementation of the map.

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <unistd.h>

#include "../../../src/gpgpu-sim/gpu_topology.h"
#include "../../../src/option_parser.h"

TEST(GpuTopology, RoundTripReducedGpcFourSmsOneCpc) {
  gpu_topology_t topo;
  topo.build(/*num_gpcs=*/1, /*num_sms_per_gpc=*/4, /*cpcs_per_gpc=*/1);
  EXPECT_EQ(topo.num_gpcs(), 1u);
  EXPECT_EQ(topo.num_sms(), 4u);
  EXPECT_EQ(topo.num_sms_in_gpc(0), 4u);
  EXPECT_EQ(topo.num_cpc_slots_in_gpc(0), 6u);
  EXPECT_EQ(topo.cpcs_per_gpc(), 1u);

  for (unsigned sm = 0; sm < 4; sm++) {
    const sm_location_t loc = topo.locate_sm(sm);
    EXPECT_EQ(loc.sm_id, sm);
    EXPECT_EQ(loc.gpc_id, 0u);
    EXPECT_EQ(loc.local_sm_id, sm);
    EXPECT_EQ(loc.cpc_id, 0u);
    EXPECT_EQ(loc.cpc_slot, sm);
    EXPECT_EQ(topo.sm_id_at(loc.gpc_id, loc.local_sm_id), sm);
    EXPECT_EQ(topo.gpc_id_of_sm(sm), loc.gpc_id);
    EXPECT_EQ(topo.local_sm_of_sm(sm), loc.local_sm_id);
  }
  EXPECT_EQ(topo.sm_id_at(0, 4), 4u);
}

TEST(GpuTopology, PgdSlotsHaveNoEnabledSm) {
  gpu_topology_t topo;
  topo.build(1, 4, 1);
  EXPECT_TRUE(topo.slot_is_enabled(0, 0, 0));
  EXPECT_TRUE(topo.slot_is_enabled(0, 0, 3));
  EXPECT_FALSE(topo.slot_is_enabled(0, 0, 4));
  EXPECT_FALSE(topo.slot_is_enabled(0, 0, 5));
  // Construction instantiates only enabled local SMs (num_sms_in_gpc).
  EXPECT_EQ(topo.num_sms_in_gpc(0), 4u);
}

TEST(GpuTopology, DefaultThreeCpcsLeavesExtraSlotsPgd) {
  gpu_topology_t topo;
  topo.build(2, 4, 3);
  EXPECT_EQ(topo.num_cpc_slots_in_gpc(0), 18u);
  EXPECT_EQ(topo.num_sms(), 8u);
  EXPECT_TRUE(topo.slot_is_enabled(0, 0, 3));
  EXPECT_FALSE(topo.slot_is_enabled(0, 0, 4));
  EXPECT_FALSE(topo.slot_is_enabled(0, 1, 0));
  EXPECT_FALSE(topo.slot_is_enabled(1, 2, 5));
  EXPECT_EQ(topo.sm_id_at(1, 0), 4u);
  EXPECT_EQ(topo.locate_sm(7).gpc_id, 1u);
  EXPECT_EQ(topo.locate_sm(7).local_sm_id, 3u);
}

TEST(GpuTopology, ShaderIcntNodeIsEnabledSm) {
  gpu_topology_t topo;
  topo.build(4, 4, 3);
  EXPECT_EQ(topo.num_sms(), 16u);
  std::set<unsigned> sm_nodes;
  for (unsigned sm = 0; sm < 16; sm++) {
    EXPECT_EQ(topo.global_sm_node_id(sm), sm);
    sm_nodes.insert(topo.global_sm_node_id(sm));
    EXPECT_NE(topo.global_l2_node_id(0), topo.global_sm_node_id(sm));
  }
  EXPECT_EQ(sm_nodes.size(), 16u);
  EXPECT_EQ(topo.global_l2_node_id(0), topo.num_sms());
  EXPECT_EQ(topo.gpc_id_of_sm(3), 0u);
  EXPECT_NE(topo.global_sm_node_id(3), topo.gpc_id_of_sm(3));
}

TEST(GpuTopology, ShaderIcntNodeCountIndependentOfGpcGrouping) {
  gpu_topology_t packed;
  gpu_topology_t sliced;
  packed.build(4, 4, 3);
  sliced.build(16, 1, 3);
  EXPECT_EQ(packed.num_sms(), 16u);
  EXPECT_EQ(sliced.num_sms(), 16u);
  EXPECT_EQ(packed.num_gpcs(), 4u);
  EXPECT_EQ(sliced.num_gpcs(), 16u);
  std::set<unsigned> packed_nodes;
  std::set<unsigned> sliced_nodes;
  for (unsigned sm = 0; sm < 16; sm++) {
    packed_nodes.insert(packed.global_sm_node_id(sm));
    sliced_nodes.insert(sliced.global_sm_node_id(sm));
    EXPECT_NE(packed.global_l2_node_id(0), packed.global_sm_node_id(sm));
    EXPECT_NE(sliced.global_l2_node_id(0), sliced.global_sm_node_id(sm));
  }
  EXPECT_EQ(packed_nodes.size(), sliced_nodes.size());
  EXPECT_EQ(packed_nodes.size(), 16u);
  EXPECT_EQ(packed.global_l2_node_id(0), packed.num_sms());
  EXPECT_EQ(sliced.global_l2_node_id(0), sliced.num_sms());
}

TEST(GpuTopology, GpcSmResponseFifoNoCrossSmHol) {
  gpc_sm_response_fifos_t fifos;
  fifos.init(/*n_local_sms=*/2, /*ejection_limit=*/2);
  mem_fetch *a = reinterpret_cast<mem_fetch *>(0x10);
  mem_fetch *b = reinterpret_cast<mem_fetch *>(0x20);
  mem_fetch *c = reinterpret_cast<mem_fetch *>(0x30);
  fifos.push(0, a);
  fifos.push(0, b);
  EXPECT_TRUE(fifos.full(0));
  EXPECT_FALSE(fifos.full(1));
  fifos.push(1, c);
  EXPECT_EQ(fifos.size(1), 1u);
  EXPECT_EQ(fifos.front(1), c);
  fifos.pop(1);
  EXPECT_TRUE(fifos.empty(1));
  EXPECT_EQ(fifos.front(0), a);
  EXPECT_TRUE(fifos.full(0));
}

static std::string read_file_if_exists(const std::string &path) {
  std::ifstream in(path.c_str());
  if (!in) return std::string();
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

TEST(GpuTopology, OrdinaryCtaIssueVisitsEachMemberSmOnce) {
  for (unsigned n = 2; n <= 4; n++) {
    for (unsigned start = 0; start < n; start++) {
      unsigned seen[4] = {};
      for (unsigned i = 0; i < n; i++) {
        unsigned core = gpc_cta_issue_visit(i, start, n);
        ASSERT_LT(core, n);
        seen[core]++;
      }
      for (unsigned sm = 0; sm < n; sm++) {
        EXPECT_EQ(seen[sm], 1u) << "n=" << n << " start=" << start
                                << " sm=" << sm;
      }
    }
  }

  std::string src;
  char cwd[4096];
  if (getcwd(cwd, sizeof(cwd))) {
    std::string dir(cwd);
    for (int i = 0; i < 8 && src.empty() && !dir.empty(); i++) {
      src = read_file_if_exists(dir + "/src/gpgpu-sim/shader.cc");
      auto slash = dir.find_last_of('/');
      if (slash == std::string::npos || slash == 0) break;
      dir.resize(slash);
    }
  }
  ASSERT_FALSE(src.empty());
  auto ord = src.find("unsigned simt_core_cluster::issue_block2core()");
  auto nextfn = src.find("\nvoid simt_core_cluster::cache_flush", ord);
  ASSERT_NE(ord, std::string::npos);
  ASSERT_NE(nextfn, std::string::npos);
  const std::string body = src.substr(ord, nextfn - ord);
  EXPECT_NE(body.find("gpc_cta_issue_visit(i, rr_start, n)"), std::string::npos);
  EXPECT_NE(body.find("const unsigned rr_start = m_cta_issue_next_core"),
            std::string::npos);
}

TEST(GpuTopology, ResolveAliasesAgree) {
  unsigned gpcs = 0, sms = 0;
  char err[128];
  ASSERT_TRUE(gpc_resolve_topology_aliases(true, 4, true, 4, true, 2, true, 2,
                                           &gpcs, &sms, err, sizeof(err)));
  EXPECT_EQ(gpcs, 4u);
  EXPECT_EQ(sms, 2u);
}

TEST(GpuTopology, ResolveAliasesNewOnly) {
  unsigned gpcs = 0, sms = 0;
  char err[128];
  ASSERT_TRUE(gpc_resolve_topology_aliases(false, 10, true, 4, false, 3, true, 2,
                                           &gpcs, &sms, err, sizeof(err)));
  EXPECT_EQ(gpcs, 4u);
  EXPECT_EQ(sms, 2u);
}

TEST(GpuTopology, ResolveAliasesConflict) {
  unsigned gpcs = 0, sms = 0;
  char err[256];
  err[0] = 0;
  ASSERT_FALSE(gpc_resolve_topology_aliases(true, 4, true, 2, true, 4, true, 4,
                                            &gpcs, &sms, err, sizeof(err)));
  EXPECT_TRUE(std::string(err).find("-gpgpu_n_clusters") != std::string::npos);
  EXPECT_TRUE(std::string(err).find("-gpgpu_num_gpcs") != std::string::npos);
}

static void parse_and_apply(const char *args, unsigned *n_clusters,
                            unsigned *n_cores, unsigned *gpcs, unsigned *sms) {
  option_parser_t opp = option_parser_create();
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, n_clusters,
                         "gpcs", "10");
  option_parser_register(opp, "-gpgpu_num_gpcs", OPT_UINT32, gpcs, "gpcs", "0");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, n_cores,
                         "sms", "3");
  option_parser_register(opp, "-gpgpu_num_sms_per_gpc", OPT_UINT32, sms, "sms",
                         "0");
  option_parser_delimited_string(opp, args, " ");
  gpc_apply_topology_aliases(opp, n_clusters, n_cores, gpcs, sms);
  option_parser_destroy(opp);
}

TEST(GpuTopology, OptionParserOldKnobsOnly) {
  unsigned n_clusters = 10, n_cores = 3, gpcs = 0, sms = 0;
  parse_and_apply("-gpgpu_n_clusters 4 -gpgpu_n_cores_per_cluster 4",
                  &n_clusters, &n_cores, &gpcs, &sms);
  EXPECT_EQ(n_clusters, 4u);
  EXPECT_EQ(n_cores, 4u);
  EXPECT_EQ(gpcs, 4u);
  EXPECT_EQ(sms, 4u);
}

static void apply_conflicting_gpc_counts() {
  unsigned n_clusters = 10, n_cores = 3, gpcs = 0, sms = 0;
  parse_and_apply(
      "-gpgpu_n_clusters 4 -gpgpu_num_gpcs 2 -gpgpu_n_cores_per_cluster 4",
      &n_clusters, &n_cores, &gpcs, &sms);
}

TEST(GpuTopology, OptionParserConflictAborts) {
  ASSERT_DEATH(apply_conflicting_gpc_counts(), "disagree");
}
