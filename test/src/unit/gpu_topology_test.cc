// Unit tests for shipped gpu_topology_t. Links src/gpgpu-sim/gpu_topology.cc
// and src/option_parser.cc — no local reimplementation of the map.

#include <gtest/gtest.h>

#include <string>

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

TEST(GpuTopology, ShaderIcntNodeIsGpcNotSm) {
  gpu_topology_t topo;
  topo.build(4, 4, 3);
  EXPECT_EQ(topo.global_sm_node_id(0), 0u);
  EXPECT_EQ(topo.global_sm_node_id(3), 0u);
  EXPECT_EQ(topo.global_sm_node_id(4), 1u);
  EXPECT_EQ(topo.global_sm_node_id(15), 3u);
  EXPECT_EQ(topo.global_l2_node_id(0), 4u);
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
