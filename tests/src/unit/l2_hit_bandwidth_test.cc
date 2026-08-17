#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gpgpu-sim/addrdec.h"
#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/l2cache.h"
#include "gpgpu-sim/mem_fetch.h"

namespace {

using ConfigEntries = std::vector<std::pair<std::string, std::string>>;

std::filesystem::path FindRepositoryRoot() {
  std::filesystem::path current = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::is_regular_file(
            current / "configs/SM100_B200/gpgpusim.config"))
      return current;
    const std::filesystem::path parent = current.parent_path();
    if (parent == current) return {};
    current = parent;
  }
}

ConfigEntries ReadConfig(const std::filesystem::path &path) {
  std::ifstream input(path);
  EXPECT_TRUE(input.is_open()) << "cannot open " << path;
  ConfigEntries entries;
  std::string line;
  while (std::getline(input, line)) {
    const std::string::size_type comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream fields(line);
    std::string option;
    std::string value;
    if (!(fields >> option >> value) || option.front() != '-') continue;
    entries.emplace_back(std::move(option), std::move(value));
  }
  return entries;
}

std::string ConfigValue(const ConfigEntries &entries,
                        const std::string &option) {
  std::string value;
  unsigned matches = 0;
  for (const auto &entry : entries) {
    if (entry.first != option) continue;
    value = entry.second;
    ++matches;
  }
  EXPECT_EQ(matches, 1u) << "expected one " << option;
  return value;
}

unsigned long long ConfigUnsigned(const ConfigEntries &entries,
                                  const std::string &option) {
  std::istringstream input(ConfigValue(entries, option));
  unsigned long long value = 0;
  const bool parsed = static_cast<bool>(input >> value);
  input >> std::ws;
  EXPECT_TRUE(parsed && input.eof()) << "invalid unsigned value for " << option;
  return value;
}

mem_fetch *FakeRequest(unsigned long &storage) {
  return reinterpret_cast<mem_fetch *>(&storage);
}

void AcceptHit(l2_multi_issue_ports &ports) {
  ASSERT_TRUE(ports.can_accept_lookup(1));
  ASSERT_TRUE(ports.data_port_has_capacity());
  ports.accept_lookup(1);
  ASSERT_EQ(ports.accept_data(1, L2_MULTI_ISSUE_HIT_DATA), 1u);
}

TEST(B200L2ConfigTest, CheckedInConfigDrivesPortModelAndValidTopology) {
  const std::filesystem::path repository_root = FindRepositoryRoot();
  ASSERT_FALSE(repository_root.empty())
      << "cannot locate configs/SM100_B200/gpgpusim.config from "
      << std::filesystem::current_path();
  const ConfigEntries config = ReadConfig(
      repository_root / "configs/SM100_B200/gpgpusim.config");
  ASSERT_FALSE(config.empty());

  const unsigned long long memory_channels =
      ConfigUnsigned(config, "-gpgpu_n_mem");
  const unsigned long long slices_per_channel =
      ConfigUnsigned(config, "-gpgpu_n_sub_partition_per_mchannel");
  ASSERT_GT(memory_channels, 0u);
  ASSERT_GT(slices_per_channel, 0u);
  const unsigned long long l2_instances =
      memory_channels * slices_per_channel;
  ASSERT_GT(l2_instances, 0u);

  char cache_type = '\0';
  unsigned long long sets_per_slice = 0;
  unsigned long long line_bytes = 0;
  unsigned long long ways = 0;
  const std::string l2_config = ConfigValue(config, "-gpgpu_cache:dl2");
  ASSERT_EQ(std::sscanf(l2_config.c_str(), "%c:%llu:%llu:%llu", &cache_type,
                        &sets_per_slice, &line_bytes, &ways),
            4);
  EXPECT_EQ(cache_type, 'S');
  ASSERT_GT(sets_per_slice, 0u);
  ASSERT_GT(line_bytes, 0u);
  ASSERT_GT(ways, 0u);
  EXPECT_EQ(line_bytes % SECTOR_SIZE, 0u);
  EXPECT_GT(l2_instances * sets_per_slice * line_bytes * ways, 0u);

  const unsigned long long port_model =
      ConfigUnsigned(config, "-gpgpu_l2_multi_issue_port_model");
  ASSERT_TRUE(l2_multi_issue_port_model_enabled(port_model));
  const unsigned long long lookup_width =
      ConfigUnsigned(config, "-gpgpu_l2_lookup_sectors_per_cycle");
  const unsigned long long data_width =
      ConfigUnsigned(config, "-gpgpu_l2_data_port_sectors_per_cycle");
  const unsigned long long fill_width =
      ConfigUnsigned(config, "-gpgpu_l2_fill_port_sectors_per_cycle");
  ASSERT_GT(lookup_width, 0u);
  ASSERT_GT(data_width, 0u);
  ASSERT_GT(fill_width, 0u);

  l2_multi_issue_ports ports;
  ports.configure(lookup_width, data_width, fill_width);
  ASSERT_TRUE(ports.can_accept_lookup(lookup_width));
  ports.accept_lookup(lookup_width);
  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_EQ(ports.accept_data(data_width + 1, L2_MULTI_ISSUE_HIT_DATA),
            data_width);
  EXPECT_EQ(ports.accept_fill(fill_width + 1), fill_width);
  EXPECT_EQ(ports.lookup_remaining(), 0u);
  EXPECT_EQ(ports.data_remaining(), 0u);
  EXPECT_EQ(ports.fill_remaining(), 0u);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, lookup_width);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, data_width);
  EXPECT_EQ(ports.stats().fill_port_accepted_sectors, fill_width);

  ports.begin_cycle();
  EXPECT_EQ(ports.lookup_remaining(), lookup_width);
  EXPECT_EQ(ports.data_remaining(), data_width);
  EXPECT_EQ(ports.fill_remaining(), fill_width);

  if (!l2_slice_mapping::is_power_of_two(slices_per_channel)) {
    const unsigned long long indexing =
        ConfigUnsigned(config, "-gpgpu_memory_partition_indexing");
    const unsigned long long slice_mapping =
        ConfigUnsigned(config, "-gpgpu_non_power2_l2_slice_mapping");
    EXPECT_EQ(indexing, static_cast<unsigned>(CONSECUTIVE));
    EXPECT_GE(slice_mapping,
              static_cast<unsigned>(NON_POWER2_L2_SLICE_PLAIN));
    EXPECT_LE(slice_mapping,
              static_cast<unsigned>(NON_POWER2_L2_SLICE_STABLE_ROTATION));
  }
}

TEST(B200TMAConfigTest, FullAndReducedCreditWindowsRemainIntentional) {
  const std::filesystem::path repository_root = FindRepositoryRoot();
  ASSERT_FALSE(repository_root.empty())
      << "cannot locate B200 configs from "
      << std::filesystem::current_path();

  const ConfigEntries full = ReadConfig(
      repository_root / "configs/SM100_B200/gpgpusim.config");
  const ConfigEntries reduced = ReadConfig(
      repository_root / "configs/SM100_B200_REDUCED/gpgpusim.config");
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(reduced.empty());

  constexpr unsigned long long kCalibrationStages = 12;
  const unsigned long long transaction_quota =
      ConfigUnsigned(full, "-gpgpu_tma_tx_quota");
  EXPECT_EQ(transaction_quota, 48u);
  EXPECT_EQ(ConfigUnsigned(full, "-gpgpu_tma_max_inflight"),
            kCalibrationStages * transaction_quota);
  EXPECT_EQ(ConfigUnsigned(full, "-gpgpu_tma_request_granularity"), 32u);
  EXPECT_EQ(ConfigUnsigned(full, "-gpgpu_tma_request_width"), 4u);
  EXPECT_EQ(ConfigUnsigned(full, "-gpgpu_tma_response_width"), 4u);

  // The one-SM reduced model is functional-only; zero deliberately exercises
  // the parser's unlimited-credit boundary instead of importing calibration.
  EXPECT_EQ(ConfigUnsigned(reduced, "-gpgpu_tma_max_inflight"), 0u);
}

TEST(L2PortModelSelectionTest, LegacyAndMultiIssueModesAreMutuallyExclusive) {
  EXPECT_FALSE(l2_multi_issue_port_model_enabled(0));
  EXPECT_TRUE(l2_multi_issue_port_model_enabled(1));
  EXPECT_DEATH_IF_SUPPORTED(l2_multi_issue_port_model_enabled(2), "mode <= 1");
}

TEST(L2PortModelSelectionTest,
     LegacyModeDoesNotActivateMultiIssueSectorAccounting) {
  l2_multi_issue_ports ports;
  ports.configure(3, 3, 3);

  if (l2_multi_issue_port_model_enabled(0)) {
    AcceptHit(ports);
  }

  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 0u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 0u);
  EXPECT_EQ(ports.stats().fill_port_accepted_sectors, 0u);
}

TEST(L2MultiIssuePortsTest, DataWidthsOneTwoAndThreeAcceptThatManySectors) {
  for (unsigned width = 1; width <= 3; ++width) {
    l2_multi_issue_ports ports;
    ports.configure(/*lookup_width=*/4, width, /*fill_width=*/1);
    EXPECT_EQ(ports.accept_data(/*pending_sectors=*/4, L2_MULTI_ISSUE_HIT_DATA),
              width);
    EXPECT_EQ(ports.data_remaining(), 0u);
    EXPECT_EQ(ports.stats().data_port_accepted_sectors, width);
  }
}

TEST(L2MultiIssuePortsTest, ThreeIndependentSectorHitsIssueInOneTick) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3, /*fill_width=*/1);

  AcceptHit(ports);
  AcceptHit(ports);
  AcceptHit(ports);

  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 3u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorDemandUsesTwoPlusOneAtWidthTwo) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/2, /*data_width=*/2, /*fill_width=*/1);

  AcceptHit(ports);
  AcceptHit(ports);
  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());

  ports.begin_cycle();
  AcceptHit(ports);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 3u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorOperationCompletesAtWidthThree) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/3, /*fill_width=*/1);
  l2_multi_issue_pending_operation operation;
  operation.start(3);

  EXPECT_TRUE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(operation.remaining_sectors(), 0u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorOperationUsesTwoPlusOneAtWidthTwo) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/2, /*fill_width=*/1);
  l2_multi_issue_pending_operation operation;
  operation.start(3);

  EXPECT_FALSE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(operation.remaining_sectors(), 1u);
  ports.begin_cycle();
  EXPECT_TRUE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, DataAndFillPortsProgressIndependentlyInOneTick) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/2, /*fill_width=*/3);
  l2_multi_issue_pending_operation data;
  l2_multi_issue_pending_operation fill;
  data.start(2);
  fill.start(3);

  EXPECT_TRUE(data.service_data(ports, L2_MULTI_ISSUE_HIT_DATA));
  EXPECT_TRUE(fill.service_fill(ports));
  EXPECT_EQ(ports.data_remaining(), 0u);
  EXPECT_EQ(ports.fill_remaining(), 0u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 2u);
  EXPECT_EQ(ports.stats().fill_port_accepted_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, HitAndDirtyEvictionShareTheDataWidth) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3, /*fill_width=*/1);

  AcceptHit(ports);
  l2_multi_issue_pending_operation dirty_eviction;
  dirty_eviction.start(3);
  EXPECT_FALSE(
      dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 1u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 2u);
}

TEST(L2MultiIssuePortsTest,
     FourSectorDirtyEvictionAtWidthThreeIsNotVisibleEarly) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3, /*fill_width=*/1);
  l2_multi_issue_pending_operation dirty_eviction;
  dirty_eviction.start(4);
  bool downstream_visible = false;

  if (dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION))
    downstream_visible = true;
  EXPECT_FALSE(downstream_visible);
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 1u);

  ports.begin_cycle();
  if (dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION))
    downstream_visible = true;
  EXPECT_TRUE(downstream_visible);
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 0u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 4u);
}

TEST(L2MultiIssuePortsTest, CleanMissConsumesOnlyLookupSectorWork) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3, /*fill_width=*/3);

  ASSERT_TRUE(ports.can_accept_lookup(1));
  ports.accept_lookup(1);

  EXPECT_EQ(ports.lookup_remaining(), 2u);
  EXPECT_EQ(ports.data_remaining(), 3u);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 0u);
}

TEST(L2MultiIssuePortsTest,
     ReadyForwardBypassesExhaustedDataPortAndFictionalDirtyCandidate) {
  constexpr new_addr_type kAddress = 0x2000;
  mshr_table mshrs(/*num_entries=*/1, /*max_merged=*/4);
  unsigned long first_storage = 0;
  unsigned long older_merged_storage = 0;
  unsigned long late_storage = 0;
  mem_fetch *first = FakeRequest(first_storage);
  mem_fetch *older_merged = FakeRequest(older_merged_storage);
  mem_fetch *late = FakeRequest(late_storage);

  mshrs.add(kAddress, first, /*is_atomic=*/false);
  mshrs.add(kAddress, older_merged, /*is_atomic=*/false);
  bool has_atomic = true;
  mshrs.mark_ready(kAddress, has_atomic);
  ASSERT_FALSE(has_atomic);
  ASSERT_EQ(mshrs.next_access(), first);
  ASSERT_TRUE(mshrs.probe_ready(kAddress));

  const bool ready_read_forward = l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/false, /*is_atomic=*/false, MISS);
  ASSERT_TRUE(ready_read_forward);
  EXPECT_FALSE(l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/true, /*is_atomic=*/false, MISS));
  EXPECT_FALSE(l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/false, /*is_atomic=*/true, MISS));

  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/2, /*data_width=*/1, /*fill_width=*/1);

  AcceptHit(ports);
  ASSERT_EQ(ports.data_remaining(), 0u);

  // A tag-only probe can nominate a dirty victim after the original filled
  // line has been evicted. A late ordinary read instead joins the ready MSHR,
  // so that prospective replacement must not become data-port work.
  ASSERT_FALSE(l2_multi_issue_needs_data_port(
      MISS, /*prospective_dirty_eviction_sectors=*/4, ready_read_forward));
  ASSERT_TRUE(ports.can_accept_lookup(1));
  ports.accept_lookup(1);
  mshrs.add_ready(kAddress, late);

  EXPECT_EQ(ports.lookup_remaining(), 0u);
  EXPECT_EQ(ports.data_remaining(), 0u);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 2u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 0u);
  EXPECT_EQ(ports.stats().data_port_width_stall_cycles, 0u);

  EXPECT_TRUE(l2_multi_issue_needs_data_port(
      MISS, /*prospective_dirty_eviction_sectors=*/4,
      /*ready_read_forward=*/false));

  EXPECT_EQ(mshrs.next_access(), older_merged);
  EXPECT_EQ(mshrs.next_access(), late);
  EXPECT_FALSE(mshrs.access_ready());
}

TEST(L2MultiIssuePortsTest, WidthStallsCountOncePerPortPerTick) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/1, /*fill_width=*/1);
  AcceptHit(ports);
  l2_multi_issue_pending_operation fill;
  fill.start(2);

  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_FALSE(fill.service_fill(ports));
  EXPECT_EQ(ports.stats().lookup_width_stall_cycles, 1u);
  EXPECT_EQ(ports.stats().data_port_width_stall_cycles, 1u);
  EXPECT_EQ(ports.stats().fill_port_width_stall_cycles, 1u);

  ports.begin_cycle();
  EXPECT_TRUE(fill.service_fill(ports));
  EXPECT_EQ(ports.stats().fill_port_width_stall_cycles, 1u);
}

TEST(L2MultiIssuePortsTest, AcceptedSectorStatisticsAreConservative) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/4, /*data_width=*/4, /*fill_width=*/2);
  ports.accept_lookup(1);  // clean miss
  AcceptHit(ports);
  l2_multi_issue_pending_operation dirty;
  dirty.start(2);
  EXPECT_TRUE(dirty.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  l2_multi_issue_pending_operation fill;
  fill.start(2);
  EXPECT_TRUE(fill.service_fill(ports));

  const l2_multi_issue_port_stats &stats = ports.stats();
  EXPECT_EQ(stats.lookup_accepted_sectors, 2u);
  EXPECT_EQ(stats.data_port_accepted_sectors, 3u);
  EXPECT_EQ(
      stats.data_port_accepted_sectors,
      stats.data_port_hit_sectors + stats.data_port_dirty_eviction_sectors);
  EXPECT_EQ(stats.fill_port_accepted_sectors, 2u);
}

}  // namespace
