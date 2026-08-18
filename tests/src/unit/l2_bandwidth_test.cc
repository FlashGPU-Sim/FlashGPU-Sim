#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/l2cache.h"

namespace {

using ConfigEntries = std::vector<std::pair<std::string, std::string>>;

std::filesystem::path FindRepositoryRoot() {
  std::filesystem::path current = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::is_regular_file(current /
                                         "configs/SM100_B200/gpgpusim.config"))
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

double ConfigDouble(const ConfigEntries &entries, const std::string &option) {
  std::istringstream input(ConfigValue(entries, option));
  double value = 0.0;
  const bool parsed = static_cast<bool>(input >> value);
  input >> std::ws;
  EXPECT_TRUE(parsed && input.eof()) << "invalid floating value for " << option;
  return value;
}

double ConfigL2ClockMHz(const ConfigEntries &entries) {
  double core_mhz = 0.0;
  double interconnect_mhz = 0.0;
  double l2_mhz = 0.0;
  double dram_mhz = 0.0;
  const std::string clocks = ConfigValue(entries, "-gpgpu_clock_domains");
  EXPECT_EQ(std::sscanf(clocks.c_str(), "%lf:%lf:%lf:%lf", &core_mhz,
                        &interconnect_mhz, &l2_mhz, &dram_mhz),
            4)
      << "invalid -gpgpu_clock_domains";
  return l2_mhz;
}

TEST(B200L2BandwidthTest, CheckedInConfigSustainsPeakWithAbove99PercentHits) {
  const std::filesystem::path repository_root = FindRepositoryRoot();
  ASSERT_FALSE(repository_root.empty())
      << "cannot locate configs/SM100_B200/gpgpusim.config from "
      << std::filesystem::current_path();
  const ConfigEntries config =
      ReadConfig(repository_root / "configs/SM100_B200/gpgpusim.config");
  ASSERT_FALSE(config.empty());
  const ConfigEntries expectations = ReadConfig(
      repository_root / "tests/configs/SM100_B200/l2_bandwidth.config");
  ASSERT_FALSE(expectations.empty());
  const double expected_tb_per_second =
      ConfigDouble(expectations, "-gpgpu_l2_expected_bandwidth_TBp");
  ASSERT_GT(expected_tb_per_second, 0.0);

  const unsigned long long memory_channels =
      ConfigUnsigned(config, "-gpgpu_n_mem");
  const unsigned long long slices_per_channel =
      ConfigUnsigned(config, "-gpgpu_n_sub_partition_per_mchannel");
  ASSERT_GT(memory_channels, 0u);
  ASSERT_GT(slices_per_channel, 0u);
  ASSERT_LE(memory_channels, std::numeric_limits<unsigned long long>::max() /
                                 slices_per_channel);
  const unsigned long long l2_instances = memory_channels * slices_per_channel;
  ASSERT_LE(l2_instances, static_cast<unsigned long long>(
                              std::numeric_limits<std::size_t>::max()));

  char cache_type = '\0';
  unsigned long long sets_per_slice = 0;
  unsigned long long line_bytes = 0;
  unsigned long long ways = 0;
  const std::string l2_config = ConfigValue(config, "-gpgpu_cache:dl2");
  ASSERT_EQ(std::sscanf(l2_config.c_str(), "%c:%llu:%llu:%llu", &cache_type,
                        &sets_per_slice, &line_bytes, &ways),
            4);
  ASSERT_EQ(cache_type, 'S')
      << "the multi-issue path services 32-byte sector work";
  ASSERT_GT(sets_per_slice, 0u);
  ASSERT_GT(line_bytes, 0u);
  ASSERT_GT(ways, 0u);
  ASSERT_EQ(line_bytes % SECTOR_SIZE, 0u);

  const unsigned long long port_model =
      ConfigUnsigned(config, "-gpgpu_l2_multi_issue_port_model");
  ASSERT_TRUE(l2_multi_issue_port_model_enabled(port_model));

  const unsigned long long lookup_width_config =
      ConfigUnsigned(config, "-gpgpu_l2_lookup_sectors_per_cycle");
  const unsigned long long data_width_config =
      ConfigUnsigned(config, "-gpgpu_l2_data_port_sectors_per_cycle");
  const unsigned long long fill_width_config =
      ConfigUnsigned(config, "-gpgpu_l2_fill_port_sectors_per_cycle");
  ASSERT_GT(lookup_width_config, 0u);
  ASSERT_GT(data_width_config, 0u);
  ASSERT_GT(fill_width_config, 0u);
  ASSERT_LE(lookup_width_config, std::numeric_limits<unsigned>::max());
  ASSERT_LE(data_width_config, std::numeric_limits<unsigned>::max());
  ASSERT_LE(fill_width_config, std::numeric_limits<unsigned>::max());

  const unsigned lookup_width = static_cast<unsigned>(lookup_width_config);
  const unsigned data_width = static_cast<unsigned>(data_width_config);
  const unsigned fill_width = static_cast<unsigned>(fill_width_config);
  const unsigned peak_hit_width = std::min(lookup_width, data_width);
  const double l2_clock_mhz = ConfigL2ClockMHz(config);
  ASSERT_GT(l2_clock_mhz, 0.0);

  // This drives the exact per-subpartition port implementation used by
  // memory_sub_partition::service_l2_requests_multi_issue(). Access outcomes
  // are supplied directly so the test excludes SM/ICNT queues and DRAM by
  // construction. A sparse miss stream keeps the measured hit rate above 99%
  // without requiring a physically impossible all-hit workload.
  constexpr unsigned kMeasurementCycles = 1024;
  constexpr unsigned kMissPeriodSectors = 128;
  std::vector<l2_multi_issue_ports> ports(
      static_cast<std::size_t>(l2_instances));
  for (l2_multi_issue_ports &instance : ports)
    instance.configure(lookup_width, data_width, fill_width);

  unsigned long long offered_sectors = 0;
  unsigned long long hit_sectors = 0;
  unsigned long long miss_sectors = 0;
  for (unsigned cycle = 0; cycle < kMeasurementCycles; ++cycle) {
    for (l2_multi_issue_ports &instance : ports) {
      instance.begin_cycle();
      for (unsigned sector = 0; sector < peak_hit_width; ++sector) {
        ASSERT_TRUE(instance.can_accept_lookup(1));
        instance.accept_lookup(1);
        ++offered_sectors;

        const bool hit = (offered_sectors % kMissPeriodSectors) != 0;
        if (!hit) {
          ++miss_sectors;
          continue;
        }

        ASSERT_TRUE(instance.data_port_has_capacity());
        ASSERT_EQ(instance.accept_data(1, L2_MULTI_ISSUE_HIT_DATA), 1u);
        ++hit_sectors;
      }
    }
  }

  l2_multi_issue_port_stats aggregate;
  for (const l2_multi_issue_ports &instance : ports)
    aggregate += instance.stats();

  const unsigned long long theoretical_peak_sectors =
      l2_instances * kMeasurementCycles * peak_hit_width;
  ASSERT_EQ(offered_sectors, theoretical_peak_sectors);
  ASSERT_GT(miss_sectors, 0u);
  ASSERT_EQ(hit_sectors + miss_sectors, offered_sectors);
  EXPECT_EQ(aggregate.lookup_accepted_sectors, offered_sectors);
  EXPECT_EQ(aggregate.data_port_accepted_sectors, hit_sectors);
  EXPECT_EQ(aggregate.data_port_hit_sectors, hit_sectors);
  EXPECT_EQ(aggregate.data_port_dirty_eviction_sectors, 0u);
  EXPECT_EQ(aggregate.fill_port_accepted_sectors, 0u);
  EXPECT_EQ(aggregate.lookup_width_stall_cycles, 0u);
  EXPECT_EQ(aggregate.data_port_width_stall_cycles, 0u);
  EXPECT_EQ(aggregate.fill_port_width_stall_cycles, 0u);

  const double hit_rate = static_cast<double>(hit_sectors) / offered_sectors;
  EXPECT_GT(hit_rate, 0.99);

  const double theoretical_tb_per_second = static_cast<double>(l2_instances) *
                                           peak_hit_width * SECTOR_SIZE *
                                           l2_clock_mhz * 1.0e6 / 1.0e12;
  const double measured_tb_per_second =
      (static_cast<double>(hit_sectors) / kMeasurementCycles) * SECTOR_SIZE *
      l2_clock_mhz * 1.0e6 / 1.0e12;
  EXPECT_GE(theoretical_tb_per_second, expected_tb_per_second);
  EXPECT_GE(measured_tb_per_second, expected_tb_per_second * 0.99);
  EXPECT_LE(measured_tb_per_second, theoretical_tb_per_second);

  std::printf(
      "L2 config-driven peak: instances=%llu lookup=%u data=%u fill=%u "
      "clock=%.3f MHz hit_rate=%.5f expected=%.6f TB/s "
      "theoretical=%.6f TB/s "
      "measured=%.6f TB/s\n",
      l2_instances, lookup_width, data_width, fill_width, l2_clock_mhz,
      hit_rate, expected_tb_per_second, theoretical_tb_per_second,
      measured_tb_per_second);
}

}  // namespace
