#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <deque>
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

struct ClockDomainsMHz {
  unsigned long long core;
  unsigned long long interconnect;
  unsigned long long l2;
  unsigned long long dram;
};

ClockDomainsMHz ConfigClockDomainsMHz(const ConfigEntries &entries) {
  ClockDomainsMHz clocks = {};
  const std::string config_text = ConfigValue(entries, "-gpgpu_clock_domains");
  EXPECT_EQ(
      std::sscanf(config_text.c_str(), "%llu:%llu:%llu:%llu", &clocks.core,
                  &clocks.interconnect, &clocks.l2, &clocks.dram),
      4)
      << "invalid -gpgpu_clock_domains";
  return clocks;
}

unsigned long long CeilMultiplyDivide(unsigned long long value,
                                      unsigned long long multiplier,
                                      unsigned long long divisor) {
  if (divisor == 0) {
    ADD_FAILURE() << "zero clock-domain divisor";
    return 0;
  }
  const unsigned __int128 product =
      static_cast<unsigned __int128>(value) * multiplier;
  const unsigned __int128 result = (product + divisor - 1) / divisor;
  if (result > std::numeric_limits<unsigned long long>::max()) {
    ADD_FAILURE() << "clock-domain conversion overflow";
    return 0;
  }
  return static_cast<unsigned long long>(result);
}

struct SimpleDramPipeline {
  SimpleDramPipeline(unsigned numerator, unsigned denominator,
                     unsigned long long max_request_atoms)
      : issue_budget(numerator, denominator, max_request_atoms),
        return_budget(numerator, denominator, max_request_atoms),
        issued_atoms(0),
        returned_atoms(0),
        max_inflight(0) {}

  simple_dram_service_budget issue_budget;
  simple_dram_service_budget return_budget;
  std::deque<unsigned long long> ready_core_cycles;
  unsigned long long issued_atoms;
  unsigned long long returned_atoms;
  unsigned long long max_inflight;
};

TEST(B200DramBandwidthTest,
     CheckedInConfigSustainsNominalPeakThroughIssueAndReturnBudgets) {
  const std::filesystem::path repository_root = FindRepositoryRoot();
  ASSERT_FALSE(repository_root.empty())
      << "cannot locate configs/SM100_B200/gpgpusim.config from "
      << std::filesystem::current_path();
  const ConfigEntries config =
      ReadConfig(repository_root / "configs/SM100_B200/gpgpusim.config");
  ASSERT_FALSE(config.empty());
  const double nominal_expected_tb_per_second =
      ConfigDouble(config, "-gpgpu_dram_expected_bandwidth_TBp");
  ASSERT_GT(nominal_expected_tb_per_second, 0.0);
  ASSERT_EQ(ConfigUnsigned(config, "-gpgpu_simple_dram_model"), 1u);

  const unsigned long long memory_partitions =
      ConfigUnsigned(config, "-gpgpu_n_mem");
  const unsigned long long rate_numerator =
      ConfigUnsigned(config, "-gpgpu_simple_dram_service_rate_num");
  const unsigned long long rate_denominator =
      ConfigUnsigned(config, "-gpgpu_simple_dram_service_rate_den");
  const unsigned long long max_inflight_requests =
      ConfigUnsigned(config, "-gpgpu_simple_dram_max_inflight");
  const unsigned long long dram_latency =
      ConfigUnsigned(config, "-dram_latency");
  const unsigned long long bus_width_bytes =
      ConfigUnsigned(config, "-gpgpu_dram_buswidth");
  const unsigned long long burst_length =
      ConfigUnsigned(config, "-gpgpu_dram_burst_length");
  const unsigned long long memories_per_controller =
      ConfigUnsigned(config, "-gpgpu_n_mem_per_ctrlr");
  const ClockDomainsMHz clocks = ConfigClockDomainsMHz(config);

  ASSERT_GT(memory_partitions, 0u);
  ASSERT_GT(rate_numerator, 0u);
  ASSERT_GT(rate_denominator, 0u);
  ASSERT_LE(rate_numerator, std::numeric_limits<unsigned>::max());
  ASSERT_LE(rate_denominator, std::numeric_limits<unsigned>::max());
  ASSERT_GT(max_inflight_requests, 0u);
  ASSERT_GT(dram_latency, 0u);
  ASSERT_GT(bus_width_bytes, 0u);
  ASSERT_GT(burst_length, 0u);
  ASSERT_GT(memories_per_controller, 0u);
  ASSERT_GT(clocks.core, 0u);
  ASSERT_GT(clocks.dram, 0u);
  ASSERT_LE(bus_width_bytes,
            std::numeric_limits<unsigned long long>::max() / burst_length);
  const unsigned long long atom_bytes = bus_width_bytes * burst_length;
  ASSERT_LE(atom_bytes, std::numeric_limits<unsigned long long>::max() /
                            memories_per_controller);
  const unsigned long long dram_atom_bytes =
      atom_bytes * memories_per_controller;
  ASSERT_GT(dram_atom_bytes, 0u);
  const unsigned long long max_request_atoms =
      (MAX_MEMORY_ACCESS_SIZE + dram_atom_bytes - 1) / dram_atom_bytes;

  ASSERT_LE(memory_partitions, static_cast<unsigned long long>(
                                   std::numeric_limits<std::size_t>::max()));
  std::vector<SimpleDramPipeline> pipelines;
  pipelines.reserve(static_cast<std::size_t>(memory_partitions));
  for (unsigned long long partition = 0; partition < memory_partitions;
       ++partition) {
    pipelines.emplace_back(static_cast<unsigned>(rate_numerator),
                           static_cast<unsigned>(rate_denominator),
                           max_request_atoms);
  }

  // Build one lightweight pipeline per configured memory partition and supply
  // each pipeline with unlimited one-atom requests. The issue and return sides
  // use the exact service-budget class used by simple_dram_model_cycle(); the
  // deque below reproduces its return-before-issue fixed-latency ordering and
  // per-partition in-flight cap. This measures the aggregate simple-DRAM
  // endpoint without instantiating SM, interconnect, L2, DRAM banks, row
  // buffers, or FR-FCFS scheduling.
  constexpr unsigned long long kSettlingCoreCycles = 64;
  constexpr unsigned long long kMeasurementTicks = 4096;
  ASSERT_LE(dram_latency, std::numeric_limits<unsigned long long>::max() -
                              kSettlingCoreCycles);
  const unsigned long long warmup_core_cycles =
      dram_latency + kSettlingCoreCycles;
  const unsigned long long measurement_begin =
      CeilMultiplyDivide(warmup_core_cycles, clocks.dram, clocks.core);
  ASSERT_LE(measurement_begin,
            std::numeric_limits<unsigned long long>::max() - kMeasurementTicks);
  const unsigned long long measurement_end =
      measurement_begin + kMeasurementTicks;
  unsigned long long measured_issue_atoms = 0;
  unsigned long long measured_return_atoms = 0;

  for (unsigned long long dram_tick = 0; dram_tick < measurement_end;
       ++dram_tick) {
    // This is the exact integer form of the production clock scheduler. A
    // DRAM event at time dram_tick/dram_freq observes the number of CORE events
    // that occurred at earlier physical times. When both domains fire at the
    // same time, simple_dram_model_cycle() runs before gpu_sim_cycle advances.
    const unsigned long long core_cycle =
        CeilMultiplyDivide(dram_tick, clocks.core, clocks.dram);
    const bool measuring =
        dram_tick >= measurement_begin && dram_tick < measurement_end;
    for (SimpleDramPipeline &pipeline : pipelines) {
      pipeline.issue_budget.begin_tick();
      pipeline.return_budget.begin_tick();

      bool ready_return_seen = false;
      while (!pipeline.ready_core_cycles.empty() &&
             pipeline.ready_core_cycles.front() <= core_cycle) {
        ready_return_seen = true;
        if (!pipeline.return_budget.can_service(1)) break;
        pipeline.return_budget.consume(1);
        pipeline.ready_core_cycles.pop_front();
        ++pipeline.returned_atoms;
        if (measuring) ++measured_return_atoms;
      }
      if (!ready_return_seen) pipeline.return_budget.discard_idle_credit();

      while (pipeline.ready_core_cycles.size() < max_inflight_requests &&
             pipeline.issue_budget.can_service(1)) {
        pipeline.issue_budget.consume(1);
        ASSERT_LE(core_cycle, std::numeric_limits<unsigned long long>::max() -
                                  dram_latency);
        pipeline.ready_core_cycles.push_back(core_cycle + dram_latency);
        ++pipeline.issued_atoms;
        if (measuring) ++measured_issue_atoms;
      }
      pipeline.max_inflight = std::max<unsigned long long>(
          pipeline.max_inflight, pipeline.ready_core_cycles.size());
      ASSERT_EQ(pipeline.issued_atoms,
                pipeline.returned_atoms + pipeline.ready_core_cycles.size());
    }
  }

  unsigned long long total_issued_atoms = 0;
  unsigned long long total_returned_atoms = 0;
  unsigned long long total_inflight_atoms = 0;
  unsigned long long observed_max_inflight = 0;
  for (const SimpleDramPipeline &pipeline : pipelines) {
    EXPECT_EQ(pipeline.issue_budget.numerator(), rate_numerator);
    EXPECT_EQ(pipeline.issue_budget.denominator(), rate_denominator);
    EXPECT_EQ(pipeline.return_budget.numerator(), rate_numerator);
    EXPECT_EQ(pipeline.return_budget.denominator(), rate_denominator);
    EXPECT_LT(pipeline.issue_budget.credit(), rate_denominator);
    EXPECT_LT(pipeline.return_budget.credit(), rate_denominator);
    EXPECT_LT(pipeline.max_inflight, max_inflight_requests);
    total_issued_atoms += pipeline.issued_atoms;
    total_returned_atoms += pipeline.returned_atoms;
    total_inflight_atoms += pipeline.ready_core_cycles.size();
    observed_max_inflight =
        std::max(observed_max_inflight, pipeline.max_inflight);
  }
  EXPECT_EQ(total_issued_atoms, total_returned_atoms + total_inflight_atoms);
  EXPECT_EQ(measured_issue_atoms, measured_return_atoms);

  const double configured_atoms_per_tick =
      static_cast<double>(rate_numerator) / rate_denominator;
  const double modeled_bdp_requests = configured_atoms_per_tick * dram_latency *
                                      static_cast<double>(clocks.dram) /
                                      clocks.core;
  const unsigned long long minimum_bdp_requests =
      static_cast<unsigned long long>(std::floor(modeled_bdp_requests));
  const unsigned long long maximum_quantized_bdp_requests =
      static_cast<unsigned long long>(std::ceil(modeled_bdp_requests)) +
      static_cast<unsigned long long>(std::ceil(configured_atoms_per_tick));
  EXPECT_GT(max_inflight_requests,
            static_cast<unsigned long long>(std::ceil(modeled_bdp_requests)));
  EXPECT_GE(observed_max_inflight, minimum_bdp_requests);
  EXPECT_LE(observed_max_inflight, maximum_quantized_bdp_requests);

  const double theoretical_tb_per_second =
      static_cast<double>(memory_partitions) * configured_atoms_per_tick *
      dram_atom_bytes * clocks.dram * 1.0e6 / 1.0e12;
  const double measured_tb_per_second =
      (static_cast<double>(measured_return_atoms) / kMeasurementTicks) *
      dram_atom_bytes * clocks.dram * 1.0e6 / 1.0e12;
  const double utilization = measured_tb_per_second / theoretical_tb_per_second;

  // The external 7.7 TB/s target is rounded to one decimal place. The exact
  // checked-in 15/4 rate is 7.67232 TB/s, so the nominal comparison allows the
  // requested 1% tolerance. A separate comparison requires the driven budget
  // implementation to sustain at least 99% of its exact configured peak.
  EXPECT_GE(theoretical_tb_per_second, nominal_expected_tb_per_second * 0.99);
  EXPECT_GE(measured_tb_per_second, nominal_expected_tb_per_second * 0.99);
  EXPECT_GE(measured_tb_per_second, theoretical_tb_per_second * 0.99);
  EXPECT_LE(measured_tb_per_second,
            theoretical_tb_per_second * (1.0 + 1.0e-12));

  std::printf(
      "DRAM config-driven peak: partitions=%llu atom=%llu B rate=%llu/%llu "
      "core_clock=%llu MHz dram_clock=%llu MHz latency=%llu "
      "modeled_bdp=%.3f max_inflight=%llu observed_inflight=%llu "
      "expected_nominal=%.6f TB/s theoretical=%.6f TB/s measured=%.6f "
      "TB/s utilization=%.5f\n",
      memory_partitions, dram_atom_bytes, rate_numerator, rate_denominator,
      clocks.core, clocks.dram, dram_latency, modeled_bdp_requests,
      max_inflight_requests, observed_max_inflight,
      nominal_expected_tb_per_second, theoretical_tb_per_second,
      measured_tb_per_second, utilization);
}

}  // namespace
