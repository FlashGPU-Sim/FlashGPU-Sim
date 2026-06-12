#ifndef GPU_SIM_PROFILER_H
#define GPU_SIM_PROFILER_H

#include <chrono>
#include <stdio.h>

namespace flash_gpgpu_sim {

struct gpgpu_sim_profile_progress_t {
  unsigned long long cycle = 0;
  unsigned long long cta_launched = 0;
  unsigned long long cta_completed = 0;
  unsigned long long tma_tx_started = 0;
  unsigned long long tma_read_tx_started = 0;
  unsigned long long tma_write_tx_started = 0;
  unsigned long long tma_tx_completed = 0;
  unsigned long long tma_read_tx_completed = 0;
  unsigned long long tma_write_tx_completed = 0;
  unsigned long long tma_mf_issued = 0;
  unsigned long long tma_read_mf_issued = 0;
  unsigned long long tma_write_mf_issued = 0;
  unsigned long long tma_mf_responses = 0;
  unsigned long long tma_read_mf_responses = 0;
  unsigned long long tma_write_mf_responses = 0;
  unsigned long long tma_bytes_issued = 0;
  unsigned long long tma_bytes_completed = 0;
};

// Profiler for tracking CPU time consumed in different cycle steps
struct gpgpu_sim_profiler_t {
  // Accumulated time for each step (in milliseconds)
  double total_icnt_cycle_time = 0.0;
  double total_mem_to_icnt_time = 0.0;
  double total_dram_cycle_time = 0.0;
  double total_l2_cache_time = 0.0;
  double total_icnt_transfer_time = 0.0;
  double total_core_cycle_time = 0.0;
  double total_other_time = 0.0;
  double total_gem5_simulate_time = 0.0;
  unsigned long profile_cycle_count = 0;
  bool has_last_progress = false;
  gpgpu_sim_profile_progress_t last_progress;

  // Timing points
  std::chrono::high_resolution_clock::time_point step_start;
  std::chrono::high_resolution_clock::time_point step_end;

  // Start timing a step
  void start_step() { step_start = std::chrono::high_resolution_clock::now(); }

  // End timing and add to the specified counter
  void end_step(double &counter) {
    step_end = std::chrono::high_resolution_clock::now();
    counter += std::chrono::duration<double, std::milli>(step_end - step_start)
                   .count();
  }

  static unsigned long long progress_delta(unsigned long long now,
                                           unsigned long long last) {
    return now >= last ? now - last : now;
  }

  void print_progress(const gpgpu_sim_profile_progress_t &progress) {
    unsigned long long cta_launched_delta = has_last_progress
                                                ? progress_delta(
                                                      progress.cta_launched,
                                                      last_progress.cta_launched)
                                                : progress.cta_launched;
    unsigned long long cta_completed_delta =
        has_last_progress
            ? progress_delta(progress.cta_completed,
                             last_progress.cta_completed)
            : progress.cta_completed;
    unsigned long long tma_tx_started_delta =
        has_last_progress
            ? progress_delta(progress.tma_tx_started,
                             last_progress.tma_tx_started)
            : progress.tma_tx_started;
    unsigned long long tma_read_tx_started_delta =
        has_last_progress
            ? progress_delta(progress.tma_read_tx_started,
                             last_progress.tma_read_tx_started)
            : progress.tma_read_tx_started;
    unsigned long long tma_write_tx_started_delta =
        has_last_progress
            ? progress_delta(progress.tma_write_tx_started,
                             last_progress.tma_write_tx_started)
            : progress.tma_write_tx_started;
    unsigned long long tma_tx_completed_delta =
        has_last_progress
            ? progress_delta(progress.tma_tx_completed,
                             last_progress.tma_tx_completed)
            : progress.tma_tx_completed;
    unsigned long long tma_read_tx_completed_delta =
        has_last_progress
            ? progress_delta(progress.tma_read_tx_completed,
                             last_progress.tma_read_tx_completed)
            : progress.tma_read_tx_completed;
    unsigned long long tma_write_tx_completed_delta =
        has_last_progress
            ? progress_delta(progress.tma_write_tx_completed,
                             last_progress.tma_write_tx_completed)
            : progress.tma_write_tx_completed;
    unsigned long long tma_mf_issued_delta =
        has_last_progress
            ? progress_delta(progress.tma_mf_issued,
                             last_progress.tma_mf_issued)
            : progress.tma_mf_issued;
    unsigned long long tma_read_mf_issued_delta =
        has_last_progress
            ? progress_delta(progress.tma_read_mf_issued,
                             last_progress.tma_read_mf_issued)
            : progress.tma_read_mf_issued;
    unsigned long long tma_write_mf_issued_delta =
        has_last_progress
            ? progress_delta(progress.tma_write_mf_issued,
                             last_progress.tma_write_mf_issued)
            : progress.tma_write_mf_issued;
    unsigned long long tma_mf_responses_delta =
        has_last_progress
            ? progress_delta(progress.tma_mf_responses,
                             last_progress.tma_mf_responses)
            : progress.tma_mf_responses;
    unsigned long long tma_read_mf_responses_delta =
        has_last_progress
            ? progress_delta(progress.tma_read_mf_responses,
                             last_progress.tma_read_mf_responses)
            : progress.tma_read_mf_responses;
    unsigned long long tma_write_mf_responses_delta =
        has_last_progress
            ? progress_delta(progress.tma_write_mf_responses,
                             last_progress.tma_write_mf_responses)
            : progress.tma_write_mf_responses;
    unsigned long long tma_bytes_issued_delta =
        has_last_progress
            ? progress_delta(progress.tma_bytes_issued,
                             last_progress.tma_bytes_issued)
            : progress.tma_bytes_issued;
    unsigned long long tma_bytes_completed_delta =
        has_last_progress
            ? progress_delta(progress.tma_bytes_completed,
                             last_progress.tma_bytes_completed)
            : progress.tma_bytes_completed;

    printf("Progress: cycle=%llu CTA launched +%llu=%llu completed "
           "+%llu=%llu\n",
           progress.cycle, cta_launched_delta, progress.cta_launched,
           cta_completed_delta, progress.cta_completed);
    printf("Progress TMA tx: started +%llu=%llu (R +%llu=%llu, W "
           "+%llu=%llu) completed +%llu=%llu (R +%llu=%llu, W +%llu=%llu)\n",
           tma_tx_started_delta, progress.tma_tx_started,
           tma_read_tx_started_delta, progress.tma_read_tx_started,
           tma_write_tx_started_delta, progress.tma_write_tx_started,
           tma_tx_completed_delta, progress.tma_tx_completed,
           tma_read_tx_completed_delta, progress.tma_read_tx_completed,
           tma_write_tx_completed_delta, progress.tma_write_tx_completed);
    printf("Progress TMA mf: issued +%llu=%llu (R +%llu=%llu, W "
           "+%llu=%llu) responses +%llu=%llu (R +%llu=%llu, W +%llu=%llu)\n",
           tma_mf_issued_delta, progress.tma_mf_issued,
           tma_read_mf_issued_delta, progress.tma_read_mf_issued,
           tma_write_mf_issued_delta, progress.tma_write_mf_issued,
           tma_mf_responses_delta, progress.tma_mf_responses,
           tma_read_mf_responses_delta, progress.tma_read_mf_responses,
           tma_write_mf_responses_delta, progress.tma_write_mf_responses);
    printf("Progress TMA bytes: issued +%llu=%llu done +%llu=%llu\n",
           tma_bytes_issued_delta, progress.tma_bytes_issued,
           tma_bytes_completed_delta, progress.tma_bytes_completed);

    last_progress = progress;
    has_last_progress = true;
  }

  // Print profiling statistics
  void print_stats(const gpgpu_sim_profile_progress_t *progress = nullptr) {
    double total_time = total_icnt_cycle_time + total_mem_to_icnt_time +
                        total_dram_cycle_time + total_l2_cache_time +
                        total_icnt_transfer_time + total_core_cycle_time +
                        total_gem5_simulate_time + total_other_time;

    printf("\n========== Cycle Profiling Statistics (Last 10000 cycles) "
           "==========\n");
    printf("Total CPU time: %.2f ms\n", total_time);
    printf("  ICNT Cycle (CORE):           %.2f ms (%.1f%%)\n",
           total_icnt_cycle_time, (total_icnt_cycle_time / total_time) * 100.0);
    printf("  Mem to ICNT (ICNT):          %.2f ms (%.1f%%)\n",
           total_mem_to_icnt_time,
           (total_mem_to_icnt_time / total_time) * 100.0);
    printf("  DRAM Cycle (DRAM):           %.2f ms (%.1f%%)\n",
           total_dram_cycle_time, (total_dram_cycle_time / total_time) * 100.0);
    printf("  L2 Cache (L2):               %.2f ms (%.1f%%)\n",
           total_l2_cache_time, (total_l2_cache_time / total_time) * 100.0);
    printf("  ICNT Transfer (ICNT):        %.2f ms (%.1f%%)\n",
           total_icnt_transfer_time,
           (total_icnt_transfer_time / total_time) * 100.0);
    printf("  Core Cycle (CORE):           %.2f ms (%.1f%%)\n",
           total_core_cycle_time, (total_core_cycle_time / total_time) * 100.0);
    printf("  gem5 Simulate:               %.2f ms (%.1f%%)\n",
           total_gem5_simulate_time,
           (total_gem5_simulate_time / total_time) * 100.0);
    printf("  Other Operations:            %.2f ms (%.1f%%)\n",
           total_other_time, (total_other_time / total_time) * 100.0);
    printf("Average time per cycle: %.4f ms\n", total_time / 10000.0);
    if (progress != nullptr)
      print_progress(*progress);
    printf("==================================================================="
           "=\n\n");
    fflush(stdout);
  }

  // Reset all counters
  void reset() {
    total_icnt_cycle_time = 0.0;
    total_mem_to_icnt_time = 0.0;
    total_dram_cycle_time = 0.0;
    total_l2_cache_time = 0.0;
    total_icnt_transfer_time = 0.0;
    total_gem5_simulate_time = 0.0;
    total_core_cycle_time = 0.0;
    total_other_time = 0.0;
  }

  bool should_print_next() const {
    return (profile_cycle_count + 1) % 10000 == 0;
  }

  // Increment cycle count and check if we should print stats
  void increment_and_check(
      const gpgpu_sim_profile_progress_t *progress = nullptr) {
    profile_cycle_count++;
    if (profile_cycle_count % 10000 == 0) {
      print_stats(progress);
      reset();
    }
  }
};
} // namespace flash_gpgpu_sim

#endif // GPU_SIM_PROFILER_H
