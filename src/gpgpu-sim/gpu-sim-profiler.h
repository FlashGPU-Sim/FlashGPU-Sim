#ifndef GPU_SIM_PROFILER_H
#define GPU_SIM_PROFILER_H

#include <chrono>
#include <stdio.h>

namespace flash_gpgpu_sim {
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
  unsigned long profile_cycle_count = 0;

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

  // Print profiling statistics
  void print_stats() {
    double total_time = total_icnt_cycle_time + total_mem_to_icnt_time +
                        total_dram_cycle_time + total_l2_cache_time +
                        total_icnt_transfer_time + total_core_cycle_time +
                        total_other_time;

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
    printf("  Other Operations:            %.2f ms (%.1f%%)\n",
           total_other_time, (total_other_time / total_time) * 100.0);
    printf("Average time per cycle: %.4f ms\n", total_time / 10000.0);
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
    total_core_cycle_time = 0.0;
    total_other_time = 0.0;
  }

  // Increment cycle count and check if we should print stats
  void increment_and_check() {
    profile_cycle_count++;
    if (profile_cycle_count % 10000 == 0) {
      print_stats();
      reset();
    }
  }
};
} // namespace flash_gpgpu_sim

#endif // GPU_SIM_PROFILER_H
