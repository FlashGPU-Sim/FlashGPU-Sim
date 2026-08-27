#ifndef FLASH_GPGPU_SIM_ICACHE_PREFETCHER_H
#define FLASH_GPGPU_SIM_ICACHE_PREFETCHER_H

#include <cstdio>
#include <map>
#include <set>
#include <vector>

#include "../gpu-cache.h"

class gpgpu_sim;
class memory_config;
class read_only_cache;
class shader_core_config;

namespace flash_gpgpu_sim {

struct icache_prefetch_stats_t {
  unsigned long long issued = 0;
  unsigned long long hit_in_cache = 0;
  unsigned long long hit_in_mshr = 0;
  unsigned long long useful = 0;
  unsigned long long streams_started = 0;
  unsigned long long drain_cycles = 0;

  icache_prefetch_stats_t &operator+=(const icache_prefetch_stats_t &other) {
    issued += other.issued;
    hit_in_cache += other.hit_in_cache;
    hit_in_mshr += other.hit_in_mshr;
    useful += other.useful;
    streams_started += other.streams_started;
    drain_cycles += other.drain_cycles;
    return *this;
  }
};

class icache_prefetcher_t {
 public:
  icache_prefetcher_t(read_only_cache *l1i, unsigned sid, unsigned tpc,
                      const memory_config *mem_config,
                      const shader_core_config *shader_config,
                      gpgpu_sim *gpu);

  void on_demand_miss(new_addr_type block_addr,
                      unsigned long long stream_id);
  bool check_useful(new_addr_type block_addr);
  void cycle();
  void on_fill(new_addr_type block_addr);
  void invalidate();

  const icache_prefetch_stats_t &stats() const { return m_stats; }
  bool enabled() const { return m_enabled; }
  bool has_pending() const { return !m_outstanding_streams.empty(); }
  void inc_drain_cycles() { m_stats.drain_cycles++; }

 private:
  struct stream_buffer_t {
    bool active = false;
    new_addr_type next_addr = 0;
    unsigned outstanding = 0;
    unsigned long long stream_id = 0;
  };

  void retire_fill(new_addr_type block_addr);

  read_only_cache *m_l1i;
  unsigned m_sid;
  unsigned m_tpc;
  const memory_config *m_mem_config;
  const shader_core_config *m_shader_config;
  gpgpu_sim *m_gpu;

  bool m_enabled;
  unsigned m_num_streams;
  unsigned m_depth;
  unsigned m_line_sz;

  std::vector<stream_buffer_t> m_streams;
  std::set<new_addr_type> m_prefetched_lines;
  std::map<new_addr_type, unsigned> m_outstanding_streams;
  icache_prefetch_stats_t m_stats;
};

}  // namespace flash_gpgpu_sim

#endif  // FLASH_GPGPU_SIM_ICACHE_PREFETCHER_H
