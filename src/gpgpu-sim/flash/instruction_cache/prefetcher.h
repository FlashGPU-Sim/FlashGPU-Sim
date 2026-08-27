#ifndef FLASH_GPGPU_SIM_INSTRUCTION_CACHE_PREFETCHER_H
#define FLASH_GPGPU_SIM_INSTRUCTION_CACHE_PREFETCHER_H

#include <cstdint>
#include <map>

#include "stream_buffer.h"

class gpgpu_sim;
class mem_fetch;
class memory_config;
class read_only_cache;

namespace flash_gpgpu_sim {

class instruction_prefetcher {
public:
  instruction_prefetcher(bool enabled, unsigned streams, unsigned depth,
                         unsigned issue_width, unsigned line_size, unsigned sid,
                         unsigned tpc, const memory_config *memory_config,
                         gpgpu_sim *gpu, read_only_cache *cache);

  void observe_demand(uint64_t context, uint64_t memory_stream,
                      uint64_t address, bool demand_miss);
  void cycle(bool demand_reservation_failed);
  void fill(const mem_fetch *mf);
  void reset();

  bool enabled() const { return m_enabled; }
  const instruction_stream_buffer_stats &stats() const {
    return m_stream_buffer.stats();
  }

private:
  instruction_prefetch_request request_from(const mem_fetch *mf) const;
  void activate_context(uint64_t context, uint64_t memory_stream);

  bool m_enabled;
  unsigned m_line_size;
  unsigned m_sid;
  unsigned m_tpc;
  const memory_config *m_memory_config;
  gpgpu_sim *m_gpu;
  read_only_cache *m_cache;
  instruction_stream_buffer m_stream_buffer;
  uint64_t m_active_context;
  std::map<uint64_t, uint64_t> m_context_streams;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_INSTRUCTION_CACHE_PREFETCHER_H
