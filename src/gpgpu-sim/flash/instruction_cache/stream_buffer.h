#ifndef FLASH_GPGPU_SIM_INSTRUCTION_CACHE_STREAM_BUFFER_H
#define FLASH_GPGPU_SIM_INSTRUCTION_CACHE_STREAM_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

namespace flash_gpgpu_sim {

enum class instruction_prefetch_state { kQueued, kInflight, kReady };

enum class instruction_demand_state {
  kUntracked,
  kQueued,
  kInflight,
  kReady,
};

struct instruction_prefetch_request {
  uint64_t address = 0;
  uint64_t context = 0;
  uint64_t generation = 0;
  unsigned stream = 0;
};

struct instruction_stream_buffer_config {
  uint64_t line_size = 128;
  unsigned streams = 1;
  unsigned depth = 4;
  unsigned issue_width = 1;
  unsigned gcc_preload_lines = 0;
};

struct instruction_stream_buffer_stats {
  uint64_t streams_started = 0;
  uint64_t streams_replaced = 0;
  uint64_t requests_issued = 0;
  uint64_t useful = 0;
  uint64_t late = 0;
  uint64_t resident = 0;
  uint64_t retries = 0;
  uint64_t stale_fills = 0;
  uint64_t canceled_entries = 0;
  uint64_t gcc_preload_hits = 0;
  uint64_t gcc_preload_misses = 0;

  instruction_stream_buffer_stats &
  operator+=(const instruction_stream_buffer_stats &other) {
    streams_started += other.streams_started;
    streams_replaced += other.streams_replaced;
    requests_issued += other.requests_issued;
    useful += other.useful;
    late += other.late;
    resident += other.resident;
    retries += other.retries;
    stale_fills += other.stale_fills;
    canceled_entries += other.canceled_entries;
    gcc_preload_hits += other.gcc_preload_hits;
    gcc_preload_misses += other.gcc_preload_misses;
    return *this;
  }
};

class instruction_stream_buffer {
public:
  explicit instruction_stream_buffer(
      const instruction_stream_buffer_config &config);

  instruction_demand_state observe_demand(uint64_t context, uint64_t address,
                                          bool demand_miss);
  std::vector<instruction_prefetch_request> issue();
  void retry(const instruction_prefetch_request &request);
  void mark_resident(const instruction_prefetch_request &request);
  void fill(const instruction_prefetch_request &request);
  void cancel_context(uint64_t context);
  void reset();

  bool current(const instruction_prefetch_request &request) const;
  bool gcc_preload_contains(const instruction_prefetch_request &request) const;
  void record_gcc_preload_hit() { ++m_stats.gcc_preload_hits; }
  void record_gcc_preload_miss() { ++m_stats.gcc_preload_misses; }
  size_t pending_entries() const;
  size_t pending_entries(unsigned stream) const;
  const instruction_stream_buffer_stats &stats() const { return m_stats; }

private:
  struct entry {
    uint64_t address = 0;
    instruction_prefetch_state state = instruction_prefetch_state::kQueued;
  };

  struct stream_state {
    bool valid = false;
    uint64_t context = 0;
    uint64_t generation = 0;
    uint64_t demand_line = 0;
    uint64_t last_use = 0;
    std::deque<entry> entries;
  };

  uint64_t line_address(uint64_t address) const;
  void start_stream(uint64_t context, uint64_t demand_line);
  void refill_window(stream_state &stream);
  void invalidate(stream_state &stream);
  entry *find_entry(stream_state &stream, uint64_t address);
  const entry *find_entry(const stream_state &stream, uint64_t address) const;
  stream_state *request_stream(const instruction_prefetch_request &request);
  const stream_state *
  request_stream(const instruction_prefetch_request &request) const;

  instruction_stream_buffer_config m_config;
  std::vector<stream_state> m_streams;
  unsigned m_issue_cursor = 0;
  uint64_t m_clock = 0;
  std::map<uint64_t, uint64_t> m_preload_base_lines;
  instruction_stream_buffer_stats m_stats;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_INSTRUCTION_CACHE_STREAM_BUFFER_H
