#ifndef FLASH_GPGPU_SIM_TCGEN05_TIMING_H
#define FLASH_GPGPU_SIM_TCGEN05_TIMING_H

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace flash_gpgpu_sim {

enum tcgen05_op_kind_t {
  TCGEN05_TIMING_MMA,
  TCGEN05_TIMING_CP,
  TCGEN05_TIMING_SHIFT,
  TCGEN05_TIMING_LD,
  TCGEN05_TIMING_ST,
  TCGEN05_TIMING_OP_COUNT
};

struct tcgen05_thread_stream_key_t {
  unsigned hw_cta_id = 0;
  unsigned warp_id = 0;
  unsigned lane_id = 0;
  unsigned cta_group = 1;

  bool operator<(const tcgen05_thread_stream_key_t &other) const;
};

struct tcgen05_warp_stream_key_t {
  unsigned hw_cta_id = 0;
  unsigned warp_id = 0;

  bool operator<(const tcgen05_warp_stream_key_t &other) const;
};

struct tcgen05_timing_config_t {
  unsigned mma_issue_interval = 1;
  unsigned mma_completion_base = 0;
  unsigned mma_f16_flops_per_cycle = 1;
  unsigned async_queue_depth = 0;
};

struct tcgen05_op_t {
  tcgen05_op_kind_t kind = TCGEN05_TIMING_MMA;
  uint64_t work = 0;
  unsigned completion_latency = 1;
  unsigned initiation_interval = 1;
};

struct tcgen05_completion_event_t {
  enum event_kind_t { MBARRIER_ARRIVAL, RELEASE_LD_WAIT, RELEASE_ST_WAIT };

  event_kind_t kind = MBARRIER_ARRIVAL;
  unsigned hw_cta_id = 0;
  unsigned warp_id = 0;
  uint32_t mbarrier_addr = 0;
};

struct tcgen05_timing_stats_t {
  uint64_t issued[TCGEN05_TIMING_OP_COUNT] = {};
  uint64_t completed[TCGEN05_TIMING_OP_COUNT] = {};
  uint64_t backend_busy_cycles = 0;
  uint64_t queue_full_stall_cycles = 0;
  uint64_t issue_interval_stall_cycles = 0;
  uint64_t commit_wait_cycles = 0;
  uint64_t ld_wait_cycles = 0;
  uint64_t st_wait_cycles = 0;
  unsigned max_queue_occupancy = 0;
};

// Per-SM TCGen05 asynchronous timing state. Functional data updates remain in
// cuda-sim; this class only controls performance-visible completion.
class tcgen05_unit_t {
public:
  explicit tcgen05_unit_t(const tcgen05_timing_config_t &config);

  bool can_enqueue(tcgen05_op_kind_t kind, uint64_t cycle);
  uint64_t enqueue_thread_op(const tcgen05_thread_stream_key_t &stream,
                             const tcgen05_op_t &op, uint64_t cycle);
  uint64_t enqueue_warp_mem_op(const tcgen05_warp_stream_key_t &stream,
                               const tcgen05_op_t &op, uint64_t cycle);

  void commit(const tcgen05_thread_stream_key_t &stream,
              uint32_t mbarrier_addr);
  bool wait_ld(const tcgen05_warp_stream_key_t &stream);
  bool wait_st(const tcgen05_warp_stream_key_t &stream);

  void cycle(uint64_t cycle);
  std::vector<tcgen05_completion_event_t> take_completion_events();
  void cleanup_cta(unsigned hw_cta_id);

  unsigned queue_occupancy() const;
  const tcgen05_timing_stats_t &stats() const { return m_stats; }

private:
  struct pending_op_t {
    bool thread_scoped = true;
    tcgen05_thread_stream_key_t thread_stream;
    tcgen05_warp_stream_key_t warp_stream;
    tcgen05_op_kind_t kind = TCGEN05_TIMING_MMA;
    uint64_t sequence = 0;
    uint64_t backend_done_cycle = 0;
    uint64_t completion_cycle = 0;
  };

  struct pending_commit_t {
    tcgen05_thread_stream_key_t stream;
    uint64_t cutoff = 0;
    uint32_t mbarrier_addr = 0;
  };

  struct pending_wait_t {
    tcgen05_warp_stream_key_t stream;
    tcgen05_op_kind_t kind = TCGEN05_TIMING_LD;
    uint64_t cutoff = 0;
  };

  bool thread_cutoff_satisfied(const tcgen05_thread_stream_key_t &stream,
                               uint64_t cutoff) const;
  bool warp_cutoff_satisfied(const tcgen05_warp_stream_key_t &stream,
                             tcgen05_op_kind_t kind, uint64_t cutoff) const;
  uint64_t last_warp_sequence(const tcgen05_warp_stream_key_t &stream,
                              tcgen05_op_kind_t kind) const;
  void resolve_controls();

  tcgen05_timing_config_t m_config;
  std::vector<pending_op_t> m_pending_ops;
  std::vector<pending_commit_t> m_pending_commits;
  std::vector<pending_wait_t> m_pending_waits;
  std::vector<tcgen05_completion_event_t> m_completion_events;
  std::map<tcgen05_thread_stream_key_t, uint64_t> m_thread_next_sequence;
  std::map<tcgen05_warp_stream_key_t, uint64_t> m_ld_next_sequence;
  std::map<tcgen05_warp_stream_key_t, uint64_t> m_st_next_sequence;
  uint64_t m_next_issue_cycle[TCGEN05_TIMING_OP_COUNT] = {};
  uint64_t m_mma_backend_available_cycle = 0;
  uint64_t m_mma_busy_period_start_cycle = 0;
  uint64_t m_mma_busy_period_work = 0;
  uint64_t m_last_queue_full_stall_cycle = std::numeric_limits<uint64_t>::max();
  uint64_t m_last_issue_interval_stall_cycle =
      std::numeric_limits<uint64_t>::max();
  tcgen05_timing_stats_t m_stats;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_TIMING_H
