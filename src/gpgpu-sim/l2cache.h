// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"
#include "mem_transport_budget.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <list>
#include <queue>
#include <vector>

// ROP-delay arbitration and accounting belong to the L2 sub-partition.
enum rop_delay_output_stop_reason {
  ROP_DELAY_OUTPUT_NO_READY_WORK,
  ROP_DELAY_OUTPUT_WIDTH_LIMITED,
  ROP_DELAY_OUTPUT_DOWNSTREAM_FULL,
};

struct rop_delay_output_service_result {
  rop_delay_output_service_result()
      : reason(ROP_DELAY_OUTPUT_NO_READY_WORK),
        accepted_items(0),
        accepted_sectors(0) {}

  rop_delay_output_stop_reason reason;
  unsigned accepted_items;
  unsigned accepted_sectors;
};

struct rop_delay_output_service_stats {
  rop_delay_output_service_stats()
      : accepted_items(0),
        accepted_sectors(0),
        service_ticks(0),
        max_sectors_per_tick(0),
        width_limited_ticks(0),
        downstream_full_ticks(0),
        queue_full_ticks(0) {}

  void record(const rop_delay_output_service_result &result) {
    accepted_items += result.accepted_items;
    accepted_sectors += result.accepted_sectors;
    if (result.accepted_items != 0) ++service_ticks;
    max_sectors_per_tick =
        std::max<unsigned long long>(max_sectors_per_tick,
                                     result.accepted_sectors);
    if (result.reason == ROP_DELAY_OUTPUT_WIDTH_LIMITED)
      ++width_limited_ticks;
    if (result.reason == ROP_DELAY_OUTPUT_DOWNSTREAM_FULL) {
      ++downstream_full_ticks;
      // The production ROP-delay consumer has exactly one downstream: the
      // bounded icnt-to-L2 FIFO. Keep the exact queue-full count separate so a
      // future downstream can be distinguished without changing this ABI.
      ++queue_full_ticks;
    }
  }

  rop_delay_output_service_stats &operator+=(
      const rop_delay_output_service_stats &rhs) {
    accepted_items += rhs.accepted_items;
    accepted_sectors += rhs.accepted_sectors;
    service_ticks += rhs.service_ticks;
    max_sectors_per_tick =
        std::max(max_sectors_per_tick, rhs.max_sectors_per_tick);
    width_limited_ticks += rhs.width_limited_ticks;
    downstream_full_ticks += rhs.downstream_full_ticks;
    queue_full_ticks += rhs.queue_full_ticks;
    return *this;
  }

  void print(FILE *fout, const char *name) const {
    fprintf(fout, "%s_accepted_items = %llu\n", name, accepted_items);
    fprintf(fout, "%s_accepted_sectors = %llu\n", name, accepted_sectors);
    fprintf(fout, "%s_service_ticks = %llu\n", name, service_ticks);
    fprintf(fout, "%s_max_sectors_per_tick = %llu\n", name,
            max_sectors_per_tick);
    fprintf(fout, "%s_width_limited_ticks = %llu\n", name,
            width_limited_ticks);
    fprintf(fout, "%s_downstream_full_ticks = %llu\n", name,
            downstream_full_ticks);
    fprintf(fout, "%s_queue_full_ticks = %llu\n", name, queue_full_ticks);
  }

  unsigned long long accepted_items;
  unsigned long long accepted_sectors;
  unsigned long long service_ticks;
  unsigned long long max_sectors_per_tick;
  unsigned long long width_limited_ticks;
  unsigned long long downstream_full_ticks;
  unsigned long long queue_full_ticks;
};

// The local and remote priority queues retain the legacy arbitration rule:
// the earlier ready cycle wins across queues, and a tie selects local. Within
// each queue, insertion order is deterministic FIFO for equal ready cycles.
template <typename T>
class rop_delay_output_queue {
 public:
  rop_delay_output_queue() : m_next_sequence(0) {}

  void push(T item, unsigned long long ready_cycle, bool remote) {
    entry value = {ready_cycle, m_next_sequence++, item};
    if (remote)
      m_remote.push(value);
    else
      m_local.push(value);
  }

  bool empty() const { return m_local.empty() && m_remote.empty(); }
  std::size_t size() const { return m_local.size() + m_remote.size(); }

  bool has_ready(unsigned long long cycle) const {
    return next_ready_queue(cycle) != NULL;
  }

  // Width one is deliberately the legacy compatibility mode: it accepts one
  // ready queue item per tick even if a non-sector cache represented that item
  // with more than one 32-byte sector. Widths above one require every item to
  // be exactly one sector, making the configured unit unambiguous.
  template <typename SectorCount, typename DownstreamFull, typename Accept>
  rop_delay_output_service_result service(unsigned long long cycle,
                                           unsigned width,
                                           SectorCount sector_count,
                                           DownstreamFull downstream_full,
                                           Accept accept) {
    assert(width > 0);
    rop_delay_output_service_result result;

    while (true) {
      queue_type *queue = next_ready_queue(cycle);
      if (queue == NULL) {
        result.reason = ROP_DELAY_OUTPUT_NO_READY_WORK;
        return result;
      }

      const unsigned sectors = sector_count(queue->top().item);
      assert(sectors > 0);
      if (width == 1) {
        if (result.accepted_items == 1) {
          result.reason = ROP_DELAY_OUTPUT_WIDTH_LIMITED;
          return result;
        }
      } else {
        assert(sectors == 1 &&
               "multi-issue ROP output requires 32-byte sector children");
        if (result.accepted_sectors + sectors > width) {
          result.reason = ROP_DELAY_OUTPUT_WIDTH_LIMITED;
          return result;
        }
      }

      if (downstream_full()) {
        result.reason = ROP_DELAY_OUTPUT_DOWNSTREAM_FULL;
        return result;
      }

      const T item = queue->top().item;
      queue->pop();
      accept(item);
      ++result.accepted_items;
      result.accepted_sectors += sectors;
    }
  }

 private:
  struct entry {
    unsigned long long ready_cycle;
    unsigned long long sequence;
    T item;
  };

  static bool earlier(const entry &lhs, const entry &rhs) {
    if (lhs.ready_cycle != rhs.ready_cycle)
      return lhs.ready_cycle < rhs.ready_cycle;
    return lhs.sequence < rhs.sequence;
  }

  struct later_first {
    bool operator()(const entry &lhs, const entry &rhs) const {
      return earlier(rhs, lhs);
    }
  };

  typedef std::priority_queue<entry, std::vector<entry>, later_first>
      queue_type;

  static bool ready(const queue_type &queue, unsigned long long cycle) {
    return !queue.empty() && cycle >= queue.top().ready_cycle;
  }

  queue_type *next_ready_queue(unsigned long long cycle) {
    const bool local_ready = ready(m_local, cycle);
    const bool remote_ready = ready(m_remote, cycle);
    if (!local_ready && !remote_ready) return NULL;
    if (local_ready && remote_ready)
      return m_local.top().ready_cycle <= m_remote.top().ready_cycle
                 ? &m_local
                 : &m_remote;
    return local_ready ? &m_local : &m_remote;
  }

  const queue_type *next_ready_queue(unsigned long long cycle) const {
    const bool local_ready = ready(m_local, cycle);
    const bool remote_ready = ready(m_remote, cycle);
    if (!local_ready && !remote_ready) return NULL;
    if (local_ready && remote_ready)
      return m_local.top().ready_cycle <= m_remote.top().ready_cycle
                 ? &m_local
                 : &m_remote;
    return local_ready ? &m_local : &m_remote;
  }

  queue_type m_local;
  queue_type m_remote;
  unsigned long long m_next_sequence;
};

enum l2_multi_issue_data_work {
  L2_MULTI_ISSUE_HIT_DATA = 0,
  L2_MULTI_ISSUE_DIRTY_EVICTION,
};

enum class l2_port_model_kind {
  legacy = 0,
  multi_issue,
};

inline l2_port_model_kind l2_port_model_from_config(unsigned mode) {
  switch (mode) {
    case 0:
      return l2_port_model_kind::legacy;
    case 1:
      return l2_port_model_kind::multi_issue;
    default:
      assert(mode <= 1);
      std::abort();
  }
}

inline bool l2_multi_issue_port_model_enabled(unsigned mode) {
  return l2_port_model_from_config(mode) == l2_port_model_kind::multi_issue;
}

inline bool l2_multi_issue_needs_data_port(
    cache_request_status tag_probe_status,
    unsigned prospective_dirty_eviction_sectors, bool ready_read_forward) {
  assert(!ready_read_forward ||
         (tag_probe_status != HIT && tag_probe_status != RESERVATION_FAIL));
  return !ready_read_forward &&
         (tag_probe_status == HIT || prospective_dirty_eviction_sectors > 0);
}

struct l2_multi_issue_port_stats {
  unsigned long long lookup_accepted_sectors;
  unsigned long long data_port_accepted_sectors;
  unsigned long long data_port_hit_sectors;
  unsigned long long data_port_dirty_eviction_sectors;
  unsigned long long fill_port_accepted_sectors;
  unsigned long long lookup_width_stall_cycles;
  unsigned long long data_port_width_stall_cycles;
  unsigned long long fill_port_width_stall_cycles;

  l2_multi_issue_port_stats()
      : lookup_accepted_sectors(0),
        data_port_accepted_sectors(0),
        data_port_hit_sectors(0),
        data_port_dirty_eviction_sectors(0),
        fill_port_accepted_sectors(0),
        lookup_width_stall_cycles(0),
        data_port_width_stall_cycles(0),
        fill_port_width_stall_cycles(0) {}

  l2_multi_issue_port_stats &operator+=(const l2_multi_issue_port_stats &rhs) {
    lookup_accepted_sectors += rhs.lookup_accepted_sectors;
    data_port_accepted_sectors += rhs.data_port_accepted_sectors;
    data_port_hit_sectors += rhs.data_port_hit_sectors;
    data_port_dirty_eviction_sectors += rhs.data_port_dirty_eviction_sectors;
    fill_port_accepted_sectors += rhs.fill_port_accepted_sectors;
    lookup_width_stall_cycles += rhs.lookup_width_stall_cycles;
    data_port_width_stall_cycles += rhs.data_port_width_stall_cycles;
    fill_port_width_stall_cycles += rhs.fill_port_width_stall_cycles;
    return *this;
  }
};

// Per-L2-instance sector service for the optional multi-issue port model.
// A sector is a 32-byte L2 work package. These counters describe internal
// sector service, not physical or logical request counts; callers retain all
// mem_fetch ownership and completion semantics.
class l2_multi_issue_ports {
 public:
  l2_multi_issue_ports()
      : m_lookup_width(1),
        m_data_width(1),
        m_fill_width(1),
        m_lookup_remaining(1),
        m_data_remaining(1),
        m_fill_remaining(1),
        m_lookup_stall_recorded(false),
        m_data_stall_recorded(false),
        m_fill_stall_recorded(false) {}

  void configure(unsigned lookup_width, unsigned data_width,
                 unsigned fill_width) {
    assert(lookup_width > 0);
    assert(data_width > 0);
    assert(fill_width > 0);
    m_lookup_width = lookup_width;
    m_data_width = data_width;
    m_fill_width = fill_width;
    begin_cycle();
  }

  void begin_cycle() {
    m_lookup_remaining = m_lookup_width;
    m_data_remaining = m_data_width;
    m_fill_remaining = m_fill_width;
    m_lookup_stall_recorded = false;
    m_data_stall_recorded = false;
    m_fill_stall_recorded = false;
  }

  bool can_accept_lookup(unsigned sectors) {
    assert(sectors > 0);
    if (sectors <= m_lookup_remaining) return true;
    record_once(m_stats.lookup_width_stall_cycles, m_lookup_stall_recorded);
    return false;
  }

  void accept_lookup(unsigned sectors) {
    assert(sectors > 0 && sectors <= m_lookup_remaining);
    m_lookup_remaining -= sectors;
    m_stats.lookup_accepted_sectors += sectors;
  }

  bool data_port_has_capacity() {
    if (m_data_remaining > 0) return true;
    record_once(m_stats.data_port_width_stall_cycles, m_data_stall_recorded);
    return false;
  }

  unsigned accept_data(unsigned pending_sectors,
                       l2_multi_issue_data_work work) {
    assert(pending_sectors > 0);
    const unsigned accepted = std::min(pending_sectors, m_data_remaining);
    m_data_remaining -= accepted;
    m_stats.data_port_accepted_sectors += accepted;
    if (work == L2_MULTI_ISSUE_HIT_DATA)
      m_stats.data_port_hit_sectors += accepted;
    else
      m_stats.data_port_dirty_eviction_sectors += accepted;
    if (accepted < pending_sectors)
      record_once(m_stats.data_port_width_stall_cycles, m_data_stall_recorded);
    return accepted;
  }

  unsigned accept_fill(unsigned pending_sectors) {
    assert(pending_sectors > 0);
    const unsigned accepted = std::min(pending_sectors, m_fill_remaining);
    m_fill_remaining -= accepted;
    m_stats.fill_port_accepted_sectors += accepted;
    if (accepted < pending_sectors)
      record_once(m_stats.fill_port_width_stall_cycles, m_fill_stall_recorded);
    return accepted;
  }

  unsigned lookup_remaining() const { return m_lookup_remaining; }
  unsigned data_remaining() const { return m_data_remaining; }
  unsigned fill_remaining() const { return m_fill_remaining; }
  const l2_multi_issue_port_stats &stats() const { return m_stats; }

 private:
  static void record_once(unsigned long long &counter, bool &recorded) {
    if (recorded) return;
    ++counter;
    recorded = true;
  }

  unsigned m_lookup_width;
  unsigned m_data_width;
  unsigned m_fill_width;
  unsigned m_lookup_remaining;
  unsigned m_data_remaining;
  unsigned m_fill_remaining;
  bool m_lookup_stall_recorded;
  bool m_data_stall_recorded;
  bool m_fill_stall_recorded;
  l2_multi_issue_port_stats m_stats;
};

class l2_multi_issue_pending_operation {
 public:
  l2_multi_issue_pending_operation() : m_remaining_sectors(0) {}

  void start(unsigned sectors) {
    assert(!active());
    assert(sectors > 0);
    m_remaining_sectors = sectors;
  }

  bool service_data(l2_multi_issue_ports &ports,
                    l2_multi_issue_data_work work) {
    assert(active());
    m_remaining_sectors -= ports.accept_data(m_remaining_sectors, work);
    return !active();
  }

  bool service_fill(l2_multi_issue_ports &ports) {
    assert(active());
    m_remaining_sectors -= ports.accept_fill(m_remaining_sectors);
    return !active();
  }

  bool active() const { return m_remaining_sectors != 0; }
  unsigned remaining_sectors() const { return m_remaining_sectors; }

 private:
  unsigned m_remaining_sectors;
};

class mem_fetch;

enum mem_sub_partition_full_stat {
  MSP_FULL_ICNT_TO_L2_NOT_ENOUGH_SECTOR_SLOTS = 0,
  MSP_FULL_ICNT_TO_L2_QUEUE_FULL,
  MSP_FULL_ICNT_TO_L2_QUEUE_NEAR_FULL,
  MSP_FULL_L2_DRAM_QUEUE_FULL,
  MSP_FULL_DRAM_L2_QUEUE_FULL,
  MSP_FULL_L2_ICNT_QUEUE_FULL,
  MSP_FULL_L2_DATA_PORT_BUSY,
  MSP_FULL_L2_FILL_PORT_BUSY,
  MSP_FULL_L2_READY_BLOCKED_BY_L2_ICNT_QUEUE,
  NUM_MEM_SUB_PARTITION_FULL_STATS
};

const char *mem_sub_partition_full_stat_str(
    enum mem_sub_partition_full_stat stat);

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        memory_stats_manager_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) const;
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }

  class gpgpu_sim *get_mgpu() const { return m_gpu; }

 private:
  unsigned m_id;
  const memory_config *m_config;
  memory_stats_manager_t *m_mem_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  // Fixed-latency aggregate-service state. Credits are represented in units
  // of the configured service-rate denominator to avoid floating-point drift.
  unsigned long long m_simple_dram_issue_credit;
  unsigned long long m_simple_dram_return_credit;
  unsigned long long m_simple_dram_cycles;
  unsigned long long m_simple_dram_issue_requests;
  unsigned long long m_simple_dram_issue_atoms;
  unsigned long long m_simple_dram_return_requests;
  unsigned long long m_simple_dram_return_atoms;
  unsigned long long m_simple_dram_issue_no_request_cycles;
  unsigned long long m_simple_dram_issue_backpressure_cycles;
  unsigned long long m_simple_dram_return_not_ready_cycles;
  unsigned long long m_simple_dram_return_backpressure_cycles;
  unsigned long long m_simple_dram_queue_length_sum;
  unsigned long long m_simple_dram_queue_length_max;

  class gpgpu_sim *m_gpu;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       memory_stats_manager_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void record_full_state(unsigned size);
  void accumulate_full_state_stats(unsigned long long *stats) const;
  void accumulate_l2_multi_issue_port_stats(
      l2_multi_issue_port_stats &stats) const;
  void accumulate_rop_delay_output_stats(
      rop_delay_output_service_stats &stats) const;
  void accumulate_l2_partition_stats(unsigned long long &remote_accesses,
                                     unsigned long long &extra_latency) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  const memory_config *m_config;
  const l2_port_model_kind m_l2_port_model;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // Fixed-ready-cycle ROP delay and its independently configured output
  // service. Local/remote entries retain separate FIFO queues; the earlier
  // ready cycle wins across queues and equal cycles retain legacy local-first
  // arbitration.
  rop_delay_output_queue<class mem_fetch *> m_rop_delay_output;
  rop_delay_output_service_stats m_rop_delay_output_stats;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

  unsigned long long m_full_state_stats[NUM_MEM_SUB_PARTITION_FULL_STATS];
  l2_multi_issue_ports m_l2_multi_issue_ports;
  mem_fetch *m_pending_l2_writeback;
  l2_multi_issue_pending_operation m_pending_l2_writeback_work;
  mem_fetch *m_pending_l2_fill;
  l2_multi_issue_pending_operation m_pending_l2_fill_work;
  unsigned long long m_l2_partition_remote_accesses;
  unsigned long long m_l2_partition_extra_latency_cycles;

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_manager_t *m_mem_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  unsigned l2_partition_extra_latency(const mem_fetch *mf) const;
  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);
  void push_rop_delay(mem_fetch *mf, unsigned long long ready_cycle,
                      bool remote);
  void process_l2_access_result(mem_fetch *mf, cache_request_status status,
                                const std::list<cache_event> &events);
  void service_ready_l2_response();
  void cycle_legacy_l2_port_model();
  void cycle_multi_issue_l2_port_model();
  void service_dram_to_l2_legacy();
  void service_dram_to_l2_multi_issue();
  void service_l2_requests_legacy();
  void service_l2_requests_multi_issue();
  void enqueue_ready_rop(unsigned cycle);
  bool l2_data_port_busy() const;
  bool l2_fill_port_busy() const;

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    m_unit->m_L2_dram_queue->push(mf);
  }

 private:
  memory_sub_partition *m_unit;
};

#endif
