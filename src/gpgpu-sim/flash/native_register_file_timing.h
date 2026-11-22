#ifndef FLASH_GPGPU_SIM_NATIVE_REGISTER_FILE_TIMING_H_
#define FLASH_GPGPU_SIM_NATIVE_REGISTER_FILE_TIMING_H_

#include "panic.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

namespace flash_gpgpu_sim {

// Timing state for the fixed-latency register-file path described by Huerta
// et al., MICRO 2025.  It intentionally does not know about SASS: a frontend
// supplies physical register numbers, regular source-operand positions, and
// compiler retain (.reuse) decisions through source_t.
//
// One subcore owns an instance.  Regular-register reads are reserved atomically
// over a fixed future window instead of being collected incrementally.  This
// keeps bank conflicts in the Allocate stage and leaves execution latency
// deterministic once an instruction advances.
class native_register_file_timing {
public:
  static constexpr unsigned kDefaultBanks = 2;
  static constexpr unsigned kDefaultReadPortsPerBank = 1;
  static constexpr unsigned kDefaultWritePortsPerBank = 1;
  static constexpr unsigned kDefaultReadWindow = 3;
  static constexpr unsigned kDefaultResultQueueDepth = 8;
  static constexpr unsigned kDefaultResultQueueMaxPops = 1;
  static constexpr unsigned kReuseSlots = 3;

  struct source_t {
    unsigned reg = 0;
    unsigned slot = 0;
    bool retain = false;
    bool cacheable = true;
  };

  struct allocation_t {
    bool ready = false;
    unsigned cache_hits = 0;
    std::vector<unsigned> reads_per_bank;
    std::vector<source_t> sources;
    uint64_t warp_id = 0;
  };

  explicit native_register_file_timing(
      unsigned banks = kDefaultBanks,
      unsigned read_ports_per_bank = kDefaultReadPortsPerBank,
      unsigned read_window = kDefaultReadWindow,
      unsigned result_queue_depth = kDefaultResultQueueDepth,
      unsigned result_queue_max_pops = kDefaultResultQueueMaxPops,
      unsigned write_ports_per_bank = kDefaultWritePortsPerBank)
      : banks_(banks), read_ports_per_bank_(read_ports_per_bank),
        read_window_(read_window), result_queue_depth_(result_queue_depth),
        result_queue_max_pops_(result_queue_max_pops),
        write_ports_per_bank_(write_ports_per_bank),
        reserved_reads_(banks, std::vector<unsigned>(read_window + 1, 0)),
        writes_this_cycle_(banks, 0),
        reuse_entries_(kReuseSlots, std::vector<reuse_entry_t>(banks)) {
    assert(banks_ != 0);
    assert(read_ports_per_bank_ != 0);
    assert(read_window_ != 0);
    assert(result_queue_depth_ != 0);
    assert(result_queue_max_pops_ != 0);
    assert(write_ports_per_bank_ != 0);
  }

  unsigned bank(unsigned reg) const { return reg % banks_; }

  // Probe RF and reuse-cache demand without modifying either state.  Cache
  // replacement is delayed until commit(), because an instruction stalled in
  // Allocate must not perturb the cache.
  allocation_t prepare(const std::vector<source_t> &sources, uint64_t warp_id,
                       unsigned read_cycles = 1) const {
    assert(read_cycles != 0);
    allocation_t result;
    result.reads_per_bank.resize(banks_, 0);
    result.sources = sources;
    result.warp_id = warp_id;

    std::vector<std::set<unsigned>> registers_by_bank(banks_);
    for (const source_t &source : sources) {
      if (source.cacheable) {
        assert(source.slot < kReuseSlots);
        if (reuse_entries_[source.slot][bank(source.reg)].matches(warp_id,
                                                                  source.reg)) {
          ++result.cache_hits;
          continue;
        }
      }
      registers_by_bank[bank(source.reg)].insert(source.reg);
    }

    result.ready = true;
    for (unsigned bank_id = 0; bank_id < banks_; ++bank_id) {
      result.reads_per_bank[bank_id] =
          registers_by_bank[bank_id].size() * read_cycles;
      if (!can_reserve(bank_id, result.reads_per_bank[bank_id]))
        result.ready = false;
    }
    return result;
  }

  // Reserve all requested reads and apply RFC replacement as one Allocate
  // transaction.  Every cacheable access invalidates its bank/operand-position
  // entry; the current value survives only when retain is set.
  void commit(const allocation_t &allocation) {
    assert(allocation.ready);
    assert(allocation.reads_per_bank.size() == banks_);
    for (unsigned bank_id = 0; bank_id < banks_; ++bank_id)
      reserve(bank_id, allocation.reads_per_bank[bank_id]);

    for (const source_t &source : allocation.sources) {
      if (!source.cacheable)
        continue;
      reuse_entry_t &entry = reuse_entries_[source.slot][bank(source.reg)];
      entry.valid = false;
      if (source.retain) {
        entry.valid = true;
        entry.warp_id = allocation.warp_id;
        entry.reg = source.reg;
      }
    }
  }

  void cycle() {
    for (auto &bank_reservations : reserved_reads_) {
      for (unsigned offset = 0; offset < read_window_; ++offset)
        bank_reservations[offset] = bank_reservations[offset + 1];
      bank_reservations[read_window_] = 0;
    }
    results_popped_this_cycle_ = 0;
    std::fill(writes_this_cycle_.begin(), writes_this_cycle_.end(), 0);
  }

  void flush_reuse_cache() {
    for (auto &slot : reuse_entries_)
      for (reuse_entry_t &entry : slot)
        entry.valid = false;
  }

  void reset() {
    for (auto &bank_reservations : reserved_reads_)
      std::fill(bank_reservations.begin(), bank_reservations.end(), 0);
    flush_reuse_cache();
    pending_results_ = 0;
    results_popped_this_cycle_ = 0;
    std::fill(writes_this_cycle_.begin(), writes_this_cycle_.end(), 0);
  }

  bool can_enqueue_result() const {
    return pending_results_ < result_queue_depth_;
  }

  void enqueue_result() {
    assert(can_enqueue_result());
    ++pending_results_;
  }

  bool can_pop_result() const {
    return results_popped_this_cycle_ < result_queue_max_pops_;
  }

  void retire_result() {
    assert(pending_results_ != 0);
    assert(can_pop_result());
    --pending_results_;
    ++results_popped_this_cycle_;
  }

  unsigned pending_results() const { return pending_results_; }

  bool can_write_result(const std::vector<unsigned> &destinations) const {
    const std::vector<unsigned> demand = write_demand(destinations);
    for (unsigned bank_id = 0; bank_id < banks_; ++bank_id) {
      if (writes_this_cycle_[bank_id] + demand[bank_id] > write_ports_per_bank_)
        return false;
    }
    return true;
  }

  unsigned write_result(const std::vector<unsigned> &destinations) {
    assert(can_write_result(destinations));
    const std::vector<unsigned> demand = write_demand(destinations);
    unsigned writes = 0;
    for (unsigned bank_id = 0; bank_id < banks_; ++bank_id) {
      writes_this_cycle_[bank_id] += demand[bank_id];
      writes += demand[bank_id];
    }
    return writes;
  }

  bool reuse_hit(unsigned slot, uint64_t warp_id, unsigned reg) const {
    assert(slot < kReuseSlots);
    return reuse_entries_[slot][bank(reg)].matches(warp_id, reg);
  }

  unsigned reserved_reads(unsigned bank_id, unsigned cycle_offset) const {
    if (bank_id >= reserved_reads_.size() ||
        cycle_offset >= reserved_reads_[bank_id].size())
      flash_gpgpu_sim::panic(
          "native register-file reservation is out of range");
    return reserved_reads_[bank_id][cycle_offset];
  }

private:
  struct reuse_entry_t {
    bool valid = false;
    uint64_t warp_id = 0;
    unsigned reg = 0;

    bool matches(uint64_t candidate_warp, unsigned candidate_reg) const {
      return valid && warp_id == candidate_warp && reg == candidate_reg;
    }
  };

  bool can_reserve(unsigned bank_id, unsigned reads) const {
    for (unsigned offset = 1; offset <= read_window_; ++offset) {
      const unsigned available =
          read_ports_per_bank_ - reserved_reads_[bank_id][offset];
      reads -= std::min(reads, available);
    }
    return reads == 0;
  }

  void reserve(unsigned bank_id, unsigned reads) {
    assert(can_reserve(bank_id, reads));
    for (unsigned offset = 1; offset <= read_window_ && reads != 0; ++offset) {
      const unsigned available =
          read_ports_per_bank_ - reserved_reads_[bank_id][offset];
      const unsigned allocated = std::min(reads, available);
      reserved_reads_[bank_id][offset] += allocated;
      reads -= allocated;
    }
    assert(reads == 0);
  }

  std::vector<unsigned>
  write_demand(const std::vector<unsigned> &destinations) const {
    std::vector<unsigned> demand(banks_, 0);
    const unsigned modeled_writes =
        std::min<unsigned>(destinations.size(), banks_ * write_ports_per_bank_);
    for (unsigned index = 0; index < modeled_writes; ++index)
      ++demand[bank(destinations[index])];
    return demand;
  }

  unsigned banks_;
  unsigned read_ports_per_bank_;
  unsigned read_window_;
  unsigned result_queue_depth_;
  unsigned result_queue_max_pops_;
  unsigned write_ports_per_bank_;
  unsigned pending_results_ = 0;
  unsigned results_popped_this_cycle_ = 0;
  std::vector<std::vector<unsigned>> reserved_reads_;
  std::vector<unsigned> writes_this_cycle_;
  std::vector<std::vector<reuse_entry_t>> reuse_entries_;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_NATIVE_REGISTER_FILE_TIMING_H_
