#ifndef FLASH_GPGPU_SIM_TCGEN05_TMEM_H
#define FLASH_GPGPU_SIM_TCGEN05_TMEM_H

#include "collector.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace flash_gpgpu_sim {

class tcgen05_tmem_spinlock_t {
public:
  tcgen05_tmem_spinlock_t() noexcept { m_locked.store(false); }

  void lock() noexcept {
    bool expected = false;
    while (!m_locked.compare_exchange_weak(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
      expected = false;
    }
  }

  void unlock() noexcept { m_locked.store(false, std::memory_order_release); }

private:
  std::atomic<bool> m_locked;
};

struct tcgen05_tmem_address_t {
  uint32_t lane;
  uint32_t column;
};

tcgen05_tmem_address_t tcgen05_decode_tmem_address(uint32_t address);
uint32_t tcgen05_encode_tmem_address(uint32_t lane, uint32_t column);

// Register transfers select their warp's fixed quarter, even when all warps
// use the same allocated base. Row bits 5:6 cannot select another quarter.
inline uint32_t tcgen05_tmem_warp_base(uint32_t base, unsigned warp) {
  return (base & ~0x00600000u) | ((warp % 4) << 21);
}
uint32_t tcgen05_tmem_register_address(uint32_t base_address,
                                       uint32_t warp_lane, uint32_t shape_lanes,
                                       uint32_t shape_bits,
                                       uint32_t register_index,
                                       bool packed = false);

struct tcgen05_tmem_scope_t {
  unsigned sm_id;
  unsigned cta_id;
  unsigned cta_group = 1;

  bool operator<(const tcgen05_tmem_scope_t &other) const {
    if (sm_id != other.sm_id)
      return sm_id < other.sm_id;
    // One physical TMEM per CTA: ld/st have no cta_group qualifier.
    return cta_id < other.cta_id;
  }
};

class tcgen05_tmem_manager_t {
public:
  struct allocation_t {
    uint32_t base;
    uint32_t ncols;
  };

  void init_native_backing(const tcgen05_tmem_scope_t &scope, uint32_t ncols);
  uint32_t alloc(const tcgen05_tmem_scope_t &scope, uint32_t ncols);
  void dealloc(const tcgen05_tmem_scope_t &scope, uint32_t base,
               uint32_t ncols);
  void relinquish_alloc_permit(const tcgen05_tmem_scope_t &scope);

  bool permit_relinquished(const tcgen05_tmem_scope_t &scope) const;
  bool has_allocation(const tcgen05_tmem_scope_t &scope, uint32_t base) const;
  bool contains_range(const tcgen05_tmem_scope_t &scope, uint32_t address,
                      uint32_t nwords) const;
  unsigned allocation_count(const tcgen05_tmem_scope_t &scope) const;

  void write_words(const tcgen05_tmem_scope_t &scope, uint32_t address,
                   const std::vector<uint32_t> &values);
  std::vector<uint32_t> read_words(const tcgen05_tmem_scope_t &scope,
                                   uint32_t address, uint32_t nwords) const;
  void write_matrix_words(const tcgen05_tmem_scope_t &scope, uint32_t address,
                          const std::vector<uint32_t> &values, uint32_t rows,
                          uint32_t columns);
  std::vector<uint32_t> read_matrix_words(const tcgen05_tmem_scope_t &scope,
                                          uint32_t address, uint32_t rows,
                                          uint32_t columns) const;
  std::vector<uint16_t>
  read_matrix_packed_u16(const tcgen05_tmem_scope_t &scope, uint32_t address,
                         uint32_t rows, uint32_t columns) const;

  std::vector<uint8_t> read_mxf4_scale_matrix(const tcgen05_tmem_scope_t &scope,
                                              uint32_t address,
                                              uint32_t logical_rows,
                                              uint32_t scales_per_row,
                                              uint8_t scale_factor_id) const;

  // Logical accumulator layout shared by PTX and SASS adapters. Layout B
  // folds CTA2 M128 across the upper datapaths; layout F spaces M64 rows.
  std::vector<uint32_t> read_accumulator(const tcgen05_tmem_scope_t &scope,
                                         uint32_t base, unsigned rows,
                                         unsigned columns, bool folded) const;
  void write_accumulator(const tcgen05_tmem_scope_t &scope, uint32_t base,
                         const std::vector<uint32_t> &values, unsigned rows,
                         unsigned columns, bool folded,
                         const uint32_t *disable_lanes = nullptr);

  bool ws_half_b(const tcgen05_tmem_scope_t &scope, unsigned owner,
                 tcgen05_ws_collector_control control,
                 tcgen05_ws_half_signature signature,
                 std::vector<uint16_t> &values);

  void clear_cta(unsigned sm_id, unsigned cta_id);
  void reset();

private:
  struct scope_state_t {
    tcgen05_ws_half_collectors ws_collectors;
    uint32_t next_base = 0;
    uint32_t last_allocation_ncols = 0;
    bool permit_relinquished = false;
    std::map<uint32_t, allocation_t> allocations;
    std::map<uint32_t, allocation_t> retired_allocations;
    std::map<uint32_t, uint32_t> words;
  };

  scope_state_t &get_or_create_scope(const tcgen05_tmem_scope_t &scope);
  const scope_state_t *find_scope(const tcgen05_tmem_scope_t &scope) const;
  const allocation_t *find_allocation_containing(const scope_state_t &state,
                                                 uint32_t address,
                                                 uint32_t nwords) const;
  const allocation_t *find_accessible_allocation_containing(
      const scope_state_t &state, uint32_t address, uint32_t nwords) const;

  std::map<tcgen05_tmem_scope_t, scope_state_t> m_scopes;
  mutable tcgen05_tmem_spinlock_t m_mutex;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_TMEM_H
