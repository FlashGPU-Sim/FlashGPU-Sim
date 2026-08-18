#include "tmem.h"

#include <cassert>

namespace flash_gpgpu_sim {

namespace {

constexpr uint32_t kTcgen05TmemColumnMask = 0xffff;
constexpr uint32_t kTcgen05TmemColumnCount = kTcgen05TmemColumnMask + 1;
constexpr uint32_t kTcgen05TmemMinAllocColumns = 32;
constexpr uint32_t kTcgen05TmemMaxAllocColumns = 512;
constexpr uint32_t kTcgen05TmemLaneMask = 0xffff;
constexpr unsigned kTcgen05TmemLaneShift = 16;

bool is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

tcgen05_tmem_address_t tcgen05_decode_tmem_address(uint32_t address) {
  return tcgen05_tmem_address_t{address >> kTcgen05TmemLaneShift,
                                address & kTcgen05TmemColumnMask};
}

uint32_t tcgen05_encode_tmem_address(uint32_t lane, uint32_t column) {
  assert(lane <= kTcgen05TmemLaneMask &&
         "TCGen05 TMEM lane index exceeds 16-bit address field");
  assert(column <= kTcgen05TmemColumnMask &&
         "TCGen05 TMEM column index exceeds 16-bit address field");
  return (lane << kTcgen05TmemLaneShift) | column;
}

std::vector<uint32_t>
tcgen05_warpx4_32x128b_words(const std::vector<uint32_t> &source) {
  constexpr uint32_t kSourceDataPaths = 32;
  constexpr uint32_t kBitsPerDataPath = 128;
  constexpr uint32_t kBroadcastCopies = 4;
  constexpr uint32_t kWordsPerDataPath = kBitsPerDataPath / 32;
  assert(source.size() == kSourceDataPaths * kWordsPerDataPath &&
         "TCGen05 32x128b.warpx4 source must contain 4096 bits");

  std::vector<uint32_t> result(
      kSourceDataPaths * kBroadcastCopies * kWordsPerDataPath, 0);
  for (uint32_t source_dp = 0; source_dp < kSourceDataPaths; ++source_dp) {
    for (uint32_t broadcast = 0; broadcast < kBroadcastCopies; ++broadcast) {
      uint32_t destination_dp = source_dp + broadcast * kSourceDataPaths;
      for (uint32_t word = 0; word < kWordsPerDataPath; ++word) {
        result[destination_dp * kWordsPerDataPath + word] =
            source[source_dp * kWordsPerDataPath + word];
      }
    }
  }
  return result;
}

uint32_t tcgen05_tmem_manager_t::alloc(const tcgen05_tmem_scope_t &scope,
                                       uint32_t ncols) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  scope_state_t &state = get_or_create_scope(scope);
  assert(!state.permit_relinquished &&
         "TCGen05 TMEM alloc after relinquish_alloc_permit is illegal");
  assert(ncols >= kTcgen05TmemMinAllocColumns &&
         ncols <= kTcgen05TmemMaxAllocColumns &&
         "TCGen05 TMEM allocation ncols must be in [32, 512]");
  assert(is_power_of_two(ncols) &&
         "TCGen05 TMEM allocation ncols must be a power of two");
  assert((state.last_allocation_ncols == 0 ||
          ncols <= state.last_allocation_ncols) &&
         "TCGen05 TMEM allocation ncols must not increase within a CTA");
  assert(state.next_base <= kTcgen05TmemColumnCount - ncols &&
         "TCGen05 TMEM allocation address overflow");

  uint32_t base = state.next_base;
  state.next_base += ncols;
  state.last_allocation_ncols = ncols;
  auto inserted = state.allocations.emplace(base, allocation_t{base, ncols});
  assert(inserted.second && "TCGen05 TMEM allocation base already exists");
  return base;
}

void tcgen05_tmem_manager_t::dealloc(const tcgen05_tmem_scope_t &scope,
                                     uint32_t base, uint32_t ncols) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  scope_state_t *state = &get_or_create_scope(scope);
  assert(ncols >= kTcgen05TmemMinAllocColumns &&
         ncols <= kTcgen05TmemMaxAllocColumns &&
         "TCGen05 TMEM deallocation ncols must be in [32, 512]");
  assert(is_power_of_two(ncols) &&
         "TCGen05 TMEM deallocation ncols must be a power of two");
  uint32_t base_column = tcgen05_decode_tmem_address(base).column;
  auto it = state->allocations.find(base_column);
  assert(it != state->allocations.end() &&
         "TCGen05 TMEM dealloc of unknown base");
  assert(it->second.ncols == ncols &&
         "TCGen05 TMEM dealloc size does not match allocation");

  // Functional simulation can replay warp-specialized code out of the real
  // hardware order. Keep retired ranges addressable until CTA teardown so
  // late-replayed TCGen05 ops do not fail after a producer warp deallocates.
  state->retired_allocations[base_column] = it->second;
  state->allocations.erase(it);
}

void tcgen05_tmem_manager_t::relinquish_alloc_permit(
    const tcgen05_tmem_scope_t &scope) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  get_or_create_scope(scope).permit_relinquished = true;
}

bool tcgen05_tmem_manager_t::permit_relinquished(
    const tcgen05_tmem_scope_t &scope) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  const scope_state_t *state = find_scope(scope);
  return state && state->permit_relinquished;
}

bool tcgen05_tmem_manager_t::has_allocation(const tcgen05_tmem_scope_t &scope,
                                            uint32_t base) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  const scope_state_t *state = find_scope(scope);
  uint32_t base_column = tcgen05_decode_tmem_address(base).column;
  return state &&
         state->allocations.find(base_column) != state->allocations.end();
}

bool tcgen05_tmem_manager_t::contains_range(const tcgen05_tmem_scope_t &scope,
                                            uint32_t address,
                                            uint32_t nwords) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  const scope_state_t *state = find_scope(scope);
  return state &&
         find_allocation_containing(*state, address, nwords) != nullptr;
}

unsigned tcgen05_tmem_manager_t::allocation_count(
    const tcgen05_tmem_scope_t &scope) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  const scope_state_t *state = find_scope(scope);
  return state ? state->allocations.size() : 0;
}

void tcgen05_tmem_manager_t::write_words(const tcgen05_tmem_scope_t &scope,
                                         uint32_t address,
                                         const std::vector<uint32_t> &values) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  assert(!values.empty() && "TCGen05 TMEM write must be non-empty");
  scope_state_t &state = get_or_create_scope(scope);
  assert(find_accessible_allocation_containing(state, address, values.size()) !=
             nullptr &&
         "TCGen05 TMEM write outside allocation");

  tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
  for (uint32_t i = 0; i < values.size(); ++i) {
    state.words[tcgen05_encode_tmem_address(decoded.lane, decoded.column + i)] =
        values[i];
  }
}

std::vector<uint32_t>
tcgen05_tmem_manager_t::read_words(const tcgen05_tmem_scope_t &scope,
                                   uint32_t address, uint32_t nwords) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  assert(nwords > 0 && "TCGen05 TMEM read must be non-empty");
  const scope_state_t *state = find_scope(scope);
  assert(state && "TCGen05 TMEM read from unknown scope");
  assert(find_accessible_allocation_containing(*state, address, nwords) !=
             nullptr &&
         "TCGen05 TMEM read outside allocation");

  std::vector<uint32_t> result(nwords, 0);
  tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
  for (uint32_t i = 0; i < nwords; ++i) {
    auto it = state->words.find(
        tcgen05_encode_tmem_address(decoded.lane, decoded.column + i));
    if (it != state->words.end())
      result[i] = it->second;
  }
  return result;
}

void tcgen05_tmem_manager_t::write_matrix_words(
    const tcgen05_tmem_scope_t &scope, uint32_t address,
    const std::vector<uint32_t> &values, uint32_t rows, uint32_t columns) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  assert(rows > 0 && columns > 0 &&
         "TCGen05 TMEM matrix write must be non-empty");
  assert(values.size() == static_cast<size_t>(rows) * columns &&
         "TCGen05 TMEM matrix write size does not match shape");
  scope_state_t &state = get_or_create_scope(scope);
  assert(find_accessible_allocation_containing(state, address, columns) !=
             nullptr &&
         "TCGen05 TMEM matrix write outside allocation");

  tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      state.words[tcgen05_encode_tmem_address(decoded.lane + row,
                                              decoded.column + column)] =
          values[row * columns + column];
    }
  }
}

std::vector<uint32_t>
tcgen05_tmem_manager_t::read_matrix_words(const tcgen05_tmem_scope_t &scope,
                                          uint32_t address, uint32_t rows,
                                          uint32_t columns) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  assert(rows > 0 && columns > 0 &&
         "TCGen05 TMEM matrix read must be non-empty");
  const scope_state_t *state = find_scope(scope);
  assert(state && "TCGen05 TMEM matrix read from unknown scope");
  assert(find_accessible_allocation_containing(*state, address, columns) !=
             nullptr &&
         "TCGen05 TMEM matrix read outside allocation");

  std::vector<uint32_t> result(rows * columns, 0);
  tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      auto it = state->words.find(tcgen05_encode_tmem_address(
          decoded.lane + row, decoded.column + column));
      if (it != state->words.end())
        result[row * columns + column] = it->second;
    }
  }
  return result;
}

std::vector<uint16_t> tcgen05_tmem_manager_t::read_matrix_packed_u16(
    const tcgen05_tmem_scope_t &scope, uint32_t address, uint32_t rows,
    uint32_t columns) const {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  assert(rows > 0 && columns > 0 &&
         "TCGen05 TMEM packed-u16 matrix read must be non-empty");

  uint32_t word_columns = (columns + 1) / 2;
  const scope_state_t *state = find_scope(scope);
  assert(state && "TCGen05 TMEM matrix read from unknown scope");
  assert(find_accessible_allocation_containing(*state, address, word_columns) !=
             nullptr &&
         "TCGen05 TMEM matrix read outside allocation");

  std::vector<uint32_t> words(rows * word_columns, 0);
  tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < word_columns; ++column) {
      auto it = state->words.find(tcgen05_encode_tmem_address(
          decoded.lane + row, decoded.column + column));
      if (it != state->words.end())
        words[row * word_columns + column] = it->second;
    }
  }

  std::vector<uint16_t> result(rows * columns, 0);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      uint32_t word = words[row * word_columns + column / 2];
      result[row * columns + column] = static_cast<uint16_t>(
          (column & 0x1) ? (word >> 16) : (word & 0xffff));
    }
  }
  return result;
}

std::vector<uint8_t> tcgen05_tmem_manager_t::read_mxf4_scale_matrix(
    const tcgen05_tmem_scope_t &scope, uint32_t address, uint32_t logical_rows,
    uint32_t scales_per_row, uint8_t scale_factor_id) const {
  constexpr uint32_t kTmemDataPaths = 128;
  constexpr uint32_t kDataPathsPerSubpartition = 32;
  constexpr uint32_t kScaleBytesPerWord = 4;
  assert(logical_rows > 0 && scales_per_row > 0 &&
         "TCGen05 MXFP4 scale matrix read must be non-empty");
  // SFA_ID/SFB_ID are instruction-descriptor fields that select a byte
  // sub-column within each TMEM word.  The PTX scale-address operand names
  // the containing TMEM matrix; it does not have to repeat the selected ID
  // in its two most-significant bits.  CUTLASS can derive the descriptor ID
  // from those bits as a convenience, while DeepGEMM deliberately keeps the
  // address fixed and changes the descriptor ID for consecutive K slices.
  assert(scale_factor_id + scales_per_row <= kScaleBytesPerWord &&
         "TCGen05 MXFP4 scale factors cross a TMEM word");

  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  const scope_state_t *state = find_scope(scope);
  assert(state && "TCGen05 MXFP4 scale read from unknown scope");
  tcgen05_tmem_address_t base =
      tcgen05_decode_tmem_address(address & 0x00ffffff);
  uint32_t word_columns = (logical_rows + kDataPathsPerSubpartition - 1) /
                          kDataPathsPerSubpartition;
  assert(find_accessible_allocation_containing(
             *state, tcgen05_encode_tmem_address(base.lane, base.column),
             word_columns) != nullptr &&
         "TCGen05 MXFP4 scale read outside allocation");

  std::vector<uint8_t> result(logical_rows * scales_per_row, 0);
  for (uint32_t row = 0; row < logical_rows; ++row) {
    // Compact scale-factor rows use 32 data paths per subpartition. Rows
    // beyond the first 32 advance the column; warpx4 leaves equivalent
    // replicas in the other three subpartitions.
    uint32_t data_path = base.lane + row % kDataPathsPerSubpartition;
    assert(data_path < kTmemDataPaths &&
           "TCGen05 MXFP4 scale TMEM data path is out of range");
    uint32_t column = base.column + row / kDataPathsPerSubpartition;
    uint32_t word = 0;
    auto it = state->words.find(tcgen05_encode_tmem_address(data_path, column));
    if (it != state->words.end())
      word = it->second;
    for (uint32_t scale = 0; scale < scales_per_row; ++scale) {
      uint32_t byte = scale_factor_id + scale;
      result[row * scales_per_row + scale] =
          static_cast<uint8_t>((word >> (byte * 8)) & 0xff);
    }
  }
  return result;
}

void tcgen05_tmem_manager_t::clear_cta(unsigned sm_id, unsigned cta_id) {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  for (auto it = m_scopes.begin(); it != m_scopes.end();) {
    if (it->first.sm_id == sm_id && it->first.cta_id == cta_id) {
      it = m_scopes.erase(it);
    } else {
      ++it;
    }
  }
}

void tcgen05_tmem_manager_t::reset() {
  std::lock_guard<tcgen05_tmem_spinlock_t> lock(m_mutex);
  m_scopes.clear();
}

tcgen05_tmem_manager_t::scope_state_t &
tcgen05_tmem_manager_t::get_or_create_scope(const tcgen05_tmem_scope_t &scope) {
  return m_scopes[scope];
}

const tcgen05_tmem_manager_t::scope_state_t *
tcgen05_tmem_manager_t::find_scope(const tcgen05_tmem_scope_t &scope) const {
  auto it = m_scopes.find(scope);
  return it == m_scopes.end() ? nullptr : &it->second;
}

const tcgen05_tmem_manager_t::allocation_t *
tcgen05_tmem_manager_t::find_allocation_containing(const scope_state_t &state,
                                                   uint32_t address,
                                                   uint32_t nwords) const {
  if (nwords == 0)
    return nullptr;
  uint64_t begin = tcgen05_decode_tmem_address(address).column;
  uint64_t end = begin + nwords;
  if (end > kTcgen05TmemColumnCount)
    return nullptr;

  for (const auto &entry : state.allocations) {
    const allocation_t &allocation = entry.second;
    uint64_t alloc_begin = allocation.base;
    uint64_t alloc_end = alloc_begin + allocation.ncols;
    if (begin >= alloc_begin && end <= alloc_end)
      return &allocation;
  }
  return nullptr;
}

const tcgen05_tmem_manager_t::allocation_t *
tcgen05_tmem_manager_t::find_accessible_allocation_containing(
    const scope_state_t &state, uint32_t address, uint32_t nwords) const {
  const allocation_t *allocation =
      find_allocation_containing(state, address, nwords);
  if (allocation)
    return allocation;

  if (nwords == 0)
    return nullptr;
  uint64_t begin = tcgen05_decode_tmem_address(address).column;
  uint64_t end = begin + nwords;
  if (end > kTcgen05TmemColumnCount)
    return nullptr;

  for (const auto &entry : state.retired_allocations) {
    const allocation_t &retired = entry.second;
    uint64_t alloc_begin = retired.base;
    uint64_t alloc_end = alloc_begin + retired.ncols;
    if (begin >= alloc_begin && end <= alloc_end)
      return &retired;
  }
  return nullptr;
}

} // namespace flash_gpgpu_sim
