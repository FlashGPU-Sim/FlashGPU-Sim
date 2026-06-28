#include "tcgen05.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace flash_gpgpu_sim {

namespace {

constexpr uint32_t kTcgen05DescriptorMask = 0x3fff;
constexpr uint32_t kTcgen05TmemColumnMask = 0xffff;
constexpr uint32_t kTcgen05TmemColumnCount = kTcgen05TmemColumnMask + 1;
constexpr uint32_t kTcgen05TmemMinAllocColumns = 32;
constexpr uint32_t kTcgen05TmemMaxAllocColumns = 512;
constexpr uint32_t kTcgen05TmemLaneMask = 0xffff;
constexpr unsigned kTcgen05TmemLaneShift = 16;

bool tcgen05_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

uint32_t tcgen05_decode_matrix_descriptor_field(uint64_t desc, unsigned shift) {
  return static_cast<uint32_t>(((desc >> shift) & kTcgen05DescriptorMask) << 4);
}

float tcgen05_input_to_f32(uint16_t value, uint8_t type) {
  switch (type) {
  case TCGEN05_MMA_TYPE_FIELD_F16:
    return tcgen05_f16_to_f32(value);
  case TCGEN05_MMA_TYPE_FIELD_ONE:
    return tcgen05_bf16_to_f32(value);
  default:
    assert(false && "Unsupported TCGen05 f16 MMA input data type");
    return 0.0f;
  }
}

} // namespace

tcgen05_shared_descriptor_t tcgen05_decode_shared_descriptor(uint64_t desc) {
  tcgen05_shared_descriptor_t decoded;
  decoded.start_address = tcgen05_decode_matrix_descriptor_field(desc, 0);
  decoded.leading_dimension_byte_offset =
      tcgen05_decode_matrix_descriptor_field(desc, 16);
  decoded.stride_dimension_byte_offset =
      tcgen05_decode_matrix_descriptor_field(desc, 32);
  decoded.fixed_constant = static_cast<uint8_t>((desc >> 46) & 0x7);
  decoded.base_offset = static_cast<uint8_t>((desc >> 49) & 0x7);
  decoded.leading_dimension_absolute = ((desc >> 52) & 0x1) != 0;
  decoded.swizzle_mode = static_cast<uint8_t>((desc >> 61) & 0x7);

  assert(decoded.fixed_constant == 1 &&
         "TCGen05 shared descriptor fixed bits must be 0b001");
  assert((decoded.swizzle_mode == 0 || decoded.swizzle_mode == 1 ||
          decoded.swizzle_mode == 2 || decoded.swizzle_mode == 4 ||
          decoded.swizzle_mode == 6) &&
         "TCGen05 shared descriptor swizzle mode is invalid");
  return decoded;
}

tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group) {
  tcgen05_mma_descriptor_t decoded;
  decoded.sparsity_selector = idesc & 0x3;
  decoded.sparse = ((idesc >> 2) & 0x1) != 0;
  decoded.d_type = static_cast<uint8_t>((idesc >> 4) & 0x3);
  decoded.a_type = static_cast<uint8_t>((idesc >> 7) & 0x7);
  decoded.b_type = static_cast<uint8_t>((idesc >> 10) & 0x7);
  decoded.negate_a = ((idesc >> 13) & 0x1) != 0;
  decoded.negate_b = ((idesc >> 14) & 0x1) != 0;
  decoded.transpose_a = ((idesc >> 15) & 0x1) != 0;
  decoded.transpose_b = ((idesc >> 16) & 0x1) != 0;
  decoded.n = ((idesc >> 17) & 0x3f) << 3;
  decoded.m = ((idesc >> 24) & 0x1f) << 4;
  decoded.k = decoded.sparse ? 32 : 16;

  assert(cta_group == 1 && "Only TCGen05 cta_group::1 is supported");
  assert(!decoded.sparse && "Sparse TCGen05 f16 MMA is not supported");
  assert(((idesc >> 3) & 0x1) == 0 &&
         "TCGen05 f16 MMA saturate bit must be zero");
  assert(((idesc >> 6) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 6 must be zero");
  assert(((idesc >> 23) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 23 must be zero");
  assert(((idesc >> 29) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 29 must be zero");
  assert(decoded.d_type == TCGEN05_MMA_TYPE_FIELD_ONE &&
         "Only TCGen05 f16 MMA with f32 D is supported");
  assert((decoded.a_type == TCGEN05_MMA_TYPE_FIELD_F16 ||
          decoded.a_type == TCGEN05_MMA_TYPE_FIELD_ONE) &&
         "Only TCGen05 f16/BF16 A input is supported");
  assert((decoded.b_type == TCGEN05_MMA_TYPE_FIELD_F16 ||
          decoded.b_type == TCGEN05_MMA_TYPE_FIELD_ONE) &&
         "Only TCGen05 f16/BF16 B input is supported");
  assert((decoded.m == 64 || decoded.m == 128) &&
         "TCGen05 f16 dense cta_group::1 supports M=64 or M=128");
  assert(decoded.n >= 8 && decoded.n <= 256 && decoded.n % 8 == 0 &&
         "TCGen05 f16 dense cta_group::1 supports N in steps of 8");
  return decoded;
}

float tcgen05_f16_to_f32(uint16_t f16) {
  uint32_t sign = (f16 >> 15) & 0x1;
  uint32_t exp = (f16 >> 10) & 0x1f;
  uint32_t frac = f16 & 0x3ff;

  if (exp == 0) {
    if (frac == 0)
      return sign ? -0.0f : 0.0f;
    float result = frac / 1024.0f / 16384.0f;
    return sign ? -result : result;
  }
  if (exp == 31)
    return frac ? NAN : (sign ? -INFINITY : INFINITY);

  uint32_t f32_exp = exp - 15 + 127;
  uint32_t f32_frac = frac << 13;
  uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;
  return tcgen05_bits_to_f32(f32_bits);
}

float tcgen05_bf16_to_f32(uint16_t bf16) {
  return tcgen05_bits_to_f32(static_cast<uint32_t>(bf16) << 16);
}

uint16_t tcgen05_f32_to_f16(float f32) {
  uint32_t f32_bits = tcgen05_f32_to_bits(f32);
  uint32_t sign = (f32_bits >> 31) & 0x1;
  int32_t exp = ((f32_bits >> 23) & 0xff) - 127 + 15;
  uint32_t frac = (f32_bits >> 13) & 0x3ff;

  if (exp <= 0)
    return sign << 15;
  if (exp >= 31)
    return (sign << 15) | 0x7c00;

  return static_cast<uint16_t>((sign << 15) | (exp << 10) | frac);
}

uint32_t tcgen05_f32_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float tcgen05_bits_to_f32(uint32_t value) {
  float bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<uint32_t> tcgen05_mma_f16_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint16_t> &a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input_d,
    bool enable_input_d) {
  assert(desc.d_type == TCGEN05_MMA_TYPE_FIELD_ONE &&
         "Only f32 TCGen05 MMA outputs are supported");
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MMA A matrix size does not match descriptor");
  assert(b.size() == desc.k * desc.n &&
         "TCGen05 MMA B matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MMA D matrix size does not match descriptor");

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum = enable_input_d
                      ? tcgen05_bits_to_f32(input_d[row * desc.n + col])
                      : 0.0f;
      for (uint32_t k = 0; k < desc.k; ++k) {
        uint32_t a_index =
            desc.transpose_a ? (k * desc.m + row) : (row * desc.k + k);
        uint32_t b_index =
            desc.transpose_b ? (col * desc.k + k) : (k * desc.n + col);
        float a_value = tcgen05_input_to_f32(a[a_index], desc.a_type);
        float b_value = tcgen05_input_to_f32(b[b_index], desc.b_type);
        if (desc.negate_a)
          a_value = -a_value;
        if (desc.negate_b)
          b_value = -b_value;
        sum += a_value * b_value;
      }
      output[row * desc.n + col] = tcgen05_f32_to_bits(sum);
    }
  }
  return output;
}

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

uint32_t tcgen05_tmem_manager_t::alloc(const tcgen05_tmem_scope_t &scope,
                                       uint32_t ncols) {
  scope_state_t &state = get_or_create_scope(scope);
  assert(!state.permit_relinquished &&
         "TCGen05 TMEM alloc after relinquish_alloc_permit is illegal");
  assert(ncols >= kTcgen05TmemMinAllocColumns &&
         ncols <= kTcgen05TmemMaxAllocColumns &&
         "TCGen05 TMEM allocation ncols must be in [32, 512]");
  assert(tcgen05_is_power_of_two(ncols) &&
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
  scope_state_t *state = &get_or_create_scope(scope);
  assert(ncols >= kTcgen05TmemMinAllocColumns &&
         ncols <= kTcgen05TmemMaxAllocColumns &&
         "TCGen05 TMEM deallocation ncols must be in [32, 512]");
  assert(tcgen05_is_power_of_two(ncols) &&
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
  get_or_create_scope(scope).permit_relinquished = true;
}

bool tcgen05_tmem_manager_t::permit_relinquished(
    const tcgen05_tmem_scope_t &scope) const {
  const scope_state_t *state = find_scope(scope);
  return state && state->permit_relinquished;
}

bool tcgen05_tmem_manager_t::has_allocation(const tcgen05_tmem_scope_t &scope,
                                            uint32_t base) const {
  const scope_state_t *state = find_scope(scope);
  uint32_t base_column = tcgen05_decode_tmem_address(base).column;
  return state &&
         state->allocations.find(base_column) != state->allocations.end();
}

bool tcgen05_tmem_manager_t::contains_range(const tcgen05_tmem_scope_t &scope,
                                            uint32_t address,
                                            uint32_t nwords) const {
  const scope_state_t *state = find_scope(scope);
  return state &&
         find_allocation_containing(*state, address, nwords) != nullptr;
}

unsigned tcgen05_tmem_manager_t::allocation_count(
    const tcgen05_tmem_scope_t &scope) const {
  const scope_state_t *state = find_scope(scope);
  return state ? state->allocations.size() : 0;
}

void tcgen05_tmem_manager_t::write_words(const tcgen05_tmem_scope_t &scope,
                                         uint32_t address,
                                         const std::vector<uint32_t> &values) {
  assert(!values.empty() && "TCGen05 TMEM write must be non-empty");
  scope_state_t &state = get_or_create_scope(scope);
  assert(find_accessible_allocation_containing(state, address, values.size()) !=
             nullptr &&
         "TCGen05 TMEM write outside allocation");

  for (uint32_t i = 0; i < values.size(); ++i) {
    tcgen05_tmem_address_t decoded = tcgen05_decode_tmem_address(address);
    state.words[tcgen05_encode_tmem_address(decoded.lane, decoded.column + i)] =
        values[i];
  }
}

std::vector<uint32_t>
tcgen05_tmem_manager_t::read_words(const tcgen05_tmem_scope_t &scope,
                                   uint32_t address, uint32_t nwords) const {
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
  assert(rows > 0 && columns > 0 &&
         "TCGen05 TMEM packed-u16 matrix read must be non-empty");

  uint32_t word_columns = (columns + 1) / 2;
  std::vector<uint32_t> words =
      read_matrix_words(scope, address, rows, word_columns);
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

void tcgen05_tmem_manager_t::clear_cta(unsigned sm_id, unsigned cta_id) {
  for (auto it = m_scopes.begin(); it != m_scopes.end();) {
    if (it->first.sm_id == sm_id && it->first.cta_id == cta_id) {
      it = m_scopes.erase(it);
    } else {
      ++it;
    }
  }
}

void tcgen05_tmem_manager_t::reset() { m_scopes.clear(); }

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
