#ifndef FLASH_GPGPU_SIM_TCGEN05_H
#define FLASH_GPGPU_SIM_TCGEN05_H

#include <cstdint>
#include <map>
#include <vector>

namespace flash_gpgpu_sim {

enum tcgen05_mma_type_field_t {
  TCGEN05_MMA_TYPE_FIELD_F16 = 0,
  TCGEN05_MMA_TYPE_FIELD_ONE = 1,
};

struct tcgen05_shared_descriptor_t {
  uint32_t start_address;
  uint32_t leading_dimension_byte_offset;
  uint32_t stride_dimension_byte_offset;
  uint8_t fixed_constant;
  uint8_t base_offset;
  bool leading_dimension_absolute;
  uint8_t swizzle_mode;
};

struct tcgen05_mma_descriptor_t {
  bool sparse;
  uint8_t sparsity_selector;
  uint8_t d_type;
  uint8_t a_type;
  uint8_t b_type;
  bool negate_a;
  bool negate_b;
  bool transpose_a;
  bool transpose_b;
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

struct tcgen05_tmem_address_t {
  uint32_t lane;
  uint32_t column;
};

tcgen05_shared_descriptor_t tcgen05_decode_shared_descriptor(uint64_t desc);
tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group);
tcgen05_tmem_address_t tcgen05_decode_tmem_address(uint32_t address);
uint32_t tcgen05_encode_tmem_address(uint32_t lane, uint32_t column);

float tcgen05_f16_to_f32(uint16_t f16);
float tcgen05_bf16_to_f32(uint16_t bf16);
uint16_t tcgen05_f32_to_f16(float f32);
uint32_t tcgen05_f32_to_bits(float value);
float tcgen05_bits_to_f32(uint32_t value);

std::vector<uint32_t> tcgen05_mma_f16_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint16_t> &a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input_d,
    bool enable_input_d);

struct tcgen05_tmem_scope_t {
  unsigned sm_id;
  unsigned cta_id;
  unsigned cta_group;

  bool operator<(const tcgen05_tmem_scope_t &other) const {
    if (sm_id != other.sm_id)
      return sm_id < other.sm_id;
    if (cta_id != other.cta_id)
      return cta_id < other.cta_id;
    return cta_group < other.cta_group;
  }
};

class tcgen05_tmem_manager_t {
public:
  struct allocation_t {
    uint32_t base;
    uint32_t ncols;
  };

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

  void clear_cta(unsigned sm_id, unsigned cta_id);
  void reset();

private:
  struct scope_state_t {
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
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_H
