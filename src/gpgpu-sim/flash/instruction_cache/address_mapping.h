#ifndef FLASH_GPGPU_SIM_INSTRUCTION_CACHE_ADDRESS_MAPPING_H
#define FLASH_GPGPU_SIM_INSTRUCTION_CACHE_ADDRESS_MAPPING_H

#include <cassert>
#include <cstdint>
#include <limits>

namespace flash_gpgpu_sim {

struct instruction_fetch_mapping {
  uint64_t cache_address;
  unsigned cache_bytes;
  unsigned functional_bytes;
};

class instruction_address_mapper {
public:
  instruction_address_mapper(unsigned scale, unsigned line_size)
      : m_scale(scale), m_line_size(line_size) {
    assert(m_scale != 0);
    assert(m_line_size != 0);
    assert((m_line_size & (m_line_size - 1)) == 0);
    assert(m_line_size % m_scale == 0);
  }

  instruction_fetch_mapping map_fetch(uint64_t functional_pc,
                                      unsigned max_functional_bytes,
                                      uint64_t cache_base) const {
    assert(max_functional_bytes != 0);
    assert(functional_pc <=
           (std::numeric_limits<uint64_t>::max() - cache_base) / m_scale);
    const uint64_t cache_address = cache_base + functional_pc * m_scale;
    const unsigned offset = cache_address & (m_line_size - 1);
    const uint64_t requested_cache_bytes =
        static_cast<uint64_t>(max_functional_bytes) * m_scale;
    const unsigned cache_bytes = static_cast<unsigned>(
        requested_cache_bytes < m_line_size - offset ? requested_cache_bytes
                                                     : m_line_size - offset);
    assert(cache_bytes % m_scale == 0);
    return {cache_address, cache_bytes, cache_bytes / m_scale};
  }

  uint64_t functional_pc(uint64_t cache_address, uint64_t cache_base) const {
    assert(cache_address >= cache_base);
    const uint64_t offset = cache_address - cache_base;
    assert(offset % m_scale == 0);
    return offset / m_scale;
  }

  unsigned functional_bytes(unsigned cache_bytes) const {
    assert(cache_bytes % m_scale == 0);
    return cache_bytes / m_scale;
  }

private:
  unsigned m_scale;
  unsigned m_line_size;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_INSTRUCTION_CACHE_ADDRESS_MAPPING_H
