#ifndef FLASH_GPGPU_SIM_SHARED_BULK_H
#define FLASH_GPGPU_SIM_SHARED_BULK_H
#include <cstdint>
#include <limits>
#include <vector>
namespace flash_gpgpu_sim {
// PTX st.bulk / native UMEMSETS normalize their units before entering here.
inline bool valid_shared_bulk_extent(uint64_t address, uint64_t bytes) {
  return address % 8 == 0 && bytes % 8 == 0 && bytes <= 16777216 &&
         address <= std::numeric_limits<uint64_t>::max() - bytes;
}
template <class Writer>
bool zero_shared_bulk(uint64_t address, uint64_t bytes, Writer write) {
  if (!valid_shared_bulk_extent(address, bytes))
    return false;
  if (!bytes)
    return true;
  const std::vector<uint8_t> zeros(bytes, 0);
  return write(address, zeros.data(), zeros.size());
}
} // namespace flash_gpgpu_sim
#endif
