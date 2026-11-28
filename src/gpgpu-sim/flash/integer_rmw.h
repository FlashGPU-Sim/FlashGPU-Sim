#pragma once
#include <cstdint>
#include <mutex>
namespace flash_gpgpu_sim {
enum class integer_rmw_kind { add, bit_xor };
inline uint64_t integer_rmw(uint64_t old_value, uint64_t value,
                            integer_rmw_kind kind) {
  return kind == integer_rmw_kind::add ? old_value + value : old_value ^ value;
}
// One lock for read/modify/write, not one lock per individual memory call.
inline std::mutex &global_rmw_mutex() {
  static std::mutex mutex;
  return mutex;
}
// Adapters supply only address-space access. Keep the lock across both memory
// operations and expose the old bits for instructions with a returned value.
template <typename Read, typename Write>
bool apply_integer_rmw(uint64_t value, integer_rmw_kind kind, Read read,
                       Write write, uint64_t &old_value) {
  std::lock_guard<std::mutex> lock(global_rmw_mutex());
  old_value = 0;
  if (!read(old_value))
    return false;
  return write(integer_rmw(old_value, value, kind));
}
} // namespace flash_gpgpu_sim
