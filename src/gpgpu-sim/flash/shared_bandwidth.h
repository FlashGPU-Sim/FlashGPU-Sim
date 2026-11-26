#ifndef FLASH_SHARED_BANDWIDTH_H
#define FLASH_SHARED_BANDWIDTH_H
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>

namespace flash_gpgpu_sim {
// First-order throughput model, not a claim about physical SRAM bank wiring.
// Addresses are CTA-local shared offsets; physical inter-CTA placement is not
// modeled. Each operation reserves both its source path and destination ports.
struct shared_bandwidth_t {
  using bytes_t = std::array<unsigned, 2>;
  enum path_t { LSU = 0, TMA = 1, TC = 2 };
  unsigned paths[3] = {};
  unsigned ports[2][2] = {}; // [read/write][low/high]
  void reset() { *this = shared_bandwidth_t(); }
  static bytes_t split(uint64_t address, unsigned bytes, unsigned half) {
    assert(half);
    bytes_t result{};
    while (bytes) {
      unsigned n = std::min<uint64_t>(bytes, half - address % half);
      result[(address / half) % 2] += n;
      address += n;
      bytes -= n;
    }
    return result;
  }
  static unsigned remaining(unsigned capacity, unsigned used) {
    return capacity == 0 ? ~0u : (used >= capacity ? 0 : capacity - used);
  }
  bool available(path_t path, bool write, const bytes_t &bytes,
                 unsigned path_capacity, unsigned bank_capacity) const {
    return uint64_t(bytes[0]) + bytes[1] <=
               remaining(path_capacity, paths[path]) &&
           bytes[0] <= remaining(bank_capacity, ports[write][0]) &&
           bytes[1] <= remaining(bank_capacity, ports[write][1]);
  }
  bool consume(path_t path, bool write, const bytes_t &bytes,
               unsigned path_capacity, unsigned bank_capacity) {
    if (!available(path, write, bytes, path_capacity, bank_capacity))
      return false;
    paths[path] += bytes[0] + bytes[1];
    for (unsigned i = 0; i < 2; ++i)
      ports[write][i] += bytes[i];
    return true;
  }
  // Partial service keeps the unserved bank's debt with the caller.
  unsigned consume_up_to(path_t path, bool write, unsigned bank, unsigned bytes,
                         unsigned path_capacity, unsigned bank_capacity) {
    unsigned n =
        std::min(bytes, std::min(remaining(path_capacity, paths[path]),
                                 remaining(bank_capacity, ports[write][bank])));
    paths[path] += n;
    ports[write][bank] += n;
    return n;
  }
};
// Future service reservations for TC reads and (in a separate instance) TMA
// writes. These clients receive first service; LSU spends the remaining budget.
// Reserve only future cycles: issue/response handling runs after the current
// cycle's port reset. The result is the first cycle after all bytes are served.
class tc_shared_calendar_t {
public:
  using cycle_t = unsigned long long;
  using bytes_t = shared_bandwidth_t::bytes_t;
  cycle_t reserve(cycle_t start, bytes_t remaining, unsigned path_capacity,
                  unsigned bank_capacity) {
    cycle_t cycle = start;
    while (remaining[0] || remaining[1]) {
      auto &used = slots[cycle];
      unsigned path_left =
          shared_bandwidth_t::remaining(path_capacity, used[0] + used[1]);
      for (unsigned half = 0; half < 2; ++half) {
        const unsigned grant = std::min(
            remaining[half],
            std::min(path_left,
                     shared_bandwidth_t::remaining(bank_capacity, used[half])));
        used[half] += grant;
        remaining[half] -= grant;
        path_left -= grant;
      }
      ++cycle;
    }
    return cycle;
  }
  bytes_t take(cycle_t cycle) {
    assert(slots.empty() || slots.begin()->first >= cycle);
    const auto it = slots.find(cycle);
    if (it == slots.end())
      return {};
    const auto bytes = it->second;
    slots.erase(it);
    return bytes;
  }
  void reset() { slots.clear(); }

private:
  std::map<cycle_t, bytes_t> slots;
};
} // namespace flash_gpgpu_sim
#endif
