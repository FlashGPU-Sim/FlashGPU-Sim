#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

namespace flash_gpgpu_sim {

// One budget per core tick, shared by ordinary and clustered launches.
// Zero limits preserve unrestricted upstream admission. No per-GPC allocation
// is needed in that default mode.
class cta_dispatch_budget {
public:
  cta_dispatch_budget(unsigned device_limit, unsigned gpc_limit, unsigned gpcs)
      : device_limit_(device_limit), gpc_limit_(gpc_limit),
        gpc_used_(gpc_limit ? gpcs : 0, 0) {}

  unsigned available(unsigned gpc) const {
    const auto unlimited = std::numeric_limits<unsigned>::max();
    return std::min(device_limit_ ? device_limit_ - device_used_ : unlimited,
                    gpc_limit_ ? gpc_limit_ - gpc_used_.at(gpc) : unlimited);
  }
  void consume(unsigned gpc, unsigned count) {
    assert(count <= available(gpc));
    if (device_limit_)
      device_used_ += count;
    if (gpc_limit_)
      gpc_used_.at(gpc) += count;
  }

private:
  unsigned device_limit_, gpc_limit_, device_used_ = 0;
  std::vector<unsigned> gpc_used_;
};

} // namespace flash_gpgpu_sim
