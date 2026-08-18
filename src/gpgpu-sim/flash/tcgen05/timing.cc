#include "timing.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace flash_gpgpu_sim {

uint64_t tcgen05_mxf4_dense_work(uint32_t m, uint32_t n, uint32_t k) {
  return 2ull * m * n * k;
}

unsigned tcgen05_mxf4_compute_cycles(uint32_t m, uint32_t n, uint32_t k,
                                     unsigned work_per_sm_cycle) {
  assert(m > 0 && n > 0 && k > 0);
  assert(work_per_sm_cycle > 0);
  const uint64_t work = tcgen05_mxf4_dense_work(m, n, k);
  const uint64_t cycles =
      (work + static_cast<uint64_t>(work_per_sm_cycle) - 1) / work_per_sm_cycle;
  assert(cycles <= UINT_MAX && "TCGen05 MXFP4 latency exceeds unsigned range");
  return std::max(1u, static_cast<unsigned>(cycles));
}

unsigned tcgen05_parse_mxf4_throughput(const char *value) {
  if (value == nullptr || value[0] == '\0') {
    fprintf(stderr,
            "GPGPU-Sim: ERROR missing TCGen05 MXFP4 compute throughput\n");
    std::abort();
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > UINT_MAX) {
    fprintf(stderr,
            "GPGPU-Sim: ERROR invalid TCGen05 MXFP4 compute throughput '%s'\n",
            value);
    std::abort();
  }
  return static_cast<unsigned>(parsed);
}

void tcgen05_timing_model_t::reserve(unsigned long long cycle,
                                     unsigned compute_cycles) {
  assert(compute_cycles > 0);
  assert(can_issue(cycle) && "TCGen05 backend reserved while busy");
  m_backend_ready_cycle = cycle + compute_cycles;
}

} // namespace flash_gpgpu_sim
