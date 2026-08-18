#ifndef FLASH_GPGPU_SIM_TCGEN05_TIMING_H
#define FLASH_GPGPU_SIM_TCGEN05_TIMING_H

#include <cstdint>

namespace flash_gpgpu_sim {

// Dense MXFP4 TCGen05 timing is expressed as useful floating-point work.  A
// multiply-add counts as two operations, matching NVIDIA's published FP4
// throughput convention and the existing WGMMA throughput model.
uint64_t tcgen05_mxf4_dense_work(uint32_t m, uint32_t n, uint32_t k);
unsigned tcgen05_mxf4_compute_cycles(uint32_t m, uint32_t n, uint32_t k,
                                     unsigned work_per_sm_cycle);
unsigned tcgen05_parse_mxf4_throughput(const char *value);

// kind::mxf8f6f4 covers all 25 ordered FP8/FP6/FP4 input combinations.  The
// useful-work convention is the same as strict mxf4, but hardware exposes it
// at the FP8-class rate rather than the dedicated 2x-faster strict-FP4 rate.
uint64_t tcgen05_mxf8f6f4_dense_work(uint32_t m, uint32_t n, uint32_t k);
unsigned tcgen05_mxf8f6f4_compute_cycles(uint32_t m, uint32_t n, uint32_t k,
                                         unsigned work_per_sm_cycle);
unsigned tcgen05_parse_mxf8f6f4_throughput(const char *value);

// TCGen05 is exposed through four scheduler-facing tensor pipes on Blackwell,
// but one instruction consumes a shared per-SM Tensor Core backend.  This
// small reservation model prevents four independently modelled frontends from
// multiplying the configured chip throughput by four.
class tcgen05_timing_model_t {
public:
  bool can_issue(unsigned long long cycle) const {
    return cycle >= m_backend_ready_cycle;
  }

  void reserve(unsigned long long cycle, unsigned compute_cycles);
  void reset() { m_backend_ready_cycle = 0; }
  unsigned long long backend_ready_cycle() const {
    return m_backend_ready_cycle;
  }

private:
  unsigned long long m_backend_ready_cycle = 0;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_TIMING_H
