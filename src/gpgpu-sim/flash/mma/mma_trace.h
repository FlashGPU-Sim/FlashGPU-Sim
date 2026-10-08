// Copyright (c) 2026 University of British Columbia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FLASH_MMA_TRACE_H
#define FLASH_MMA_TRACE_H

#include <string>

#include "../../../trace.h"

class core_t;
class gpgpu_sim;
class warp_inst_t;

namespace flash_gpgpu_sim {

bool mma_trace_enabled(core_t *core, const warp_inst_t &inst);

// Assemble fragment and matrix values into complete, core-scoped trace lines.
class mma_trace_record {
public:
  mma_trace_record(core_t *core, const warp_inst_t &inst);
  ~mma_trace_record();
  void set_lane(unsigned lane) { m_lane = lane; }
  void append(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

private:
  void emit(const std::string &line) const;
  gpgpu_sim *m_gpu;
  unsigned m_sid;
  unsigned m_warp;
  unsigned long long m_pc;
  int m_lane = -1;
  std::string m_pending;
};

} // namespace flash_gpgpu_sim

#if TRACING_ON
#define MMA_TRACE(core, inst, ...)                                             \
  do {                                                                         \
    if (flash_gpgpu_sim::mma_trace_enabled(core, inst)) {                      \
      flash_gpgpu_sim::mma_trace_record mma_trace(core, inst);                 \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)
#else
#define MMA_TRACE(core, inst, ...)                                             \
  do {                                                                         \
  } while (0)
#endif

#define MMA_TRACE_LANE(core, inst, lane, ...)                                  \
  MMA_TRACE(core, inst, {                                                      \
    mma_trace.set_lane(lane);                                                  \
    __VA_ARGS__;                                                               \
  })

#endif
