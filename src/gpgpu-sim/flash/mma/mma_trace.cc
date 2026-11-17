// Copyright (c) 2026 University of British Columbia
// SPDX-License-Identifier: BSD-3-Clause

#include "mma_trace.h"

#include <cstdarg>
#include <cstdio>
#include <vector>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_sim.h"
#include "../../gpu-sim.h"

namespace flash_gpgpu_sim {
namespace {

unsigned mma_warp_id(core_t *core, const warp_inst_t &inst) {
  return core->get_gpu()->is_functional_sim() ? inst.warp_id_func()
                                              : inst.warp_id();
}

ptx_thread_info *mma_trace_thread(core_t *core, const warp_inst_t &inst) {
  const unsigned base = mma_warp_id(core, inst) * core->get_warp_size();
  for (unsigned lane = 0; lane < core->get_warp_size(); ++lane) {
    if (inst.active(lane) && core->get_thread_info()[base + lane])
      return core->get_thread_info()[base + lane];
  }
  return nullptr;
}

} // namespace

bool mma_trace_enabled(core_t *core, const warp_inst_t &inst) {
  if (!GPTRACE(MMA))
    return false;
  const auto *thread = mma_trace_thread(core, inst);
  return thread && GPTRACE_CORE(MMA, thread->get_hw_sid());
}

mma_trace_record::mma_trace_record(core_t *core, const warp_inst_t &inst)
    : m_gpu(core->get_gpu()), m_sid(mma_trace_thread(core, inst)->get_hw_sid()),
      m_warp(mma_warp_id(core, inst)), m_pc(inst.pc) {}

mma_trace_record::~mma_trace_record() {
  if (!m_pending.empty())
    emit(m_pending);
}

void mma_trace_record::emit(const std::string &line) const {
  if (line.empty())
    return;
  if (m_lane >= 0) {
    GPPRINTF_GPU_CORE(m_gpu, m_sid, MMA, "warp=%u pc=0x%llx lane=%d %s\n",
                      m_warp, m_pc, m_lane, line.c_str());
    return;
  }
  GPPRINTF_GPU_CORE(m_gpu, m_sid, MMA, "warp=%u pc=0x%llx %s\n", m_warp, m_pc,
                    line.c_str());
}

void mma_trace_record::append(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list copy;
  va_copy(copy, args);
  const int length = vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  if (length >= 0) {
    std::vector<char> buffer(static_cast<size_t>(length) + 1);
    vsnprintf(buffer.data(), buffer.size(), fmt, args);
    m_pending.append(buffer.data(), static_cast<size_t>(length));
  }
  va_end(args);
  size_t newline;
  while ((newline = m_pending.find('\n')) != std::string::npos) {
    emit(m_pending.substr(0, newline));
    m_pending.erase(0, newline + 1);
  }
}

} // namespace flash_gpgpu_sim
