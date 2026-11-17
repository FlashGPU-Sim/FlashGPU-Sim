// Copyright (c) 2009-2013, Tor M. Aamodt, Timothy Rogers,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// This file is inspired by the trace system in gem5.
// This is a highly simplified version adpated for gpgpusim

#ifndef __TRACE_H__
#define __TRACE_H__

class mem_fetch;

namespace Trace {

#define TS_TUP_BEGIN(X) enum X {
#define TS_TUP(X) X
#define TS_TUP_END(X) \
  }                   \
  ;
#include "trace_streams.tup"
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END

extern bool enabled;
extern int sampling_core;
extern int sampling_memory_partition;
extern const char* trace_streams_str[];
extern bool trace_streams_enabled[NUM_TRACE_STREAMS];
extern const char* config_str;

void init();

void tma_transaction_event(unsigned long long cycle, unsigned sm_id,
                           const char *event, unsigned tx_uid, const char *kind,
                           unsigned tma_type, unsigned long long pc,
                           unsigned cta, unsigned warp, unsigned lane,
                           unsigned tid, unsigned long long src,
                           unsigned long long dst, unsigned size, unsigned mbar,
                           unsigned mf_uid, unsigned long long mf_addr,
                           unsigned mf_size, unsigned issued_mf,
                           unsigned received_mf, unsigned bytes_completed,
                           unsigned global_inflight, unsigned response_fifo);
void gem5_mem_fetch_event(unsigned long long tick, const char *event,
                          const mem_fetch *mf, unsigned input, unsigned output,
                          unsigned long long pending_queue);

void l2_request_event(const char *event, unsigned long long cycle,
                      unsigned subpart, const mem_fetch *mf,
                      const char *status);
bool request_noc_sample_accepts(const mem_fetch *mf);
void request_noc_packet_event(unsigned long long icnt_cycle, const char *event,
                              unsigned input, unsigned output, unsigned subpart,
                              unsigned requesters, unsigned queued_packets,
                              unsigned input_occupancy,
                              unsigned output_occupancy, const mem_fetch *mf,
                              unsigned packet_size, const char *status);
void request_noc_prearb_event(unsigned long long icnt_cycle, unsigned input,
                              unsigned output, unsigned subpart,
                              unsigned requesters, unsigned queued_packets,
                              const mem_fetch *mf);

const char *named_barrier_type_name(unsigned type);
const char *instruction_issue_op_name(unsigned op);
const char *instruction_issue_producer_name(unsigned producer);

}  // namespace Trace

#if TRACING_ON

#define SIM_PRINT_STR "GPGPU-Sim Cycle %llu: %s - "
#define GPTRACE(x) ((Trace::trace_streams_enabled[Trace::x]) && Trace::enabled)
// Core-scoped diagnostics for functional and timing components.
#define GPTRACE_CORE(x, sid)                    \
  (GPTRACE(x) && (Trace::sampling_core == -1 || \
                  Trace::sampling_core == static_cast<int>(sid)))
#define GPPRINTF_GPU_CORE(gpu, sid, x, ...)                              \
  do {                                                                   \
    if (GPTRACE_CORE(x, sid)) {                                          \
      flockfile(stdout);                                                 \
      printf(SIM_PRINT_STR "Core %d - ",                                 \
             (gpu)->gpu_sim_cycle + (gpu)->gpu_tot_sim_cycle,            \
             Trace::trace_streams_str[Trace::x], static_cast<int>(sid)); \
      printf(__VA_ARGS__);                                               \
      funlockfile(stdout);                                               \
    }                                                                    \
  } while (0)

#define GPPRINTF(x, ...)                                                     \
  do {                                                                       \
    if (GPTRACE(x)) {                                                        \
      flockfile(stdout);                                                     \
      printf(SIM_PRINT_STR, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);                            \
      printf(__VA_ARGS__);                                                   \
      funlockfile(stdout);                                                   \
    }                                                                        \
  } while (0)

#define GPPRINTF_GPU(m_gpu, x, fmt, ...)                                        \
  do {                                                                         \
    if (GPTRACE(x)) {                                                           \
      printf(SIM_PRINT_STR fmt,                                                \
             (m_gpu)->gpu_sim_cycle + (m_gpu)->gpu_tot_sim_cycle,              \
             Trace::trace_streams_str[Trace::x], __VA_ARGS__);                 \
    }                                                                          \
  } while (0)

#define GPPRINTF_THREAD(x, thread, fmt, ...)                                    \
  do {                                                                         \
    auto m_gpu = thread->get_gpu();                                            \
    auto flat_ctaid = thread->get_flat_ctaid();                                \
    auto flat_tid = thread->get_flat_tid();                                    \
    auto warp_size = thread->get_core()->get_warp_size();                      \
    auto warpid = flat_tid / warp_size;                                        \
    auto laneid = flat_tid % warp_size;                                        \
    auto sm_id = thread->get_hw_sid();                                         \
                                                                               \
    if (GPTRACE(x)) {                                                           \
      printf(SIM_PRINT_STR "[%3d,%3d,%3d] SM %3d " fmt,                               \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,                  \
             Trace::trace_streams_str[Trace::x], flat_ctaid, warpid, laneid, sm_id,   \
             __VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

#define GPPRINTF_NoGPU(x, ...)                                         \
  do {                                                                 \
    if (GPTRACE(x)) {                                                  \
      flockfile(stdout);                                               \
      printf(SIM_PRINT_STR, -1ll, Trace::trace_streams_str[Trace::x]); \
      printf(__VA_ARGS__);                                             \
      funlockfile(stdout);                                             \
    }                                                                  \
  } while (0)

#define GPPRINTFG(x, ...)                                       \
  do {                                                         \
    if (GPTRACE(x)) {                                           \
      flockfile(stdout);                                       \
      printf(SIM_PRINT_STR, gpu_sim_cycle + gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);              \
      printf(__VA_ARGS__);                                     \
      funlockfile(stdout);                                     \
    }                                                          \
  } while (0)

#else

#define GPTRACE(x) (false)
#define GPTRACE_CORE(x, sid) (false)
#define GPPRINTF_GPU_CORE(gpu, sid, x, ...) \
  do {                                      \
  } while (0)
#define GPPRINTF(x, ...) \
  do {                   \
  } while (0)
#define GPPRINTF_GPU(m_gpu, x, fmt, ...)                                        \
  do {                                                                         \
  } while (0)
#define GPPRINTF_THREAD(m_gpu, x, fmt, ...)                                     \
  do {                                                                         \
  } while (0)
#define GPPRINTF_NoGPU(x, ...) \
  do {                  \
  } while (0)
#define GPPRINTFG(x, ...) \
  do {                   \
  } while (0)

#endif

#define GPPRINTF_INST_EXEC(x, fmt, ...)                                         \
  GPPRINTF_THREAD(x, thread, fmt, __VA_ARGS__)

#define GPPRINTF_THREAD_CORE(x, thread, fmt, ...)                           \
  GPPRINTF_GPU_CORE(                                                        \
      (thread)->get_gpu(), (thread)->get_hw_sid(), x, "[%3d,%3d,%3d] " fmt, \
      (thread)->get_flat_ctaid(),                                           \
      (thread)->get_flat_tid() / (thread)->get_core()->get_warp_size(),     \
      (thread)->get_flat_tid() % (thread)->get_core()->get_warp_size(),     \
      __VA_ARGS__)

#define GPPRINTF_TMA(x, ...)                                             \
  GPPRINTF_GPU_CORE(m_shader_ctx->get_gpu(), m_shader_ctx->get_sid(), x, \
                    __VA_ARGS__)

#endif
