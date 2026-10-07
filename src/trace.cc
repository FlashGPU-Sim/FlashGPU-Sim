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

#include "trace.h"
#include <algorithm>
#include <sstream>
#include <string>
#include "gpgpu-sim/mem_fetch.h"
#include "gpgpu-sim/scoreboard.h"

namespace Trace {

void tma_transaction_event(unsigned long long cycle, unsigned sm_id,
                           const char *event, unsigned tx_uid, const char *kind,
                           unsigned tma_type, unsigned long long pc,
                           unsigned cta, unsigned warp, unsigned lane,
                           unsigned tid, unsigned long long src,
                           unsigned long long dst, unsigned size, unsigned mbar,
                           unsigned mf_uid, unsigned long long mf_addr,
                           unsigned mf_size, unsigned issued_mf,
                           unsigned received_mf, unsigned bytes_completed,
                           unsigned global_inflight, unsigned response_fifo) {
#if TRACING_ON
  if (!GPTRACE_CORE(TMA, sm_id)) return;
  flockfile(stdout);
  printf(SIM_PRINT_STR
         "Core %u - TMA event=%s tx_uid=%u kind=%s tma_type=%u "
         "pc=0x%llx cta=%u warp=%u lane=%u tid=%u src=0x%llx dst=0x%llx "
         "size=%u mbar=0x%x mf_uid=%u mf_addr=0x%llx mf_size=%u "
         "issued_mf=%u received_mf=%u bytes_completed=%u global_inflight=%u "
         "response_fifo=%u\n",
         cycle, trace_streams_str[TMA], sm_id, event, tx_uid, kind, tma_type,
         pc, cta, warp, lane, tid, src, dst, size, mbar, mf_uid, mf_addr,
         mf_size, issued_mf, received_mf, bytes_completed, global_inflight,
         response_fifo);
  funlockfile(stdout);
#endif
}

void gem5_mem_fetch_event(unsigned long long tick, const char *event,
                          const mem_fetch *mf, unsigned input, unsigned output,
                          unsigned long long pending_queue) {
#if TRACING_ON
  if (!mf || !GPTRACE_CORE(INTERCONNECT, mf->get_sid())) return;
  flockfile(stdout);
  printf(
      "GPGPU-Sim: %s - Core %d - GEM5 gem5_tick=%llu event=%s mf_uid=%u "
      "type=%s addr=0x%llx access_size=%u data_size=%u is_write=%u "
      "input=%u output=%u pending_queue=%llu\n",
      trace_streams_str[INTERCONNECT], static_cast<int>(mf->get_sid()), tick,
      event, mf->get_request_uid(), mem_access_type_str(mf->get_access_type()),
      static_cast<unsigned long long>(mf->get_addr()), mf->get_access_size(),
      mf->get_data_size(), mf->get_is_write() ? 1u : 0u, input, output,
      pending_queue);
  funlockfile(stdout);
#endif
}

void l2_request_event(const char *event, unsigned long long cycle,
                      unsigned subpart, const mem_fetch *mf,
                      const char *status) {
#if TRACING_ON
  if (!mf || !GPTRACE_CORE(MEMORY_SUBPARTITION_UNIT, mf->get_sid()) ||
      (sampling_memory_partition != -1 &&
       sampling_memory_partition != static_cast<int>(subpart)))
    return;
  const mem_fetch *original = const_cast<mem_fetch *>(mf)->get_original_mf();
  const unsigned original_uid =
      original ? original->get_request_uid() : mf->get_request_uid();
  flockfile(stdout);
  printf(SIM_PRINT_STR
         "Core %d - L2 event=%s subpart=%u mf_subpart=%u "
         "addr=0x%llx partition_addr=0x%llx tpc=%u wid=%u uid=%u orig_uid=%u "
         "type=%s is_write=%u data_size=%u access_size=%u sector_mask=0x%lx "
         "status=%s\n",
         cycle, trace_streams_str[MEMORY_SUBPARTITION_UNIT],
         static_cast<int>(mf->get_sid()), event, subpart,
         mf->get_sub_partition_id(),
         static_cast<unsigned long long>(mf->get_addr()),
         static_cast<unsigned long long>(mf->get_partition_addr()),
         mf->get_tpc(), mf->get_wid(), mf->get_request_uid(), original_uid,
         mem_access_type_str(mf->get_access_type()), mf->get_is_write(),
         mf->get_data_size(), mf->get_access_size(),
         mf->get_access_sector_mask().to_ulong(), status);
  funlockfile(stdout);
#endif
}

bool request_noc_sample_accepts(const mem_fetch *mf) {
  return mf && GPTRACE_CORE(INTERCONNECT, mf->get_sid());
}

void request_noc_packet_event(unsigned long long icnt_cycle, const char *event,
                              unsigned input, unsigned output, unsigned subpart,
                              unsigned requesters, unsigned queued_packets,
                              unsigned input_occupancy,
                              unsigned output_occupancy, const mem_fetch *mf,
                              unsigned packet_size, const char *status) {
#if TRACING_ON
  if (!request_noc_sample_accepts(mf)) return;
  const mem_fetch *original = const_cast<mem_fetch *>(mf)->get_original_mf();
  const unsigned original_uid =
      original ? original->get_request_uid() : mf->get_request_uid();
  flockfile(stdout);
  printf(
      "GPGPU-Sim: %s - Core %d - NoC icnt_cycle=%llu gpu_push_cycle=%llu "
      "event=%s input=%u output=%u subpart=%u requesters=%u queued_pkts=%u "
      "in_occ=%u out_occ=%u addr=0x%llx partition_addr=0x%llx tpc=%u "
      "wid=%u uid=%u orig_uid=%u type=%s is_write=%u packet_size=%u "
      "data_size=%u access_size=%u sector_mask=0x%lx status=%s\n",
      trace_streams_str[INTERCONNECT], static_cast<int>(mf->get_sid()),
      icnt_cycle, mf->get_status_change(), event, input, output, subpart,
      requesters, queued_packets, input_occupancy, output_occupancy,
      static_cast<unsigned long long>(mf->get_addr()),
      static_cast<unsigned long long>(mf->get_partition_addr()), mf->get_tpc(),
      mf->get_wid(), mf->get_request_uid(), original_uid,
      mem_access_type_str(mf->get_access_type()), mf->get_is_write(),
      packet_size, mf->get_data_size(), mf->get_access_size(),
      mf->get_access_sector_mask().to_ulong(), status);
  funlockfile(stdout);
#endif
}

void request_noc_prearb_event(unsigned long long icnt_cycle, unsigned input,
                              unsigned output, unsigned subpart,
                              unsigned requesters, unsigned queued_packets,
                              const mem_fetch *mf) {
#if TRACING_ON
  if (requesters < 2) return;
  request_noc_packet_event(icnt_cycle, "PRE_ARB", input, output, subpart,
                           requesters, queued_packets, 0, 0, mf, 0,
                           "REQUESTERS");
#endif
}

const char *named_barrier_type_name(unsigned type) {
  switch (type) {
    case SYNC:
      return "sync";
    case ARRIVE:
      return "arrive";
    case RED:
      return "red";
    default:
      return "not_bar";
  }
}

#define TS_TUP_BEGIN(X) const char *trace_streams_str[] = {
#define TS_TUP(X) #X
#define TS_TUP_END(X) \
  }                   \
  ;
#include "trace_streams.tup"
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END

bool enabled = false;
int sampling_core = 0;
int sampling_memory_partition = -1;
bool trace_streams_enabled[NUM_TRACE_STREAMS] = {false};
const char *config_str;

void init() {
  std::fill_n(trace_streams_enabled, NUM_TRACE_STREAMS, false);
  std::istringstream components(config_str ? config_str : "");
  std::string component;
  while (std::getline(components, component, ',')) {
    const auto first = component.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) continue;
    const auto last = component.find_last_not_of(" \t\r\n");
    component = component.substr(first, last - first + 1);
    for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) {
      if (component == trace_streams_str[i]) {
        trace_streams_enabled[i] = true;
        break;
      }
    }
  }
}

const char *instruction_issue_op_name(unsigned op) {
  switch (op) {
    case SP_OP:
      return "SP";
    case SFU_OP:
      return "SFU";
    case ALU_SFU_OP:
      return "ALU_SFU";
    case DP_OP:
      return "DP";
    case INTP_OP:
      return "INT";
    case ALU_OP:
      return "ALU";
    case LOAD_OP:
      return "LD";
    case STORE_OP:
      return "ST";
    case MEMORY_BARRIER_OP:
      return "MEMBAR";
    case TENSOR_CORE_LOAD_OP:
      return "LDSM";
    case TENSOR_CORE_STORE_OP:
      return "STMATRIX";
    case TENSOR_CORE_OP:
      return "MMA";
    case TENSOR_MEMORY_ACCELERATOR_OP:
      return "TMA";
    case ASYNC_COPY_OP:
      return "CP_ASYNC";
    case TENSOR_MAP_OP:
      return "TENSOR_MAP";
    default:
      return "OTHER";
  }
}

const char *instruction_issue_producer_name(unsigned producer) {
  switch (producer) {
    case PROD_MEM_GLOBAL:
      return "MEM_GLOBAL";
    case PROD_MEM_SHARED:
      return "MEM_SHARED";
    case PROD_TENSOR_CORE:
      return "TENSOR";
    case PROD_SP_INT:
      return "SP_INT";
    case PROD_SFU:
      return "SFU";
    case PROD_TMA:
      return "TMA";
    case PROD_TENSOR_MAP:
      return "TENSOR_MAP";
    default:
      return "OTHER";
  }
}
}  // namespace Trace
