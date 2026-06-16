// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
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

#include <algorithm>
#include <limits.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>
#include <utility>

#include "local_interconnect.h"
#include "mem_fetch.h"

namespace {

class req_noc_trace {
 public:
  static req_noc_trace &instance() {
    static req_noc_trace trace;
    return trace;
  }

  bool enabled() const { return m_enabled; }
  bool trace_push() const { return m_enabled && m_trace_push; }
  bool trace_grant() const { return m_enabled && m_trace_grant; }
  bool trace_prearb() const { return m_enabled && m_trace_prearb; }
  unsigned min_requesters() const { return m_min_requesters; }
  bool accepts(const mem_fetch *mf) const {
    if (!mf) return false;
    const mem_access_type access_type = mf->get_access_type();
    return !m_tma_only || access_type == TMA_ACC_R || access_type == TMA_ACC_W;
  }

  void log_packet(unsigned long long icnt_cycle, const char *event,
                  unsigned input_node, unsigned output_node, unsigned subpart,
                  unsigned requesters, unsigned queued_pkts, unsigned in_occ,
                  unsigned out_occ, const mem_fetch *mf, unsigned packet_size,
                  const char *status) {
    if (!m_enabled || icnt_cycle > m_max_icnt_cycle) return;
    if (!mf) return;

    if (!accepts(mf)) return;
    const mem_access_type access_type = mf->get_access_type();

    mem_fetch *original_mf = const_cast<mem_fetch *>(mf)->get_original_mf();
    const unsigned original_uid =
        original_mf ? original_mf->get_request_uid() : mf->get_request_uid();
    const unsigned long sector_mask =
        mf->get_access_sector_mask().to_ulong();

    flockfile(m_file);
    fprintf(m_file,
            "%llu,%llu,%s,%u,%u,%u,%u,%u,%u,%u,0x%llx,0x%llx,%u,%u,%u,%u,%u,"
            "%s,%u,%u,%u,%u,0x%lx,%s\n",
            icnt_cycle, mf->get_status_change(), event, input_node, output_node,
            subpart, requesters, queued_pkts, in_occ, out_occ,
            (unsigned long long)mf->get_addr(),
            (unsigned long long)mf->get_partition_addr(), mf->get_sid(),
            mf->get_tpc(), mf->get_wid(), mf->get_request_uid(), original_uid,
            mem_access_type_str(access_type), mf->get_is_write(), packet_size,
            mf->get_data_size(), mf->get_access_size(), sector_mask, status);
    flush_periodically();
    funlockfile(m_file);
  }

  void log_prearb(unsigned long long icnt_cycle, unsigned input_node,
                  unsigned output_node, unsigned subpart, unsigned requesters,
                  unsigned queued_pkts, const mem_fetch *front_mf) {
    if (!m_enabled || !m_trace_prearb || icnt_cycle > m_max_icnt_cycle) return;
    if (requesters < m_min_requesters) return;

    if (front_mf && !accepts(front_mf)) return;

    unsigned long long gpu_push_cycle = 0;
    unsigned long long addr = 0;
    unsigned long long partition_addr = 0;
    unsigned requestor_sm = UINT_MAX;
    unsigned tpc = UINT_MAX;
    unsigned wid = UINT_MAX;
    unsigned uid = 0;
    unsigned orig_uid = 0;
    const char *type = "NONE";
    unsigned is_write = 0;
    unsigned data_size = 0;
    unsigned access_size = 0;
    unsigned long sector_mask = 0;
    if (front_mf) {
      mem_fetch *original_mf =
          const_cast<mem_fetch *>(front_mf)->get_original_mf();
      gpu_push_cycle = front_mf->get_status_change();
      addr = front_mf->get_addr();
      partition_addr = front_mf->get_partition_addr();
      requestor_sm = front_mf->get_sid();
      tpc = front_mf->get_tpc();
      wid = front_mf->get_wid();
      uid = front_mf->get_request_uid();
      orig_uid =
          original_mf ? original_mf->get_request_uid() : front_mf->get_request_uid();
      type = mem_access_type_str(front_mf->get_access_type());
      is_write = front_mf->get_is_write();
      data_size = front_mf->get_data_size();
      access_size = front_mf->get_access_size();
      sector_mask = front_mf->get_access_sector_mask().to_ulong();
    }

    flockfile(m_file);
    fprintf(m_file,
            "%llu,%llu,PRE_ARB,%u,%u,%u,%u,%u,0,0,0x%llx,0x%llx,%u,%u,%u,%u,"
            "%u,%s,%u,0,%u,%u,0x%lx,%s\n",
            icnt_cycle, gpu_push_cycle, input_node, output_node, subpart,
            requesters, queued_pkts, addr, partition_addr, requestor_sm, tpc,
            wid, uid, orig_uid, type, is_write, data_size, access_size,
            sector_mask, "REQUESTERS");
    flush_periodically();
    funlockfile(m_file);
  }

 private:
  req_noc_trace()
      : m_enabled(false),
        m_trace_push(true),
        m_trace_grant(true),
        m_trace_prearb(true),
        m_tma_only(false),
        m_min_requesters(2),
        m_max_icnt_cycle(ULLONG_MAX),
        m_file(NULL),
        m_buffer(NULL),
        m_lines(0) {
    const char *path = getenv("FLASHGPU_REQ_NOC_TRACE_CSV");
    if (!path || path[0] == '\0') return;

    const char *max_cycle = getenv("FLASHGPU_REQ_NOC_TRACE_MAX_ICNT_CYCLE");
    if (max_cycle && max_cycle[0] != '\0')
      m_max_icnt_cycle = strtoull(max_cycle, NULL, 0);

    m_trace_push = env_flag_default("FLASHGPU_REQ_NOC_TRACE_PUSH", true);
    m_trace_grant = env_flag_default("FLASHGPU_REQ_NOC_TRACE_GRANT", true);
    m_trace_prearb = env_flag_default("FLASHGPU_REQ_NOC_TRACE_PREARB", true);
    m_tma_only = env_flag_default("FLASHGPU_REQ_NOC_TRACE_TMA_ONLY", false);

    const char *min_req = getenv("FLASHGPU_REQ_NOC_TRACE_MIN_REQUESTERS");
    if (min_req && min_req[0] != '\0')
      m_min_requesters = strtoul(min_req, NULL, 0);

    m_file = fopen(path, "w");
    if (!m_file) {
      perror("FLASHGPU_REQ_NOC_TRACE_CSV");
      return;
    }

    m_buffer = (char *)malloc(16 * 1024 * 1024);
    if (m_buffer) setvbuf(m_file, m_buffer, _IOFBF, 16 * 1024 * 1024);

    fprintf(m_file,
            "icnt_cycle,gpu_push_cycle,event,input_node,output_node,subpart,"
            "requesters,queued_pkts,in_occ,out_occ,addr,partition_addr,"
            "requestor_sm,tpc,wid,uid,orig_uid,type,is_write,packet_size,"
            "data_size,access_size,sector_mask,status\n");
    m_enabled = true;
    printf("FLASHGPU_REQ_NOC_TRACE_CSV enabled: path=%s max_icnt_cycle=%llu "
           "tma_only=%u push=%u grant=%u prearb=%u min_requesters=%u\n",
           path, m_max_icnt_cycle, m_tma_only ? 1 : 0, m_trace_push ? 1 : 0,
           m_trace_grant ? 1 : 0, m_trace_prearb ? 1 : 0, m_min_requesters);
  }

  ~req_noc_trace() {
    if (m_file) {
      fflush(m_file);
      fclose(m_file);
    }
    free(m_buffer);
  }

  static bool env_flag_default(const char *name, bool default_value) {
    const char *value = getenv(name);
    if (!value || value[0] == '\0') return default_value;
    return strcmp(value, "0") != 0;
  }

  void flush_periodically() {
    ++m_lines;
    if ((m_lines & ((1ULL << 20) - 1)) == 0) fflush(m_file);
  }

  bool m_enabled;
  bool m_trace_push;
  bool m_trace_grant;
  bool m_trace_prearb;
  bool m_tma_only;
  unsigned m_min_requesters;
  unsigned long long m_max_icnt_cycle;
  FILE *m_file;
  char *m_buffer;
  unsigned long long m_lines;
};

}  // namespace

xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config &m_localinct_config) {
  m_id = router_id;
  router_type = m_type;
  _n_mem = n_mem;
  _n_shader = n_shader;
  total_nodes = n_shader + n_mem;
  verbose = m_localinct_config.verbose;
  grant_cycles = m_localinct_config.grant_cycles;
  grant_cycles_count = m_localinct_config.grant_cycles;
  use_voq = m_localinct_config.use_voq != 0;
  in_buffers.resize(total_nodes);
  const unsigned queues_per_input = use_voq ? total_nodes : 1;
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffers[i].resize(queues_per_input);
  }
  out_buffers.resize(total_nodes);
  in_buffer_occupancy.assign(total_nodes, 0);
  next_node.resize(total_nodes, 0);
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  arbit_type = m_localinct_config.arbiter_algo;
  next_node_id = 0;
  if (m_type == REQ_NET) {
    active_in_buffers = n_shader;
    active_out_buffers = n_mem;
    active_in_buffer_base = 0;
    active_out_buffer_base = n_shader;
  } else if (m_type == REPLY_NET) {
    active_in_buffers = n_mem;
    active_out_buffers = n_shader;
    active_in_buffer_base = n_shader;
    active_out_buffer_base = 0;
  }

  cycles = 0;
  conflicts = 0;
  out_buffer_full = 0;
  in_buffer_full = 0;
  out_buffer_util = 0;
  in_buffer_util = 0;
  packets_num = 0;
  conflicts_util = 0;
  cycles_util = 0;
  reqs_util = 0;
  input_pushes.assign(total_nodes, 0);
  output_pushes.assign(total_nodes, 0);
  input_grants.assign(total_nodes, 0);
  output_grants.assign(total_nodes, 0);
  input_full_events.assign(total_nodes, 0);
  output_full_events.assign(total_nodes, 0);
  max_input_occupancy.assign(total_nodes, 0);
  max_output_occupancy.assign(total_nodes, 0);
}

xbar_router::~xbar_router() {}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void *data, unsigned int size) {
  assert(input_deviceID < total_nodes);
  assert(output_deviceID < total_nodes);
  in_buffers[input_deviceID][InputQueueIndex(output_deviceID)].push_back(
      Packet(data, output_deviceID, size));
  in_buffer_occupancy[input_deviceID]++;
  packets_num++;
  input_pushes[input_deviceID]++;
  output_pushes[output_deviceID]++;
  max_input_occupancy[input_deviceID] =
      std::max(max_input_occupancy[input_deviceID],
               in_buffer_occupancy[input_deviceID]);
  if (router_type == REQ_NET && req_noc_trace::instance().trace_push()) {
    req_noc_trace::instance().log_packet(
        cycles, "PUSH", input_deviceID, output_deviceID,
        output_deviceID - active_out_buffer_base, 0, 0,
        in_buffer_occupancy[input_deviceID], out_buffers[output_deviceID].size(),
        static_cast<mem_fetch *>(data), size, "IN_NOC");
  }
}

void *xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  void *data = NULL;

  if (!out_buffers[ouput_deviceID].empty()) {
    const Packet packet = out_buffers[ouput_deviceID].front();
    data = packet.data;
    out_buffers[ouput_deviceID].pop_front();
  }

  return data;
}

bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size,
                                bool update_counter) {
  assert(input_deviceID < total_nodes);

  bool has_buffer = (in_buffer_occupancy[input_deviceID] + size <=
                     in_buffer_limit);
  if (update_counter && !has_buffer) {
    in_buffer_full++;
    input_full_events[input_deviceID]++;
  }

  return has_buffer;
}

bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size) {
  return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
}

void xbar_router::Advance() {
  if (arbit_type == NAIVE_RR)
    RR_Advance();
  else if (arbit_type == iSLIP)
    iSLIP_Advance();
  else
    assert(0);
}

void xbar_router::RR_Advance() {
  bool active = false;
  vector<bool> issued(total_nodes, false);
  unsigned conflict_sub = 0;
  unsigned reqs = 0;
  CollectRequestStats(&active, &conflict_sub);

  for (unsigned i = 0; i < total_nodes; ++i) {
    unsigned node_id = (i + next_node_id) % total_nodes;

    if (node_id >= active_in_buffer_base &&
        node_id < active_in_buffer_base + active_in_buffers &&
        InputHasPackets(node_id)) {
      const unsigned output = FirstReadyOutput(node_id);
      assert(output < total_nodes);
      if (Has_Buffer_Out(output, 1)) {
        if (!issued[output]) {
          TransferPacket(node_id, output);
          issued[output] = true;
          reqs++;
        }
      } else {
        out_buffer_full++;
        output_full_events[output]++;
      }
    }
  }
  next_node_id = next_node_id + 1;
  next_node_id = (next_node_id % total_nodes);

  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
    reqs_util += reqs;
  }

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffer_occupancy[i];
    out_buffer_util += out_buffers[i].size();
  }

  cycles++;
}
// iSLIP algorithm
// McKeown, Nick. "The iSLIP scheduling algorithm for input-queued switches."
// IEEE/ACM transactions on networking 2 (1999): 188-201.
// https://www.cs.rutgers.edu/~sn624/552-F18/papers/islip.pdf
void xbar_router::iSLIP_Advance() {
  bool active = false;

  unsigned conflict_sub = 0;
  unsigned reqs = 0;
  CollectRequestStats(&active, &conflict_sub);

  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
  }
  // do iSLIP
  for (unsigned i = active_out_buffer_base;
       i < active_out_buffer_base + active_out_buffers; ++i) {
    if (Has_Buffer_Out(i, 1)) {

      // Only check the input buffers.
      for (unsigned j = 0; j < active_in_buffers; ++j) {
        unsigned node_id =
            (j + next_node[i]) % active_in_buffers + active_in_buffer_base;

        if (InputHasPacketForOutput(node_id, i)) {
          TransferPacket(node_id, i);
          if (verbose)
            printf("%d : cycle %llu : send req from %d to %d\n", m_id, cycles,
                   node_id, i - _n_shader);
          if (grant_cycles_count == 1) {
            if (use_voq) {
              next_node[i] =
                  (node_id - active_in_buffer_base + 1) % active_in_buffers;
            } else {
              next_node[i] = (node_id + 1) % active_in_buffers;
            }
          }
          if (verbose) {
            for (unsigned k = j + 1; k < total_nodes; ++k) {
              unsigned node_id2 = (k + next_node[i]) % total_nodes;
              if (node_id2 >= active_in_buffer_base &&
                  node_id2 < active_in_buffer_base + active_in_buffers &&
                  InputHasPacketForOutput(node_id2, i)) {
                printf("%d : cycle %llu : cannot send req from %d to %d\n",
                       m_id, cycles, node_id2, i - _n_shader);
              }
            }
          }

          reqs++;
          break;
        }
      }
    } else {
      out_buffer_full++;
      output_full_events[i]++;
    }
  }

  if (active) {
    reqs_util += reqs;
  }

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);

  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  else if (active)
    grant_cycles_count--;

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffer_occupancy[i];
    out_buffer_util += out_buffers[i].size();
  }

  cycles++;
}

bool xbar_router::InputHasPackets(unsigned input_deviceID) const {
  assert(input_deviceID < total_nodes);
  return in_buffer_occupancy[input_deviceID] > 0;
}

bool xbar_router::InputHasPacketForOutput(unsigned input_deviceID,
                                          unsigned output_deviceID) const {
  assert(input_deviceID < total_nodes);
  assert(output_deviceID < total_nodes);

  const deque<Packet> &input_queue =
      in_buffers[input_deviceID][InputQueueIndex(output_deviceID)];
  return !input_queue.empty() &&
         (use_voq || input_queue.front().output_deviceID == output_deviceID);
}

unsigned xbar_router::FirstReadyOutput(unsigned input_deviceID) const {
  assert(input_deviceID < total_nodes);
  assert(InputHasPackets(input_deviceID));

  if (!use_voq) {
    assert(!in_buffers[input_deviceID][0].empty());
    return in_buffers[input_deviceID][0].front().output_deviceID;
  }

  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    if (!in_buffers[input_deviceID][output].empty())
      return output;
  }

  assert(0);
  return 0;
}

unsigned xbar_router::InputQueueIndex(unsigned output_deviceID) const {
  return use_voq ? output_deviceID : 0;
}

void xbar_router::TransferPacket(unsigned input_deviceID,
                                 unsigned output_deviceID) {
  assert(input_deviceID < total_nodes);
  assert(output_deviceID < total_nodes);
  deque<Packet> &input_queue =
      in_buffers[input_deviceID][InputQueueIndex(output_deviceID)];
  assert(!input_queue.empty());
  assert(in_buffer_occupancy[input_deviceID] > 0);

  Packet packet = input_queue.front();
  assert(packet.output_deviceID == output_deviceID);
  if (router_type == REQ_NET && req_noc_trace::instance().trace_grant()) {
    req_noc_trace::instance().log_packet(
        cycles, "GRANT", input_deviceID, output_deviceID,
        output_deviceID - active_out_buffer_base, 0, input_queue.size(),
        in_buffer_occupancy[input_deviceID], out_buffers[output_deviceID].size(),
        static_cast<mem_fetch *>(packet.data), packet.size, "TO_L2_OUTPUT");
  }
  out_buffers[output_deviceID].push_back(packet);
  max_output_occupancy[output_deviceID] =
      std::max<unsigned>(max_output_occupancy[output_deviceID],
                         out_buffers[output_deviceID].size());
  input_grants[input_deviceID]++;
  output_grants[output_deviceID]++;
  input_queue.pop_front();
  in_buffer_occupancy[input_deviceID]--;
}

void xbar_router::CollectRequestStats(bool *active,
                                      unsigned *conflicts) const {
  *active = false;
  *conflicts = 0;

  if (!use_voq) {
    std::set<unsigned> requested_outputs;
    for (unsigned input = active_in_buffer_base;
         input < active_in_buffer_base + active_in_buffers; ++input) {
      if (!InputHasPackets(input)) continue;

      *active = true;
      const Packet &packet = in_buffers[input][0].front();
      if (!requested_outputs.insert(packet.output_deviceID).second)
        (*conflicts)++;
    }
    if (router_type == REQ_NET && req_noc_trace::instance().trace_prearb()) {
      for (unsigned output = active_out_buffer_base;
           output < active_out_buffer_base + active_out_buffers; ++output) {
        unsigned requesters = 0;
        unsigned queued_pkts = 0;
        unsigned first_input = UINT_MAX;
        mem_fetch *front_mf = NULL;
        for (unsigned input = active_in_buffer_base;
             input < active_in_buffer_base + active_in_buffers; ++input) {
          if (!InputHasPackets(input)) continue;
          const Packet &packet = in_buffers[input][0].front();
          if (packet.output_deviceID != output) continue;
          requesters++;
          queued_pkts++;
          mem_fetch *mf = static_cast<mem_fetch *>(packet.data);
          if (first_input == UINT_MAX && req_noc_trace::instance().accepts(mf)) {
            first_input = input;
            front_mf = mf;
          }
        }
        req_noc_trace::instance().log_prearb(
            cycles, first_input, output, output - active_out_buffer_base,
            requesters, queued_pkts, front_mf);
      }
    }
    return;
  }

  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    if (InputHasPackets(input)) {
      *active = true;
    }
  }

  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    unsigned requesters = 0;
    unsigned queued_pkts = 0;
    unsigned first_input = UINT_MAX;
    mem_fetch *front_mf = NULL;
    for (unsigned input = active_in_buffer_base;
         input < active_in_buffer_base + active_in_buffers; ++input) {
      const deque<Packet> &queue = in_buffers[input][output];
      if (!queue.empty()) {
        requesters++;
        queued_pkts += queue.size();
        mem_fetch *mf = static_cast<mem_fetch *>(queue.front().data);
        if (first_input == UINT_MAX && req_noc_trace::instance().accepts(mf)) {
          first_input = input;
          front_mf = mf;
        }
      }
    }
    if (requesters > 0)
      *conflicts += requesters - 1;
    if (router_type == REQ_NET && req_noc_trace::instance().trace_prearb()) {
      req_noc_trace::instance().log_prearb(
          cycles, first_input, output, output - active_out_buffer_base,
          requesters, queued_pkts, front_mf);
    }
  }
}

void xbar_router::DisplayStats(const char *name) const {
  printf("%s_Network_injected_packets_num = %lld\n", name, packets_num);
  printf("%s_Network_cycles = %lld\n", name, cycles);
  printf("%s_Network_injected_packets_per_cycle = %12.4f%s\n", name,
         (float)(packets_num) / (cycles), router_type == REQ_NET ? " " : "");
  printf("%s_Network_conflicts_per_cycle = %12.4f\n", name,
         (float)(conflicts) / (cycles));
  printf("%s_Network_conflicts_per_cycle_util = %12.4f\n", name,
         (float)(conflicts_util) / (cycles_util));
  printf("%s_Bank_Level_Parallism = %12.4f\n", name,
         (float)(reqs_util) / (cycles_util));
  printf("%s_Network_in_buffer_full_per_cycle = %12.4f\n", name,
         (float)(in_buffer_full) / (cycles));
  printf("%s_Network_in_buffer_avg_util = %12.4f\n", name,
         ((float)(in_buffer_util) / (cycles) / active_in_buffers));
  printf("%s_Network_out_buffer_full_per_cycle = %12.4f\n", name,
         (float)(out_buffer_full) / (cycles));
  printf("%s_Network_out_buffer_avg_util = %12.4f\n", name,
         ((float)(out_buffer_util) / (cycles) / active_out_buffers));

  auto print_top = [&](const char *label,
                       const std::vector<unsigned long long> &values,
                       unsigned base, unsigned count) {
    std::vector<std::pair<unsigned long long, unsigned> > ranked;
    for (unsigned i = base; i < base + count; ++i) {
      if (values[i] != 0) ranked.push_back(std::make_pair(values[i], i));
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<unsigned long long, unsigned> &a,
                 const std::pair<unsigned long long, unsigned> &b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
    printf("%s_Network_%s_top =", name, label);
    const unsigned n = std::min<unsigned>(ranked.size(), 8);
    for (unsigned i = 0; i < n; ++i) {
      printf(" %u:%llu", ranked[i].second, ranked[i].first);
    }
    printf("\n");
  };

  auto print_top_unsigned = [&](const char *label,
                                const std::vector<unsigned> &values,
                                unsigned base, unsigned count) {
    std::vector<std::pair<unsigned, unsigned> > ranked;
    for (unsigned i = base; i < base + count; ++i) {
      if (values[i] != 0) ranked.push_back(std::make_pair(values[i], i));
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<unsigned, unsigned> &a,
                 const std::pair<unsigned, unsigned> &b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
    printf("%s_Network_%s_top =", name, label);
    const unsigned n = std::min<unsigned>(ranked.size(), 8);
    for (unsigned i = 0; i < n; ++i) {
      printf(" %u:%u", ranked[i].second, ranked[i].first);
    }
    printf("\n");
  };

  print_top("input_pushes", input_pushes, active_in_buffer_base,
            active_in_buffers);
  print_top("output_pushes", output_pushes, active_out_buffer_base,
            active_out_buffers);
  print_top("input_grants", input_grants, active_in_buffer_base,
            active_in_buffers);
  print_top("output_grants", output_grants, active_out_buffer_base,
            active_out_buffers);
  print_top("input_full_events", input_full_events, active_in_buffer_base,
            active_in_buffers);
  print_top("output_full_events", output_full_events, active_out_buffer_base,
            active_out_buffers);
  print_top_unsigned("max_input_occupancy", max_input_occupancy,
                     active_in_buffer_base, active_in_buffers);
  print_top_unsigned("max_output_occupancy", max_output_occupancy,
                     active_out_buffer_base, active_out_buffers);
}

bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (in_buffer_occupancy[i] > 0)
      return true;

    if (!out_buffers[i].empty())
      return true;
  }
  return false;
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// assume all the packets are one flit
#define LOCAL_INCT_FLIT_SIZE 40

LocalInterconnect *
LocalInterconnect::New(const struct inct_config &m_localinct_config) {
  LocalInterconnect *icnt_interface = new LocalInterconnect(m_localinct_config);

  return icnt_interface;
}

LocalInterconnect::LocalInterconnect(
    const struct inct_config &m_localinct_config)
    : m_inct_config(m_localinct_config) {
  n_shader = 0;
  n_mem = 0;
  n_subnets = m_localinct_config.subnets;
}

LocalInterconnect::~LocalInterconnect() {
  for (unsigned i = 0; i < m_inct_config.subnets; ++i) {
    delete net[i];
  }
}

void LocalInterconnect::CreateInterconnect(unsigned m_n_shader,
                                           unsigned m_n_mem) {
  n_shader = m_n_shader;
  n_mem = m_n_mem;

  net.resize(n_subnets);
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i] = new xbar_router(i, static_cast<Interconnect_type>(i), m_n_shader,
                             m_n_mem, m_inct_config);
  }
}

void LocalInterconnect::Init() {
  // empty
  // there is nothing to do
}

void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                             void *data, unsigned int size) {
  unsigned subnet;
  if (n_subnets == 1) {
    subnet = 0;
  } else {
    if (input_deviceID < n_shader) {
      subnet = 0;
    } else {
      subnet = 1;
    }
  }

  // it should have free buffer
  // assume all the packets have size of one
  // no flits are implemented
  assert(net[subnet]->Has_Buffer_In(input_deviceID, 1));

  net[subnet]->Push(input_deviceID, output_deviceID, data, size);
}

void *LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  if (ouput_deviceID < n_shader)
    subnet = 1;

  return net[subnet]->Pop(ouput_deviceID);
}

void LocalInterconnect::Advance() {
#ifdef FLASH_GPGPU_SIM_OMP
#pragma omp parallel for schedule(static)
#endif
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i]->Advance();
  }
}

bool LocalInterconnect::Busy() const {
  for (unsigned i = 0; i < n_subnets; ++i) {
    if (net[i]->Busy())
      return true;
  }
  return false;
}

bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  bool has_buffer = false;

  if ((n_subnets > 1) && deviceID >= n_shader) // deviceID is memory node
    has_buffer = net[REPLY_NET]->Has_Buffer_In(deviceID, 1, true);
  else
    has_buffer = net[REQ_NET]->Has_Buffer_In(deviceID, 1, true);

  return has_buffer;
}

void LocalInterconnect::DisplayStats() const {
  net[REQ_NET]->DisplayStats("Req");
  printf("\n");
  net[REPLY_NET]->DisplayStats("Reply");
}

void LocalInterconnect::DisplayOverallStats() const {}

unsigned LocalInterconnect::GetFlitSize() const { return LOCAL_INCT_FLIT_SIZE; }

void LocalInterconnect::DisplayState(FILE *fp) const {
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
}
