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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <utility>

#include "local_interconnect.h"
#include "mem_fetch.h"

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
}

xbar_router::~xbar_router() {}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void *data, unsigned int size) {
  assert(input_deviceID < total_nodes);
  assert(output_deviceID < total_nodes);
  in_buffers[input_deviceID][InputQueueIndex(output_deviceID)].push_back(
      Packet(data, output_deviceID));
  in_buffer_occupancy[input_deviceID]++;
  packets_num++;
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
  if (update_counter && !has_buffer)
    in_buffer_full++;

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
  out_buffers[output_deviceID].push_back(packet);
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
    for (unsigned input = active_in_buffer_base;
         input < active_in_buffer_base + active_in_buffers; ++input) {
      if (!in_buffers[input][output].empty()) requesters++;
    }
    if (requesters > 0)
      *conflicts += requesters - 1;
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
