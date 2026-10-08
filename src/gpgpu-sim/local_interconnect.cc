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

#include "../trace.h"
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
  allow_multi_grant =
      m_type == REQ_NET ? m_localinct_config.multi_grant_request != 0
                        : m_localinct_config.multi_grant_reply != 0;
  input_sector_width = m_type == REQ_NET
                           ? m_localinct_config.request_input_sectors_per_cycle
                           : m_localinct_config.reply_input_sectors_per_cycle;
  output_sector_width =
      m_type == REQ_NET ? m_localinct_config.request_output_sectors_per_cycle
                        : m_localinct_config.reply_output_sectors_per_cycle;
  tma_request_multicast =
      m_type == REQ_NET && m_localinct_config.tma_request_multicast != 0;
  tma_response_multicast =
      m_type == REPLY_NET && m_localinct_config.tma_response_multicast != 0;
  tma_multicast_master_sectors = 0;
  tma_multicast_merged_sectors = 0;
  tma_multicast_max_waiters = 0;
  tma_response_multicast_masters = 0;
  tma_response_multicast_waiters = 0;
  tma_response_multicast_max_waiters = 0;
  in_buffers.resize(total_nodes);
  const unsigned queues_per_input = use_voq ? total_nodes : 1;
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffers[i].resize(queues_per_input);
  }
  out_buffers.resize(total_nodes);
  multicast_delivery_buffers.resize(total_nodes);
  in_buffer_occupancy.assign(total_nodes, 0);
  next_node.resize(total_nodes, 0);
  next_output.resize(total_nodes, 0);
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  arbit_type = m_localinct_config.arbiter_algo;
  next_node_id = 0;
  next_output_id = 0;
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
  input_service_stats.resize(total_nodes);
  output_service_stats.resize(total_nodes);
  input_budgets.resize(total_nodes);
  output_budgets.resize(total_nodes);
  input_tick_service_slots.assign(total_nodes, 0);
  output_tick_service_slots.assign(total_nodes, 0);
}

xbar_router::~xbar_router() {}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void *data, unsigned int size, unsigned data_sectors) {
  assert(input_deviceID < total_nodes);
  assert(output_deviceID < total_nodes);
  mem_fetch *mf = static_cast<mem_fetch *>(data);
  if (tma_request_multicast) {
    // Shader clusters inject concurrently under FLASH_GPGPU_SIM_OMP. The
    // ordinary input queues are cluster-private, but multicast generations
    // are shared across all request-network inputs.
    std::lock_guard<std::mutex> lock(tma_multicast_mutex);
    const unsigned long long sector_addr =
        mf->get_addr() - (mf->get_addr() % SECTOR_SIZE);
    if (!mf->get_is_write() && mf->get_access_type() == TMA_ACC_R &&
        mf->get_data_size() == SECTOR_SIZE) {
      unsigned long long waiter_count = 0;
      if (!tma_multicast_groups.admit(sector_addr, data, waiter_count)) {
        ++tma_multicast_merged_sectors;
        tma_multicast_max_waiters =
            std::max(tma_multicast_max_waiters, waiter_count);
        return;
      }
      ++tma_multicast_master_sectors;
    } else {
      // A non-eligible access establishes an ordering boundary for this
      // address. This includes writes, atomics, and non-sector TMA requests.
      tma_multicast_groups.close_address(sector_addr);
    }
  }
  in_buffers[input_deviceID][InputQueueIndex(output_deviceID)].push_back(
      Packet(data, output_deviceID, size, data_sectors));
  in_buffer_occupancy[input_deviceID]++;
  packets_num++;
  input_pushes[input_deviceID]++;
  output_pushes[output_deviceID]++;
  max_input_occupancy[input_deviceID] = std::max(
      max_input_occupancy[input_deviceID], in_buffer_occupancy[input_deviceID]);
  if (router_type == REQ_NET && GPTRACE(INTERCONNECT)) {
    Trace::request_noc_packet_event(
        cycles, "PUSH", input_deviceID, output_deviceID,
        output_deviceID - active_out_buffer_base, 0, 0,
        in_buffer_occupancy[input_deviceID],
        out_buffers[output_deviceID].size(), static_cast<mem_fetch *>(data),
        size, "IN_NOC");
  }
}

void xbar_router::PushMulticast(
    unsigned input_deviceID, unsigned output_deviceID, void *data,
    unsigned int size, unsigned data_sectors,
    const std::vector<std::pair<unsigned, void *> > &destinations) {
  assert(tma_response_multicast);
  assert(router_type == REPLY_NET);
  assert(data != NULL);
  {
    std::lock_guard<std::mutex> lock(tma_multicast_mutex);
    assert(tma_response_multicast_deliveries.find(data) ==
           tma_response_multicast_deliveries.end());
    std::deque<Packet> &deliveries = tma_response_multicast_deliveries[data];
    for (std::vector<std::pair<unsigned, void *> >::const_iterator it =
             destinations.begin();
         it != destinations.end(); ++it) {
      assert(it->first < total_nodes);
      assert(it->second != NULL);
      deliveries.push_back(Packet(it->second, it->first, size, data_sectors));
    }
    ++tma_response_multicast_masters;
    tma_response_multicast_waiters += destinations.size();
    tma_response_multicast_max_waiters =
        std::max<unsigned long long>(tma_response_multicast_max_waiters,
                                     destinations.size());
  }
  // Only the master occupies reply-network input and output bandwidth. The
  // associated requester responses are materialized at their target xbar
  // outputs when the master arrives there.
  Push(input_deviceID, output_deviceID, data, size, data_sectors);
}

void *xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  if (tma_request_multicast || tma_response_multicast) {
    std::lock_guard<std::mutex> lock(tma_multicast_mutex);
    if (!multicast_delivery_buffers[ouput_deviceID].empty()) {
      const Packet packet = multicast_delivery_buffers[ouput_deviceID].front();
      multicast_delivery_buffers[ouput_deviceID].pop_front();
      return packet.data;
    }

    if (!out_buffers[ouput_deviceID].empty()) {
      const Packet packet = out_buffers[ouput_deviceID].front();
      out_buffers[ouput_deviceID].pop_front();
      if (tma_request_multicast) {
        std::deque<void *> waiters;
        if (tma_multicast_groups.close_master(packet.data, waiters)) {
          while (!waiters.empty()) {
            multicast_delivery_buffers[ouput_deviceID].push_back(
                Packet(waiters.front(), packet.output_deviceID, packet.size,
                       packet.data_sectors));
            waiters.pop_front();
          }
        }
      }
      if (tma_response_multicast) {
        std::unordered_map<void *, std::deque<Packet> >::iterator deliveries =
            tma_response_multicast_deliveries.find(packet.data);
        if (deliveries != tma_response_multicast_deliveries.end()) {
          while (!deliveries->second.empty()) {
            const Packet waiter = deliveries->second.front();
            deliveries->second.pop_front();
            multicast_delivery_buffers[waiter.output_deviceID].push_back(
                waiter);
          }
          tma_response_multicast_deliveries.erase(deliveries);
        }
      }
      return packet.data;
    }
    return NULL;
  }

  void *data = NULL;
  if (!out_buffers[ouput_deviceID].empty()) {
    data = out_buffers[ouput_deviceID].front().data;
    out_buffers[ouput_deviceID].pop_front();
  }
  return data;
}

void *xbar_router::Top(unsigned output_deviceID) const {
  assert(output_deviceID < total_nodes);
  if (tma_request_multicast || tma_response_multicast) {
    std::lock_guard<std::mutex> lock(tma_multicast_mutex);
    if (!multicast_delivery_buffers[output_deviceID].empty())
      return multicast_delivery_buffers[output_deviceID].front().data;
  }
  if (out_buffers[output_deviceID].empty()) return NULL;
  return out_buffers[output_deviceID].front().data;
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
  std::fill(input_tick_service_slots.begin(), input_tick_service_slots.end(), 0);
  std::fill(output_tick_service_slots.begin(), output_tick_service_slots.end(),
            0);
  if (input_sector_width != 0 || output_sector_width != 0) {
    NumericAdvance(arbit_type == iSLIP);
    return;
  }
  if (arbit_type == NAIVE_RR)
    RR_Advance();
  else if (arbit_type == iSLIP)
    iSLIP_Advance();
  else
    assert(0);
  FinalizeLegacyServiceStats();
}

void xbar_router::NumericAdvance(bool is_islip) {
  assert(arbit_type == NAIVE_RR || arbit_type == iSLIP);

  bool active = false;
  unsigned conflict_sub = 0;
  unsigned reqs = 0;
  CollectRequestStats(&active, &conflict_sub);

  vector<unsigned> legacy_input_grants(total_nodes, 0);
  vector<unsigned> legacy_output_grants(total_nodes, 0);
  vector<vector<bool> > legacy_pairs;
  if (input_sector_width == 0 && allow_multi_grant && is_islip)
    legacy_pairs.assign(total_nodes, vector<bool>(total_nodes));
  vector<bool> downstream_seen(total_nodes, false);
  // A full destination is an input-side stall statistic, but it must not clear
  // credit accumulated for an independent VOQ on the same input.
  vector<bool> input_downstream_seen(total_nodes, false);
  vector<bool> input_credit_reserved(total_nodes, false);
  vector<bool> output_credit_reserved(total_nodes, false);

  unsigned queued_packets = 0;
  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    input_budgets[input].begin_tick(input_sector_width);
    queued_packets += in_buffer_occupancy[input];
  }
  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    output_budgets[output].begin_tick(output_sector_width);
  }

  // Each round transfers at least one packet or terminates.  Bounding the
  // rounds by the number present at tick start makes the arbitration finite
  // without imposing a separate packet-count constant.
  for (unsigned round = 0; round < queued_packets; ++round) {
    bool progress = false;

    if (is_islip) {
      for (unsigned output_offset = 0; output_offset < active_out_buffers;
           ++output_offset) {
        const unsigned output =
            active_out_buffer_base +
            (output_offset + next_output_id + round) % active_out_buffers;
        if (output_credit_reserved[output]) continue;

        if (!Has_Buffer_Out(output, 1)) {
          bool requested = false;
          for (unsigned input = active_in_buffer_base;
               input < active_in_buffer_base + active_in_buffers; ++input) {
            if (InputHasPacketForOutput(input, output)) {
              requested = true;
              input_downstream_seen[input] = true;
            }
          }
          if (requested && !downstream_seen[output]) {
            downstream_seen[output] = true;
            output_budgets[output].note_downstream_full();
            ++out_buffer_full;
            ++output_full_events[output];
          }
          continue;
        }

        for (unsigned input_offset = 0; input_offset < active_in_buffers;
             ++input_offset) {
          const unsigned input =
              active_in_buffer_base +
              (input_offset + next_node[output]) % active_in_buffers;
          if (input_credit_reserved[input]) continue;
          const Packet *packet = InputPacketForOutput(input, output);
          if (!packet) continue;
          const unsigned slots =
              memory_transport_service_slots(packet->data_sectors);
          const bool input_needs_credit =
              input_sector_width != 0 && slots > input_sector_width &&
              !input_budgets[input].can_accept(packet->data_sectors);
          const bool output_needs_credit =
              output_sector_width != 0 && slots > output_sector_width &&
              !output_budgets[output].can_accept(packet->data_sectors);
          if (input_needs_credit || output_needs_credit) {
            if (input_needs_credit) {
              input_budgets[input].note_width_limited(packet->data_sectors);
              input_credit_reserved[input] = true;
            }
            if (output_needs_credit) {
              output_budgets[output].note_width_limited(packet->data_sectors);
              output_credit_reserved[output] = true;
              break;
            }
            continue;
          }
          if (!InputCanGrant(input, output, *packet, legacy_input_grants,
                             legacy_pairs) ||
              !OutputCanGrant(output, *packet, legacy_output_grants)) {
            continue;
          }

          if (input_sector_width != 0)
            input_budgets[input].consume(packet->data_sectors);
          if (output_sector_width != 0)
            output_budgets[output].consume(packet->data_sectors);
          TransferPacket(input, output);
          ++legacy_input_grants[input];
          ++legacy_output_grants[output];
          if (!legacy_pairs.empty()) legacy_pairs[input][output] = true;
          if (grant_cycles_count == 1)
            next_node[output] =
                (input - active_in_buffer_base + 1) % active_in_buffers;
          ++reqs;
          progress = true;
          break;
        }
      }
    } else {
      for (unsigned input_offset = 0; input_offset < active_in_buffers;
           ++input_offset) {
        const unsigned input =
            active_in_buffer_base +
            (input_offset + next_node_id + round) % active_in_buffers;
        if (input_credit_reserved[input]) continue;
        if (!InputHasPackets(input)) continue;

        for (unsigned output_offset = 0; output_offset < active_out_buffers;
             ++output_offset) {
          unsigned output = 0;
          if (use_voq) {
            output = active_out_buffer_base +
                     (output_offset + next_output[input]) % active_out_buffers;
          } else {
            output = FirstReadyOutput(input);
            if (output_offset != 0) break;
          }

          const Packet *packet = InputPacketForOutput(input, output);
          if (!packet) continue;
          if (output_credit_reserved[output]) continue;
          if (!Has_Buffer_Out(output, 1)) {
            input_downstream_seen[input] = true;
            if (!downstream_seen[output]) {
              downstream_seen[output] = true;
              output_budgets[output].note_downstream_full();
              ++out_buffer_full;
              ++output_full_events[output];
            }
            continue;
          }
          const unsigned slots =
              memory_transport_service_slots(packet->data_sectors);
          const bool input_needs_credit =
              input_sector_width != 0 && slots > input_sector_width &&
              !input_budgets[input].can_accept(packet->data_sectors);
          const bool output_needs_credit =
              output_sector_width != 0 && slots > output_sector_width &&
              !output_budgets[output].can_accept(packet->data_sectors);
          if (input_needs_credit || output_needs_credit) {
            if (input_needs_credit) {
              input_budgets[input].note_width_limited(packet->data_sectors);
              input_credit_reserved[input] = true;
            }
            if (output_needs_credit) {
              output_budgets[output].note_width_limited(packet->data_sectors);
              output_credit_reserved[output] = true;
            }
            // An output-only credit wait must not head-of-line block the
            // input's other VOQs.  Input credit, in contrast, reserves the
            // shared input budget and therefore stops this input for the tick.
            if (input_needs_credit) break;
            continue;
          }
          if (!InputCanGrant(input, output, *packet, legacy_input_grants,
                             legacy_pairs) ||
              !OutputCanGrant(output, *packet, legacy_output_grants)) {
            continue;
          }

          if (input_sector_width != 0)
            input_budgets[input].consume(packet->data_sectors);
          if (output_sector_width != 0)
            output_budgets[output].consume(packet->data_sectors);
          TransferPacket(input, output);
          ++legacy_input_grants[input];
          ++legacy_output_grants[output];
          if (!legacy_pairs.empty()) legacy_pairs[input][output] = true;
          next_output[input] =
              (output - active_out_buffer_base + 1) % active_out_buffers;
          ++reqs;
          progress = true;
          break;
        }
      }
    }

    if (!progress) break;
  }

  // Identify resources whose remaining budget, rather than backpressure,
  // leaves a packet queued.  The remaining credit is retained only in this
  // case so an oversized head packet eventually advances.
  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    if (!InputHasPackets(input)) continue;
    if (input_sector_width == 0) {
      const bool one_grant_per_tick =
          arbit_type == NAIVE_RR || !allow_multi_grant;
      if ((one_grant_per_tick && legacy_input_grants[input] != 0) ||
          (!one_grant_per_tick &&
           legacy_input_grants[input] >= active_out_buffers))
        input_budgets[input].note_width_limited(0);
    } else if (use_voq) {
      for (unsigned output = active_out_buffer_base;
           output < active_out_buffer_base + active_out_buffers; ++output) {
        const Packet *packet = InputPacketForOutput(input, output);
        if (packet && (!input_budgets[input].can_accept(packet->data_sectors) ||
                       memory_transport_service_slots(packet->data_sectors) >
                           input_sector_width))
          input_budgets[input].note_width_limited(packet->data_sectors);
      }
    } else {
      const unsigned output = FirstReadyOutput(input);
      const Packet *packet = InputPacketForOutput(input, output);
      if (packet && (!input_budgets[input].can_accept(packet->data_sectors) ||
                     memory_transport_service_slots(packet->data_sectors) >
                         input_sector_width))
        input_budgets[input].note_width_limited(packet->data_sectors);
    }
  }

  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    bool pending = false;
    for (unsigned input = active_in_buffer_base;
         input < active_in_buffer_base + active_in_buffers; ++input) {
      const Packet *packet = InputPacketForOutput(input, output);
      if (!packet) continue;
      pending = true;
      if (output_sector_width != 0 &&
          (!output_budgets[output].can_accept(packet->data_sectors) ||
           memory_transport_service_slots(packet->data_sectors) >
               output_sector_width))
        output_budgets[output].note_width_limited(packet->data_sectors);
    }
    if (pending && output_sector_width == 0 &&
        legacy_output_grants[output] != 0)
      output_budgets[output].note_width_limited(0);
  }

  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    input_budgets[input].end_tick(&input_service_stats[input]);
    if (input_sector_width == 0)
      input_service_stats[input].record_tick_service(
          input_tick_service_slots[input]);
    if (input_downstream_seen[input])
      ++input_service_stats[input].downstream_full_ticks;
  }
  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    output_budgets[output].end_tick(&output_service_stats[output]);
    if (output_sector_width == 0)
      output_service_stats[output].record_tick_service(
          output_tick_service_slots[output]);
  }

  next_node_id = (next_node_id + 1) % active_in_buffers;
  next_output_id = (next_output_id + 1) % active_out_buffers;
  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    ++cycles_util;
    reqs_util += reqs;
  }
  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  else if (active)
    --grant_cycles_count;

  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffer_occupancy[i];
    out_buffer_util += out_buffers[i].size();
  }
  ++cycles;
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
  vector<bool> input_granted(total_nodes, false);

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

        if ((allow_multi_grant || !input_granted[node_id]) &&
            InputHasPacketForOutput(node_id, i)) {
          TransferPacket(node_id, i);
          if (!allow_multi_grant) input_granted[node_id] = true;
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

const xbar_router::Packet *xbar_router::InputPacketForOutput(
    unsigned input_deviceID, unsigned output_deviceID) const {
  if (!InputHasPacketForOutput(input_deviceID, output_deviceID)) return NULL;
  const deque<Packet> &queue =
      in_buffers[input_deviceID][InputQueueIndex(output_deviceID)];
  return &queue.front();
}

bool xbar_router::InputCanGrant(
    unsigned input_deviceID, unsigned output_deviceID, const Packet &packet,
    const std::vector<unsigned> &legacy_input_grants,
    const std::vector<std::vector<bool> > &legacy_pairs) const {
  if (input_sector_width != 0) {
    if (input_budgets[input_deviceID].has_reserved_credit() &&
        memory_transport_service_slots(packet.data_sectors) <=
            input_sector_width)
      return false;
    return input_budgets[input_deviceID].can_accept(packet.data_sectors);
  }
  if (arbit_type == NAIVE_RR || !allow_multi_grant)
    return legacy_input_grants[input_deviceID] == 0;
  assert(!legacy_pairs.empty());
  return !legacy_pairs[input_deviceID][output_deviceID];
}

bool xbar_router::OutputCanGrant(
    unsigned output_deviceID, const Packet &packet,
    const std::vector<unsigned> &legacy_output_grants) const {
  if (output_sector_width != 0) {
    if (output_budgets[output_deviceID].has_reserved_credit() &&
        memory_transport_service_slots(packet.data_sectors) <=
            output_sector_width)
      return false;
    return output_budgets[output_deviceID].can_accept(packet.data_sectors);
  }
  return legacy_output_grants[output_deviceID] == 0;
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
  if (router_type == REQ_NET && GPTRACE(INTERCONNECT)) {
    Trace::request_noc_packet_event(
        cycles, "GRANT", input_deviceID, output_deviceID,
        output_deviceID - active_out_buffer_base, 0, input_queue.size(),
        in_buffer_occupancy[input_deviceID],
        out_buffers[output_deviceID].size(),
        static_cast<mem_fetch *>(packet.data), packet.size, "TO_L2_OUTPUT");
  }
  out_buffers[output_deviceID].push_back(packet);
  if (tma_response_multicast) {
    std::lock_guard<std::mutex> lock(tma_multicast_mutex);
    std::unordered_map<void *, std::deque<Packet> >::iterator deliveries =
        tma_response_multicast_deliveries.find(packet.data);
    if (deliveries != tma_response_multicast_deliveries.end()) {
      std::deque<Packet> same_output;
      while (!deliveries->second.empty()) {
        const Packet waiter = deliveries->second.front();
        deliveries->second.pop_front();
        if (waiter.output_deviceID == output_deviceID)
          same_output.push_back(waiter);
        else
          multicast_delivery_buffers[waiter.output_deviceID].push_back(
              waiter);
      }
      if (same_output.empty()) {
        tma_response_multicast_deliveries.erase(deliveries);
      } else {
        deliveries->second.swap(same_output);
      }
    }
  }
  max_output_occupancy[output_deviceID] =
      std::max<unsigned>(max_output_occupancy[output_deviceID],
                         out_buffers[output_deviceID].size());
  input_grants[input_deviceID]++;
  output_grants[output_deviceID]++;
  input_service_stats[input_deviceID].record_accept(packet.data_sectors);
  output_service_stats[output_deviceID].record_accept(packet.data_sectors);
  const unsigned service_slots =
      memory_transport_service_slots(packet.data_sectors);
  input_tick_service_slots[input_deviceID] += service_slots;
  output_tick_service_slots[output_deviceID] += service_slots;
  input_queue.pop_front();
  in_buffer_occupancy[input_deviceID]--;
}

void xbar_router::FinalizeLegacyServiceStats() {
  vector<bool> input_width_limited(total_nodes, false);
  vector<bool> input_downstream_full(total_nodes, false);
  vector<bool> output_width_limited(total_nodes, false);
  vector<bool> output_downstream_full(total_nodes, false);

  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    if (!InputHasPackets(input)) continue;
    if (use_voq) {
      for (unsigned output = active_out_buffer_base;
           output < active_out_buffer_base + active_out_buffers; ++output) {
        if (!InputHasPacketForOutput(input, output)) continue;
        if (Has_Buffer_Out(output, 1)) {
          if ((arbit_type == NAIVE_RR || !allow_multi_grant) &&
              input_tick_service_slots[input] != 0)
            input_width_limited[input] = true;
          if (output_tick_service_slots[output] != 0)
            output_width_limited[output] = true;
        } else {
          input_downstream_full[input] = true;
          output_downstream_full[output] = true;
        }
      }
    } else {
      const unsigned output = FirstReadyOutput(input);
      if (Has_Buffer_Out(output, 1)) {
        if ((arbit_type == NAIVE_RR || !allow_multi_grant) &&
            input_tick_service_slots[input] != 0)
          input_width_limited[input] = true;
        if (output_tick_service_slots[output] != 0)
          output_width_limited[output] = true;
      } else {
        input_downstream_full[input] = true;
        output_downstream_full[output] = true;
      }
    }
  }

  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input) {
    input_service_stats[input].record_tick_service(
        input_tick_service_slots[input]);
    if (input_width_limited[input])
      ++input_service_stats[input].width_limited_ticks;
    if (input_downstream_full[input])
      ++input_service_stats[input].downstream_full_ticks;
  }
  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output) {
    output_service_stats[output].record_tick_service(
        output_tick_service_slots[output]);
    if (output_width_limited[output])
      ++output_service_stats[output].width_limited_ticks;
    if (output_downstream_full[output])
      ++output_service_stats[output].downstream_full_ticks;
  }
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
    if (router_type == REQ_NET && GPTRACE(INTERCONNECT)) {
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
          if (first_input == UINT_MAX &&
              Trace::request_noc_sample_accepts(mf)) {
            first_input = input;
            front_mf = mf;
          }
        }
        Trace::request_noc_prearb_event(cycles, first_input, output,
                                        output - active_out_buffer_base,
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
        if (first_input == UINT_MAX && Trace::request_noc_sample_accepts(mf)) {
          first_input = input;
          front_mf = mf;
        }
      }
    }
    if (requesters > 0) *conflicts += requesters - 1;
    if (router_type == REQ_NET && GPTRACE(INTERCONNECT)) {
      Trace::request_noc_prearb_event(cycles, first_input, output,
                                      output - active_out_buffer_base,
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
  printf("%s_Network_tma_multicast_master_sectors = %llu\n", name,
         tma_multicast_master_sectors);
  printf("%s_Network_tma_multicast_merged_sectors = %llu\n", name,
         tma_multicast_merged_sectors);
  printf("%s_Network_tma_multicast_max_waiters = %llu\n", name,
         tma_multicast_max_waiters);
  printf("%s_Network_tma_response_multicast_masters = %llu\n", name,
         tma_response_multicast_masters);
  printf("%s_Network_tma_response_multicast_waiters = %llu\n", name,
         tma_response_multicast_waiters);
  printf("%s_Network_tma_response_multicast_max_waiters = %llu\n", name,
         tma_response_multicast_max_waiters);

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

  memory_transport_service_stats input_total;
  memory_transport_service_stats output_total;
  for (unsigned input = active_in_buffer_base;
       input < active_in_buffer_base + active_in_buffers; ++input)
    input_total.add(input_service_stats[input]);
  for (unsigned output = active_out_buffer_base;
       output < active_out_buffer_base + active_out_buffers; ++output)
    output_total.add(output_service_stats[output]);
  std::string input_name = std::string(name) + "_input_transport";
  std::string output_name = std::string(name) + "_output_transport";
  input_total.print(stdout, input_name.c_str());
  output_total.print(stdout, output_name.c_str());
}

bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (in_buffer_occupancy[i] > 0)
      return true;

    if (!out_buffers[i].empty())
      return true;

    if (!multicast_delivery_buffers[i].empty())
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

  assert(data != NULL);
  const mem_fetch *mf = static_cast<const mem_fetch *>(data);
  net[subnet]->Push(input_deviceID, output_deviceID, data, size,
                    memory_transport_data_sectors(mf));
}

void LocalInterconnect::PushMulticast(
    unsigned input_deviceID, unsigned output_deviceID, void *data,
    unsigned int size,
    const std::vector<std::pair<unsigned, void *> > &destinations) {
  assert(n_subnets > REPLY_NET);
  assert(input_deviceID >= n_shader);
  assert(output_deviceID < n_shader);
  assert(net[REPLY_NET]->Has_Buffer_In(input_deviceID, 1));
  assert(data != NULL);
  const mem_fetch *mf = static_cast<const mem_fetch *>(data);
  net[REPLY_NET]->PushMulticast(input_deviceID, output_deviceID, data, size,
                                memory_transport_data_sectors(mf),
                                destinations);
}

void *LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  if (ouput_deviceID < n_shader)
    subnet = 1;

  return net[subnet]->Pop(ouput_deviceID);
}

void *LocalInterconnect::Top(unsigned output_deviceID) const {
  int subnet = output_deviceID < n_shader ? REPLY_NET : REQ_NET;
  return net[subnet]->Top(output_deviceID);
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
