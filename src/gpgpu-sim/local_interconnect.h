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

#ifndef _LOCAL_INTERCONNECT_HPP_
#define _LOCAL_INTERCONNECT_HPP_

#include <deque>
#include <iostream>
#include <map>
#include <vector>

#include "mem_transport_budget.h"
using namespace std;

enum Interconnect_type { REQ_NET = 0, REPLY_NET = 1 };

enum Arbiteration_type { NAIVE_RR = 0, iSLIP = 1 };

struct inct_config {
  // config for local interconnect
  unsigned in_buffer_limit;
  unsigned out_buffer_limit;
  unsigned subnets;
  Arbiteration_type arbiter_algo;
  unsigned verbose;
  unsigned grant_cycles;
  unsigned use_voq;
  unsigned multi_grant_request;
  unsigned multi_grant_reply;
  unsigned request_input_sectors_per_cycle;
  unsigned request_output_sectors_per_cycle;
  unsigned reply_input_sectors_per_cycle;
  unsigned reply_output_sectors_per_cycle;
};

class xbar_router {
  struct Packet;

 public:
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config);
  ~xbar_router();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, unsigned data_sectors = 1);
  void* Pop(unsigned ouput_deviceID);
  void* Top(unsigned output_deviceID) const;
  void Advance();

  bool Busy() const;
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);
  void DisplayStats(const char* name) const;

  // some stats
  unsigned long long cycles;
  unsigned long long conflicts;
  unsigned long long conflicts_util;
  unsigned long long cycles_util;
  unsigned long long reqs_util;
  unsigned long long out_buffer_full;
  unsigned long long out_buffer_util;
  unsigned long long in_buffer_full;
  unsigned long long in_buffer_util;
  unsigned long long packets_num;
  std::vector<unsigned long long> input_pushes;
  std::vector<unsigned long long> output_pushes;
  std::vector<unsigned long long> input_grants;
  std::vector<unsigned long long> output_grants;
  std::vector<unsigned long long> input_full_events;
  std::vector<unsigned long long> output_full_events;
  std::vector<unsigned> max_input_occupancy;
  std::vector<unsigned> max_output_occupancy;
  std::vector<memory_transport_service_stats> input_service_stats;
  std::vector<memory_transport_service_stats> output_service_stats;

 private:
  void iSLIP_Advance();
  void RR_Advance();
  void NumericAdvance(bool is_islip);
  void FinalizeLegacyServiceStats();
  void CollectRequestStats(bool* active, unsigned* conflicts) const;
  bool InputHasPackets(unsigned input_deviceID) const;
  bool InputHasPacketForOutput(unsigned input_deviceID,
                               unsigned output_deviceID) const;
  unsigned FirstReadyOutput(unsigned input_deviceID) const;
  unsigned InputQueueIndex(unsigned output_deviceID) const;
  const Packet* InputPacketForOutput(unsigned input_deviceID,
                                     unsigned output_deviceID) const;
  bool InputCanGrant(unsigned input_deviceID, unsigned output_deviceID,
                     const Packet& packet,
                     const std::vector<unsigned>& legacy_input_grants,
                     const std::vector<std::vector<bool> >& legacy_pairs) const;
  bool OutputCanGrant(unsigned output_deviceID, const Packet& packet,
                      const std::vector<unsigned>& legacy_output_grants) const;
  void TransferPacket(unsigned input_deviceID, unsigned output_deviceID);

  struct Packet {
    Packet(void* m_data, unsigned m_output_deviceID, unsigned m_size,
           unsigned m_data_sectors) {
      data = m_data;
      output_deviceID = m_output_deviceID;
      size = m_size;
      data_sectors = m_data_sectors;
    }
    void* data;
    unsigned output_deviceID;
    unsigned size;
    unsigned data_sectors;
  };
  vector<vector<deque<Packet> > > in_buffers;
  vector<deque<Packet> > out_buffers;
  vector<unsigned> in_buffer_occupancy;
  unsigned _n_shader, _n_mem, total_nodes;
  unsigned in_buffer_limit, out_buffer_limit;
  vector<unsigned> next_node;  // used for iSLIP arbit
  vector<unsigned> next_output;  // used for numeric RR output fairness
  unsigned next_node_id;       // used for RR arbit
  unsigned next_output_id;     // rotates numeric iSLIP output priority
  unsigned m_id;
  enum Interconnect_type router_type;
  unsigned active_in_buffers, active_out_buffers;

  // Help us to avoid iterate all buffers every time.
  unsigned active_in_buffer_base, active_out_buffer_base;
  unsigned round_in_idx(unsigned idx) const {
    return (idx - active_in_buffer_base) % active_in_buffers +
           active_in_buffer_base;
  }
  unsigned round_out_idx(unsigned idx) const {
    return (idx - active_out_buffer_base) % active_out_buffers +
           active_out_buffer_base;
  }

  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;
  bool use_voq;
  bool allow_multi_grant;
  unsigned input_sector_width;
  unsigned output_sector_width;
  vector<memory_transport_service_budget> input_budgets;
  vector<memory_transport_service_budget> output_budgets;
  vector<unsigned> input_tick_service_slots;
  vector<unsigned> output_tick_service_slots;

  friend class LocalInterconnect;
};

class LocalInterconnect {
 public:
  LocalInterconnect(const struct inct_config& m_localinct_config);
  ~LocalInterconnect();
  static LocalInterconnect* New(const struct inct_config& m_inct_config);
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void* Top(unsigned output_deviceID) const;
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

 protected:
  const inct_config& m_inct_config;

  unsigned n_shader, n_mem;
  unsigned n_subnets;
  vector<xbar_router*> net;
};

#endif
