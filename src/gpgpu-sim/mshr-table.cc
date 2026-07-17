// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#include "gpu-cache.h"

bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator entry = m_data.find(block_addr);
  return entry != m_data.end() && !entry->second.m_ready;
}

bool mshr_table::probe_ready(new_addr_type block_addr) const {
  table::const_iterator entry = m_data.find(block_addr);
  return entry != m_data.end() && entry->second.m_ready;
}

bool mshr_table::ready_for_forward(new_addr_type block_addr) const {
  table::const_iterator entry = m_data.find(block_addr);
  return entry != m_data.end() && entry->second.m_ready &&
         entry->second.m_list.size() < m_max_merged;
}

bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator entry = m_data.find(block_addr);
  if (entry != m_data.end() && entry->second.m_ready)
    return true;
  else if (entry != m_data.end())
    return entry->second.m_list.size() >= m_max_merged;
  else
    return m_data.size() >= m_num_entries;
}

void mshr_table::add(new_addr_type block_addr, mem_fetch *mf, bool is_atomic) {
  assert(!probe_ready(block_addr));
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  if (is_atomic) m_data[block_addr].m_has_atomic = true;
}

void mshr_table::add_ready(new_addr_type block_addr, mem_fetch *mf) {
  assert(ready_for_forward(block_addr));
  m_data[block_addr].m_list.push_back(mf);
}

void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  assert(!busy());
  table::iterator entry = m_data.find(block_addr);
  assert(entry != m_data.end());
  assert(!entry->second.m_ready);
  entry->second.m_ready = true;
  m_current_response.push_back(block_addr);
  has_atomic = entry->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

mem_fetch *mshr_table::next_access() {
  assert(access_ready());
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();
  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    m_data.erase(block_addr);
    m_current_response.pop_front();
  }
  return result;
}
