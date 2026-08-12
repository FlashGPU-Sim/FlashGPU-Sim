// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <algorithm>
#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

const char *mem_sub_partition_full_stat_str(
    enum mem_sub_partition_full_stat stat) {
  switch (stat) {
    case MSP_FULL_ICNT_TO_L2_NOT_ENOUGH_SECTOR_SLOTS:
      return "ICNT_TO_L2_NOT_ENOUGH_SECTOR_SLOTS";
    case MSP_FULL_ICNT_TO_L2_QUEUE_FULL:
      return "ICNT_TO_L2_QUEUE_FULL";
    case MSP_FULL_ICNT_TO_L2_QUEUE_NEAR_FULL:
      return "ICNT_TO_L2_QUEUE_NEAR_FULL";
    case MSP_FULL_L2_DRAM_QUEUE_FULL:
      return "L2_DRAM_QUEUE_FULL";
    case MSP_FULL_DRAM_L2_QUEUE_FULL:
      return "DRAM_L2_QUEUE_FULL";
    case MSP_FULL_L2_ICNT_QUEUE_FULL:
      return "L2_ICNT_QUEUE_FULL";
    case MSP_FULL_L2_DATA_PORT_BUSY:
      return "L2_DATA_PORT_BUSY";
    case MSP_FULL_L2_FILL_PORT_BUSY:
      return "L2_FILL_PORT_BUSY";
    case MSP_FULL_L2_READY_BLOCKED_BY_L2_ICNT_QUEUE:
      return "L2_READY_BLOCKED_BY_L2_ICNT_QUEUE";
    case NUM_MEM_SUB_PARTITION_FULL_STATS:
      break;
  }
  return "UNKNOWN";
}

namespace {

class l2_request_trace {
 public:
  static l2_request_trace &instance() {
    static l2_request_trace trace;
    return trace;
  }

  bool cache_accept_enabled() const { return m_enabled && m_trace_cache_accept; }

  void log(const char *event, unsigned long long cycle, unsigned subpart_id,
           mem_fetch *mf, const char *status) {
    if (!m_enabled || !mf || cycle > m_max_cycle) return;

    const mem_access_type access_type = mf->get_access_type();
    if (m_tma_only && access_type != TMA_ACC_R && access_type != TMA_ACC_W &&
        access_type != CP_ASYNC_ACC_R)
      return;

    const mem_fetch *original_mf = mf->get_original_mf();
    const unsigned original_uid =
        original_mf ? original_mf->get_request_uid() : mf->get_request_uid();
    const unsigned long sector_mask =
        mf->get_access_sector_mask().to_ulong();

    flockfile(m_file);
    fprintf(m_file,
            "%llu,%s,%u,%u,0x%llx,0x%llx,%u,%u,%u,%u,%u,%s,%u,%u,%u,0x%lx,%s\n",
            cycle, event, subpart_id, mf->get_sub_partition_id(),
            (unsigned long long)mf->get_addr(),
            (unsigned long long)mf->get_partition_addr(), mf->get_sid(),
            mf->get_tpc(), mf->get_wid(), mf->get_request_uid(), original_uid,
            mem_access_type_str(access_type), mf->get_is_write(),
            mf->get_data_size(), mf->get_access_size(), sector_mask, status);
    ++m_lines;
    if ((m_lines & ((1ULL << 20) - 1)) == 0) fflush(m_file);
    funlockfile(m_file);
  }

 private:
  l2_request_trace()
      : m_enabled(false),
        m_trace_cache_accept(false),
        m_tma_only(false),
        m_max_cycle(ULLONG_MAX),
        m_file(NULL),
        m_buffer(NULL),
        m_lines(0) {
    const char *path = getenv("FLASHGPU_L2_TRACE_CSV");
    if (!path || path[0] == '\0') return;

    const char *max_cycle = getenv("FLASHGPU_L2_TRACE_MAX_CYCLE");
    if (max_cycle && max_cycle[0] != '\0')
      m_max_cycle = strtoull(max_cycle, NULL, 0);

    const char *cache_accept = getenv("FLASHGPU_L2_TRACE_CACHE_ACCEPT");
    m_trace_cache_accept =
        cache_accept && cache_accept[0] != '\0' && strcmp(cache_accept, "0");

    const char *tma_only = getenv("FLASHGPU_L2_TRACE_TMA_ONLY");
    m_tma_only = tma_only && tma_only[0] != '\0' && strcmp(tma_only, "0");

    m_file = fopen(path, "w");
    if (!m_file) {
      perror("FLASHGPU_L2_TRACE_CSV");
      return;
    }

    m_buffer = (char *)malloc(16 * 1024 * 1024);
    if (m_buffer) setvbuf(m_file, m_buffer, _IOFBF, 16 * 1024 * 1024);

    fprintf(m_file,
            "cycle,event,subpart,mf_subpart,addr,partition_addr,requestor_sm,"
            "tpc,wid,uid,orig_uid,type,is_write,data_size,access_size,"
            "sector_mask,status\n");
    m_enabled = true;
    printf("FLASHGPU_L2_TRACE_CSV enabled: path=%s max_cycle=%llu "
           "tma_only=%u cache_accept=%u\n",
           path, m_max_cycle, m_tma_only ? 1 : 0,
           m_trace_cache_accept ? 1 : 0);
  }

  ~l2_request_trace() {
    if (m_file) {
      fflush(m_file);
      fclose(m_file);
    }
    free(m_buffer);
  }

  bool m_enabled;
  bool m_trace_cache_accept;
  bool m_tma_only;
  unsigned long long m_max_cycle;
  FILE *m_file;
  char *m_buffer;
  unsigned long long m_lines;
};

static void trace_l2_event(const char *event, unsigned long long cycle,
                           unsigned subpart_id, mem_fetch *mf,
                           const char *status) {
  l2_request_trace::instance().log(event, cycle, subpart_id, mf, status);
}

static unsigned coarse_l2_partition_id(unsigned id, unsigned total,
                                       unsigned partition_count) {
  assert(partition_count > 0);
  assert(total > 0);
  unsigned long long partition =
      (static_cast<unsigned long long>(id) * partition_count) / total;
  if (partition >= partition_count) partition = partition_count - 1;
  return static_cast<unsigned>(partition);
}

static unsigned l2_sector_work_packages(unsigned bytes) {
  assert(bytes > 0);
  return (bytes + SECTOR_SIZE - 1) / SECTOR_SIZE;
}

}  // namespace

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             memory_stats_manager_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),
      m_config(config),
      m_mem_stats(stats),
      m_arbitration_metadata(config),
      m_simple_dram_issue_credit(0),
      m_simple_dram_return_credit(0),
      m_simple_dram_cycles(0),
      m_simple_dram_issue_requests(0),
      m_simple_dram_issue_atoms(0),
      m_simple_dram_return_requests(0),
      m_simple_dram_return_atoms(0),
      m_simple_dram_issue_no_request_cycles(0),
      m_simple_dram_issue_backpressure_cycles(0),
      m_simple_dram_return_not_ready_cycles(0),
      m_simple_dram_return_backpressure_cycles(0),
      m_simple_dram_queue_length_sum(0),
      m_simple_dram_queue_length_max(0),
      m_gpu(gpu) {
  m_dram = new dram_t(m_id, m_config, stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_GPPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  if (config->simple_dram_model && config->simple_dram_max_inflight != 0) {
    // One private credit is reserved for each subpartition. The remainder is
    // shared, keeping the configured value equal to the total in-flight cap.
    m_shared_credit_limit = config->simple_dram_max_inflight -
                            config->m_n_sub_partition_per_memory_channel;
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = !m_dram_latency_queue.empty();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_GPPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  const unsigned long long now =
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  const unsigned long long numerator = m_config->simple_dram_service_rate_num;
  const unsigned long long denominator = m_config->simple_dram_service_rate_den;
  const unsigned long long max_request_atoms =
      (MAX_MEMORY_ACCESS_SIZE + m_config->dram_atom_size - 1) /
      m_config->dram_atom_size;
  const unsigned long long max_service_atoms =
      (numerator + denominator - 1) / denominator;
  const unsigned long long max_credit_atoms =
      std::max(max_service_atoms, max_request_atoms);
  const unsigned long long whole_credit_cap = max_credit_atoms * denominator;
  const unsigned long long credit_cap = whole_credit_cap + denominator - 1;

  m_simple_dram_cycles++;
  m_simple_dram_issue_credit += numerator;
  m_simple_dram_return_credit += numerator;
  if (m_simple_dram_issue_credit > credit_cap) {
    m_simple_dram_issue_credit =
        whole_credit_cap + m_simple_dram_issue_credit % denominator;
  }
  if (m_simple_dram_return_credit > credit_cap) {
    m_simple_dram_return_credit =
        whole_credit_cap + m_simple_dram_return_credit % denominator;
  }

  // Complete as many fixed-latency requests as the return service permits.
  bool saw_ready_return = false;
  bool return_blocked = false;
  while (!m_dram_latency_queue.empty() &&
         now >= m_dram_latency_queue.front().ready_cycle) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    const unsigned long long atoms = std::max(
        1ull, (static_cast<unsigned long long>(mf_return->get_data_size()) +
               m_config->dram_atom_size - 1) /
                  m_config->dram_atom_size);
    const unsigned long long cost = atoms * denominator;
    saw_ready_return = true;
    if (m_simple_dram_return_credit < cost) break;

    const bool is_writeback = mf_return->get_access_type() == L1_WRBK_ACC ||
                              mf_return->get_access_type() == L2_WRBK_ACC;
    if (!is_writeback) {
      const unsigned dest_global_spid = mf_return->get_sub_partition_id();
      const int dest_spid =
          global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      if (m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        return_blocked = true;
        break;
      }

      mf_return->set_reply();
      m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
      mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, now);
      m_arbitration_metadata.return_credit(dest_spid);
      MEMPART_GPPRINTF(
          "mem_fetch request %p return from dram to sub partition %d\n",
          mf_return, dest_spid);
    } else {
      this->set_done(mf_return);
      delete mf_return;
    }

    m_dram_latency_queue.pop_front();
    m_simple_dram_return_credit -= cost;
    m_simple_dram_return_requests++;
    m_simple_dram_return_atoms += atoms;
  }
  if (return_blocked) {
    m_simple_dram_return_backpressure_cycles++;
  } else if (!saw_ready_return) {
    m_simple_dram_return_not_ready_cycles++;
    // Do not accumulate an unbounded burst while the fixed-latency pipeline is
    // empty. Preserve only the fractional phase of the configured rate.
    m_simple_dram_return_credit %= denominator;
  }

  // Admit requests from L2 in round-robin order until the aggregate service
  // credit is exhausted. Each request consumes ceil(bytes / atom_size) atoms.
  bool saw_issue_request = false;
  bool issue_backpressured = false;
  while (true) {
    bool issued = false;
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
         p++) {
      const int spid = (p + last_issued_partition + 1) %
                       m_config->m_n_sub_partition_per_memory_channel;
      if (m_sub_partition[spid]->L2_dram_queue_empty()) continue;
      saw_issue_request = true;
      if (!can_issue_to_dram(spid)) {
        issue_backpressured = true;
        continue;
      }

      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      const unsigned long long atoms =
          std::max(1ull, (static_cast<unsigned long long>(mf->get_data_size()) +
                          m_config->dram_atom_size - 1) /
                             m_config->dram_atom_size);
      const unsigned long long cost = atoms * denominator;
      // Preserve round-robin fairness across mixed request sizes. Skipping a
      // large request here would let a continuous stream of one-atom requests
      // consume every new credit and starve it indefinitely.
      if (m_simple_dram_issue_credit < cost) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_GPPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = now + m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, now);
      m_arbitration_metadata.borrow_credit(spid);
      m_mem_stats->get_stats()->memlatstat_dram_access(mf);
      m_simple_dram_issue_credit -= cost;
      m_simple_dram_issue_requests++;
      m_simple_dram_issue_atoms += atoms;
      issued = true;
      break;
    }
    if (!issued) break;
  }
  if (!saw_issue_request) {
    m_simple_dram_issue_no_request_cycles++;
    m_simple_dram_issue_credit %= denominator;
  } else if (issue_backpressured) {
    m_simple_dram_issue_backpressure_cycles++;
  }

  const unsigned long long queue_length = m_dram_latency_queue.size();
  m_simple_dram_queue_length_sum += queue_length;
  m_simple_dram_queue_length_max =
      std::max(m_simple_dram_queue_length_max, queue_length);
}

void memory_partition_unit::print_stat(FILE *fp) const {
  if (!m_config->simple_dram_model) {
    m_dram->print_stat(fp);
    return;
  }

  const double issue_util =
      m_simple_dram_cycles == 0
          ? 0.0
          : 100.0 * m_simple_dram_issue_atoms *
                m_config->simple_dram_service_rate_den /
                (m_simple_dram_cycles * m_config->simple_dram_service_rate_num);
  const double return_util =
      m_simple_dram_cycles == 0
          ? 0.0
          : 100.0 * m_simple_dram_return_atoms *
                m_config->simple_dram_service_rate_den /
                (m_simple_dram_cycles * m_config->simple_dram_service_rate_num);
  const double queue_avg =
      m_simple_dram_cycles == 0
          ? 0.0
          : static_cast<double>(m_simple_dram_queue_length_sum) /
                m_simple_dram_cycles;
  fprintf(fp,
          "simple_dram_service[%u] rate=%u/%u cycles=%llu "
          "issue_requests=%llu issue_atoms=%llu issue_util=%.4f%% "
          "return_requests=%llu return_atoms=%llu return_util=%.4f%% "
          "issue_no_request_cycles=%llu issue_backpressure_cycles=%llu "
          "return_not_ready_cycles=%llu return_backpressure_cycles=%llu "
          "queue_avg=%.4f queue_max=%llu\n",
          m_id, m_config->simple_dram_service_rate_num,
          m_config->simple_dram_service_rate_den, m_simple_dram_cycles,
          m_simple_dram_issue_requests, m_simple_dram_issue_atoms, issue_util,
          m_simple_dram_return_requests, m_simple_dram_return_atoms,
          return_util, m_simple_dram_issue_no_request_cycles,
          m_simple_dram_issue_backpressure_cycles,
          m_simple_dram_return_not_ready_cycles,
          m_simple_dram_return_backpressure_cycles, queue_avg,
          m_simple_dram_queue_length_max);
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_GPPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
    }
  } else {
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_GPPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_GPPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
}

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const memory_config *config,
                                           memory_stats_manager_t *stats,
                                           class gpgpu_sim *gpu)
    : m_id(sub_partition_id),
      m_config(config),
      m_l2_port_model(
          l2_port_model_from_config(config->l2_multi_issue_port_model)) {
  m_mem_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;
  m_next_rop_sequence = 0;
  memset(m_full_state_stats, 0, sizeof(m_full_state_stats));
  m_l2_partition_remote_accesses = 0;
  m_l2_partition_extra_latency_cycles = 0;
  m_pending_l2_writeback = NULL;
  m_pending_l2_fill = NULL;
  if (m_l2_port_model == l2_port_model_kind::multi_issue) {
    assert(!m_config->m_L2_config.disabled());
    assert(m_config->m_L2_config.m_cache_type == SECTOR);
    m_l2_multi_issue_ports.configure(m_config->l2_lookup_sectors_per_cycle,
                                     m_config->l2_data_port_sectors_per_cycle,
                                     m_config->l2_fill_port_sectors_per_cycle);
  }

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
  delete m_pending_l2_writeback;
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::process_l2_access_result(
    mem_fetch *mf, cache_request_status status,
    const std::list<cache_event> &events) {
  if (status != RESERVATION_FAIL &&
      l2_request_trace::instance().cache_accept_enabled()) {
    trace_l2_event("CACHE_ACCEPT",
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, m_id, mf,
                   cache_request_status_str(status));
  }
  const bool write_sent = was_write_sent(events);
  const bool read_sent = was_read_sent(events);
  MEM_SUBPART_GPPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                       mf->get_addr(), status);

  if (status == HIT) {
    if (!write_sent) {
      assert(!read_sent);
      if (mf->get_access_type() == L1_WRBK_ACC) {
        m_request_tracker.erase(mf);
        delete mf;
      } else {
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      }
      m_icnt_L2_queue->pop();
    } else {
      assert(write_sent);
      m_icnt_L2_queue->pop();
    }
  } else if (status != RESERVATION_FAIL) {
    if (mf->is_write() &&
        (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
         m_config->m_L2_config.m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        !was_writeallocate_sent(events)) {
      if (mf->get_access_type() == L1_WRBK_ACC) {
        m_request_tracker.erase(mf);
        delete mf;
      } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      }
    }
    m_icnt_L2_queue->pop();
  } else {
    assert(!write_sent);
    assert(!read_sent);
  }
}

void memory_sub_partition::service_ready_l2_response() {
  if (m_config->m_L2_config.disabled() || !m_L2cache->access_ready() ||
      m_L2_icnt_queue->full())
    return;

  mem_fetch *mf = m_L2cache->next_access();
  if (mf->get_access_type() !=
      L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                        // upper level cache
    mf->set_reply();
    mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_L2_icnt_queue->push(mf);
    return;
  }

  if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
    mem_fetch *original_wr_mf = mf->get_original_wr_mf();
    assert(original_wr_mf);
    original_wr_mf->set_reply();
    original_wr_mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                               m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_L2_icnt_queue->push(original_wr_mf);
  }
  m_request_tracker.erase(mf);
  delete mf;
}

void memory_sub_partition::service_dram_to_l2_legacy() {
  if (m_dram_L2_queue->empty()) return;

  mem_fetch *mf = m_dram_L2_queue->top();
  if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
    if (m_L2cache->fill_port_free()) {
      mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                              m_memcpy_cycle_offset);
      m_dram_L2_queue->pop();
    }
  } else if (!m_L2_icnt_queue->full()) {
    if (mf->is_write() && mf->get_type() == WRITE_ACK)
      mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_L2_icnt_queue->push(mf);
    m_dram_L2_queue->pop();
  }
}

void memory_sub_partition::service_dram_to_l2_multi_issue() {
  bool continue_fill_service = true;
  while (continue_fill_service && !m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (m_L2cache->waiting_for_fill(mf)) {
      if (m_pending_l2_fill == NULL) {
        m_pending_l2_fill = mf;
        m_pending_l2_fill_work.start(
            l2_sector_work_packages(mf->get_data_size()));
      } else {
        assert(m_pending_l2_fill == mf);
        assert(m_pending_l2_fill_work.active());
      }

      if (!m_pending_l2_fill_work.service_fill(m_l2_multi_issue_ports)) break;

      mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2cache->fill_multi_issue_port_model(mf, m_gpu->gpu_sim_cycle +
                                                     m_gpu->gpu_tot_sim_cycle +
                                                     m_memcpy_cycle_offset);
      m_dram_L2_queue->pop();
      m_pending_l2_fill = NULL;
      assert(!m_pending_l2_fill_work.active());
    } else {
      assert(m_pending_l2_fill == NULL);
      assert(!m_pending_l2_fill_work.active());
      if (m_L2_icnt_queue->full()) break;
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
      // The bypass path is not an L2 fill-port operation. Preserve its
      // legacy single-response-per-tick behavior.
      continue_fill_service = false;
    }
  }
}

void memory_sub_partition::service_l2_requests_legacy() {
  if (m_L2_dram_queue->full() || m_icnt_L2_queue->empty()) return;

  mem_fetch *mf = m_icnt_L2_queue->top();
  if (!m_config->m_L2_config.disabled() &&
      ((m_config->m_L2_texure_only && mf->istexture()) ||
       !m_config->m_L2_texure_only)) {
    // L2 is enabled and access is for L2.
    const bool output_full = m_L2_icnt_queue->full();
    const bool port_free = m_L2cache->data_port_free();
    if (!output_full && port_free) {
      std::list<cache_event> events;
      const cache_request_status status =
          m_L2cache->access(mf->get_addr(), mf,
                            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset,
                            events);
      process_l2_access_result(mf, status, events);
    }
    return;
  }

  // L2 is disabled or this is a non-texture access to a texture-only L2.
  mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  m_L2_dram_queue->push(mf);
  m_icnt_L2_queue->pop();
}

void memory_sub_partition::service_l2_requests_multi_issue() {
  bool continue_l2_service = true;

  // A dirty victim is removed from the cache's miss queue when generated. It
  // becomes visible downstream only after every dirty sector uses the data
  // port shared with L2 hits.
  if (m_pending_l2_writeback != NULL) {
    assert(m_pending_l2_writeback_work.active());
    if (m_pending_l2_writeback_work.service_data(
            m_l2_multi_issue_ports, L2_MULTI_ISSUE_DIRTY_EVICTION)) {
      m_L2cache->release_deferred_writeback(m_pending_l2_writeback);
      m_pending_l2_writeback = NULL;
      assert(!m_pending_l2_writeback_work.active());
    } else {
      continue_l2_service = false;
    }
  }

  while (continue_l2_service && !m_L2_dram_queue->full() &&
         !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         !m_config->m_L2_texure_only)) {
      if (m_L2_icnt_queue->full()) break;

      // push() breaks a sector-L2 demand into 32-byte internal mem_fetch
      // children. They remain associated with their parent; these counters
      // represent sector work packages rather than request counts.
      assert(mf->get_data_size() == SECTOR_SIZE);
      unsigned prospective_dirty_eviction_sectors = 0;
      const cache_request_status probe_status = m_L2cache->probe(
          mf->get_addr(), mf, prospective_dirty_eviction_sectors);
      const bool ready_read_forward =
          m_L2cache->ready_read_forward_eligible(mf, probe_status);
      const bool needs_data_port = l2_multi_issue_needs_data_port(
          probe_status, prospective_dirty_eviction_sectors, ready_read_forward);
      const bool lookup_available = m_l2_multi_issue_ports.can_accept_lookup(1);
      bool data_port_available = true;
      if (needs_data_port)
        data_port_available = m_l2_multi_issue_ports.data_port_has_capacity();
      if (!lookup_available || !data_port_available) break;

      std::list<cache_event> events;
      mem_fetch *deferred_writeback = NULL;
      unsigned deferred_writeback_sectors = 0;
      const cache_request_status status = m_L2cache->access_multi_issue(
          mf->get_addr(), mf,
          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
              m_memcpy_cycle_offset,
          events, deferred_writeback, deferred_writeback_sectors);
      m_l2_multi_issue_ports.accept_lookup(1);

      if (probe_status == HIT)
        assert(status == HIT || status == RESERVATION_FAIL);
      if (ready_read_forward) {
        // The authoritative predicate is shared with send_read_request(), so
        // this access joins the ready MSHR without touching the tag replacement
        // candidate. Actual events guard against charging a fictional dirty
        // eviction.
        assert(status != RESERVATION_FAIL);
        assert(status != HIT);
        assert(deferred_writeback == NULL);
        assert(deferred_writeback_sectors == 0);
        assert(events.empty());
      }
      if (status == HIT) {
        const unsigned accepted =
            m_l2_multi_issue_ports.accept_data(1, L2_MULTI_ISSUE_HIT_DATA);
        assert(accepted == 1);
        assert(deferred_writeback == NULL);
      }

      if (deferred_writeback != NULL) {
        assert(status != RESERVATION_FAIL);
        assert(m_pending_l2_writeback == NULL);
        m_pending_l2_writeback = deferred_writeback;
        assert(deferred_writeback_sectors ==
               prospective_dirty_eviction_sectors);
        m_pending_l2_writeback_work.start(deferred_writeback_sectors);
        if (m_pending_l2_writeback_work.service_data(
                m_l2_multi_issue_ports, L2_MULTI_ISSUE_DIRTY_EVICTION)) {
          m_L2cache->release_deferred_writeback(m_pending_l2_writeback);
          m_pending_l2_writeback = NULL;
          assert(!m_pending_l2_writeback_work.active());
        } else {
          continue_l2_service = false;
        }
      } else if (status != RESERVATION_FAIL && !ready_read_forward) {
        assert(prospective_dirty_eviction_sectors == 0);
        assert(deferred_writeback_sectors == 0);
      }

      process_l2_access_result(mf, status, events);
      if (status == RESERVATION_FAIL) continue_l2_service = false;
    } else {
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
      continue_l2_service = false;
    }
  }
}

void memory_sub_partition::cycle_legacy_l2_port_model() {
  service_dram_to_l2_legacy();
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();
  service_l2_requests_legacy();
}

void memory_sub_partition::cycle_multi_issue_l2_port_model() {
  m_l2_multi_issue_ports.begin_cycle();
  service_dram_to_l2_multi_issue();
  if (!m_config->m_L2_config.disabled())
    m_L2cache->cycle_multi_issue_port_model();
  service_l2_requests_multi_issue();
}

void memory_sub_partition::enqueue_ready_rop(unsigned cycle) {
  // Keep local and remote traffic separate so a far-L2 request cannot block an
  // already-ready local request.
  mem_fetch *mf = NULL;
  if (!m_icnt_L2_queue->full() && pop_ready_rop(cycle, mf)) {
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  service_ready_l2_response();

  switch (m_l2_port_model) {
    case l2_port_model_kind::legacy:
      cycle_legacy_l2_port_model();
      break;
    case l2_port_model_kind::multi_issue:
      cycle_multi_issue_l2_port_model();
      break;
  }

  enqueue_ready_rop(cycle);
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::l2_data_port_busy() const {
  assert(!m_config->m_L2_config.disabled());
  switch (m_l2_port_model) {
    case l2_port_model_kind::legacy:
      return !m_L2cache->data_port_free();
    case l2_port_model_kind::multi_issue:
      return m_pending_l2_writeback != NULL ||
             m_l2_multi_issue_ports.data_remaining() == 0;
  }
  abort();
}

bool memory_sub_partition::l2_fill_port_busy() const {
  assert(!m_config->m_L2_config.disabled());
  switch (m_l2_port_model) {
    case l2_port_model_kind::legacy:
      return !m_L2cache->fill_port_free();
    case l2_port_model_kind::multi_issue:
      return m_pending_l2_fill != NULL ||
             m_l2_multi_issue_ports.fill_remaining() == 0;
  }
  abort();
}

void memory_sub_partition::record_full_state(unsigned size) {
  if (!full(size)) return;

  m_full_state_stats[MSP_FULL_ICNT_TO_L2_NOT_ENOUGH_SECTOR_SLOTS]++;
  if (m_icnt_L2_queue->full()) {
    m_full_state_stats[MSP_FULL_ICNT_TO_L2_QUEUE_FULL]++;
  } else {
    m_full_state_stats[MSP_FULL_ICNT_TO_L2_QUEUE_NEAR_FULL]++;
  }

  if (m_L2_dram_queue->full())
    m_full_state_stats[MSP_FULL_L2_DRAM_QUEUE_FULL]++;
  if (m_dram_L2_queue->full())
    m_full_state_stats[MSP_FULL_DRAM_L2_QUEUE_FULL]++;
  if (m_L2_icnt_queue->full())
    m_full_state_stats[MSP_FULL_L2_ICNT_QUEUE_FULL]++;

  if (!m_config->m_L2_config.disabled()) {
    if (l2_data_port_busy()) m_full_state_stats[MSP_FULL_L2_DATA_PORT_BUSY]++;
    if (l2_fill_port_busy()) m_full_state_stats[MSP_FULL_L2_FILL_PORT_BUSY]++;
    if (m_L2cache->access_ready() && m_L2_icnt_queue->full()) {
      m_full_state_stats[MSP_FULL_L2_READY_BLOCKED_BY_L2_ICNT_QUEUE]++;
    }
  }
}

void memory_sub_partition::accumulate_full_state_stats(
    unsigned long long *stats) const {
  for (unsigned i = 0; i < NUM_MEM_SUB_PARTITION_FULL_STATS; ++i)
    stats[i] += m_full_state_stats[i];
}

void memory_sub_partition::accumulate_l2_multi_issue_port_stats(
    l2_multi_issue_port_stats &stats) const {
  stats += m_l2_multi_issue_ports.stats();
}

void memory_sub_partition::accumulate_l2_partition_stats(
    unsigned long long &remote_accesses,
    unsigned long long &extra_latency) const {
  remote_accesses += m_l2_partition_remote_accesses;
  extra_latency += m_l2_partition_extra_latency_cycles;
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const {
  return !m_request_tracker.empty() || m_pending_l2_writeback != NULL;
}

unsigned memory_sub_partition::l2_partition_extra_latency(
    const mem_fetch *mf) const {
  if (m_config->l2_partition_extra_latency == 0 ||
      m_config->l2_partition_count <= 1 || mf == NULL)
    return 0;

  const unsigned sid = mf->get_sid();
  const unsigned num_sms = m_gpu->get_config().num_shader();
  if (sid >= num_sms) return 0;

  const unsigned sm_partition =
      coarse_l2_partition_id(sid, num_sms, m_config->l2_partition_count);
  const unsigned l2_partition =
      coarse_l2_partition_id(m_id, m_config->m_n_mem_sub_partition,
                             m_config->l2_partition_count);
  if (sm_partition == l2_partition) return 0;

  return m_config->l2_partition_extra_latency;
}

void memory_sub_partition::push_rop_delay(mem_fetch *mf,
                                          unsigned long long ready_cycle,
                                          bool remote) {
  rop_delay_t r;
  r.ready_cycle = ready_cycle;
  r.sequence = m_next_rop_sequence++;
  r.req = mf;
  if (remote)
    m_rop_remote.push(r);
  else
    m_rop_local.push(r);
}

bool memory_sub_partition::pop_ready_rop(unsigned long long cycle,
                                         mem_fetch *&mf) {
  const bool local_ready =
      !m_rop_local.empty() && cycle >= m_rop_local.top().ready_cycle;
  const bool remote_ready =
      !m_rop_remote.empty() && cycle >= m_rop_remote.top().ready_cycle;
  if (!local_ready && !remote_ready) return false;

  rop_delay_queue_t *queue = NULL;
  if (local_ready && remote_ready) {
    const rop_delay_t &local = m_rop_local.top();
    const rop_delay_t &remote = m_rop_remote.top();
    queue = (local.ready_cycle <= remote.ready_cycle) ? &m_rop_local
                                                     : &m_rop_remote;
  } else {
    queue = local_ready ? &m_rop_local : &m_rop_remote;
  }

  rop_delay_t r = queue->top();
  queue->pop();
  mf = r.req;
  return true;
}

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();

  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else {
    if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
      sector_mask.set();
      // This is for constant cache.
    } else if (mf->get_data_size() == 2 * SECTOR_SIZE &&
               (sector_mask.all() || sector_mask.none())) {
      sector_mask.reset();
      const unsigned start =
          (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0) ? 0 : 2;
      sector_mask.set(start);
      sector_mask.set(start + 1);
    }

    const new_addr_type line_base =
        mf->get_addr() - (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE);

    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            line_base + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_mem_stats->get_stats()->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      trace_l2_event("REQ", cycle, m_id, req, "ICNT_TO_L2");
      const unsigned extra_latency = l2_partition_extra_latency(req);
      if (extra_latency > 0) {
        m_l2_partition_remote_accesses++;
        m_l2_partition_extra_latency_cycles += extra_latency;
      }
      if (req->istexture() && extra_latency == 0) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        unsigned long long ready_cycle = cycle + extra_latency;
        if (!req->istexture()) ready_cycle += m_config->rop_latency;
        push_rop_delay(req, ready_cycle, extra_latency > 0);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  if (mf) {
    trace_l2_event("RESP", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
                   m_id, mf, "L2_TO_ICNT");
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  auto m_stats = m_mem_stats->get_stats();
  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}
