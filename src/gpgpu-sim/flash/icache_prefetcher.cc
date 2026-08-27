#include "icache_prefetcher.h"

#include <climits>
#include <list>

#include "../../abstract_hardware_model.h"
#include "../gpu-sim.h"
#include "../mem_fetch.h"
#include "../shader.h"

namespace flash_gpgpu_sim {

icache_prefetcher_t::icache_prefetcher_t(
    read_only_cache *l1i, unsigned sid, unsigned tpc,
    const memory_config *mem_config,
    const shader_core_config *shader_config, gpgpu_sim *gpu)
    : m_l1i(l1i),
      m_sid(sid),
      m_tpc(tpc),
      m_mem_config(mem_config),
      m_shader_config(shader_config),
      m_gpu(gpu),
      m_enabled(shader_config->icache_prefetch_enable),
      m_num_streams(shader_config->icache_prefetch_num_streams),
      m_depth(shader_config->icache_prefetch_depth),
      m_line_sz(shader_config->m_L1I_config.get_line_sz()) {
  if (m_num_streams == 0 || m_depth == 0) {
    m_enabled = false;
  }
  m_streams.resize(m_num_streams);
}

void icache_prefetcher_t::on_demand_miss(new_addr_type block_addr,
                                         unsigned long long stream_id) {
  if (!m_enabled) return;

  new_addr_type prefetch_addr = block_addr + m_line_sz;
  for (unsigned i = 0; i < m_num_streams; i++) {
    if (m_streams[i].active && m_streams[i].next_addr == prefetch_addr) {
      return;
    }
  }

  int target = -1;
  for (unsigned i = 0; i < m_num_streams; i++) {
    if (!m_streams[i].active) {
      target = i;
      break;
    }
  }
  if (target < 0) {
    unsigned min_outstanding = UINT_MAX;
    for (unsigned i = 0; i < m_num_streams; i++) {
      if (m_streams[i].outstanding < min_outstanding) {
        min_outstanding = m_streams[i].outstanding;
        target = i;
      }
    }
  }

  stream_buffer_t &stream = m_streams[target];
  stream.active = true;
  stream.next_addr = prefetch_addr;
  stream.outstanding = 0;
  stream.stream_id = stream_id;
  m_stats.streams_started++;
}

bool icache_prefetcher_t::check_useful(new_addr_type block_addr) {
  if (!m_enabled) return false;
  new_addr_type aligned_addr = m_l1i->get_config().block_addr(block_addr);
  std::set<new_addr_type>::iterator line = m_prefetched_lines.find(aligned_addr);
  if (line == m_prefetched_lines.end()) {
    return false;
  }

  m_prefetched_lines.erase(line);
  m_stats.useful++;
  return true;
}

void icache_prefetcher_t::cycle() {
  if (!m_enabled) return;

  unsigned long long time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  for (unsigned i = 0; i < m_num_streams; i++) {
    stream_buffer_t &stream = m_streams[i];
    if (!stream.active || stream.outstanding >= m_depth) continue;

    new_addr_type addr = stream.next_addr;
    unsigned cache_idx = 0;
    mem_access_sector_mask_t sector_mask;
    sector_mask.set();
    enum cache_request_status probe_status =
        m_l1i->get_tag_array()->probe(addr, cache_idx, sector_mask, false);
    if (probe_status == HIT) {
      stream.active = false;
      m_stats.hit_in_cache++;
      continue;
    }

    new_addr_type mshr_addr = m_l1i->get_config().mshr_addr(addr);
    const mshr_table &mshr = m_l1i->get_mshr();
    if (mshr.probe(mshr_addr) || mshr.probe_ready(mshr_addr)) {
      stream.active = false;
      m_stats.hit_in_mshr++;
      continue;
    }

    mem_access_t access(INST_ACC_R, addr, m_line_sz, false, m_gpu->gpgpu_ctx);
    mem_fetch *mf =
        new mem_fetch(access, NULL, stream.stream_id, READ_PACKET_SIZE, 0,
                      m_sid, m_tpc, m_mem_config, time);
    mf->set_is_prefetch(true);

    std::list<cache_event> events;
    enum cache_request_status status = m_l1i->access(addr, mf, time, events);
    if (status == MISS) {
      stream.outstanding++;
      stream.next_addr += m_line_sz;
      m_outstanding_streams[mshr_addr] = i;
      m_stats.issued++;
    } else {
      delete mf;
      if (status == HIT) {
        stream.active = false;
        m_stats.hit_in_cache++;
      }
    }
  }
}

void icache_prefetcher_t::retire_fill(new_addr_type block_addr) {
  std::map<new_addr_type, unsigned>::iterator outstanding =
      m_outstanding_streams.find(block_addr);
  if (outstanding == m_outstanding_streams.end()) {
    return;
  }
  unsigned stream_id = outstanding->second;
  if (stream_id < m_streams.size() && m_streams[stream_id].outstanding > 0) {
    m_streams[stream_id].outstanding--;
  }
  m_outstanding_streams.erase(outstanding);
}

void icache_prefetcher_t::on_fill(new_addr_type block_addr) {
  if (!m_enabled) return;
  new_addr_type aligned_addr = m_l1i->get_config().block_addr(block_addr);
  m_prefetched_lines.insert(aligned_addr);
  retire_fill(m_l1i->get_config().mshr_addr(block_addr));
}

void icache_prefetcher_t::invalidate() {
  for (unsigned i = 0; i < m_num_streams; i++) {
    m_streams[i] = stream_buffer_t();
  }
  m_prefetched_lines.clear();
  m_outstanding_streams.clear();
}

}  // namespace flash_gpgpu_sim
