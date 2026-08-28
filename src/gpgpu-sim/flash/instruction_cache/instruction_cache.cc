#include "instruction_cache.h"

#include <algorithm>

#include "../../gpu-sim.h"
#include "../../mem_fetch.h"

namespace flash_gpgpu_sim {

instruction_cache::instruction_cache(
    const char *name, cache_config &config, int core_id, int type_id,
    mem_fetch_interface *memport, enum mem_fetch_status status,
    enum cache_gpu_level level, gpgpu_sim *gpu)
    : read_only_cache(name, config, core_id, type_id, memport, status, level,
                      gpu) {}

bool instruction_cache::schedule_preload_fill(mem_fetch *mf,
                                              uint64_t ready_cycle) {
  const auto miss = std::find(m_miss_queue.begin(), m_miss_queue.end(), mf);
  if (miss == m_miss_queue.end()) return false;
  m_miss_queue.erase(miss);

  if (m_config.get_mshr_type() == ASSOC) {
    mf->set_reply();
    m_preload_fills.push_back({mf, ready_cycle});
    return true;
  }
  assert(m_config.get_mshr_type() == SECTOR_ASSOC);

  // Mirror the L2 request breakdown so baseline_cache::fill retires the
  // original ICC MSHR only after every sector reply has arrived.
  const new_addr_type line_base = m_config.block_addr(mf->get_addr());
  for (unsigned sector = 0; sector < m_config.get_line_sz() / SECTOR_SIZE;
       ++sector) {
    mem_access_byte_mask_t byte_mask;
    for (unsigned byte = sector * SECTOR_SIZE;
         byte < (sector + 1) * SECTOR_SIZE; ++byte) {
      byte_mask.set(byte);
    }
    mem_access_sector_mask_t sector_mask;
    sector_mask.set(sector);
    mem_access_t access(
        mf->get_access_type(), line_base + sector * SECTOR_SIZE, SECTOR_SIZE,
        false, mf->get_access_warp_mask(),
        mf->get_access_byte_mask() & byte_mask, sector_mask, m_gpu->gpgpu_ctx);
    mem_fetch *reply =
        new mem_fetch(access, nullptr, mf->get_streamID(), mf->get_ctrl_size(),
                      mf->get_wid(), mf->get_sid(), mf->get_tpc(),
                      mf->get_mem_config(), ready_cycle, mf);
    reply->set_reply();

    const auto position = std::upper_bound(
        m_preload_fills.begin(), m_preload_fills.end(), ready_cycle,
        [](uint64_t cycle, const preload_fill &fill) {
          return cycle < fill.ready_cycle;
        });
    m_preload_fills.insert(position, {reply, ready_cycle});
  }
  return true;
}

void instruction_cache::cycle() {
  const uint64_t now = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  while (!m_preload_fills.empty() &&
         m_preload_fills.front().ready_cycle <= now && fill_port_free()) {
    mem_fetch *mf = m_preload_fills.front().mf;
    m_preload_fills.pop_front();
    fill(mf, static_cast<unsigned>(now));
  }
  baseline_cache::cycle();
}

}  // namespace flash_gpgpu_sim
