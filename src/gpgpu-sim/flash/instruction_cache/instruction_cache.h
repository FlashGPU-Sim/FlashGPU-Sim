#ifndef FLASH_GPGPU_SIM_INSTRUCTION_CACHE_INSTRUCTION_CACHE_H
#define FLASH_GPGPU_SIM_INSTRUCTION_CACHE_INSTRUCTION_CACHE_H

#include <cstdint>
#include <deque>

#include "../../gpu-cache.h"

namespace flash_gpgpu_sim {

class instruction_cache : public read_only_cache {
 public:
  instruction_cache(const char *name, cache_config &config, int core_id,
                    int type_id, mem_fetch_interface *memport,
                    enum mem_fetch_status status, enum cache_gpu_level level,
                    gpgpu_sim *gpu);

  bool schedule_preload_fill(mem_fetch *mf, uint64_t ready_cycle);
  void cycle();

 private:
  struct preload_fill {
    mem_fetch *mf;
    uint64_t ready_cycle;
  };

  std::deque<preload_fill> m_preload_fills;
};

}  // namespace flash_gpgpu_sim

#endif  // FLASH_GPGPU_SIM_INSTRUCTION_CACHE_INSTRUCTION_CACHE_H
