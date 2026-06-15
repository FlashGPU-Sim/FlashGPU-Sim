#include "dyn_ptx_inst.h"

#include "ptx_ir.h"

#ifdef FLASH_GPGPU_SIM_OMP
#include <omp.h>
#endif

#include <signal.h>

namespace flash_gpgpu_sim {

static dyn_ptx_inst_manager g_dyn_ptx_inst_manager;

#ifdef FLASH_GPGPU_SIM_OMP
dyn_ptx_inst_manager::dyn_ptx_inst_manager()
    : m_thread_cnt(FLASH_GPGPU_SIM_OMP_MAX_THREADS) {
#else
dyn_ptx_inst_manager::dyn_ptx_inst_manager() : m_thread_cnt(1) {
#endif
  m_all_inst_mem.resize(m_thread_cnt);
}

dyn_ptx_inst_manager::~dyn_ptx_inst_manager() {
  for (auto &thread_inst_mem : m_all_inst_mem) {
    for (auto &entry : thread_inst_mem) {
      if (entry.second) {
        delete entry.second;
      }
    }
  }
}

const ptx_instruction *
dyn_ptx_inst_manager::get_or_allocate(int thread_id, uint64_t pc,
                                      const ptx_instruction *static_inst) {
  if (thread_id < 0 || thread_id >= m_thread_cnt) {
    printf("Error: invalid thread_id %d\n", thread_id);
    abort();
  }
  auto &thread_inst_mem = m_all_inst_mem[thread_id];
  auto it = thread_inst_mem.find(pc);
  if (it == thread_inst_mem.end()) {
    // allocate a new dynamic instruction
    auto *inst = new ptx_instruction(*static_inst);
    it = thread_inst_mem.emplace(pc, inst).first;
#ifdef FLASH_GPGPU_SIM_OMP
    // printf("GPGPU-Sim PTX: Thread %d allocated dyn ptx_inst for PC 0x%llx\n",
    //        thread_id, pc);
#endif
  }
  return it->second;
}

const ptx_instruction *
dyn_ptx_inst_manager::get_or_allocate(uint64_t pc,
                                      const ptx_instruction *static_inst) {
#ifdef FLASH_GPGPU_SIM_OMP
  int thread_id = omp_get_thread_num();
  assert(thread_id < FLASH_GPGPU_SIM_OMP_MAX_THREADS);
#else
  int thread_id = 0;
#endif
  return g_dyn_ptx_inst_manager.get_or_allocate(thread_id, pc, static_inst);
}

void dyn_ptx_inst_manager::update_predecoded_inst_impl(
    uint64_t pc, ptx_instruction *static_inst) {
  for (int thread_id = 0; thread_id < m_thread_cnt; thread_id++) {
    auto &thread_inst_mem = m_all_inst_mem[thread_id];
    auto it = thread_inst_mem.find(pc);
    if (it == thread_inst_mem.end()) {
      continue;
    }
    *it->second = *static_inst;
#ifdef FLASH_GPGPU_SIM_OMP
    // printf("GPGPU-Sim PTX: Thread %d updated dyn ptx_inst for PC 0x%llx\n",
    //        thread_id, pc);
#endif
  }
}

void dyn_ptx_inst_manager::update_predecoded_inst(
    uint64_t pc, ptx_instruction *static_inst) {
  g_dyn_ptx_inst_manager.update_predecoded_inst_impl(pc, static_inst);
}

} // namespace flash_gpgpu_sim

const ptx_instruction *function_info::get_dyn_inst(unsigned PC) const {
  unsigned index = PC - m_start_PC;
  if (index < m_instr_mem_size) {
    auto static_inst = m_instr_mem[index];
    return flash_gpgpu_sim::dyn_ptx_inst_manager::get_or_allocate(PC,
                                                                  static_inst);
  }
  return NULL;
}

void function_info::update_dyn_inst(ptx_instruction *inst) {
  flash_gpgpu_sim::dyn_ptx_inst_manager::update_predecoded_inst(inst->get_PC(),
                                                                inst);
}
