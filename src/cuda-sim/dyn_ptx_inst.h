#ifndef DYN_PTX_INST_H
#define DYN_PTX_INST_H

/**
 * GPGPU-Sim does not distinguish between dynamic and static instructions,
 * which causes race condition during parallel simulation.
 *
 * To solve this problem, we introduce dyn_ptx_inst_manager to create a copy
 * of the dynamic instruction for each thread at runtime.
 * This wastes space, but it is simple and effective.
 */
#include <cstdint>
#include <vector>

class ptx_instruction;
namespace flash_gpgpu_sim {
class dyn_ptx_inst_manager {
public:
  static const ptx_instruction *
  get_or_allocate(uint64_t pc, const ptx_instruction *static_inst);
  dyn_ptx_inst_manager();
  ~dyn_ptx_inst_manager();

private:

  using thread_inst_mem_t = std::vector<const ptx_instruction *>;
  using all_inst_mem_t = std::vector<thread_inst_mem_t>;
  int m_thread_cnt;

  all_inst_mem_t m_all_inst_mem;

  const ptx_instruction *get_or_allocate(int thread_id, uint64_t pc,
                                         const ptx_instruction *static_inst);
};

} // namespace flash_gpgpu_sim

#endif