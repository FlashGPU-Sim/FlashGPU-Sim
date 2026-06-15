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
#include <unordered_map>
#include <vector>

class ptx_instruction;
namespace flash_gpgpu_sim {
class dyn_ptx_inst_manager {
public:
  static const ptx_instruction *
  get_or_allocate(uint64_t pc, const ptx_instruction *static_inst);

  /**
   * It is possible that some threads already allocated dynamic instructions
   * due to misspeculation, e.g. continuous fetched into next function.
   *
   * This could happen before the misspeculated kernel is pre-decoded and first
   * executed. Therefore, some fields in the inst_t is not valid.
   *
   * We need to update for such dynamic instructions when pre-decoding the
   * kernel.
   */
  static void update_predecoded_inst(uint64_t pc, ptx_instruction *static_inst);

  dyn_ptx_inst_manager();
  ~dyn_ptx_inst_manager();

private:
  using thread_inst_mem_t = std::unordered_map<uint64_t, ptx_instruction *>;
  using all_inst_mem_t = std::vector<thread_inst_mem_t>;
  int m_thread_cnt;

  all_inst_mem_t m_all_inst_mem;

  const ptx_instruction *get_or_allocate(int thread_id, uint64_t pc,
                                         const ptx_instruction *static_inst);
  void update_predecoded_inst_impl(uint64_t pc, ptx_instruction *static_inst);
};

} // namespace flash_gpgpu_sim

#endif
