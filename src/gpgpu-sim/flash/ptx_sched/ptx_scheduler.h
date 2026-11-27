#ifndef FLASH_GPGPU_SIM_PTX_SCHEDULER_H
#define FLASH_GPGPU_SIM_PTX_SCHEDULER_H

#include <list>
#include <set>
class function_info;
class ptx_instruction;
class symbol;

namespace flash_gpgpu_sim {

namespace detail {
void collect_ptx_registers(const ptx_instruction *inst,
                           std::set<const symbol *> &uses,
                           std::set<const symbol *> &defs);

struct ptx_schedule_priority_t {
  unsigned estimated_issue;
  unsigned remaining_path;
  unsigned original_index;
};

inline bool
ptx_schedule_priority_precedes(const ptx_schedule_priority_t &candidate,
                               const ptx_schedule_priority_t &incumbent) {
  return candidate.estimated_issue < incumbent.estimated_issue ||
         (candidate.estimated_issue == incumbent.estimated_issue &&
          (candidate.remaining_path > incumbent.remaining_path ||
           (candidate.remaining_path == incumbent.remaining_path &&
            candidate.original_index < incumbent.original_index)));
}

} // namespace detail

void run_ptx_reorder(function_info *func);
// Read-only control-flow annotation; never rewrites the PTX instruction stream.
void mark_explicit_elected_tma_regions(
    std::list<ptx_instruction *> &instructions);

} // namespace flash_gpgpu_sim

#endif
