#include "reg_alloc.h"

#include "../../../libcuda/gpgpu_context.h"
#include "../../cuda-sim/opcodes.h"
#include "../../cuda-sim/ptx_ir.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace {

bool ptx_opcode_has_dst(int opcode) {
  switch (opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                             \
  case OP:                                                                     \
    return DST != 0;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                           \
  case OP:                                                                     \
    return DST != 0;
#include "../../cuda-sim/opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
  default:
    return false;
  }
}

typedef std::set<const symbol *> reg_symbol_set;

struct bb_live_info {
  reg_symbol_set use;
  reg_symbol_set def;
  reg_symbol_set live_in;
  reg_symbol_set live_out;
};

struct live_interval {
  unsigned start;
  unsigned end;
  int group;
  bool valid;

  live_interval() : start(0), end(0), group(-1), valid(false) {}
};

struct sorted_interval {
  const symbol *sym;
  unsigned start;
  unsigned end;
  int group;
};

struct group_alloc_stats {
  unsigned virt;
  unsigned physical;
  unsigned safe_physical;
  unsigned pinned;

  group_alloc_stats() : virt(0), physical(0), safe_physical(0), pinned(0) {}
};

struct physical_slot {
  const symbol *representative;
  unsigned end;

  physical_slot(const symbol *rep, unsigned last_pc)
      : representative(rep), end(last_pc) {}
};

void add_liveness_reg(reg_symbol_set &regs, const symbol *sym) {
  if (sym == NULL)
    return;
  if (!sym->is_reg() || sym->is_non_arch_reg())
    return;
  if (sym->name() == "_")
    return;
  regs.insert(sym);
}

void collect_operand_liveness_regs(const operand_info &op,
                                   reg_symbol_set &regs) {
  if (op.is_vector()) {
    for (unsigned i = 0; i < op.get_vect_nelem(); ++i) {
      add_liveness_reg(regs, op.vec_symbol(i));
    }
    return;
  }

  if (op.is_reg() || op.get_type() == address_t || op.is_memory_operand()) {
    add_liveness_reg(regs, op.get_symbol());
  }
}

void collect_inst_liveness_regs(const ptx_instruction *inst,
                                reg_symbol_set &uses, reg_symbol_set &defs) {
  if (inst == NULL || inst->is_label())
    return;

  if (inst->has_pred())
    collect_operand_liveness_regs(inst->get_pred(), uses);

  const bool has_dst = ptx_opcode_has_dst(inst->get_opcode());
  const bool dst_is_accumulator = inst->get_opcode() == MMA_OP ||
                                  inst->get_opcode() == TENSOR_MMA_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_SP_OP;

  const std::vector<operand_info> &operands = inst->get_operands();
  for (unsigned i = 0; i < operands.size(); ++i) {
    const operand_info &op = operands[i];
    if (has_dst && i == 0) {
      collect_operand_liveness_regs(op, defs);
      if (dst_is_accumulator || op.get_operand_lohi() != 0 ||
          op.get_double_operand_type() != 0) {
        collect_operand_liveness_regs(op, uses);
      }
    } else {
      collect_operand_liveness_regs(op, uses);
      // Some PTXPlus memory address forms update the address register.
      if (op.is_memory_operand2() && (op.get_double_operand_type() == 2 ||
                                      op.get_double_operand_type() == 3)) {
        collect_operand_liveness_regs(op, defs);
      }
    }
  }
}

int reg_alloc_group(const symbol *sym) {
  if (sym == NULL || sym->type() == NULL)
    return -1;
  const int scalar_type = sym->type()->get_key().scalar_type();
  return static_cast<int>(sym->get_size_in_bytes()) * 4096 + scalar_type;
}

const char *reg_alloc_group_name(int group) {
  static char buf[64];
  const int bytes = group / 4096;
  const int scalar_type = group % 4096;
  snprintf(buf, sizeof(buf), "%dB/type%d", bytes, scalar_type);
  return buf;
}

void record_interval(std::map<const symbol *, live_interval> &intervals,
                     const symbol *sym, unsigned pc) {
  live_interval &range = intervals[sym];
  if (!range.valid) {
    range.start = pc;
    range.end = pc;
    range.group = reg_alloc_group(sym);
    range.valid = true;
  } else {
    range.start = std::min(range.start, pc);
    range.end = std::max(range.end, pc);
  }
}

std::vector<sorted_interval>
sort_intervals(const std::map<const symbol *, live_interval> &intervals) {
  std::vector<sorted_interval> sorted;
  sorted.reserve(intervals.size());
  for (std::map<const symbol *, live_interval>::const_iterator i =
           intervals.begin();
       i != intervals.end(); ++i) {
    if (!i->second.valid)
      continue;
    sorted_interval item;
    item.sym = i->first;
    item.start = i->second.start;
    item.end = i->second.end;
    item.group = i->second.group;
    sorted.push_back(item);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const sorted_interval &a, const sorted_interval &b) {
              if (a.group != b.group)
                return a.group < b.group;
              if (a.start != b.start)
                return a.start < b.start;
              if (a.end != b.end)
                return a.end < b.end;
              return a.sym->uid() < b.sym->uid();
            });
  return sorted;
}

unsigned allocate_intervals(
    const std::vector<sorted_interval> &sorted,
    const reg_symbol_set *pinned_regs,
    std::map<int, group_alloc_stats> *group_stats,
    std::unordered_map<const symbol *, const symbol *> *aliases) {
  unsigned total_physical = 0;
  size_t index = 0;

  while (index < sorted.size()) {
    const int group = sorted[index].group;
    std::vector<physical_slot> active;
    unsigned pinned = 0;
    unsigned virt = 0;

    while (index < sorted.size() && sorted[index].group == group) {
      const sorted_interval &interval = sorted[index++];
      virt++;

      const bool is_pinned =
          pinned_regs != NULL &&
          pinned_regs->find(interval.sym) != pinned_regs->end();
      if (is_pinned) {
        pinned++;
        continue;
      }

      bool reused = false;
      for (std::vector<physical_slot>::iterator slot = active.begin();
           slot != active.end(); ++slot) {
        if (slot->end < interval.start) {
          if (aliases != NULL && slot->representative != interval.sym) {
            (*aliases)[interval.sym] = slot->representative;
          }
          slot->end = interval.end;
          reused = true;
          break;
        }
      }

      if (!reused) {
        active.push_back(physical_slot(interval.sym, interval.end));
      }
    }

    const unsigned physical = static_cast<unsigned>(active.size()) + pinned;
    total_physical += physical;
    if (group_stats != NULL) {
      group_alloc_stats &stats = (*group_stats)[group];
      stats.virt = virt;
      if (pinned_regs == NULL) {
        stats.physical = physical;
      } else {
        stats.safe_physical = physical;
        stats.pinned = pinned;
      }
    }
  }

  return total_physical;
}

} // namespace

namespace flash_gpgpu_sim {

void run_ptx_register_allocation(function_info *func) {
  if (func == NULL)
    return;

  func->m_reg_alloc_aliases.clear();

  gpgpu_context *ctx = func->gpgpu_ctx;
  const bool enable = ctx != NULL && ctx->ptx_register_allocator_enabled;
  const bool stats_enabled = ctx != NULL && ctx->ptx_register_allocator_stats;
  if (!enable && !stats_enabled)
    return;
  if (func->m_basic_blocks.empty() || func->m_instr_mem == NULL)
    return;

  std::vector<bb_live_info> bb_live(func->m_basic_blocks.size());
  std::map<const symbol *, live_interval> intervals;
  reg_symbol_set all_regs;
  reg_symbol_set linear_defined;
  reg_symbol_set linear_read_before_def;
  unsigned real_inst_count = 0;

  for (unsigned bb_id = 0; bb_id < func->m_basic_blocks.size(); ++bb_id) {
    basic_block_t *bb = func->m_basic_blocks[bb_id];
    if (bb->is_exit || bb->ptx_begin == NULL || bb->ptx_end == NULL)
      continue;
    const unsigned begin = bb->ptx_begin->get_m_instr_mem_index();
    const unsigned end = bb->ptx_end->get_m_instr_mem_index();

    for (unsigned pc = begin; pc <= end;) {
      ptx_instruction *inst = func->m_instr_mem[pc];
      if (inst == NULL) {
        ++pc;
        continue;
      }

      reg_symbol_set uses;
      reg_symbol_set defs;
      collect_inst_liveness_regs(inst, uses, defs);
      if (!inst->is_label())
        ++real_inst_count;

      for (reg_symbol_set::const_iterator u = uses.begin(); u != uses.end();
           ++u) {
        all_regs.insert(*u);
        if (bb_live[bb_id].def.find(*u) == bb_live[bb_id].def.end())
          bb_live[bb_id].use.insert(*u);
        if (linear_defined.find(*u) == linear_defined.end())
          linear_read_before_def.insert(*u);
        record_interval(intervals, *u, pc);
      }

      for (reg_symbol_set::const_iterator d = defs.begin(); d != defs.end();
           ++d) {
        all_regs.insert(*d);
        bb_live[bb_id].def.insert(*d);
        linear_defined.insert(*d);
        record_interval(intervals, *d, pc);
      }

      const unsigned inst_size = inst->inst_size();
      pc += inst_size == 0 ? 1 : inst_size;
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (int bb_id = static_cast<int>(func->m_basic_blocks.size()) - 1;
         bb_id >= 0; --bb_id) {
      reg_symbol_set new_live_out;
      for (std::set<int>::const_iterator s =
               func->m_basic_blocks[bb_id]->successor_ids.begin();
           s != func->m_basic_blocks[bb_id]->successor_ids.end(); ++s) {
        if (*s >= 0 && static_cast<unsigned>(*s) < bb_live.size()) {
          new_live_out.insert(bb_live[*s].live_in.begin(),
                              bb_live[*s].live_in.end());
        }
      }

      reg_symbol_set new_live_in = bb_live[bb_id].use;
      for (reg_symbol_set::const_iterator r = new_live_out.begin();
           r != new_live_out.end(); ++r) {
        if (bb_live[bb_id].def.find(*r) == bb_live[bb_id].def.end())
          new_live_in.insert(*r);
      }

      if (new_live_in != bb_live[bb_id].live_in ||
          new_live_out != bb_live[bb_id].live_out) {
        bb_live[bb_id].live_in.swap(new_live_in);
        bb_live[bb_id].live_out.swap(new_live_out);
        changed = true;
      }
    }
  }

  for (unsigned bb_id = 0; bb_id < func->m_basic_blocks.size(); ++bb_id) {
    const reg_symbol_set &live_in = bb_live[bb_id].live_in;
    if (live_in.empty())
      continue;
    basic_block_t *bb = func->m_basic_blocks[bb_id];
    if (bb->ptx_begin == NULL)
      continue;
    const unsigned begin = bb->ptx_begin->get_m_instr_mem_index();
    for (reg_symbol_set::const_iterator r = live_in.begin(); r != live_in.end();
         ++r) {
      record_interval(intervals, *r, begin);
    }
  }

  unsigned max_live = 0;
  for (int bb_id = static_cast<int>(func->m_basic_blocks.size()) - 1;
       bb_id >= 0; --bb_id) {
    basic_block_t *bb = func->m_basic_blocks[bb_id];
    if (bb->is_exit || bb->ptx_begin == NULL || bb->ptx_end == NULL)
      continue;
    reg_symbol_set live = bb_live[bb_id].live_out;
    max_live = std::max<unsigned>(max_live, live.size());

    for (int pc = static_cast<int>(bb->ptx_end->get_m_instr_mem_index());
         pc >= static_cast<int>(bb->ptx_begin->get_m_instr_mem_index()); --pc) {
      ptx_instruction *inst = func->m_instr_mem[pc];
      if (inst == NULL)
        continue;

      reg_symbol_set uses;
      reg_symbol_set defs;
      collect_inst_liveness_regs(inst, uses, defs);
      for (reg_symbol_set::const_iterator d = defs.begin(); d != defs.end();
           ++d) {
        live.erase(*d);
      }
      live.insert(uses.begin(), uses.end());
      max_live = std::max<unsigned>(max_live, live.size());
    }
  }

  std::vector<sorted_interval> sorted = sort_intervals(intervals);
  std::map<int, group_alloc_stats> group_stats;
  const unsigned total_physical =
      allocate_intervals(sorted, NULL, &group_stats, NULL);

  std::unordered_map<const symbol *, const symbol *> aliases;
  const unsigned total_safe_physical = allocate_intervals(
      sorted, &linear_read_before_def, &group_stats, enable ? &aliases : NULL);

  if (enable)
    func->m_reg_alloc_aliases.swap(aliases);

  unsigned entry_live = 0;
  if (!bb_live.empty())
    entry_live = bb_live.front().live_in.size();

  if (stats_enabled) {
    printf(
        "GPGPU-Sim PTX: register allocator stats for '%s': enabled=%u "
        "inst=%u bb=%zu virtual=%zu physical=%u safe_physical=%u aliases=%zu "
        "max_live=%u entry_live=%u read_before_def=%zu\n",
        func->m_name.c_str(), enable ? 1 : 0, real_inst_count,
        func->m_basic_blocks.size(), all_regs.size(), total_physical,
        total_safe_physical, func->m_reg_alloc_aliases.size(), max_live,
        entry_live, linear_read_before_def.size());

    for (std::map<int, group_alloc_stats>::const_iterator g =
             group_stats.begin();
         g != group_stats.end(); ++g) {
      printf("GPGPU-Sim PTX:   group %s virtual=%u physical=%u "
             "safe_physical=%u pinned=%u\n",
             reg_alloc_group_name(g->first), g->second.virt, g->second.physical,
             g->second.safe_physical, g->second.pinned);
    }
    fflush(stdout);
  }
}

} // namespace flash_gpgpu_sim
