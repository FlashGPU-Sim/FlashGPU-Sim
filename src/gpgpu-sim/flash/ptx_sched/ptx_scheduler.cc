#include "ptx_scheduler.h"

#include "../../../../libcuda/gpgpu_context.h"
#include "../../../cuda-sim/opcodes.h"
#include "../../../cuda-sim/ptx_ir.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace {

enum class inst_class_t {
  tensor,
  ldmatrix,
  sfu,
  fp32,
  shfl,
  mem,
  intp,
  control,
  other,
  boundary
};

enum class pipe_t {
  tensor,
  sfu,
  ldst,
  fp32,
  xbar,
  intp,
  control,
  other,
  count
};

typedef std::set<const symbol *> reg_set_t;

struct sched_inst_t {
  ptx_instruction *inst;
  unsigned original_index;
  inst_class_t cls;
  reg_set_t defs;
  reg_set_t uses;
};

struct dep_edge_t {
  unsigned src;
  unsigned dst;
  unsigned latency;
};

struct dep_graph_t {
  std::vector<std::set<unsigned>> succ;
  std::vector<unsigned> indeg;
  std::vector<dep_edge_t> edges;
};

struct reorder_stats_t {
  unsigned segments;
  unsigned skipped_segments;
  unsigned total_insts;
  unsigned moved_slots;
  unsigned max_segment;
  unsigned edges;

  reorder_stats_t()
      : segments(0), skipped_segments(0), total_insts(0), moved_slots(0),
        max_segment(0), edges(0) {}
};

bool ptx_opcode_has_dst(int opcode) {
  switch (opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                             \
  case OP:                                                                     \
    return DST != 0;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                           \
  case OP:                                                                     \
    return DST != 0;
#include "../../../cuda-sim/opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
  default:
    return false;
  }
}

void add_reg(reg_set_t &regs, const symbol *sym) {
  if (sym == NULL)
    return;
  if (!sym->is_reg() || sym->is_non_arch_reg())
    return;
  if (sym->name() == "_")
    return;
  regs.insert(sym);
}

void collect_operand_regs(const operand_info &op, reg_set_t &regs) {
  if (op.is_vector()) {
    for (unsigned i = 0; i < op.get_vect_nelem(); ++i)
      add_reg(regs, op.vec_symbol(i));
    return;
  }

  if (op.is_reg() || op.get_type() == address_t)
    add_reg(regs, op.get_symbol());
}

void collect_inst_regs(const ptx_instruction *inst, reg_set_t &uses,
                       reg_set_t &defs) {
  if (inst == NULL || inst->is_label())
    return;

  if (inst->has_pred())
    collect_operand_regs(inst->get_pred(), uses);

  const std::vector<operand_info> &operands = inst->get_operands();
  if (inst->get_opcode() == SETP_OP && operands.size() >= 2) {
    collect_operand_regs(operands[0], defs);
    collect_operand_regs(operands[1], defs);
    for (unsigned i = 2; i < operands.size(); ++i)
      collect_operand_regs(operands[i], uses);
    return;
  }

  const bool has_dst = ptx_opcode_has_dst(inst->get_opcode());
  const bool dst_is_accumulator = inst->get_opcode() == MMA_OP ||
                                  inst->get_opcode() == TENSOR_MMA_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_SP_OP;

  for (unsigned i = 0; i < operands.size(); ++i) {
    const operand_info &op = operands[i];
    if (has_dst && i == 0) {
      collect_operand_regs(op, defs);
      if (dst_is_accumulator || op.get_operand_lohi() != 0 ||
          op.get_double_operand_type() != 0)
        collect_operand_regs(op, uses);
      continue;
    }

    collect_operand_regs(op, uses);
    if (op.is_memory_operand2() && (op.get_double_operand_type() == 2 ||
                                    op.get_double_operand_type() == 3))
      collect_operand_regs(op, defs);
  }
}

inst_class_t classify_inst(const ptx_instruction *inst) {
  if (inst == NULL || inst->is_label())
    return inst_class_t::boundary;

  switch (inst->get_opcode()) {
  case MMA_OP:
  case TENSOR_MMA_OP:
    return inst_class_t::tensor;

  case LDMATRIX_OP:
  case STMATRIX_OP:
    return inst_class_t::ldmatrix;

  case EX2_OP:
  case LG2_OP:
  case RCP_OP:
  case RSQRT_OP:
  case SQRT_OP:
  case SIN_OP:
  case COS_OP:
  case TANH_OP:
  case DIV_OP:
    return inst_class_t::sfu;

  case FMA_OP:
  case MAD_OP:
  case MUL_OP:
  case ADD_OP:
  case SUB_OP:
  case MAX_OP:
  case MIN_OP:
  case CVT_OP:
    return inst_class_t::fp32;

  case SHFL_OP:
    return inst_class_t::shfl;

  case LD_OP:
  case LDU_OP:
  case ST_OP:
  case ATOM_OP:
  case RED_OP:
  case PREFETCH_OP:
  case PREFETCHU_OP:
  case CP_ASYNC_OP:
  case TMA_OP:
  case TMA_PREFETCH_OP:
  case TENSORMAP_OP:
    return inst_class_t::mem;

  case BRA_OP:
  case BRX_OP:
  case BRKPT_OP:
  case BREAK_OP:
  case BREAKADDR_OP:
  case CALL_OP:
  case CALLP_OP:
  case RET_OP:
  case RETP_OP:
  case EXIT_OP:
  case TRAP_OP:
  case BAR_OP:
  case MBAR_OP:
  case MEMBAR_OP:
  case FENCE_OP:
  case CP_ASYNC_COMMIT_OP:
  case CP_ASYNC_WAIT_OP:
  case WGMMA_MMA_ASYNC_OP:
  case WGMMA_MMA_ASYNC_SP_OP:
  case WGMMA_FENCE_OP:
  case WGMMA_COMMIT_GROUP_OP:
  case WGMMA_WAIT_GROUP_OP:
  case SETMAXNREG_OP:
  case GRIDDEPCONTROL_OP:
    return inst_class_t::control;

  case ADDP_OP:
  case ADDC_OP:
  case AND_OP:
  case ANDN_OP:
  case BFE_OP:
  case BFI_OP:
  case BFIND_OP:
  case BREV_OP:
  case CLZ_OP:
  case CNOT_OP:
  case CVTA_OP:
  case ISSPACEP_OP:
  case MAPA_OP:
  case MOV_OP:
  case MUL24_OP:
  case NANDN_OP:
  case NORN_OP:
  case NOT_OP:
  case OR_OP:
  case ORN_OP:
  case POPC_OP:
  case PRMT_OP:
  case REM_OP:
  case SAD_OP:
  case SELP_OP:
  case SETP_OP:
  case SET_OP:
  case SHF_OP:
  case SHL_OP:
  case SHR_OP:
  case SLCT_OP:
  case VOTE_OP:
  case ACTIVEMASK_OP:
  case XOR_OP:
  case ELECT_OP:
    return inst_class_t::intp;

  default:
    return inst_class_t::other;
  }
}

bool has_unsupported_operand_form(const ptx_instruction *inst) {
  if (inst == NULL || inst->is_label())
    return false;
  const std::vector<operand_info> &operands = inst->get_operands();
  for (unsigned i = 0; i < operands.size(); ++i) {
    if (operands[i].is_memory_operand())
      return true;
  }
  return false;
}

bool is_segment_boundary(const ptx_instruction *inst) {
  inst_class_t cls = classify_inst(inst);
  return cls == inst_class_t::boundary || cls == inst_class_t::control ||
         has_unsupported_operand_form(inst);
}

bool is_memory_like(inst_class_t cls) {
  return cls == inst_class_t::mem || cls == inst_class_t::ldmatrix;
}

char class_token(inst_class_t cls) {
  switch (cls) {
  case inst_class_t::tensor:
    return 'T';
  case inst_class_t::ldmatrix:
    return 'L';
  case inst_class_t::sfu:
    return 'S';
  case inst_class_t::fp32:
    return 'F';
  case inst_class_t::shfl:
    return 'H';
  case inst_class_t::mem:
    return 'M';
  case inst_class_t::intp:
    return 'I';
  case inst_class_t::control:
    return 'C';
  case inst_class_t::other:
    return '.';
  case inst_class_t::boundary:
    return 'B';
  }
  return '?';
}

pipe_t inst_pipe(inst_class_t cls) {
  switch (cls) {
  case inst_class_t::tensor:
    return pipe_t::tensor;
  case inst_class_t::sfu:
    return pipe_t::sfu;
  case inst_class_t::ldmatrix:
  case inst_class_t::mem:
    return pipe_t::ldst;
  case inst_class_t::fp32:
    return pipe_t::fp32;
  case inst_class_t::shfl:
    return pipe_t::xbar;
  case inst_class_t::intp:
    return pipe_t::intp;
  case inst_class_t::control:
    return pipe_t::control;
  default:
    return pipe_t::other;
  }
}

unsigned inst_latency(const sched_inst_t &inst) {
  if (inst.inst->get_opcode() == CP_ASYNC_OP)
    return 1;

  switch (inst.cls) {
  case inst_class_t::tensor:
    return 32;
  case inst_class_t::sfu:
    return 28;
  case inst_class_t::ldmatrix:
  case inst_class_t::mem:
    return 8;
  case inst_class_t::fp32:
  case inst_class_t::shfl:
  case inst_class_t::intp:
    return 4;
  default:
    return 1;
  }
}

unsigned inst_initiation(const sched_inst_t &inst) {
  if (inst.inst->get_opcode() == CP_ASYNC_OP)
    return 1;

  switch (inst.cls) {
  case inst_class_t::tensor:
  case inst_class_t::sfu:
    return 8;
  default:
    return 1;
  }
}

double switch_bonus(char last, char next) {
  switch (last) {
  case 'T':
    switch (next) {
    case 'L':
      return 16.0;
    case 'F':
      return 12.0;
    case 'I':
      return 7.0;
    case 'S':
      return 5.0;
    case 'T':
      return 3.0;
    }
    break;
  case 'L':
    switch (next) {
    case 'T':
      return 18.0;
    case 'F':
      return 6.0;
    case 'I':
      return 5.0;
    case 'S':
      return 4.0;
    case 'L':
      return 2.0;
    }
    break;
  case 'F':
    switch (next) {
    case 'T':
      return 10.0;
    case 'I':
      return 10.0;
    case 'S':
      return 8.0;
    case 'M':
      return 4.0;
    case 'L':
      return 2.0;
    case 'F':
      return 2.0;
    }
    break;
  case 'S':
    switch (next) {
    case 'F':
      return 18.0;
    case 'I':
      return 8.0;
    case 'T':
      return 7.0;
    case 'L':
      return 2.0;
    }
    break;
  case 'M':
    switch (next) {
    case 'I':
      return 16.0;
    case 'F':
      return 8.0;
    case '.':
      return 4.0;
    case 'M':
      return 2.0;
    }
    break;
  case 'I':
    switch (next) {
    case 'M':
      return 12.0;
    case 'F':
      return 10.0;
    case 'T':
      return 6.0;
    case '.':
      return 5.0;
    case 'S':
      return 4.0;
    case 'L':
      return 2.0;
    case 'I':
      return 1.0;
    }
    break;
  case '.':
    switch (next) {
    case 'I':
      return 12.0;
    case 'F':
      return 7.0;
    case 'M':
      return 4.0;
    case 'T':
      return 2.0;
    }
    break;
  case 'C':
    switch (next) {
    case 'I':
      return 8.0;
    case '.':
      return 4.0;
    case 'T':
      return 2.0;
    }
    break;
  case 'H':
    switch (next) {
    case 'F':
      return 10.0;
    case 'I':
      return 2.0;
    case 'H':
      return 1.0;
    }
    break;
  }
  return 0.0;
}

void add_edge(dep_graph_t &graph,
              std::map<std::pair<unsigned, unsigned>, unsigned> &edge_index,
              unsigned src, unsigned dst, unsigned latency) {
  if (src == dst)
    return;

  std::pair<unsigned, unsigned> key(src, dst);
  std::map<std::pair<unsigned, unsigned>, unsigned>::iterator found =
      edge_index.find(key);
  if (found != edge_index.end()) {
    dep_edge_t &edge = graph.edges[found->second];
    edge.latency = std::max(edge.latency, latency);
    return;
  }

  edge_index[key] = graph.edges.size();
  graph.succ[src].insert(dst);
  ++graph.indeg[dst];
  dep_edge_t edge;
  edge.src = src;
  edge.dst = dst;
  edge.latency = latency;
  graph.edges.push_back(edge);
}

dep_graph_t build_dependency_graph(const std::vector<sched_inst_t> &chunk) {
  dep_graph_t graph;
  graph.succ.resize(chunk.size());
  graph.indeg.assign(chunk.size(), 0);
  std::map<std::pair<unsigned, unsigned>, unsigned> edge_index;

  std::map<const symbol *, unsigned> last_def;
  std::map<const symbol *, std::set<unsigned>> last_uses;
  bool have_last_mem = false;
  unsigned last_mem = 0;

  for (unsigned i = 0; i < chunk.size(); ++i) {
    const sched_inst_t &inst = chunk[i];
    for (reg_set_t::const_iterator use = inst.uses.begin();
         use != inst.uses.end(); ++use) {
      std::map<const symbol *, unsigned>::const_iterator def =
          last_def.find(*use);
      if (def != last_def.end())
        add_edge(graph, edge_index, def->second, i,
                 inst_latency(chunk[def->second]));
      last_uses[*use].insert(i);
    }

    for (reg_set_t::const_iterator def = inst.defs.begin();
         def != inst.defs.end(); ++def) {
      std::map<const symbol *, unsigned>::const_iterator old_def =
          last_def.find(*def);
      if (old_def != last_def.end())
        add_edge(graph, edge_index, old_def->second, i, 0);

      std::map<const symbol *, std::set<unsigned>>::const_iterator uses =
          last_uses.find(*def);
      if (uses != last_uses.end()) {
        for (std::set<unsigned>::const_iterator use = uses->second.begin();
             use != uses->second.end(); ++use)
          add_edge(graph, edge_index, *use, i, 0);
      }

      last_def[*def] = i;
      last_uses[*def].clear();
    }

    if (is_memory_like(inst.cls)) {
      if (have_last_mem)
        add_edge(graph, edge_index, last_mem, i, 0);
      have_last_mem = true;
      last_mem = i;
    }
  }

  return graph;
}

std::vector<sched_inst_t>
schedule_switch(const std::vector<sched_inst_t> &chunk, int ready_slack,
                unsigned *edge_count, bool *valid) {
  *valid = true;
  dep_graph_t graph = build_dependency_graph(chunk);
  *edge_count = graph.edges.size();

  std::vector<std::vector<dep_edge_t>> edge_by_src(chunk.size());
  for (unsigned i = 0; i < graph.edges.size(); ++i)
    edge_by_src[graph.edges[i].src].push_back(graph.edges[i]);

  std::vector<unsigned> height(chunk.size(), 1);
  for (int i = static_cast<int>(chunk.size()) - 1; i >= 0; --i) {
    height[i] = inst_latency(chunk[i]);
    for (std::vector<dep_edge_t>::const_iterator edge = edge_by_src[i].begin();
         edge != edge_by_src[i].end(); ++edge) {
      height[i] = std::max(height[i], edge->latency + height[edge->dst]);
    }
  }

  std::set<unsigned> ready;
  for (unsigned i = 0; i < graph.indeg.size(); ++i) {
    if (graph.indeg[i] == 0)
      ready.insert(i);
  }

  std::vector<unsigned> dep_ready(chunk.size(), 0);
  std::vector<unsigned> pipe_ready(static_cast<unsigned>(pipe_t::count), 0);
  unsigned warp_issue_ready = 0;
  std::vector<unsigned> emitted;
  std::deque<char> recent;
  char last_token = 0;

  while (!ready.empty()) {
    std::map<unsigned, unsigned> issues;
    unsigned min_issue = std::numeric_limits<unsigned>::max();
    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const sched_inst_t &inst = chunk[*it];
      const unsigned pipe_index = static_cast<unsigned>(inst_pipe(inst.cls));
      unsigned issue = std::max(dep_ready[*it], pipe_ready[pipe_index]);
      issue = std::max(issue, warp_issue_ready);
      issues[*it] = issue;
      min_issue = std::min(min_issue, issue);
    }

    bool have_pick = false;
    unsigned pick = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_neg_issue = std::numeric_limits<int>::min();
    unsigned best_height = 0;
    int best_neg_index = std::numeric_limits<int>::min();
    const unsigned slack =
        ready_slack < 0 ? 0 : static_cast<unsigned>(ready_slack);

    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const unsigned idx = *it;
      const unsigned issue = issues[idx];
      if (issue > min_issue + slack)
        continue;

      const sched_inst_t &inst = chunk[idx];
      const char token = class_token(inst.cls);
      unsigned same_recent = 0;
      for (std::deque<char>::const_iterator r = recent.begin();
           r != recent.end(); ++r) {
        if (*r == token)
          ++same_recent;
      }

      double score = -0.02 * static_cast<double>(issue) +
                     0.001 * static_cast<double>(height[idx]) -
                     0.0001 * static_cast<double>(inst.original_index);
      if (last_token != 0) {
        score += 0.6 * switch_bonus(last_token, token);
        if ((last_token == 'T' && token == 'L') ||
            (last_token == 'L' && token == 'T'))
          score += 4.0;
        score -= 2.0 * static_cast<double>(same_recent);
      }

      const int neg_issue = -static_cast<int>(issue);
      const int neg_index = -static_cast<int>(inst.original_index);
      const bool better =
          !have_pick || score > best_score + 1e-12 ||
          (std::fabs(score - best_score) <= 1e-12 &&
           (neg_issue > best_neg_issue ||
            (neg_issue == best_neg_issue &&
             (height[idx] > best_height ||
              (height[idx] == best_height && neg_index > best_neg_index)))));
      if (better) {
        have_pick = true;
        pick = idx;
        best_score = score;
        best_neg_issue = neg_issue;
        best_height = height[idx];
        best_neg_index = neg_index;
      }
    }

    if (!have_pick) {
      *valid = false;
      return chunk;
    }

    ready.erase(pick);
    emitted.push_back(pick);

    const sched_inst_t &inst = chunk[pick];
    const unsigned issue = issues[pick];
    const unsigned pipe_index = static_cast<unsigned>(inst_pipe(inst.cls));
    pipe_ready[pipe_index] = issue + inst_initiation(inst);
    warp_issue_ready = issue + 1;

    last_token = class_token(inst.cls);
    recent.push_back(last_token);
    if (recent.size() > 12)
      recent.pop_front();

    for (std::vector<dep_edge_t>::const_iterator edge =
             edge_by_src[pick].begin();
         edge != edge_by_src[pick].end(); ++edge) {
      dep_ready[edge->dst] =
          std::max(dep_ready[edge->dst], issue + edge->latency);
      if (graph.indeg[edge->dst] == 0) {
        *valid = false;
        return chunk;
      }
      --graph.indeg[edge->dst];
      if (graph.indeg[edge->dst] == 0)
        ready.insert(edge->dst);
    }
  }

  if (emitted.size() != chunk.size()) {
    *valid = false;
    return chunk;
  }

  std::vector<sched_inst_t> scheduled;
  scheduled.reserve(chunk.size());
  for (unsigned i = 0; i < emitted.size(); ++i)
    scheduled.push_back(chunk[emitted[i]]);
  return scheduled;
}

void flush_segment(std::vector<sched_inst_t> &segment,
                   std::list<ptx_instruction *> &out, int ready_slack,
                   reorder_stats_t &stats) {
  if (segment.empty())
    return;

  stats.total_insts += segment.size();
  stats.max_segment =
      std::max(stats.max_segment, static_cast<unsigned>(segment.size()));

  if (segment.size() == 1) {
    out.push_back(segment[0].inst);
    segment.clear();
    return;
  }

  unsigned edge_count = 0;
  bool valid = true;
  std::vector<sched_inst_t> scheduled =
      schedule_switch(segment, ready_slack, &edge_count, &valid);
  stats.edges += edge_count;
  ++stats.segments;

  if (!valid) {
    ++stats.skipped_segments;
    scheduled = segment;
  }

  for (unsigned i = 0; i < scheduled.size(); ++i) {
    if (scheduled[i].inst != segment[i].inst)
      ++stats.moved_slots;
    out.push_back(scheduled[i].inst);
  }
  segment.clear();
}

} // namespace

namespace flash_gpgpu_sim {

void run_ptx_reorder(function_info *func) {
  if (func == NULL || func->gpgpu_ctx == NULL ||
      !func->gpgpu_ctx->ptx_reorder_enabled)
    return;

  std::list<ptx_instruction *> reordered;
  std::vector<sched_inst_t> segment;
  reorder_stats_t stats;
  unsigned original_index = 0;

  for (std::list<ptx_instruction *>::iterator it = func->m_instructions.begin();
       it != func->m_instructions.end(); ++it, ++original_index) {
    ptx_instruction *inst = *it;
    if (is_segment_boundary(inst)) {
      flush_segment(segment, reordered,
                    func->gpgpu_ctx->ptx_reorder_ready_slack, stats);
      reordered.push_back(inst);
      continue;
    }

    sched_inst_t sched_inst;
    sched_inst.inst = inst;
    sched_inst.original_index = original_index;
    sched_inst.cls = classify_inst(inst);
    collect_inst_regs(inst, sched_inst.uses, sched_inst.defs);
    segment.push_back(sched_inst);
  }
  flush_segment(segment, reordered, func->gpgpu_ctx->ptx_reorder_ready_slack,
                stats);

  if (reordered.size() != func->m_instructions.size()) {
    printf("GPGPU-Sim PTX: reorder switch skipped function '%s' due to size "
           "mismatch (%zu vs %zu)\n",
           func->m_name.c_str(), reordered.size(), func->m_instructions.size());
    return;
  }

  func->m_instructions.swap(reordered);

  if (func->gpgpu_ctx->ptx_reorder_stats || stats.moved_slots != 0) {
    printf("GPGPU-Sim PTX: reorder switch function '%s': segments=%u "
           "skipped=%u insts=%u moved_slots=%u max_segment=%u edges=%u "
           "slack=%d\n",
           func->m_name.c_str(), stats.segments, stats.skipped_segments,
           stats.total_insts, stats.moved_slots, stats.max_segment, stats.edges,
           func->gpgpu_ctx->ptx_reorder_ready_slack);
  }
}

} // namespace flash_gpgpu_sim
