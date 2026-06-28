#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/opcodes.h"
#include "ptx.tab.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {

void fail(const std::string &message) {
  std::fprintf(stderr, "tcgen05 parser smoke failed: %s\n", message.c_str());
  std::exit(1);
}

bool has_option(const ptx_instruction *inst, int option) {
  const std::list<int> options = inst->get_options();
  return std::find(options.begin(), options.end(), option) != options.end();
}

void expect_options(const ptx_instruction *inst,
                    std::initializer_list<int> options) {
  for (int option : options) {
    if (!has_option(inst, option)) {
      fail(std::string(inst->get_opcode_cstr()) +
           " missing expected option " + std::to_string(option));
    }
  }
}

void expect_operand_count(const ptx_instruction *inst, unsigned expected) {
  if (inst->get_num_operands() != expected) {
    fail(std::string(inst->get_opcode_cstr()) + " operand count is " +
         std::to_string(inst->get_num_operands()) + ", expected " +
         std::to_string(expected));
  }
}

void expect_vector_width(const ptx_instruction *inst, unsigned operand,
                         unsigned expected) {
  const operand_info &op = inst->operand_lookup(operand);
  if (!op.is_vector()) {
    fail(std::string(inst->get_opcode_cstr()) + " operand " +
         std::to_string(operand) + " is not a vector");
  }
  if (op.get_vect_nelem() != expected) {
    fail(std::string(inst->get_opcode_cstr()) + " operand " +
         std::to_string(operand) + " vector width is " +
         std::to_string(op.get_vect_nelem()) + ", expected " +
         std::to_string(expected));
  }
}

const ptx_instruction *first(
    const std::map<int, std::vector<const ptx_instruction *>> &by_opcode,
    int opcode) {
  auto found = by_opcode.find(opcode);
  if (found == by_opcode.end() || found->second.empty()) {
    fail(std::string("missing opcode ") + g_opcode_string[opcode]);
  }
  return found->second.front();
}

void expect_opcode_count(
    const std::map<int, std::vector<const ptx_instruction *>> &by_opcode,
    int opcode, size_t expected) {
  const size_t actual =
      by_opcode.count(opcode) == 0 ? 0 : by_opcode.at(opcode).size();
  if (actual != expected) {
    fail(std::string(g_opcode_string[opcode]) + " count is " +
         std::to_string(actual) + ", expected " + std::to_string(expected));
  }
}

void assert_tcgen05_decode(gpgpu_context *ctx, symbol_table *symtab) {
  symbol *kernel_symbol = symtab->lookup("tcgen05_phase1_smoke");
  if (kernel_symbol == nullptr || kernel_symbol->get_pc() == nullptr) {
    fail("missing tcgen05_phase1_smoke entry");
  }

  function_info *kernel = kernel_symbol->get_pc();
  constexpr unsigned kMaxInstSize = 8;
  const unsigned start_pc = kernel->get_start_PC();
  const unsigned limit_pc = std::min<unsigned>(
      start_pc + kMaxInstSize * (kernel->get_function_size() + 1),
      ctx->s_g_pc_to_insn.size());

  std::map<int, std::vector<const ptx_instruction *>> by_opcode;
  for (unsigned pc = start_pc; pc < limit_pc; ++pc) {
    const ptx_instruction *inst = ctx->s_g_pc_to_insn[pc];
    if (inst == nullptr || static_cast<int>(inst->get_opcode()) < 0) {
      continue;
    }
    const std::string source = inst->get_source();
    if (source.find("tcgen05") != std::string::npos &&
        inst->get_opcode() == NOP_OP) {
      fail("TCGen05 instruction decoded as nop: " + source);
    }
    by_opcode[inst->get_opcode()].push_back(inst);
  }

  expect_opcode_count(by_opcode, TCGEN05_ALLOC_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_MMA_OP, 2);
  expect_opcode_count(by_opcode, TCGEN05_COMMIT_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_LD_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_ST_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_WAIT_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_DEALLOC_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_RELINQUISH_ALLOC_PERMIT_OP, 1);

  const ptx_instruction *alloc = first(by_opcode, TCGEN05_ALLOC_OP);
  expect_operand_count(alloc, 2);
  expect_options(alloc,
                 {TCGEN05_CTA_GROUP_1_OPTION, SYNC_OPTION, ALIGNED_OPTION,
                  CTA_OPTION});

  const ptx_instruction *mma = first(by_opcode, TCGEN05_MMA_OP);
  expect_operand_count(mma, 6);
  expect_options(mma, {TCGEN05_CTA_GROUP_1_OPTION, TCGEN05_KIND_F16_OPTION});

  const ptx_instruction *mma_with_tmem_offset =
      by_opcode.at(TCGEN05_MMA_OP).at(1);
  const operand_info &offset_operand = mma_with_tmem_offset->operand_lookup(1);
  if (!offset_operand.is_memory_operand() ||
      offset_operand.get_addr_offset() != 8 ||
      offset_operand.name() != "tmem_a") {
    fail("tcgen05.mma did not decode [tmem_a + 0x8] as a TMEM memory operand");
  }

  const ptx_instruction *commit = first(by_opcode, TCGEN05_COMMIT_OP);
  expect_operand_count(commit, 1);
  expect_options(commit, {TCGEN05_CTA_GROUP_1_OPTION,
                          TCGEN05_MBARRIER_ARRIVE_ONE_OPTION,
                          CLUSTER_OPTION});

  const ptx_instruction *ld = first(by_opcode, TCGEN05_LD_OP);
  expect_operand_count(ld, 2);
  expect_options(ld,
                 {SYNC_OPTION, ALIGNED_OPTION, TCGEN05_32X32B_OPTION,
                  X32_OPTION});
  expect_vector_width(ld, 0, 32);

  const ptx_instruction *st = first(by_opcode, TCGEN05_ST_OP);
  expect_operand_count(st, 2);
  expect_options(st,
                 {SYNC_OPTION, ALIGNED_OPTION, TCGEN05_32X32B_OPTION,
                  X16_OPTION});
  expect_vector_width(st, 1, 16);

  const ptx_instruction *wait = first(by_opcode, TCGEN05_WAIT_OP);
  expect_operand_count(wait, 0);
  expect_options(wait, {SYNC_OPTION, ALIGNED_OPTION});

  expect_operand_count(first(by_opcode, TCGEN05_DEALLOC_OP), 2);
  expect_operand_count(first(by_opcode, TCGEN05_RELINQUISH_ALLOC_PERMIT_OP),
                       0);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <ptx-file>\n", argv[0]);
    return 2;
  }

  gpgpu_context *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  if (symtab == nullptr) {
    std::fprintf(stderr, "failed to parse %s\n", argv[1]);
    return 1;
  }
  assert_tcgen05_decode(ctx, symtab);

  std::printf("parsed and validated %s\n", argv[1]);
  return 0;
}
