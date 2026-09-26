#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
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

std::map<int, std::vector<const ptx_instruction *>> collect_tcgen05_by_opcode(
    gpgpu_context *ctx, const std::string &ptx_path) {
  std::map<int, std::vector<const ptx_instruction *>> by_opcode;

  std::ifstream input(ptx_path);
  if (!input) fail("failed to reopen PTX file for line scan: " + ptx_path);

  std::string line;
  unsigned line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find("tcgen05.") == std::string::npos) {
      continue;
    }
    const ptx_instruction *inst =
        ctx->ptx_parser->ptx_instruction_lookup(ptx_path.c_str(),
                                                line_number);
    if (inst == nullptr) {
      fail("missing parsed instruction for " + ptx_path + ":" +
           std::to_string(line_number));
    }
    if (static_cast<int>(inst->get_opcode()) < 0) {
      continue;
    }
    if (inst->get_opcode() == NOP_OP) {
      fail("TCGen05 instruction decoded as nop: " + line);
    }
    by_opcode[inst->get_opcode()].push_back(inst);
  }
  return by_opcode;
}

void assert_tcgen05_phase1_decode(gpgpu_context *ctx,
                                  const std::string &ptx_path) {
  std::map<int, std::vector<const ptx_instruction *>> by_opcode =
      collect_tcgen05_by_opcode(ctx, ptx_path);
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
  expect_options(wait,
                 {TCGEN05_WAIT_ST_OPTION, SYNC_OPTION, ALIGNED_OPTION});

  expect_operand_count(first(by_opcode, TCGEN05_DEALLOC_OP), 2);
  expect_operand_count(first(by_opcode, TCGEN05_RELINQUISH_ALLOC_PERMIT_OP),
                       0);
}

void assert_tcgen05_surface_decode(gpgpu_context *ctx,
                                   const std::string &ptx_path) {
  std::map<int, std::vector<const ptx_instruction *>> by_opcode =
      collect_tcgen05_by_opcode(ctx, ptx_path);
  expect_opcode_count(by_opcode, TCGEN05_ALLOC_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_CP_OP, 6);
  expect_opcode_count(by_opcode, TCGEN05_SHIFT_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_FENCE_OP, 2);
  expect_opcode_count(by_opcode, TCGEN05_COMMIT_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_LD_OP, 2);
  expect_opcode_count(by_opcode, TCGEN05_ST_OP, 2);
  expect_opcode_count(by_opcode, TCGEN05_WAIT_OP, 2);
  expect_opcode_count(by_opcode, TCGEN05_DEALLOC_OP, 1);
  expect_opcode_count(by_opcode, TCGEN05_RELINQUISH_ALLOC_PERMIT_OP, 1);

  const ptx_instruction *cp = first(by_opcode, TCGEN05_CP_OP);
  expect_operand_count(cp, 2);
  expect_options(cp, {TCGEN05_CTA_GROUP_1_OPTION,
                      TCGEN05_128X256B_OPTION});

  bool saw_warpx2_0213 = false;
  bool saw_warpx2_0123 = false;
  bool saw_warpx4 = false;
  for (const ptx_instruction *inst : by_opcode.at(TCGEN05_CP_OP)) {
    saw_warpx2_0213 |= has_option(inst, TCGEN05_WARPX2_02_13_OPTION);
    saw_warpx2_0123 |= has_option(inst, TCGEN05_WARPX2_01_23_OPTION);
    saw_warpx4 |= has_option(inst, TCGEN05_WARPX4_OPTION);
  }
  if (!saw_warpx2_0213 || !saw_warpx2_0123 || !saw_warpx4) {
    fail("tcgen05.cp surface did not preserve expected shape/warp options");
  }

  const ptx_instruction *shift = first(by_opcode, TCGEN05_SHIFT_OP);
  expect_operand_count(shift, 1);
  expect_options(shift, {TCGEN05_CTA_GROUP_1_OPTION, DOWN_OPTION});

  const std::vector<const ptx_instruction *> &fences =
      by_opcode.at(TCGEN05_FENCE_OP);
  expect_options(fences.at(0), {TCGEN05_BEFORE_THREAD_SYNC_OPTION});
  expect_options(fences.at(1), {TCGEN05_AFTER_THREAD_SYNC_OPTION});

  const ptx_instruction *commit = first(by_opcode, TCGEN05_COMMIT_OP);
  expect_operand_count(commit, 1);
  expect_options(commit, {TCGEN05_CTA_GROUP_1_OPTION,
                          TCGEN05_MBARRIER_ARRIVE_ONE_OPTION,
                          CLUSTER_OPTION});

  bool saw_pack = false;
  for (const ptx_instruction *inst : by_opcode.at(TCGEN05_LD_OP)) {
    saw_pack |= has_option(inst, TCGEN05_PACK_16B_OPTION);
    expect_options(inst, {SYNC_OPTION, ALIGNED_OPTION,
                          TCGEN05_32X32B_OPTION, X1_OPTION});
    expect_vector_width(inst, 0, 1);
  }
  if (!saw_pack) fail("tcgen05.ld surface did not preserve .pack::16b");

  bool saw_unpack = false;
  for (const ptx_instruction *inst : by_opcode.at(TCGEN05_ST_OP)) {
    saw_unpack |= has_option(inst, TCGEN05_UNPACK_16B_OPTION);
    expect_options(inst, {SYNC_OPTION, ALIGNED_OPTION,
                          TCGEN05_32X32B_OPTION, X1_OPTION});
    expect_vector_width(inst, 1, 1);
  }
  if (!saw_unpack) fail("tcgen05.st surface did not preserve .unpack::16b");

  const std::vector<const ptx_instruction *> &waits =
      by_opcode.at(TCGEN05_WAIT_OP);
  expect_options(waits.at(0),
                 {TCGEN05_WAIT_LD_OPTION, SYNC_OPTION, ALIGNED_OPTION});
  expect_options(waits.at(1),
                 {TCGEN05_WAIT_ST_OPTION, SYNC_OPTION, ALIGNED_OPTION});
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
  const std::string ptx_path = argv[1];
  if (ptx_path.find("tcgen05_phase1_smoke") != std::string::npos) {
    assert_tcgen05_phase1_decode(ctx, ptx_path);
  } else if (ptx_path.find("tcgen05_instruction_surface_smoke") !=
                 std::string::npos ||
             ptx_path.find("tcgen05_instruction_surface_inline") !=
                 std::string::npos) {
    assert_tcgen05_surface_decode(ctx, ptx_path);
  } else {
    fail("unknown tcgen05 parser smoke PTX path");
  }

  std::printf("parsed and validated %s\n", argv[1]);
  return 0;
}
