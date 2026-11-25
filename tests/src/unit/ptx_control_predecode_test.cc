#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "../../../libcuda/gpgpu_context.h"
#include "cuda-sim/opcodes.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
void check_typeless_control(const char *opcode, int expected_opcode) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  const auto name = std::string("predecode_") + opcode;
  const auto path = name + ".ptx";
  {
    std::ofstream fixture(path);
    fixture << ".version 8.0\n.target sm_80\n.address_size 64\n"
               ".visible .entry "
            << name
            << "() {\n"
               "bra.uni done;\n"
            << opcode << ";\ndone:\nret;\n}\n";
    ASSERT_TRUE(fixture.good());
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
  std::remove(path.c_str());
  ASSERT_NE(symbols, nullptr);
  auto *func = symbols->lookup_function(name.c_str());
  ASSERT_NE(func, nullptr);
  // Predecode visits the instruction even when control flow skips it.
  func->do_pdom();
  bool found = false;
  for (unsigned pc = func->get_start_PC(), count = 0; count < 4; ++count) {
    const auto *inst = func->get_dyn_inst(pc);
    ASSERT_NE(inst, nullptr);
    if (inst->get_opcode() == expected_opcode) {
      found = true;
      EXPECT_TRUE(inst->get_scalar_type().empty());
      EXPECT_EQ(inst->oprnd_type, UN_OP);
      EXPECT_EQ(inst->sp_op, OTHER_OP);
    }
    if (inst->get_opcode() == RET_OP) break;
    pc += inst->inst_size();
  }
  EXPECT_TRUE(found);
}

TEST(PtxControlPredecode, UnexecutedTrapNeedsNoScalarType) {
  check_typeless_control("trap", TRAP_OP);
}

TEST(PtxControlPredecode, UnexecutedBreakpointNeedsNoScalarType) {
  check_typeless_control("brkpt", BRKPT_OP);
}
}  // namespace
