#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include "../../../libcuda/gpgpu_context.h"
#include "cuda-sim/ptx_ir.h"
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
namespace {
function_info *parse_coords(const char *name, const std::string &body) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  // Test the simulator parser directly, without ptxas rejecting inputs first.
  const bool reorder = ctx->ptx_reorder_enabled;
  const bool guided = ctx->ptx_reorder_sass_guided;
  ctx->ptx_reorder_enabled = false;
  ctx->ptx_reorder_sass_guided = false;
  const auto path = std::string(name) + ".ptx";
  {
    std::ofstream out(path);
    out << ".version 8.6\n.target sm_100\n.address_size 64\n.visible .entry "
        << name << "(){\n.reg .b32 x,b; .reg .b64 a;\n"
        << body << "\nret;\n}\n";
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
  std::remove(path.c_str());
  auto *f = symbols->lookup_function(name);
  f->do_pdom();
  ctx->ptx_reorder_enabled = reorder;
  ctx->ptx_reorder_sass_guided = guided;
  return f;
}
TEST(PtxTensorCoordinates, MixedIntegerCoordinatesRetainValuesAndPredecode) {
  auto *f = parse_coords("tensor_literal_coords", R"(
cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [x],[a,{x,1,-2,0}],[b];
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [a,{x,1}],[b];
cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [x],[a,{x,1,2,3,4}],[b];
)");
  auto pc = f->get_start_PC();
  unsigned ordinal = 0;
  for (unsigned count : {4u, 2u, 5u}) {
    const auto *inst = f->get_dyn_inst(pc);
    ASSERT_NE(inst, nullptr);
    const auto &v = inst->get_operands()[ordinal == 1 ? 1 : 2];
    ASSERT_TRUE(v.is_vector());
    ASSERT_EQ(v.get_vect_nelem(), count);
    EXPECT_FALSE(v.vec_is_literal(0));
    EXPECT_EQ(v.vec_literal_value(1).u32, 1u);
    if (ordinal == 0) EXPECT_EQ(v.vec_literal_value(2).s32, -2);
    if (ordinal == 2) EXPECT_EQ(v.vec_literal_value(4).s32, 4);
    pc += inst->inst_size();
    ++ordinal;
  }
}
void reject_coords(const char *name, const std::string &body) {
  // The parser prints diagnostics to stdout; death tests capture stderr.
  std::fflush(stdout);
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) std::abort();
  parse_coords(name, body);
}
TEST(PtxTensorCoordinates, RejectsFloatWidthAndCoordinateCount) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      reject_coords("bad_tensor_float",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{x,1.5}],[b];"),
      "tensor coordinate literals must be integers");
  EXPECT_DEATH(
      reject_coords("bad_tensor_width",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{x,4294967296}],[b];"),
      "tensor coordinate literal exceeds 32 bits");
  EXPECT_DEATH(
      reject_coords("bad_tensor_count",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{x,1,2}],[b];"),
      "tensor coordinate count");
}
}  // namespace
