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
TEST(PtxTensorCoordinates, OrdinaryLoadAndStoreRanksOneThroughFive) {
  const char *coordinates[] = {"0", "x,1", "x,1,-2", "x,1,-2,0", "x,1,-2,0,4294967295"};
  for (unsigned rank = 1; rank <= 5; ++rank) {
    for (bool store : {false, true}) {
      const std::string name = "literal_coords_" + std::to_string(rank) + (store ? "_store" : "_load");
      std::string body = "cp.async.bulk.tensor." + std::to_string(rank) + "d.";
      body += store ? "global.shared::cta.bulk_group [a,{" :
                      "shared::cta.global.mbarrier::complete_tx::bytes [x],[a,{";
      body += coordinates[rank - 1];
      body += "}],[b];";
      auto *f = parse_coords(name.c_str(), body);
      const auto *inst = f->get_dyn_inst(f->get_start_PC());
      ASSERT_NE(inst, nullptr);
      const auto &v = inst->get_operands()[store ? 1 : 2];
      ASSERT_TRUE(v.is_vector());
      ASSERT_EQ(v.get_vect_nelem(), rank);
      if (rank == 1) {
        EXPECT_TRUE(v.vec_is_literal(0));
        EXPECT_EQ(v.vec_literal_value(0).u32, 0u);
      } else {
        EXPECT_FALSE(v.vec_is_literal(0));
        EXPECT_EQ(v.vec_literal_value(1).u32, 1u);
      }
      if (rank >= 3) EXPECT_EQ(v.vec_literal_value(2).s32, -2);
      if (rank >= 4) EXPECT_EQ(v.vec_literal_value(3).u32, 0u);
      if (rank == 5) EXPECT_EQ(v.vec_literal_value(4).u32, UINT32_MAX);
    }
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
      reject_coords("bad_tensor_register",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{a,1}],[b];"),
      "tensor coordinate registers must be 32-bit integers");
  EXPECT_DEATH(
      reject_coords("bad_tensor_negative_width",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{x,-2147483649}],[b];"),
      "tensor coordinate literal exceeds 32 bits");
  EXPECT_DEATH(
      reject_coords("bad_tensor_count",
                   "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes [x],[a,{x,1,2}],[b];"),
      "tensor coordinate count");
}
}  // namespace
