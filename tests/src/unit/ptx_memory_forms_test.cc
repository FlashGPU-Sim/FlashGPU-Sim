#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
TEST(PtxMemoryForms, CacheHintsPreserveOperatorAndBulkIsTypeless) {
  auto *ctx=GPGPU_Context(); GPGPUSim_Context(ctx);
  const char *path="memory_forms_predecode.ptx";
  {
    std::ofstream f(path);
    f << R"(.version 9.0
.target sm_100
.address_size 64
.visible .entry memory_forms_predecode() {
.reg .u64 a, r, r1, r2, r3;
.reg .u32 n;
ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v4.u64 {r,r1,r2,r3}, [a];
ld.global.L1::no_allocate.L2::64B.u64 r, [a];
ld.global.cg.L2::128B.u64 r, [a];
st.bulk.weak.shared::cta [a], n, 0;
ret;
})";
  }
  auto *symbols=ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
  std::remove(path); ASSERT_NE(symbols,nullptr);
  auto *func=symbols->lookup_function("memory_forms_predecode");
  ASSERT_NE(func,nullptr); func->do_pdom();
  unsigned pc=func->get_start_PC();
  for(auto expected:{CACHE_L1,CACHE_ALL,CACHE_GLOBAL}) {
    auto *inst=func->get_dyn_inst(pc); ASSERT_NE(inst,nullptr);
    EXPECT_EQ(inst->get_opcode(),LD_OP); EXPECT_EQ(inst->cache_op,expected);
    pc+=inst->inst_size();
  }
  auto *inst=func->get_dyn_inst(pc); ASSERT_NE(inst,nullptr);
  EXPECT_EQ(inst->get_opcode(),ST_BULK_OP);
  EXPECT_TRUE(inst->get_scalar_type().empty());
  EXPECT_TRUE(inst->has_memory_write()); EXPECT_EQ(inst->op,STORE_OP);
  EXPECT_EQ(inst->space.get_type(),shared_space);
  EXPECT_EQ(inst->data_size,8u);
}
