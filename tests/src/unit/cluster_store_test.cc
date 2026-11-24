#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include "gpgpu-sim/flash/cluster_store.h"
#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
TEST(ClusterStoreContract, OwnerAlignmentWidthAndClusterSize) {
  flash_gpgpu_sim::cluster_store_request request;
  for (unsigned bytes : {4u, 8u, 16u}) {
    ASSERT_TRUE(flash_gpgpu_sim::decode_cluster_store(0x1000100, 0x1000080,
                                                      bytes, 2, request));
    EXPECT_EQ(request.owner, 1u);
    EXPECT_EQ(request.destination, 0x100u);
    EXPECT_EQ(request.barrier, 0x80u);
  }
  for (auto invalid : {0u, 1u, 2u, 32u})
    EXPECT_FALSE(
        flash_gpgpu_sim::decode_cluster_store(256, 128, invalid, 2, request));
  EXPECT_FALSE(flash_gpgpu_sim::decode_cluster_store(256, 128, 16, 1, request));
  EXPECT_FALSE(flash_gpgpu_sim::decode_cluster_store(257, 128, 16, 2, request));
  EXPECT_FALSE(flash_gpgpu_sim::decode_cluster_store(256, 129, 16, 2, request));
  EXPECT_FALSE(
      flash_gpgpu_sim::decode_cluster_store(0x1000100, 128, 16, 2, request));
  EXPECT_FALSE(flash_gpgpu_sim::decode_cluster_store(0x2000100, 0x2000080, 16,
                                                     2, request));
  EXPECT_FALSE(flash_gpgpu_sim::decode_cluster_store(0x100000100ULL, 128, 16, 2,
                                                     request));
}
TEST(ClusterStoreContract, PtxUsesBarrierTransportAndReadsBothAddresses) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  const char *path = "cluster_store_predecode.ptx";
  {
    std::ofstream f(path);
    f << R"(.version 9.2
.target sm_100
.address_size 64
.visible .entry cluster_store_predecode() {
.reg .b32 %r<6>;
st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%r0],{%r1,%r2,%r3,%r4},[%r5];
ret;
})";
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
  std::remove(path);
  ASSERT_NE(symbols, nullptr);
  auto *f = symbols->lookup_function("cluster_store_predecode");
  ASSERT_NE(f, nullptr);
  f->do_pdom();
  auto *inst = f->get_dyn_inst(f->get_start_PC());
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->op, MBARRIER_OP);
  EXPECT_FALSE(inst->has_memory_read());
  EXPECT_FALSE(inst->has_memory_write());
  EXPECT_EQ(inst->get_mbarrier_static_info().operation,
            inst_t::mbarrier_static_info_t::MBARRIER_COMPLETE_TX);
  EXPECT_TRUE(inst->get_mbarrier_static_info().per_lane_complete_tx);
  EXPECT_EQ(inst->incount, 6u);
  EXPECT_EQ(inst->outcount, 0u);
}
