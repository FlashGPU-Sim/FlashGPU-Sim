#include "../test_support.h"
namespace {
TEST(SassTimingProjectionTest, ProjectsPopcIncrementWithSharedAtomicProxy) {
  const auto source = official_instruction(0, "ATOMS.POPC.INC.32", "RZ[R2+UR4]");
  const timing_instruction projected(source);
  EXPECT_TRUE(uses_shared_atomic_timing_proxy(source));
  EXPECT_EQ(projected.op, LOAD_OP);
  EXPECT_EQ(projected.space.get_type(), global_space);
  EXPECT_EQ(projected.cache_op, CACHE_GLOBAL);
  EXPECT_EQ(projected.data_size, 4u);
  EXPECT_TRUE(projected.isatomic());
  EXPECT_FALSE(projected.executes_atomic_callback());
  EXPECT_EQ(projected.outcount, 0u);
  ASSERT_EQ(projected.incount, 2u);
  EXPECT_EQ(projected.arch_reg.src[0], 3);
  EXPECT_EQ(projected.arch_reg.src[1], 261);
  EXPECT_TRUE(orders_deferred_cta_barrier(source, architecture::kSm100));
  EXPECT_FALSE(uses_shared_atomic_timing_proxy(
      official_instruction(0, "ATOMS.POPC.INC.64", "RZ[R2+UR4]")));
}

} // namespace
