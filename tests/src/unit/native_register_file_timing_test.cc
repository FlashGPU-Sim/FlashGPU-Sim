#include <gtest/gtest.h>

#include "gpgpu-sim/flash/native_register_file_timing.h"

namespace flash_gpgpu_sim {
namespace {

using source_t = native_register_file_timing::source_t;

TEST(NativeRegisterFileTiming, DifferentBanksReserveInParallel) {
  native_register_file_timing rf;
  const auto allocation = rf.prepare({source_t{2, 0}, source_t{3, 1}}, 7);

  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.reads_per_bank, (std::vector<unsigned>{1, 1}));
  rf.commit(allocation);
  EXPECT_EQ(rf.reserved_reads(0, 1), 1u);
  EXPECT_EQ(rf.reserved_reads(1, 1), 1u);
}

TEST(NativeRegisterFileTiming, ThreeSameBankReadsFillReadWindow) {
  native_register_file_timing rf;
  const auto allocation =
      rf.prepare({source_t{0, 0}, source_t{2, 1}, source_t{4, 2}}, 7);

  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);
  EXPECT_EQ(rf.reserved_reads(0, 1), 1u);
  EXPECT_EQ(rf.reserved_reads(0, 2), 1u);
  EXPECT_EQ(rf.reserved_reads(0, 3), 1u);
}

TEST(NativeRegisterFileTiming, AllocateStallsWhenWindowCannotFitReads) {
  native_register_file_timing rf;
  const auto first =
      rf.prepare({source_t{0, 0}, source_t{2, 1}, source_t{4, 2}}, 7);
  ASSERT_TRUE(first.ready);
  rf.commit(first);

  EXPECT_FALSE(rf.prepare({source_t{6, 0}}, 7).ready);
  rf.cycle();
  EXPECT_TRUE(rf.prepare({source_t{6, 0}}, 7).ready);
}

TEST(NativeRegisterFileTiming, CacheHitWithoutRetainEvictsEntry) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);
  ASSERT_TRUE(rf.reuse_hit(0, 7, 2));

  allocation = rf.prepare({source_t{2, 0, false}}, 7);
  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.cache_hits, 1u);
  rf.commit(allocation);
  EXPECT_FALSE(rf.reuse_hit(0, 7, 2));
  EXPECT_EQ(rf.prepare({source_t{2, 0}}, 7).cache_hits, 0u);
}

TEST(NativeRegisterFileTiming, CacheHitWithRetainSurvives) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);

  allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.cache_hits, 1u);
  rf.commit(allocation);
  EXPECT_EQ(rf.prepare({source_t{2, 0}}, 7).cache_hits, 1u);
}

TEST(NativeRegisterFileTiming, CacheIdentityIncludesOperandSlotAndWarp) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);

  EXPECT_EQ(rf.prepare({source_t{2, 1}}, 7).cache_hits, 0u);
  EXPECT_EQ(rf.prepare({source_t{2, 0}}, 8).cache_hits, 0u);
  EXPECT_EQ(rf.prepare({source_t{2, 0}}, 7).cache_hits, 1u);
}

TEST(NativeRegisterFileTiming, SameBankAndSlotAccessReplacesEntry) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);

  allocation = rf.prepare({source_t{4, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.cache_hits, 0u);
  rf.commit(allocation);
  EXPECT_FALSE(rf.reuse_hit(0, 7, 2));
  EXPECT_TRUE(rf.reuse_hit(0, 7, 4));
}

TEST(NativeRegisterFileTiming, StalledAllocateDoesNotMutateCache) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);
  rf.cycle();

  allocation = rf.prepare(
      {source_t{0, 0, true}, source_t{4, 1}, source_t{6, 2}}, 8);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);
  const auto stalled = rf.prepare({source_t{8, 0, true}}, 8);
  ASSERT_FALSE(stalled.ready);

  EXPECT_TRUE(rf.reuse_hit(0, 8, 0));
}

TEST(NativeRegisterFileTiming, DuplicateRegisterNeedsOnePhysicalRead) {
  native_register_file_timing rf;
  const auto allocation =
      rf.prepare({source_t{2, 0}, source_t{2, 1}}, 7);

  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.reads_per_bank, (std::vector<unsigned>{1, 0}));
}

TEST(NativeRegisterFileTiming, ResetClearsReservationsAndReuseEntries) {
  native_register_file_timing rf;
  const auto allocation = rf.prepare(
      {source_t{0, 0, true}, source_t{2, 1}, source_t{4, 2}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);

  rf.reset();
  EXPECT_FALSE(rf.reuse_hit(0, 7, 0));
  const auto after_reset = rf.prepare({source_t{6, 0}}, 7);
  EXPECT_TRUE(after_reset.ready);
  EXPECT_EQ(after_reset.cache_hits, 0u);
}

TEST(NativeRegisterFileTiming, NonCacheableSourceIgnoresReuseState) {
  native_register_file_timing rf;
  auto allocation = rf.prepare({source_t{2, 0, true}}, 7);
  ASSERT_TRUE(allocation.ready);
  rf.commit(allocation);
  rf.cycle();
  ASSERT_TRUE(rf.reuse_hit(0, 7, 2));

  allocation = rf.prepare({source_t{2, 0, true, false}}, 7);
  ASSERT_TRUE(allocation.ready);
  EXPECT_EQ(allocation.cache_hits, 0u);
  EXPECT_EQ(allocation.reads_per_bank, (std::vector<unsigned>{1, 0}));
  rf.commit(allocation);
  EXPECT_TRUE(rf.reuse_hit(0, 7, 2));
}

TEST(NativeRegisterFileTiming, ResultQueueBackpressuresUntilRetirement) {
  native_register_file_timing rf(2, 1, 3, 2, 2);
  ASSERT_TRUE(rf.can_enqueue_result());
  rf.enqueue_result();
  ASSERT_TRUE(rf.can_enqueue_result());
  rf.enqueue_result();
  EXPECT_FALSE(rf.can_enqueue_result());
  EXPECT_EQ(rf.pending_results(), 2u);

  rf.retire_result();
  EXPECT_TRUE(rf.can_enqueue_result());
  EXPECT_EQ(rf.pending_results(), 1u);
  rf.reset();
  EXPECT_EQ(rf.pending_results(), 0u);
}

TEST(NativeRegisterFileTiming, ResultQueueLimitsPopsPerCycle) {
  native_register_file_timing rf(2, 1, 3, 2, 1);
  rf.enqueue_result();
  rf.enqueue_result();

  ASSERT_TRUE(rf.can_pop_result());
  rf.retire_result();
  EXPECT_FALSE(rf.can_pop_result());
  EXPECT_EQ(rf.pending_results(), 1u);

  rf.cycle();
  ASSERT_TRUE(rf.can_pop_result());
  rf.retire_result();
  EXPECT_EQ(rf.pending_results(), 0u);
}

TEST(NativeRegisterFileTiming, RegisterWritesContendWithinEachBank) {
  native_register_file_timing rf;
  ASSERT_TRUE(rf.can_write_result({2}));
  EXPECT_EQ(rf.write_result({2}), 1u);
  EXPECT_FALSE(rf.can_write_result({4}));
  EXPECT_TRUE(rf.can_write_result({3}));
  EXPECT_EQ(rf.write_result({3}), 1u);

  rf.cycle();
  EXPECT_TRUE(rf.can_write_result({4}));
}

TEST(NativeRegisterFileTiming, MultiRegisterWritesAreCappedByPortWidth) {
  native_register_file_timing rf;
  ASSERT_TRUE(rf.can_write_result({8, 9, 10, 11}));
  EXPECT_EQ(rf.write_result({8, 9, 10, 11}), 2u);
  EXPECT_FALSE(rf.can_write_result({12}));
  EXPECT_FALSE(rf.can_write_result({13}));
}

} // namespace
} // namespace flash_gpgpu_sim
