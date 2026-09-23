#include <gtest/gtest.h>

// Include the standalone bulk_group header
#include "../../../src/gpgpu-sim/flash/bulk_group.h"

using namespace flash_gpgpu_sim;

//=============================================================================
// Test Fixture
//=============================================================================

class BulkGroupTest : public ::testing::Test {
protected:
  bulk_group_manager_t manager;

  // Common test identifiers
  static constexpr unsigned CTA_0 = 0;
  static constexpr unsigned CTA_1 = 1;
  static constexpr unsigned WARP_0 = 0;
  static constexpr unsigned WARP_1 = 1;

  void SetUp() override {
    // Fresh manager for each test
  }

  void TearDown() override {
    // Cleanup handled by destructor
  }
};

//=============================================================================
// Basic Functionality Tests
//=============================================================================

// Test: Basic add and commit operations
TEST_F(BulkGroupTest, BasicAddAndCommit) {
  // Add transactions to pending group
  manager.add_tx(CTA_0, WARP_0, /*tx_uid=*/100);
  manager.add_tx(CTA_0, WARP_0, /*tx_uid=*/101);

  // Commit the group
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Should have 1 pending group
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_TRUE(manager.is_tx_committed(CTA_0, WARP_0, 100));
  EXPECT_TRUE(manager.is_tx_committed(CTA_0, WARP_0, 101));
}

// Test: Empty group commit
TEST_F(BulkGroupTest, EmptyGroupCommit) {
  // Commit without adding any transactions
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Empty group should be immediately completed
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

// Test: Multiple empty group commits
TEST_F(BulkGroupTest, MultipleEmptyGroupCommits) {
  manager.commit_bulk_group(CTA_0, WARP_0);
  manager.commit_bulk_group(CTA_0, WARP_0);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // All empty groups should be immediately completed
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Transaction completion removes from pending
TEST_F(BulkGroupTest, TransactionCompletion) {
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  // Complete the transaction
  manager.complete_tx(CTA_0, WARP_0, 100);

  // Group should now be completed
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

//=============================================================================
// Wait Mechanism Tests
//=============================================================================

TEST_F(BulkGroupTest, TxNotCommittedUntilCommitGroup) {
  manager.add_tx(CTA_0, WARP_0, 100);
  EXPECT_FALSE(manager.is_tx_committed(CTA_0, WARP_0, 100));
  manager.commit_bulk_group(CTA_0, WARP_0);
  EXPECT_TRUE(manager.is_tx_committed(CTA_0, WARP_0, 100));
}

// wait_group stays blocked until complete_tx (the TMA finalize hook).
// The deferred-complete policy forbids finalize on the commit cycle.
TEST_F(BulkGroupTest, WaitBlockedUntilCommittedTxFinalized) {
  constexpr unsigned long long kCommitCycle = 40;
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_FALSE(manager.wait_bulk_group(CTA_0, WARP_0, 0));
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_FALSE(idealized_bulk_write_can_finalize(
      /*committed=*/false, kCommitCycle, kCommitCycle));
  EXPECT_FALSE(idealized_bulk_write_can_finalize(
      /*committed=*/true, kCommitCycle, kCommitCycle));
  EXPECT_TRUE(idealized_bulk_write_can_finalize(
      /*committed=*/true, idealized_bulk_write_ready_cycle(kCommitCycle),
      kCommitCycle));
  // Eligible to finalize is not the same as wait retiring.
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  EXPECT_TRUE(manager.complete_tx(CTA_0, WARP_0, 100));
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

// Test: Wait for all groups (wait_group(0))
TEST_F(BulkGroupTest, WaitForAllGroups) {
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Wait for all groups to complete (N=0 means no incomplete groups allowed)
  bool wait_result = manager.wait_bulk_group(CTA_0, WARP_0, 0);

  // Should be waiting since tx 100 is not complete
  EXPECT_FALSE(wait_result);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  // Complete the transaction
  manager.complete_tx(CTA_0, WARP_0, 100);

  // Wait should now be satisfied
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

// Test: Wait with allowance (wait_group(1))
TEST_F(BulkGroupTest, WaitWithAllowance) {
  // Create 2 groups
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 1

  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 2

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);

  // Wait allowing 1 incomplete group
  bool wait_result = manager.wait_bulk_group(CTA_0, WARP_0, 1);

  // Should be waiting (2 incomplete > 1 allowed)
  EXPECT_FALSE(wait_result);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  // Complete group 1
  manager.complete_tx(CTA_0, WARP_0, 100);

  // Now only 1 incomplete group, wait should be satisfied
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);
}

// Test: Wait already satisfied
TEST_F(BulkGroupTest, WaitAlreadySatisfied) {
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Wait allowing 1 incomplete group (we have exactly 1)
  bool wait_result = manager.wait_bulk_group(CTA_0, WARP_0, 1);

  // Should be immediately satisfied
  EXPECT_TRUE(wait_result);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

// Test: Wait on empty warp
TEST_F(BulkGroupTest, WaitOnEmptyWarp) {
  // Wait on a warp with no groups
  bool wait_result = manager.wait_bulk_group(CTA_0, WARP_0, 0);

  // Should be immediately satisfied
  EXPECT_TRUE(wait_result);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

//=============================================================================
// Multi-Group Completion Order Tests
//=============================================================================

// Test: In-order group completion
TEST_F(BulkGroupTest, InOrderGroupCompletion) {
  // Create 3 groups
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 1

  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 2

  manager.add_tx(CTA_0, WARP_0, 300);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 3

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 3u);

  // Complete in order
  manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);

  manager.complete_tx(CTA_0, WARP_0, 200);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  manager.complete_tx(CTA_0, WARP_0, 300);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Out-of-order group completion
TEST_F(BulkGroupTest, OutOfOrderGroupCompletion) {
  // Create 3 groups
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 1

  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 2

  manager.add_tx(CTA_0, WARP_0, 300);
  manager.commit_bulk_group(CTA_0, WARP_0);  // Group 3

  // Complete out of order: 2, 3, 1
  // Note: pending_group_count tracks physically incomplete groups (groups with
  // remaining transactions), NOT logically completed groups (which require
  // sequential completion for wait semantics).
  manager.complete_tx(CTA_0, WARP_0, 200);  // Group 2 physically done
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);

  manager.complete_tx(CTA_0, WARP_0, 300);  // Group 3 physically done
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  manager.complete_tx(CTA_0, WARP_0, 100);  // Group 1 done, all logically complete
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Out-of-order completion with wait
TEST_F(BulkGroupTest, OutOfOrderCompletionWithWait) {
  // Create 3 groups
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);

  manager.add_tx(CTA_0, WARP_0, 300);
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 3u);

  // Wait for at most 1 incomplete group
  manager.wait_bulk_group(CTA_0, WARP_0, 1);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  // Complete group 3 first - doesn't satisfy wait because group 1 blocks
  // logical completion (wait uses sequential completion semantics)
  manager.complete_tx(CTA_0, WARP_0, 300);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);  // Physically 2 groups left

  // Complete group 2
  manager.complete_tx(CTA_0, WARP_0, 200);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);  // Physically 1 group left

  // Complete group 1 - now all are done, wait satisfied
  manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

//=============================================================================
// Multi-Transaction Group Tests
//=============================================================================

// Test: Group with multiple transactions
TEST_F(BulkGroupTest, GroupWithMultipleTransactions) {
  // Add multiple transactions to one group
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.add_tx(CTA_0, WARP_0, 101);
  manager.add_tx(CTA_0, WARP_0, 102);
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  // Complete transactions one by one
  manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  manager.complete_tx(CTA_0, WARP_0, 101);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  manager.complete_tx(CTA_0, WARP_0, 102);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Multiple groups with multiple transactions each
TEST_F(BulkGroupTest, MultipleGroupsMultipleTransactions) {
  // Group 1: 2 transactions
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.add_tx(CTA_0, WARP_0, 101);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Group 2: 3 transactions
  manager.add_tx(CTA_0, WARP_0, 200);
  manager.add_tx(CTA_0, WARP_0, 201);
  manager.add_tx(CTA_0, WARP_0, 202);
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);

  // Complete group 1
  manager.complete_tx(CTA_0, WARP_0, 100);
  manager.complete_tx(CTA_0, WARP_0, 101);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  // Complete group 2
  manager.complete_tx(CTA_0, WARP_0, 200);
  manager.complete_tx(CTA_0, WARP_0, 201);
  manager.complete_tx(CTA_0, WARP_0, 202);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

//=============================================================================
// Multi-Warp Independence Tests
//=============================================================================

// Test: Different warps are independent
TEST_F(BulkGroupTest, MultiWarpIndependence) {
  // Warp 0: Add and commit
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Warp 1: Add and commit
  manager.add_tx(CTA_0, WARP_1, 200);
  manager.commit_bulk_group(CTA_0, WARP_1);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_1), 1u);

  // Complete warp 0's transaction
  manager.complete_tx(CTA_0, WARP_0, 100);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_1), 1u);  // Unaffected
}

// Test: Different CTAs are independent
TEST_F(BulkGroupTest, MultiCTAIndependence) {
  // CTA 0, Warp 0
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // CTA 1, Warp 0
  manager.add_tx(CTA_1, WARP_0, 200);
  manager.commit_bulk_group(CTA_1, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);
  EXPECT_EQ(manager.get_pending_group_count(CTA_1, WARP_0), 1u);

  // Complete CTA 1's transaction
  manager.complete_tx(CTA_1, WARP_0, 200);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);  // Unaffected
  EXPECT_EQ(manager.get_pending_group_count(CTA_1, WARP_0), 0u);
}

// Test: Wait independence across warps
TEST_F(BulkGroupTest, WaitIndependenceAcrossWarps) {
  // Both warps have pending groups
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  manager.add_tx(CTA_0, WARP_1, 200);
  manager.commit_bulk_group(CTA_0, WARP_1);

  // Warp 0 waits
  manager.wait_bulk_group(CTA_0, WARP_0, 0);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_1));

  // Warp 1 completes (doesn't affect warp 0's wait)
  manager.complete_tx(CTA_0, WARP_1, 200);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  // Warp 0 completes
  manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_FALSE(manager.is_waiting(CTA_0, WARP_0));
}

//=============================================================================
// Edge Case Tests
//=============================================================================

// Test: Complete uncommitted transaction
TEST_F(BulkGroupTest, CompleteUncommittedTransaction) {
  manager.add_tx(CTA_0, WARP_0, 100);
  // Don't commit - transaction is still in pending_txs

  // This should remove from pending_txs, not cause errors
  manager.complete_tx(CTA_0, WARP_0, 100);

  // Commit empty group
  manager.commit_bulk_group(CTA_0, WARP_0);
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Mixed empty and non-empty groups
TEST_F(BulkGroupTest, MixedEmptyAndNonEmptyGroups) {
  // Non-empty group
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Empty group
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Non-empty group
  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Should have 2 pending groups (empty one auto-completed)
  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 2u);
}

// Test: Large number of groups (stress test)
TEST_F(BulkGroupTest, StressTestManyGroups) {
  const unsigned NUM_GROUPS = 100;

  for (unsigned i = 0; i < NUM_GROUPS; ++i) {
    manager.add_tx(CTA_0, WARP_0, i);
    manager.commit_bulk_group(CTA_0, WARP_0);
  }

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), NUM_GROUPS);

  // Complete all in reverse order
  for (unsigned i = NUM_GROUPS; i > 0; --i) {
    manager.complete_tx(CTA_0, WARP_0, i - 1);
  }

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

// Test: Large number of transactions per group
TEST_F(BulkGroupTest, StressTestManyTransactionsPerGroup) {
  const unsigned NUM_TXS = 100;

  for (unsigned i = 0; i < NUM_TXS; ++i) {
    manager.add_tx(CTA_0, WARP_0, i);
  }
  manager.commit_bulk_group(CTA_0, WARP_0);

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 1u);

  // Complete all
  for (unsigned i = 0; i < NUM_TXS; ++i) {
    manager.complete_tx(CTA_0, WARP_0, i);
  }

  EXPECT_EQ(manager.get_pending_group_count(CTA_0, WARP_0), 0u);
}

//=============================================================================
// Return Value Tests
//=============================================================================

// Test: complete_tx return value when not waiting
TEST_F(BulkGroupTest, CompleteTxReturnValueNotWaiting) {
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // A transaction completion is not a scheduler release unless it satisfies a
  // pending cp.async.bulk.wait_group.
  bool result = manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_FALSE(result);
}

// Test: complete_tx return value when waiting
TEST_F(BulkGroupTest, CompleteTxReturnValueWhenWaiting) {
  manager.add_tx(CTA_0, WARP_0, 100);
  manager.commit_bulk_group(CTA_0, WARP_0);

  manager.add_tx(CTA_0, WARP_0, 200);
  manager.commit_bulk_group(CTA_0, WARP_0);

  // Start waiting for all groups
  manager.wait_bulk_group(CTA_0, WARP_0, 0);
  EXPECT_TRUE(manager.is_waiting(CTA_0, WARP_0));

  // Complete first group - still waiting
  bool result1 = manager.complete_tx(CTA_0, WARP_0, 100);
  EXPECT_FALSE(result1);  // Still waiting

  // Complete second group - wait satisfied
  bool result2 = manager.complete_tx(CTA_0, WARP_0, 200);
  EXPECT_TRUE(result2);  // No longer waiting
}
