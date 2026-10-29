#ifndef FLASH_GPGPU_SIM_BULK_GROUP_H
#define FLASH_GPGPU_SIM_BULK_GROUP_H

#include <cassert>
#include <map>
#include <set>
#include <utility>

namespace flash_gpgpu_sim {

/**
 * Bulk Group Manager for TMA Store Operations
 *
 * This class implements the bulk group mechanism for asynchronous TMA store
 * operations as specified in the CUDA PTX ISA. Key concepts:
 *
 * 1. **Bulk Groups**: Collections of TMA store transactions that are committed
 *    together. Each warp maintains its own sequence of bulk groups.
 *
 * 2. **Transactions (tx)**: Individual TMA store operations identified by
 * unique IDs (tx_uid). Transactions are first added to a "pending" set, then
 * moved to a committed group when commit_group() is called.
 *
 * 3. **Group Lifecycle**:
 *    - Pending: Transactions added via add_tx() but not yet committed
 *    - Committed: Group created via commit_group(), waiting for tx completions
 *    - Completed: All transactions in the group have finished
 *
 * 4. **Wait Semantics**: wait_group(N) waits until at most N groups remain
 *    incomplete. For example, wait_group(0) waits for all groups to complete.
 *
 * Usage Example:
 *   manager.add_tx(cta, warp, tx1);  // Add tx to pending
 *   manager.add_tx(cta, warp, tx2);
 *   manager.commit_bulk_group(cta, warp);  // Commit as group 1
 *
 *   manager.add_tx(cta, warp, tx3);
 *   manager.commit_bulk_group(cta, warp);  // Commit as group 2
 *
 *   manager.wait_bulk_group(cta, warp, 1); // Wait until only 1 group remains
 *   manager.complete_tx(cta, warp, tx1);   // Mark tx1 done
 *   manager.complete_tx(cta, warp, tx2);   // Group 1 now complete, wait
 * satisfied
 */
class bulk_group_manager_t {
public:
  bulk_group_manager_t() = default;

  /**
   * Add a transaction to the current pending bulk group for the specified warp.
   * The transaction will be included in the next committed group.
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   * @param tx_uid Unique transaction identifier
   */
  void add_tx(unsigned cta_id, unsigned warp_id, unsigned tx_uid);

  /**
   * Commit the current pending transactions as a new bulk group.
   * If no transactions are pending, an empty group is created (which is
   * immediately considered complete for sequencing purposes).
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   */
  void commit_bulk_group(unsigned cta_id, unsigned warp_id);

  /**
   * Begin waiting for bulk groups to complete.
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   * @param latest_group_num Maximum number of incomplete groups allowed.
   *        0 = wait for all groups, 1 = allow 1 incomplete group, etc.
   * @return true if the wait condition is already satisfied, false if waiting
   */
  bool wait_bulk_group(unsigned cta_id, unsigned warp_id,
                       unsigned latest_group_num);

  /**
   * Mark a transaction as completed. This may trigger group completion and
   * potentially satisfy a pending wait condition.
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   * @param tx_uid Unique transaction identifier
   * @return true if no wait is pending (or wait is now satisfied), false if
   * still waiting
   */
  bool complete_tx(unsigned cta_id, unsigned warp_id, unsigned tx_uid);

  /**
   * Drop all bulk group state for a hardware CTA when it is deallocated.
   */
  void cleanup_cta(unsigned cta_id);

  /**
   * Check if a warp is currently waiting on bulk group completion.
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   * @return true if the warp is waiting, false otherwise
   */
  bool is_waiting(unsigned cta_id, unsigned warp_id) const;

  /**
   * Get the number of pending (incomplete) groups for a warp.
   *
   * @param cta_id CTA identifier
   * @param warp_id Warp identifier within the CTA
   * @return Number of groups that have been committed but not yet completed
   */
  unsigned get_pending_group_count(unsigned cta_id, unsigned warp_id) const;

  // True once commit_group has taken this tx out of the uncommitted set.
  bool is_tx_committed(unsigned cta_id, unsigned warp_id,
                       unsigned tx_uid) const;

private:
  // Represents a committed bulk group containing a set of transactions
  struct bulk_group_t {
    unsigned group_id;
    std::set<unsigned> contained_txs;
  };

  // Per-warp bulk group tracking information
  struct warp_bulk_info_t {
    unsigned next_group_id; // Next group ID to assign (starts at 1)
    std::map<unsigned, unsigned> tx_to_group; // Maps tx_uid -> group_id
    std::map<unsigned, bulk_group_t>
        pending_groups;             // Maps group_id -> bulk_group_t
    std::set<unsigned> pending_txs; // Transactions in current uncommitted group
    unsigned
        latest_completed_group_id; // Highest completed group ID (sequential)
    unsigned waiting_group_id;     // Group ID being waited on
    bool is_waiting;               // Whether warp is currently waiting

    warp_bulk_info_t();

    void commit_group();
    void add_tx(unsigned tx_uid);
    bool complete_tx(unsigned tx_uid);
    void wait_group(unsigned group_num);

    unsigned get_pending_count() const { return pending_groups.size(); }
  };

  // Maps (cta_id, warp_id) -> warp bulk info
  std::map<std::pair<unsigned, unsigned>, warp_bulk_info_t> m_warp_bulk_info;
};

// Idealized TMA stores complete the cycle after commit, never on the
// commit cycle, so wait_group can observe a still-incomplete group.
inline unsigned long long
idealized_bulk_write_ready_cycle(unsigned long long commit_cycle) {
  return commit_cycle + 1;
}

inline bool idealized_bulk_write_can_finalize(bool committed,
                                              unsigned long long now,
                                              unsigned long long commit_cycle) {
  if (!committed)
    return false;
  return now >= idealized_bulk_write_ready_cycle(commit_cycle);
}

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_BULK_GROUP_H
