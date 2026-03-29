#include "bulk_group.h"
#include "../shader.h"

namespace flash_gpgpu_sim {

//=============================================================================
// warp_bulk_info_t Implementation
//=============================================================================

bulk_group_manager_t::warp_bulk_info_t::warp_bulk_info_t()
    : next_group_id(1), latest_completed_group_id(0), waiting_group_id(0),
      is_waiting(false) {}

void bulk_group_manager_t::warp_bulk_info_t::add_tx(unsigned tx_uid) {
  pending_txs.insert(tx_uid);
}

void bulk_group_manager_t::warp_bulk_info_t::commit_group() {
  assert(pending_groups.find(next_group_id) == pending_groups.end() &&
         "Committing a bulk group that already exists");

  if (!pending_txs.empty()) {
    // Move pending transactions to a new committed group
    for (auto tx_uid : pending_txs) {
      tx_to_group[tx_uid] = next_group_id;
    }
    pending_groups.emplace(next_group_id,
                           bulk_group_t{next_group_id, std::move(pending_txs)});
    pending_txs.clear();
  } else if (next_group_id == latest_completed_group_id + 1) {
    // Empty group: immediately mark as completed if it's the next in sequence
    latest_completed_group_id++;
  }
  next_group_id++;
}

bool bulk_group_manager_t::warp_bulk_info_t::complete_tx(unsigned tx_uid) {
  auto it = tx_to_group.find(tx_uid);

  if (it != tx_to_group.end()) {
    // Transaction belongs to a committed group
    unsigned group_id = it->second;
    tx_to_group.erase(it);

    auto group_it = pending_groups.find(group_id);
    assert(group_it != pending_groups.end() &&
           "Group for the tx does not exist");
    group_it->second.contained_txs.erase(tx_uid);

    // Check if the group is now empty (all transactions complete)
    if (group_it->second.contained_txs.empty()) {
      pending_groups.erase(group_it);

      // Update latest_completed_group_id if this was the next expected group
      if (group_id == latest_completed_group_id + 1) {
        latest_completed_group_id++;

        // Skip over any empty groups that were already completed
        while (latest_completed_group_id + 1 < next_group_id &&
               pending_groups.find(latest_completed_group_id + 1) ==
                   pending_groups.end()) {
          latest_completed_group_id++;
        }

        // Check if wait condition is now satisfied
        // wait_group(N) means: wait until (next_group_id - 1 -
        // latest_completed) <= N
        if (is_waiting && next_group_id - waiting_group_id >=
                              next_group_id - latest_completed_group_id) {
          is_waiting = false;
        }
      }
    }
  } else {
    // Transaction is still in pending (uncommitted) state, just remove it
    pending_txs.erase(tx_uid);
  }

  return !is_waiting;
}

void bulk_group_manager_t::warp_bulk_info_t::wait_group(unsigned group_num) {
  assert(!is_waiting && "Warp is already waiting on a bulk group");

  // Calculate number of incomplete groups
  // incomplete_count = next_group_id - 1 - latest_completed_group_id
  unsigned incomplete_count = next_group_id - latest_completed_group_id - 1;

  if (group_num < incomplete_count) {
    // Need to wait: set the target group ID
    waiting_group_id = next_group_id - group_num - 1;
    is_waiting = true;
  }
  // If group_num >= incomplete_count, wait is immediately satisfied
}

//=============================================================================
// bulk_group_manager_t Implementation
//=============================================================================

void bulk_group_manager_t::add_tx(unsigned cta_id, unsigned warp_id,
                                  unsigned tx_uid) {
  auto key = std::make_pair(cta_id, warp_id);
  auto &info = m_warp_bulk_info[key];
  info.add_tx(tx_uid);
}

void bulk_group_manager_t::commit_bulk_group(unsigned cta_id,
                                             unsigned warp_id) {
  auto key = std::make_pair(cta_id, warp_id);
  auto it = m_warp_bulk_info.find(key);
  if (it != m_warp_bulk_info.end()) {
    it->second.commit_group();
  } else {
    // Create new warp info and commit empty group
    auto &info = m_warp_bulk_info[key];
    info.commit_group();
  }
}

bool bulk_group_manager_t::wait_bulk_group(unsigned cta_id, unsigned warp_id,
                                           unsigned latest_group_num) {
  auto key = std::make_pair(cta_id, warp_id);
  auto it = m_warp_bulk_info.find(key);

  if (it == m_warp_bulk_info.end()) {
    // No bulk groups exist for this warp, wait is immediately satisfied
    return true;
  }

  auto &info = it->second;
  info.wait_group(latest_group_num);

  return !info.is_waiting;
}

bool bulk_group_manager_t::complete_tx(unsigned cta_id, unsigned warp_id,
                                       unsigned tx_uid) {
  auto key = std::make_pair(cta_id, warp_id);
  auto it = m_warp_bulk_info.find(key);

  assert(it != m_warp_bulk_info.end() &&
         "Completing a tx for a warp with no bulk groups");

  it->second.complete_tx(tx_uid);
  return !it->second.is_waiting;
}

bool bulk_group_manager_t::is_waiting(unsigned cta_id, unsigned warp_id) const {
  auto key = std::make_pair(cta_id, warp_id);
  auto it = m_warp_bulk_info.find(key);

  if (it == m_warp_bulk_info.end()) {
    return false;
  }
  return it->second.is_waiting;
}

unsigned bulk_group_manager_t::get_pending_group_count(unsigned cta_id,
                                                       unsigned warp_id) const {
  auto key = std::make_pair(cta_id, warp_id);
  auto it = m_warp_bulk_info.find(key);

  if (it == m_warp_bulk_info.end()) {
    return 0;
  }
  return it->second.get_pending_count();
}

} // namespace flash_gpgpu_sim

//=============================================================================
// Bulk Group Methods (for TMA write operations)
//=============================================================================

void barrier_set_t::add_bulk_tx(unsigned cta_id, unsigned warp_id,
                                unsigned tx_uid) {
  m_bulk_group_manager.add_tx(cta_id, warp_id, tx_uid);
}

void barrier_set_t::complete_bulk_tx(unsigned cta_id, unsigned warp_id,
                                     unsigned tx_uid) {
  bool satisfied = m_bulk_group_manager.complete_tx(cta_id, warp_id, tx_uid);

  // If the warp was waiting and the wait is now satisfied, release the warp
  if (satisfied) {
    m_warp_at_barrier.reset(warp_id);
  }
}

void barrier_set_t::wait_bulk_group(unsigned cta_id, unsigned warp_id,
                                    unsigned latest_group_num) {
  bool satisfied =
      m_bulk_group_manager.wait_bulk_group(cta_id, warp_id, latest_group_num);

  // If the wait is not immediately satisfied, mark the warp as blocked
  if (!satisfied) {
    m_warp_at_barrier.set(warp_id);
    m_warp_barrier_type[warp_id] = BARRIER_WAIT_BULK_GROUP;
  }
}

void barrier_set_t::commit_bulk_group(unsigned cta_id, unsigned warp_id) {
  m_bulk_group_manager.commit_bulk_group(cta_id, warp_id);
}