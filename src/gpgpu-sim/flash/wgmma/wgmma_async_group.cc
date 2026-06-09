#include "wgmma_async_group.h"

#include <cassert>

#include "../../shader.h"

namespace flash_gpgpu_sim {

void wgmma_async_group_manager_t::warpgroup_info_t::add_op(
    unsigned op_uid) {
  pending_ops.insert(op_uid);
}

void wgmma_async_group_manager_t::warpgroup_info_t::commit_group() {
  group_t group;
  group.group_id = next_group_id++;
  group.op_uids.swap(pending_ops);

  if (group.op_uids.empty()) return;

  for (std::set<unsigned>::const_iterator it = group.op_uids.begin();
       it != group.op_uids.end(); ++it) {
    op_to_group[*it] = group.group_id;
  }
  pending_groups[group.group_id] = group;
}

bool wgmma_async_group_manager_t::warpgroup_info_t::wait_satisfied() const {
  return pending_groups.size() <= wait_allowance;
}

wgmma_async_group_manager_t::wait_result_t
wgmma_async_group_manager_t::warpgroup_info_t::release_if_satisfied() {
  wait_result_t result;
  if (!is_waiting || !wait_satisfied()) return result;

  result.satisfied = true;
  result.released_warps = waiting_warps;
  waiting_warps.clear();
  is_waiting = false;
  return result;
}

wgmma_async_group_manager_t::wait_result_t
wgmma_async_group_manager_t::warpgroup_info_t::wait_group(
    unsigned max_pending_groups, const unsigned *warp_ids, unsigned count) {
  assert(!is_waiting && "Warpgroup is already waiting on a WGMMA group");
  wait_allowance = max_pending_groups;

  wait_result_t result;
  if (wait_satisfied()) return result;

  result.satisfied = false;
  is_waiting = true;
  waiting_warps.assign(warp_ids, warp_ids + count);
  return result;
}

wgmma_async_group_manager_t::wait_result_t
wgmma_async_group_manager_t::warpgroup_info_t::complete_op(unsigned op_uid) {
  std::set<unsigned>::iterator pending_it = pending_ops.find(op_uid);
  if (pending_it != pending_ops.end()) {
    pending_ops.erase(pending_it);
    return release_if_satisfied();
  }

  std::map<unsigned, unsigned>::iterator op_it = op_to_group.find(op_uid);
  if (op_it == op_to_group.end()) return release_if_satisfied();

  unsigned group_id = op_it->second;
  op_to_group.erase(op_it);

  std::map<unsigned, group_t>::iterator group_it =
      pending_groups.find(group_id);
  assert(group_it != pending_groups.end());
  group_it->second.op_uids.erase(op_uid);
  if (group_it->second.op_uids.empty()) pending_groups.erase(group_it);

  return release_if_satisfied();
}

void wgmma_async_group_manager_t::add_op(unsigned cta_id,
                                         unsigned warpgroup_id,
                                         unsigned op_uid) {
  m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].add_op(op_uid);
}

void wgmma_async_group_manager_t::commit_group(unsigned cta_id,
                                               unsigned warpgroup_id) {
  m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].commit_group();
}

wgmma_async_group_manager_t::wait_result_t
wgmma_async_group_manager_t::wait_group(unsigned cta_id,
                                        unsigned warpgroup_id,
                                        unsigned max_pending_groups,
                                        const unsigned *warp_ids,
                                        unsigned count) {
  return m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].wait_group(
      max_pending_groups, warp_ids, count);
}

wgmma_async_group_manager_t::wait_result_t
wgmma_async_group_manager_t::complete_op(unsigned cta_id,
                                         unsigned warpgroup_id,
                                         unsigned op_uid) {
  std::map<key_t, warpgroup_info_t>::iterator it =
      m_warpgroup_info.find(std::make_pair(cta_id, warpgroup_id));
  if (it == m_warpgroup_info.end()) return wait_result_t();
  return it->second.complete_op(op_uid);
}

bool wgmma_async_group_manager_t::is_waiting(unsigned cta_id,
                                             unsigned warpgroup_id) const {
  std::map<key_t, warpgroup_info_t>::const_iterator it =
      m_warpgroup_info.find(std::make_pair(cta_id, warpgroup_id));
  return it != m_warpgroup_info.end() && it->second.is_waiting;
}

unsigned wgmma_async_group_manager_t::pending_group_count(
    unsigned cta_id, unsigned warpgroup_id) const {
  std::map<key_t, warpgroup_info_t>::const_iterator it =
      m_warpgroup_info.find(std::make_pair(cta_id, warpgroup_id));
  if (it == m_warpgroup_info.end()) return 0;
  return it->second.pending_groups.size();
}

void wgmma_async_group_manager_t::cleanup_cta(unsigned cta_id) {
  for (std::map<key_t, warpgroup_info_t>::iterator it =
           m_warpgroup_info.begin();
       it != m_warpgroup_info.end();) {
    if (it->first.first == cta_id) {
      it = m_warpgroup_info.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace flash_gpgpu_sim

void barrier_set_t::add_wgmma_op(unsigned cta_id, unsigned warpgroup_id,
                                 unsigned op_uid) {
  m_wgmma_group_manager.add_op(cta_id, warpgroup_id, op_uid);
}

void barrier_set_t::complete_wgmma_op(unsigned cta_id, unsigned warpgroup_id,
                                      unsigned op_uid) {
  flash_gpgpu_sim::wgmma_async_group_manager_t::wait_result_t result =
      m_wgmma_group_manager.complete_op(cta_id, warpgroup_id, op_uid);

  for (std::vector<unsigned>::const_iterator it = result.released_warps.begin();
       it != result.released_warps.end(); ++it) {
    unsigned warp_id = *it;
    if (m_warp_barrier_type[warp_id] == BARRIER_WAIT_WGMMA_GROUP)
      m_warp_at_barrier.reset(warp_id);
  }
}

void barrier_set_t::wait_wgmma_group(unsigned cta_id, unsigned warpgroup_id,
                                     unsigned max_pending_groups,
                                     const unsigned *warp_ids, unsigned count) {
  flash_gpgpu_sim::wgmma_async_group_manager_t::wait_result_t result =
      m_wgmma_group_manager.wait_group(cta_id, warpgroup_id,
                                       max_pending_groups, warp_ids, count);
  if (result.satisfied) return;

  for (unsigned i = 0; i < count; ++i) {
    m_warp_at_barrier.set(warp_ids[i]);
    m_warp_barrier_type[warp_ids[i]] = BARRIER_WAIT_WGMMA_GROUP;
  }
}

void barrier_set_t::commit_wgmma_group(unsigned cta_id,
                                       unsigned warpgroup_id) {
  m_wgmma_group_manager.commit_group(cta_id, warpgroup_id);
}

void barrier_set_t::cleanup_cta_wgmma_groups(unsigned cta_id) {
  m_wgmma_group_manager.cleanup_cta(cta_id);
}
