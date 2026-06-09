#ifndef FLASH_GPGPU_SIM_WGMMA_ASYNC_GROUP_H
#define FLASH_GPGPU_SIM_WGMMA_ASYNC_GROUP_H

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace flash_gpgpu_sim {

class wgmma_async_group_manager_t {
 public:
  struct wait_result_t {
    bool satisfied = true;
    std::vector<unsigned> released_warps;
  };

  void add_op(unsigned cta_id, unsigned warpgroup_id, unsigned op_uid);
  void commit_group(unsigned cta_id, unsigned warpgroup_id);
  wait_result_t wait_group(unsigned cta_id, unsigned warpgroup_id,
                           unsigned max_pending_groups,
                           const unsigned *warp_ids, unsigned count);
  wait_result_t complete_op(unsigned cta_id, unsigned warpgroup_id,
                            unsigned op_uid);
  bool is_waiting(unsigned cta_id, unsigned warpgroup_id) const;
  unsigned pending_group_count(unsigned cta_id, unsigned warpgroup_id) const;
  void cleanup_cta(unsigned cta_id);

 private:
  struct group_t {
    unsigned group_id = 0;
    std::set<unsigned> op_uids;
  };

  struct warpgroup_info_t {
    unsigned next_group_id = 1;
    std::set<unsigned> pending_ops;
    std::map<unsigned, unsigned> op_to_group;
    std::map<unsigned, group_t> pending_groups;
    bool is_waiting = false;
    unsigned wait_allowance = 0;
    std::vector<unsigned> waiting_warps;

    void add_op(unsigned op_uid);
    void commit_group();
    wait_result_t wait_group(unsigned max_pending_groups,
                             const unsigned *warp_ids, unsigned count);
    wait_result_t complete_op(unsigned op_uid);
    bool wait_satisfied() const;
    wait_result_t release_if_satisfied();
  };

  typedef std::pair<unsigned, unsigned> key_t;
  std::map<key_t, warpgroup_info_t> m_warpgroup_info;
};

}  // namespace flash_gpgpu_sim

#endif  // FLASH_GPGPU_SIM_WGMMA_ASYNC_GROUP_H
