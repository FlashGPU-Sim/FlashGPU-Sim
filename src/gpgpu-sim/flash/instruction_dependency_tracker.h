#ifndef FLASH_GPGPU_SIM_INSTRUCTION_DEPENDENCY_TRACKER_H_
#define FLASH_GPGPU_SIM_INSTRUCTION_DEPENDENCY_TRACKER_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "../../abstract_hardware_model.h"
#include "panic.h"

namespace flash_gpgpu_sim {

// Timing-backend state for dependency controls carried by a frontend-neutral
// inst_t. PTX leaves the controls at their defaults; native frontends can
// provide an issue stall and six explicit read/write dependency barriers.
class instruction_dependency_tracker {
public:
  bool ready(const inst_t::dependency_control_t &control, uint64_t cycle,
             op_type consumer_op = NO_OP) {
    return control_delay_ready(cycle) &&
           pending_wait_mask(control, cycle, consumer_op) == 0;
  }

  bool control_delay_ready(uint64_t cycle) const {
    return cycle >= next_issue_cycle_;
  }

  uint8_t pending_wait_mask(const inst_t::dependency_control_t &control,
                            uint64_t cycle, op_type consumer_op = NO_OP) {
    advance(cycle);
    uint8_t pending = 0;
    for (unsigned barrier = 0; barrier < barrier_owners_.size(); ++barrier) {
      if (!control.waits_on(barrier))
        continue;
      for (const auto &owner : barrier_owners_[barrier]) {
        uint8_t roles = owner.second;
        const auto forwarding = sp_forwarding_ready_cycles_.find(owner.first);
        if (consumer_op == SP_OP &&
            forwarding != sp_forwarding_ready_cycles_.end() &&
            cycle >= forwarding->second)
          roles &= ~kWriteOwner;
        if (roles == 0)
          continue;
        pending |= uint8_t{1} << barrier;
        break;
      }
    }
    return pending;
  }

  void issue(const inst_t::dependency_control_t &control, uint64_t token,
             bool executes, uint64_t cycle) {
    advance(cycle);
    next_issue_cycle_ =
        std::max(next_issue_cycle_, cycle + control.stall_cycles);
    if (!executes)
      return;
    if (control.has_write_barrier())
      reserve(control.write_barrier, token, kWriteOwner);
    if (control.has_read_barrier())
      reserve(control.read_barrier, token, kReadOwner);
  }

  // Keep a native variable-latency result's write barrier live until its
  // measured visibility cycle.  This is scheduled immediately after issue,
  // before ordinary pipeline completion can install an earlier release.
  void hold_write_barrier_until(const inst_t::dependency_control_t &control,
                                uint64_t token, uint64_t visible_cycle) {
    if (control.has_write_barrier())
      schedule_release_at(control.write_barrier, token, kWriteOwner,
                          visible_cycle);
  }

  void begin_service(const inst_t::dependency_control_t &control,
                     uint64_t token, op_type producer_op, uint64_t cycle,
                     unsigned sfu_to_sp_forwarding_latency,
                     unsigned variable_read_barrier_latency = 0) {
    advance(cycle);
    // Queued/variable execution paths may keep source operands live after the
    // ordinary collector handshake. Release their WAR barrier at the
    // calibrated service phase instead.
    if (variable_read_barrier_latency != 0 && control.has_read_barrier())
      schedule_release_at(control.read_barrier, token, kReadOwner,
                          cycle + variable_read_barrier_latency);
    if ((producer_op == SFU_OP || producer_op == ALU_SFU_OP) &&
        sfu_to_sp_forwarding_latency != 0 && control.has_write_barrier()) {
      sp_forwarding_ready_cycles_[token] = cycle + sfu_to_sp_forwarding_latency;
    }
  }

  // A native read-dependency barrier protects source registers only until the
  // operand collector has consumed them.  It must not remain live for the
  // producer's full execution latency (write barriers do).
  void operands_read(const inst_t::dependency_control_t &control,
                     uint64_t token, uint64_t cycle,
                     bool defer_read_barrier = false) {
    advance(cycle);
    if (control.has_read_barrier() && !defer_read_barrier)
      schedule_release_at(control.read_barrier, token, kReadOwner, cycle + 1);
  }

  void complete(const inst_t::dependency_control_t &control, uint64_t token,
                uint64_t cycle) {
    advance(cycle);
    if (control.has_write_barrier())
      schedule_release_at(control.write_barrier, token, kWriteOwner, cycle + 1);
    // Completion is a conservative fallback for paths that do not traverse
    // the ordinary operand collector (for example, warpgroup participants).
    if (control.has_read_barrier())
      schedule_release_at(control.read_barrier, token, kReadOwner, cycle + 1);
  }

  void reset() {
    next_issue_cycle_ = 0;
    for (auto &owners : barrier_owners_)
      owners.clear();
    pending_releases_.clear();
    sp_forwarding_ready_cycles_.clear();
  }

  uint64_t next_issue_cycle() const { return next_issue_cycle_; }
  bool barrier_pending(unsigned barrier) const {
    return barrier < barrier_owners_.size() &&
           !barrier_owners_[barrier].empty();
  }

private:
  static constexpr uint8_t kReadOwner = uint8_t{1} << 0;
  static constexpr uint8_t kWriteOwner = uint8_t{1} << 1;

  struct pending_release {
    unsigned barrier;
    uint64_t token;
    uint8_t role;
    uint64_t visible_cycle;
  };

  void reserve(unsigned barrier, uint64_t token, uint8_t role) {
    if (barrier >= barrier_owners_.size())
      flash_gpgpu_sim::panic("dependency barrier is out of range");
    barrier_owners_[barrier][token] |= role;
  }

  void schedule_release_at(unsigned barrier, uint64_t token, uint8_t role,
                           uint64_t visible_cycle) {
    if (barrier >= barrier_owners_.size())
      flash_gpgpu_sim::panic("dependency barrier is out of range");
    const auto owner = barrier_owners_[barrier].find(token);
    if (owner == barrier_owners_[barrier].end() || (owner->second & role) == 0)
      return;
    const auto already_scheduled =
        std::find_if(pending_releases_.begin(), pending_releases_.end(),
                     [=](const pending_release &release) {
                       return release.barrier == barrier &&
                              release.token == token && release.role == role;
                     });
    if (already_scheduled == pending_releases_.end())
      pending_releases_.push_back({barrier, token, role, visible_cycle});
  }

  void advance(uint64_t cycle) {
    auto release = pending_releases_.begin();
    while (release != pending_releases_.end()) {
      if (release->visible_cycle > cycle) {
        ++release;
        continue;
      }
      if (release->barrier >= barrier_owners_.size())
        flash_gpgpu_sim::panic("dependency release barrier is out of range");
      auto &owners = barrier_owners_[release->barrier];
      const auto owner = owners.find(release->token);
      if (owner != owners.end()) {
        owner->second &= ~release->role;
        if (owner->second == 0)
          owners.erase(owner);
      }
      if (release->role == kWriteOwner)
        sp_forwarding_ready_cycles_.erase(release->token);
      release = pending_releases_.erase(release);
    }
  }

  uint64_t next_issue_cycle_ = 0;
  std::array<std::map<uint64_t, uint8_t>, 6> barrier_owners_{};
  std::vector<pending_release> pending_releases_{};
  std::map<uint64_t, uint64_t> sp_forwarding_ready_cycles_{};
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_INSTRUCTION_DEPENDENCY_TRACKER_H_
