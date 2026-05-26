#ifndef __FLASH_GEM_FORGE_GPGPUSIM_GEM5_MEM_SUBSYSTEM_HH__
#define __FLASH_GEM_FORGE_GPGPUSIM_GEM5_MEM_SUBSYSTEM_HH__

// Include from gem5 codebase.
#include "mem/gpgpusim/gpgpusim_requestor.hh"
#include "sim/system.hh"

// Include from GPGPUSim codebase.
#include "../../mem_fetch.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace flash_gpgpu_sim {

class Gem5MemSubsystem {
public:
  using GPGPUSimReqVec = std::vector<gem5::GPGPUSimRequestor *>;

  Gem5MemSubsystem(gem5::System *sys, const GPGPUSimReqVec &requestors,
                   bool gmem_skip_l1d);

  /**
   * Register the interconnect interface for GPGPUSim.
   */
  void registerGPGPUSimInterconnectInterface();

  /**
   * Implement the interface to GPGPUSim.
   */
  using GPGPUSimPortId = unsigned;
  void pushMemFetch(GPGPUSimPortId input_port_id, GPGPUSimPortId output_port_id,
                    mem_fetch *mf);

  // Actually send out the pending mem_fetches to gem5 in one thread.
  void drainPendingMemFetches();

  mem_fetch *popMemFetch(GPGPUSimPortId output_port_id);

private:
  gem5::System *system;

  // Vector of GPGPUSim requestors, one per GPU port.
  GPGPUSimReqVec gpgpusim_requestors;
  bool gmem_skip_l1d;

  gem5::GPGPUSimRequestor *getRequestorForGPUPort(GPGPUSimPortId port_id) const;

  struct GPGPUSimSenderState : public gem5::Packet::SenderState {
    const GPGPUSimPortId input_port_id;
    const GPGPUSimPortId output_port_id;
    mem_fetch *const mf;
    GPGPUSimSenderState(GPGPUSimPortId input_port_id_,
                        GPGPUSimPortId output_port_id_, mem_fetch *mf_)
        : input_port_id(input_port_id_), output_port_id(output_port_id_),
          mf(mf_) {}
  };

  gem5::PacketPtr
  createGem5PacketForMemFetch(GPGPUSimSenderState *sender_state);

  /**
   * GPGPUSim parallelize simulation of SM cores across multiple threads.
   * However, gem5 memory system is not safe, and the event_queue may lose
   * events when multiple threads access it simultaneously. Therefore, we decide
   * to serialize the memory requests here.
   */
  std::mutex push_mem_fetch_mutex;
  std::deque<GPGPUSimSenderState *> pending_mem_fetch_queue;
};

} // namespace flash_gpgpu_sim

#endif
