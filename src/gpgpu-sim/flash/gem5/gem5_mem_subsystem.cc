#include "gem5_mem_subsystem.hh"

#include "debug/GPGPUSim.hh"

#include "../../icnt_wrapper.h"

namespace flash_gpgpu_sim {

namespace {

bool isTmaAccess(mem_access_type type) {
  return type == TMA_ACC_R || type == TMA_ACC_W;
}

bool isGlobalAccess(mem_access_type type) {
  return type == GLOBAL_ACC_R || type == GLOBAL_ACC_W || isTmaAccess(type);
}

bool shouldSetGLC(mem_fetch *mf, bool gmem_skip_l1d) {
  const auto access_type = mf->get_access_type();
  if (isTmaAccess(access_type)) {
    return true;
  }

  const auto &inst = mf->get_inst();
  const auto cache_op = inst.cache_op;
  if (cache_op == CACHE_GLOBAL || cache_op == CACHE_STREAMING ||
      cache_op == CACHE_LAST_USE) {
    return true;
  }

  return gmem_skip_l1d && isGlobalAccess(access_type) && cache_op != CACHE_L1;
}

} // namespace

Gem5MemSubsystem::Gem5MemSubsystem(gem5::System *sys,
                                   const GPGPUSimReqVec &requestors,
                                   bool gmem_skip_l1d_)
    : system(sys),
      gpgpusim_requestors(requestors),
      gmem_skip_l1d(gmem_skip_l1d_) {}

void Gem5MemSubsystem::registerGPGPUSimInterconnectInterface() {
  auto original_icnt_push = icnt_push;
  icnt_push = [this, original_icnt_push](unsigned input, unsigned output,
                                         void *data, unsigned int size) {
    this->pushMemFetch(input, output, static_cast<mem_fetch *>(data));
    // Keep the original behavior as well?
    // if (original_icnt_push) {
    //   original_icnt_push(input, output, data, size);
    // }
  };

  icnt_pop = [this](unsigned output) -> void * {
    return this->popMemFetch(output);
  };
}

void Gem5MemSubsystem::pushMemFetch(GPGPUSimPortId input_port_id,
                                    GPGPUSimPortId output_port_id,
                                    mem_fetch *mf) {

  /**
   * ! Be careful this may be from multiple shader threads !
   */
  gem5::curEventQueue(gem5::getEventQueue(0));

  /**
   * So far with gem5 integration, we expect only SM requests here.
   * If this is from memory, we panic.
   */
  if (input_port_id >= g_icnt_n_shader) {
    panic("Received request from port %u which is not a shader core "
          "(n_shader=%u). Memory-to-shader requests are not supported in "
          "gem5 integration.\n",
          input_port_id, g_icnt_n_shader);
  }

  DPRINTF(GPGPUSim, "Pushing mem_fetch from GPGPUSim to gem5 MemSubsystem\n");
  if (::gem5::debug::GPGPUSim) {
    mf->print(stdout);
  }

  auto sender_state =
      new GPGPUSimSenderState(input_port_id, output_port_id, mf);
  {
    std::lock_guard<std::mutex> lock(push_mem_fetch_mutex);
    pending_mem_fetch_queue.push_back(sender_state);
  }
}

void Gem5MemSubsystem::drainPendingMemFetches() {
  std::lock_guard<std::mutex> lock(push_mem_fetch_mutex);
  while (!pending_mem_fetch_queue.empty()) {
    auto sender_state = pending_mem_fetch_queue.front();
    pending_mem_fetch_queue.pop_front();

    auto pkt = createGem5PacketForMemFetch(sender_state);

    auto requestor = getRequestorForGPUPort(sender_state->input_port_id);
    requestor->sendTimingReq(pkt);
  }
}

gem5::PacketPtr Gem5MemSubsystem::createGem5PacketForMemFetch(
    GPGPUSimSenderState *sender_state) {

  // Extract information from mem_fetch
  auto mf = sender_state->mf;
  auto addr = mf->get_addr();
  auto size = mf->get_access_size();
  auto is_write = mf->get_is_write();

  // Requestor ID
  auto requestor = getRequestorForGPUPort(sender_state->input_port_id);
  auto requestor_id = requestor->requestorId;

  DPRINTF(GPGPUSim,
          "Creating packet: addr=0x%x, size=%d, write=%d, requestor=%d\n", addr,
          size, is_write, requestor_id);

  auto req = std::make_shared<gem5::Request>(
      addr, size, gem5::Request::PHYSICAL, requestor_id);
  if (shouldSetGLC(mf, gmem_skip_l1d)) {
    req->setCacheCoherenceFlags(gem5::Request::GLC_BIT);
  }

  auto cmd = is_write ? gem5::MemCmd::WriteReq : gem5::MemCmd::ReadReq;

  auto pkt = new gem5::Packet(req, cmd);

  pkt->allocate();

  pkt->pushSenderState(sender_state);

  return pkt;
}

gem5::GPGPUSimRequestor *
Gem5MemSubsystem::getRequestorForGPUPort(GPGPUSimPortId port_id) const {

  if (port_id >= gpgpusim_requestors.size()) {
    panic("Invalid GPGPUSim port ID %u requested; only %u requestors "
          "available.\n",
          port_id, gpgpusim_requestors.size());
  }
  return gpgpusim_requestors[port_id];
}

mem_fetch *Gem5MemSubsystem::popMemFetch(GPGPUSimPortId output_port_id) {

  /**
   * ! Be careful this may be from multiple shader threads !
   */
  gem5::curEventQueue(gem5::getEventQueue(0));

  /**
   * Similarly, we need to ignore all pop request from memory ports.
   * But no need to panic here, just return nullptr.
   */
  if (output_port_id >= g_icnt_n_shader) {
    return nullptr;
  }

  auto requestor = getRequestorForGPUPort(output_port_id);
  auto pkt = requestor->popRespQueue();
  if (pkt == nullptr) {
    return nullptr;
  }

  DPRINTF(GPGPUSim, "Popping mem_fetch from gem5 MemSubsystem to GPGPUSim\n");

  // Extract the SenderState.
  auto sender_state =
      dynamic_cast<GPGPUSimSenderState *>(pkt->popSenderState());
  assert(sender_state != nullptr);

  // Here input port == output port since we are popping the response.
  assert(sender_state->input_port_id == output_port_id);

  auto mf = sender_state->mf;

  // Update the mem_fetch type to reply before returning to GPGPUSim.
  // This converts READ_REQUEST -> READ_REPLY or WRITE_REQUEST -> WRITE_ACK.
  if (mf->get_access_type() != L1_WRBK_ACC &&
      mf->get_access_type() != L2_WRBK_ACC) {
    mf->set_reply();
  }

  // Clean up the packet.
  delete pkt;
  delete sender_state;

  return mf;
}

} // namespace flash_gpgpu_sim
