#include "gem5_mem_subsystem.hh"

#include "debug/GPGPUSim.hh"
#include "sim/cur_tick.hh"

#include "../../icnt_wrapper.h"

#include <cstdlib>
#include <cstring>

namespace flash_gpgpu_sim {

namespace {

bool isTmaAccess(mem_access_type type) {
  return type == TMA_ACC_R || type == TMA_ACC_W;
}

bool isCpAsyncAccess(mem_access_type type) { return type == CP_ASYNC_ACC_R; }

bool isGlobalAccess(mem_access_type type) {
  return type == GLOBAL_ACC_R || type == GLOBAL_ACC_W || isTmaAccess(type) ||
         isCpAsyncAccess(type);
}

bool shouldSetGLC(mem_fetch *mf, bool gmem_skip_l1d) {
  const auto access_type = mf->get_access_type();
  if (access_type == GLOBAL_ACC_W) {
    return true;
  }
  if (isTmaAccess(access_type) || isCpAsyncAccess(access_type)) {
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

bool shouldUseWriteThroughBypass(mem_fetch *mf) {
  return mf->get_access_type() == GLOBAL_ACC_W;
}

void gem5_mf_trace_emit(gem5::Tick tick, const char *event, mem_fetch *mf,
                        unsigned input_port, unsigned output_port,
                        size_t pending_queue) {
  const char *path = std::getenv("FLASHGPU_GEM5_MF_TRACE_CSV");
  if (path == nullptr || path[0] == '\0')
    return;

  const char *limit = std::getenv("FLASHGPU_GEM5_MF_TRACE_TICK_LIMIT");
  if (limit != nullptr && limit[0] != '\0') {
    unsigned long long limit_tick = std::strtoull(limit, nullptr, 0);
    if (limit_tick != 0 && tick > limit_tick)
      return;
  }

  static std::mutex trace_mutex;
  static FILE *trace_file = nullptr;
  static bool header_written = false;

  std::lock_guard<std::mutex> lock(trace_mutex);
  if (trace_file == nullptr) {
    trace_file = std::strcmp(path, "-") == 0 ? stdout : std::fopen(path, "a");
    if (trace_file == nullptr)
      return;
  }
  if (!header_written) {
    std::fprintf(trace_file,
                 "tick,event,mf_uid,access_type,addr,access_size,data_size,"
                 "is_write,input_port,output_port,pending_queue\n");
    header_written = true;
  }

  std::fprintf(trace_file, "%llu,%s,%u,%d,0x%llx,%u,%u,%u,%u,%u,%zu\n",
               static_cast<unsigned long long>(tick), event,
               mf->get_request_uid(), static_cast<int>(mf->get_access_type()),
               static_cast<unsigned long long>(mf->get_addr()),
               mf->get_access_size(), mf->get_data_size(),
               mf->get_is_write() ? 1 : 0, input_port, output_port,
               pending_queue);
  std::fflush(trace_file);
}

} // namespace

Gem5MemSubsystem::Gem5MemSubsystem(gem5::System *sys,
                                   const GPGPUSimReqVec &requestors,
                                   bool gmem_skip_l1d_)
    : system(sys), gpgpusim_requestors(requestors),
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
    gem5_mf_trace_emit(gem5::curTick(), "PUSH", mf, input_port_id,
                       output_port_id, pending_mem_fetch_queue.size());
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
    gem5_mf_trace_emit(gem5::curTick(), "DRAIN_SEND", sender_state->mf,
                       sender_state->input_port_id,
                       sender_state->output_port_id,
                       pending_mem_fetch_queue.size());
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
  if (shouldUseWriteThroughBypass(mf)) {
    req->setFlags(gem5::Request::NO_RUBY_SEQUENCER_COALESCE);
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
  gem5_mf_trace_emit(gem5::curTick(), "POP", mf, sender_state->input_port_id,
                     sender_state->output_port_id, 0);

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
