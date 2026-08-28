#include "prefetcher.h"

#include <limits>
#include <list>

#include "../../../abstract_hardware_model.h"
#include "../../gpu-sim.h"
#include "../../mem_fetch.h"
#include "../../shader.h"
#include "instruction_cache.h"

namespace flash_gpgpu_sim {

namespace {

instruction_stream_buffer_config make_config(unsigned streams, unsigned depth,
                                             unsigned issue_width,
                                             unsigned gcc_preload_lines,
                                             unsigned line_size) {
  instruction_stream_buffer_config config;
  config.line_size = line_size;
  config.streams = streams;
  config.depth = depth;
  config.issue_width = issue_width;
  config.gcc_preload_lines = gcc_preload_lines;
  return config;
}

} // namespace

instruction_prefetcher::instruction_prefetcher(
    bool enabled, unsigned streams, unsigned depth, unsigned issue_width,
    unsigned gcc_preload_lines, unsigned gcc_hit_latency, unsigned line_size,
    unsigned sid, unsigned tpc,
    const memory_config *memory_config, gpgpu_sim *gpu,
    instruction_cache *cache)
    : m_enabled(enabled && streams != 0 && depth != 0 && issue_width != 0),
      m_line_size(line_size), m_gcc_preload_lines(gcc_preload_lines),
      m_gcc_hit_latency(gcc_hit_latency), m_sid(sid), m_tpc(tpc),
      m_memory_config(memory_config), m_gpu(gpu), m_cache(cache),
      m_stream_buffer(make_config(streams ? streams : 1, depth ? depth : 1,
                                  issue_width ? issue_width : 1,
                                  gcc_preload_lines, line_size)),
      m_active_context(std::numeric_limits<uint64_t>::max()) {}

void instruction_prefetcher::activate_context(uint64_t context,
                                              uint64_t memory_stream) {
  if (m_active_context != context) {
    m_stream_buffer.reset();
    m_context_streams.clear();
    m_active_context = context;
  }
  m_context_streams[context] = memory_stream;
}

void instruction_prefetcher::observe_demand(uint64_t context,
                                            uint64_t memory_stream,
                                            uint64_t address,
                                            bool demand_miss) {
  if (!m_enabled)
    return;
  activate_context(context, memory_stream);
  m_stream_buffer.observe_demand(context, address, demand_miss);
}

void instruction_prefetcher::cycle(bool demand_reservation_failed) {
  if (!m_enabled || demand_reservation_failed)
    return;
  const uint64_t now = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  const std::vector<instruction_prefetch_request> requests =
      m_stream_buffer.issue();
  for (const instruction_prefetch_request &request : requests) {
    const auto stream = m_context_streams.find(request.context);
    if (stream == m_context_streams.end()) {
      m_stream_buffer.retry(request);
      continue;
    }
    mem_access_t access(INST_ACC_R, request.address, m_line_size, false,
                        m_gpu->gpgpu_ctx);
    mem_fetch *mf =
        new mem_fetch(access, nullptr, stream->second, READ_PACKET_SIZE, 0,
                      m_sid, m_tpc, m_memory_config, now);
    mf->set_instruction_prefetch(request.stream, request.generation,
                                 request.context);

    std::list<cache_event> events;
    const cache_request_status status =
        m_cache->access(request.address, mf, now, events);
    if (status == MISS) {
      if (m_stream_buffer.gcc_preload_contains(request) &&
          m_cache->schedule_preload_fill(mf, now + m_gcc_hit_latency)) {
        m_stream_buffer.record_gcc_preload_hit();
      } else if (m_gcc_preload_lines != 0) {
        m_stream_buffer.record_gcc_preload_miss();
      }
      continue;
    }
    if (status == HIT) {
      m_stream_buffer.mark_resident(request);
    } else {
      m_stream_buffer.retry(request);
    }
    delete mf;
  }
}

instruction_prefetch_request
instruction_prefetcher::request_from(const mem_fetch *mf) const {
  instruction_prefetch_request request;
  request.address = mf->get_addr();
  request.context = mf->get_instruction_prefetch_context();
  request.generation = mf->get_instruction_prefetch_generation();
  request.stream = mf->get_instruction_prefetch_stream();
  return request;
}

void instruction_prefetcher::fill(const mem_fetch *mf) {
  if (!m_enabled || !mf->is_instruction_prefetch())
    return;
  m_stream_buffer.fill(request_from(mf));
}

void instruction_prefetcher::reset() {
  m_stream_buffer.reset();
  m_context_streams.clear();
  m_active_context = std::numeric_limits<uint64_t>::max();
}

} // namespace flash_gpgpu_sim
