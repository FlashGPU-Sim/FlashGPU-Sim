#include "stream_buffer.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace flash_gpgpu_sim {

instruction_stream_buffer::instruction_stream_buffer(
    const instruction_stream_buffer_config &config)
    : m_config(config), m_streams(config.streams) {
  assert(m_config.line_size != 0);
  assert((m_config.line_size & (m_config.line_size - 1)) == 0);
  assert(m_config.streams != 0);
  assert(m_config.depth != 0);
  assert(m_config.issue_width != 0);
}

uint64_t instruction_stream_buffer::line_address(uint64_t address) const {
  return address & ~(m_config.line_size - 1);
}

instruction_stream_buffer::entry *
instruction_stream_buffer::find_entry(stream_state &stream, uint64_t address) {
  for (entry &candidate : stream.entries) {
    if (candidate.address == address)
      return &candidate;
  }
  return nullptr;
}

const instruction_stream_buffer::entry *
instruction_stream_buffer::find_entry(const stream_state &stream,
                                      uint64_t address) const {
  for (const entry &candidate : stream.entries) {
    if (candidate.address == address)
      return &candidate;
  }
  return nullptr;
}

void instruction_stream_buffer::invalidate(stream_state &stream) {
  m_stats.canceled_entries += stream.entries.size();
  stream.entries.clear();
  stream.valid = false;
  stream.context = 0;
  stream.demand_line = 0;
  stream.last_use = 0;
  ++stream.generation;
}

void instruction_stream_buffer::refill_window(stream_state &stream) {
  stream.entries.erase(
      std::remove_if(stream.entries.begin(), stream.entries.end(),
                     [&stream](const entry &candidate) {
                       return candidate.address <= stream.demand_line;
                     }),
      stream.entries.end());

  for (unsigned distance = 1; distance <= m_config.depth; ++distance) {
    if (distance > (std::numeric_limits<uint64_t>::max() - stream.demand_line) /
                       m_config.line_size) {
      break;
    }
    const uint64_t address = stream.demand_line + distance * m_config.line_size;
    if (!find_entry(stream, address)) {
      stream.entries.push_back({address, instruction_prefetch_state::kQueued});
    }
  }
  std::sort(stream.entries.begin(), stream.entries.end(),
            [](const entry &left, const entry &right) {
              return left.address < right.address;
            });
  assert(stream.entries.size() <= m_config.depth);
}

void instruction_stream_buffer::start_stream(uint64_t context,
                                             uint64_t demand_line) {
  stream_state *target = nullptr;
  for (stream_state &stream : m_streams) {
    if (!stream.valid) {
      target = &stream;
      break;
    }
  }
  if (!target) {
    target = &*std::min_element(
        m_streams.begin(), m_streams.end(),
        [](const stream_state &left, const stream_state &right) {
          return left.last_use < right.last_use;
        });
    ++m_stats.streams_replaced;
    invalidate(*target);
  }

  target->valid = true;
  target->context = context;
  target->demand_line = demand_line;
  target->last_use = ++m_clock;
  ++target->generation;
  refill_window(*target);
  ++m_stats.streams_started;
}

instruction_demand_state
instruction_stream_buffer::observe_demand(uint64_t context, uint64_t address,
                                          bool demand_miss) {
  const uint64_t line = line_address(address);
  for (stream_state &stream : m_streams) {
    if (!stream.valid || stream.context != context)
      continue;
    entry *candidate = find_entry(stream, line);
    if (!candidate)
      continue;

    instruction_demand_state result = instruction_demand_state::kUntracked;
    switch (candidate->state) {
    case instruction_prefetch_state::kQueued:
      result = instruction_demand_state::kQueued;
      ++m_stats.late;
      break;
    case instruction_prefetch_state::kInflight:
      result = instruction_demand_state::kInflight;
      ++m_stats.late;
      break;
    case instruction_prefetch_state::kReady:
      result = instruction_demand_state::kReady;
      ++m_stats.useful;
      break;
    }
    stream.demand_line = line;
    stream.last_use = ++m_clock;
    refill_window(stream);
    return result;
  }

  if (demand_miss)
    start_stream(context, line);
  return instruction_demand_state::kUntracked;
}

std::vector<instruction_prefetch_request> instruction_stream_buffer::issue() {
  std::vector<instruction_prefetch_request> requests;
  requests.reserve(m_config.issue_width);
  unsigned visited_without_issue = 0;
  while (requests.size() < m_config.issue_width &&
         visited_without_issue < m_streams.size()) {
    const unsigned stream_index = m_issue_cursor;
    m_issue_cursor = (m_issue_cursor + 1) % m_streams.size();
    stream_state &stream = m_streams[stream_index];
    entry *candidate = nullptr;
    if (stream.valid) {
      for (entry &pending : stream.entries) {
        if (pending.state == instruction_prefetch_state::kQueued) {
          candidate = &pending;
          break;
        }
      }
    }
    if (!candidate) {
      ++visited_without_issue;
      continue;
    }
    visited_without_issue = 0;
    candidate->state = instruction_prefetch_state::kInflight;
    requests.push_back(
        {candidate->address, stream.context, stream.generation, stream_index});
    ++m_stats.requests_issued;
  }
  return requests;
}

instruction_stream_buffer::stream_state *
instruction_stream_buffer::request_stream(
    const instruction_prefetch_request &request) {
  if (request.stream >= m_streams.size())
    return nullptr;
  stream_state &stream = m_streams[request.stream];
  if (!stream.valid || stream.context != request.context ||
      stream.generation != request.generation) {
    return nullptr;
  }
  return &stream;
}

const instruction_stream_buffer::stream_state *
instruction_stream_buffer::request_stream(
    const instruction_prefetch_request &request) const {
  if (request.stream >= m_streams.size())
    return nullptr;
  const stream_state &stream = m_streams[request.stream];
  if (!stream.valid || stream.context != request.context ||
      stream.generation != request.generation) {
    return nullptr;
  }
  return &stream;
}

bool instruction_stream_buffer::current(
    const instruction_prefetch_request &request) const {
  const stream_state *stream = request_stream(request);
  return stream && find_entry(*stream, request.address);
}

void instruction_stream_buffer::retry(
    const instruction_prefetch_request &request) {
  stream_state *stream = request_stream(request);
  entry *candidate = stream ? find_entry(*stream, request.address) : nullptr;
  if (!candidate || candidate->state != instruction_prefetch_state::kInflight) {
    ++m_stats.stale_fills;
    return;
  }
  candidate->state = instruction_prefetch_state::kQueued;
  ++m_stats.retries;
}

void instruction_stream_buffer::mark_resident(
    const instruction_prefetch_request &request) {
  stream_state *stream = request_stream(request);
  entry *candidate = stream ? find_entry(*stream, request.address) : nullptr;
  if (!candidate || candidate->state != instruction_prefetch_state::kInflight) {
    ++m_stats.stale_fills;
    return;
  }
  candidate->state = instruction_prefetch_state::kReady;
  ++m_stats.resident;
}

void instruction_stream_buffer::fill(
    const instruction_prefetch_request &request) {
  stream_state *stream = request_stream(request);
  entry *candidate = stream ? find_entry(*stream, request.address) : nullptr;
  if (!candidate || candidate->state != instruction_prefetch_state::kInflight) {
    ++m_stats.stale_fills;
    return;
  }
  candidate->state = instruction_prefetch_state::kReady;
  ++m_stats.resident;
}

void instruction_stream_buffer::cancel_context(uint64_t context) {
  for (stream_state &stream : m_streams) {
    if (stream.valid && stream.context == context)
      invalidate(stream);
  }
}

void instruction_stream_buffer::reset() {
  for (stream_state &stream : m_streams)
    invalidate(stream);
  m_issue_cursor = 0;
}

size_t instruction_stream_buffer::pending_entries() const {
  size_t total = 0;
  for (const stream_state &stream : m_streams)
    total += stream.entries.size();
  return total;
}

size_t instruction_stream_buffer::pending_entries(unsigned stream) const {
  assert(stream < m_streams.size());
  return m_streams[stream].entries.size();
}

} // namespace flash_gpgpu_sim
