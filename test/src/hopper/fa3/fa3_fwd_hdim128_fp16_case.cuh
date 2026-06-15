#pragma once

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "flash.h"
#include "flash_bwd_launch_template.h"
#include "flash_fwd_launch_template.h"

namespace fa3_hopper_test {

struct Fa3PrefillCase {
  const char *name;
  int batch;
  int seqlen;
  int heads;
  int head_dim;
  bool causal;
};

#define FA3_PREFILL_CASE_LIST(X)                    \
  X(H32D64FullB64S512, 64, 512, 32, 64, false)      \
  X(H32D64FullB32S1024, 32, 1024, 32, 64, false)    \
  X(H32D64FullB16S2048, 16, 2048, 32, 64, false)    \
  X(H32D64FullB8S4096, 8, 4096, 32, 64, false)      \
  X(H32D64FullB4S8192, 4, 8192, 32, 64, false)      \
  X(H32D64CausalB64S512, 64, 512, 32, 64, true)     \
  X(H32D64CausalB32S1024, 32, 1024, 32, 64, true)   \
  X(H32D64CausalB16S2048, 16, 2048, 32, 64, true)   \
  X(H32D64CausalB8S4096, 8, 4096, 32, 64, true)     \
  X(H32D64CausalB4S8192, 4, 8192, 32, 64, true)     \
  X(H16D128FullB64S512, 64, 512, 16, 128, false)    \
  X(H16D128FullB32S1024, 32, 1024, 16, 128, false)  \
  X(H16D128FullB16S2048, 16, 2048, 16, 128, false)  \
  X(H16D128FullB8S4096, 8, 4096, 16, 128, false)    \
  X(H16D128FullB4S8192, 4, 8192, 16, 128, false)    \
  X(H16D128CausalB64S512, 64, 512, 16, 128, true)   \
  X(H16D128CausalB32S1024, 32, 1024, 16, 128, true) \
  X(H16D128CausalB16S2048, 16, 2048, 16, 128, true) \
  X(H16D128CausalB8S4096, 8, 4096, 16, 128, true)   \
  X(H16D128CausalB4S8192, 4, 8192, 16, 128, true)

static constexpr int kFa3PrefillCaseCount = 20;

#define FA3_PREFILL_SMOKE_CASE_LIST(X)              \
  X(H32D64FullB2S128, 2, 128, 32, 64, false)        \
  X(H32D64CausalB2S128, 2, 128, 32, 64, true)       \
  X(H16D128FullB2S128, 2, 128, 16, 128, false)      \
  X(H16D128CausalB2S128, 2, 128, 16, 128, true)

static constexpr int kFa3PrefillSmokeCaseCount = 4;

#define FA3_PREFILL_SMALL_CASE_LIST(X)              \
  X(H32D64FullB32S256, 32, 256, 32, 64, false)      \
  X(H32D64CausalB32S256, 32, 256, 32, 64, true)     \
  X(H16D128FullB32S256, 32, 256, 16, 128, false)    \
  X(H16D128CausalB32S256, 32, 256, 16, 128, true)

static constexpr int kFa3PrefillSmallCaseCount = 4;

#define FA3_PREFILL_MEDIUM_CASE_LIST(X)             \
  X(H32D64FullB16S512, 16, 512, 32, 64, false)      \
  X(H32D64CausalB16S512, 16, 512, 32, 64, true)     \
  X(H16D128FullB16S512, 16, 512, 16, 128, false)    \
  X(H16D128CausalB16S512, 16, 512, 16, 128, true)

static constexpr int kFa3PrefillMediumCaseCount = 4;

#define FA3_PREFILL_CASE_ENTRY(name, batch, seqlen, heads, head_dim, causal) \
  Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal},
static constexpr Fa3PrefillCase kFa3PrefillCases[] = {
    FA3_PREFILL_CASE_LIST(FA3_PREFILL_CASE_ENTRY)};
static constexpr Fa3PrefillCase kFa3PrefillSmokeCases[] = {
    FA3_PREFILL_SMOKE_CASE_LIST(FA3_PREFILL_CASE_ENTRY)};
static constexpr Fa3PrefillCase kFa3PrefillSmallCases[] = {
    FA3_PREFILL_SMALL_CASE_LIST(FA3_PREFILL_CASE_ENTRY)};
static constexpr Fa3PrefillCase kFa3PrefillMediumCases[] = {
    FA3_PREFILL_MEDIUM_CASE_LIST(FA3_PREFILL_CASE_ENTRY)};
#undef FA3_PREFILL_CASE_ENTRY

static_assert(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]) ==
                  kFa3PrefillCaseCount,
              "FA3 prefill case list must contain 20 cases");
static_assert(sizeof(kFa3PrefillSmokeCases) /
                      sizeof(kFa3PrefillSmokeCases[0]) ==
                  kFa3PrefillSmokeCaseCount,
              "FA3 prefill smoke case list must contain 4 cases");
static_assert(sizeof(kFa3PrefillSmallCases) /
                      sizeof(kFa3PrefillSmallCases[0]) ==
                  kFa3PrefillSmallCaseCount,
              "FA3 prefill small case list must contain 4 cases");
static_assert(sizeof(kFa3PrefillMediumCases) /
                      sizeof(kFa3PrefillMediumCases[0]) ==
                  kFa3PrefillMediumCaseCount,
              "FA3 prefill medium case list must contain 4 cases");

inline bool is_supported_fa3_prefill_case(const Fa3PrefillCase &cfg) {
  return cfg.batch > 0 && cfg.seqlen > 0 && cfg.heads > 0 &&
         (cfg.head_dim == 64 || cfg.head_dim == 128);
}

struct Fa3RunResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  float output0 = 0.0f;
  float lse0 = 0.0f;
};

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)
struct Fa3SingleTileProfileResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  int seqlen_q = 128;
  int seqlen_k = 0;
  int heads = 1;
  int head_dim = 128;
  int block_m = 128;
  int block_n = 176;
  int k_tiles = 0;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  uint64_t clock_delta = 0;
  uint64_t mainloop_start = 0;
  uint64_t mainloop_end = 0;
  uint64_t mainloop_delta = 0;
  uint64_t epilogue_start = 0;
  uint64_t epilogue_end = 0;
  uint64_t epilogue_delta = 0;
  uint64_t qk_wait_cycles = 0;
  uint64_t qk_wgmma_issue_cycles = 0;
  uint64_t softmax_cycles = 0;
  uint64_t pv_wait_cycles = 0;
  uint64_t pv_wgmma_issue_wait_cycles = 0;
  uint64_t mainloop_iterations = 0;
  float output0 = 0.0f;
  float lse0 = 0.0f;
};

struct Fa3PrefillProfileTraceStep {
  int step = 0;
  int kind = kFlashFwdProfileTraceUnused;
  int n_block = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t qk_wait_cycles = 0;
  uint64_t qk_wgmma_issue_cycles = 0;
  uint64_t softmax_cycles = 0;
  uint64_t pv_wait_cycles = 0;
  uint64_t pv_wgmma_issue_wait_cycles = 0;
};

struct Fa3PrefillProfileTimelineEvent {
  int actor = 0;
  int step = 0;
  int n_block = 0;
  int event = 0;
  uint64_t clock = 0;
};

struct Fa3PrefillProfileRegTimelineEvent {
  int block_slot = 0;
  int block_x = 0;
  int block_y = 0;
  int block_z = 0;
  int smid = -1;
  int slot = 0;
  int step = 0;
  int n_block = 0;
  int local_thread = 0;
  int local_wg = 0;
  int warp_in_wg = 0;
  int lane = 0;
  int event = 0;
  uint64_t clock = 0;
};

struct Fa3PrefillProfileResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  std::string name;
  int batch = 0;
  int seqlen_q = 0;
  int seqlen_k = 0;
  int heads = 0;
  int head_dim = 0;
  int causal = 0;
  int block_m = 128;
  int block_n = 176;
  int m_tiles = 0;
  int k_tiles = 0;
  int logical_tiles = 0;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  uint64_t clock_delta = 0;
  uint64_t mainloop_start = 0;
  uint64_t mainloop_end = 0;
  uint64_t mainloop_delta = 0;
  uint64_t epilogue_start = 0;
  uint64_t epilogue_end = 0;
  uint64_t epilogue_delta = 0;
  uint64_t qk_wait_cycles = 0;
  uint64_t qk_wgmma_issue_cycles = 0;
  uint64_t softmax_cycles = 0;
  uint64_t pv_wait_cycles = 0;
  uint64_t pv_wgmma_issue_wait_cycles = 0;
  uint64_t mainloop_iterations = 0;
  std::vector<Fa3PrefillProfileTraceStep> trace_steps;
  std::vector<Fa3PrefillProfileTimelineEvent> timeline_events;
  std::vector<Fa3PrefillProfileRegTimelineEvent> reg_timeline_events;
  float output0 = 0.0f;
  float lse0 = 0.0f;
};

inline int fa3_profile_trace_offset(int step,
                                    FlashFwdProfileTraceFieldIndex field) {
  return kFlashFwdProfileTraceBase +
         step * kFlashFwdProfileTraceFieldCount + int(field);
}

inline std::vector<uint64_t> make_fa3_profile_clock_init() {
  std::vector<uint64_t> profile_clock(kFlashFwdProfileClockCount, 0);
  constexpr uint64_t kMaxClock = std::numeric_limits<uint64_t>::max();
  profile_clock[kFlashFwdProfileKernelStart] = kMaxClock;
  profile_clock[kFlashFwdProfileMainloopStart] = kMaxClock;
  profile_clock[kFlashFwdProfileEpilogueStart] = kMaxClock;
  return profile_clock;
}

inline uint64_t fa3_profile_clock_delta(uint64_t start, uint64_t end) {
  return start != std::numeric_limits<uint64_t>::max() && end >= start
             ? end - start
             : 0;
}

inline bool fa3_prefill_profile_clock_enabled() {
  const char *env = std::getenv("FA3_PREFILL_PROFILE_DISABLE_CLOCK");
  return env == nullptr || std::string(env).empty() || std::string(env) == "0";
}

inline std::vector<Fa3PrefillProfileTraceStep> extract_fa3_profile_trace(
    const std::vector<uint64_t> &profile_clock) {
  std::vector<Fa3PrefillProfileTraceStep> trace;
  for (int step = 0; step < kFlashFwdProfileTraceMaxSteps; ++step) {
    const int base = fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceKind);
    if (base + kFlashFwdProfileTraceFieldCount >
        static_cast<int>(profile_clock.size())) {
      break;
    }
    const auto kind = static_cast<int>(
        profile_clock[fa3_profile_trace_offset(
            step, kFlashFwdProfileTraceKind)]);
    const uint64_t start = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceStart)];
    const uint64_t end = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceEnd)];
    if (kind == kFlashFwdProfileTraceUnused && start == 0 && end == 0) {
      break;
    }
    Fa3PrefillProfileTraceStep row;
    row.step = step;
    row.kind = kind;
    const uint64_t raw_n_block = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceNBlock)];
    row.n_block =
        raw_n_block <= static_cast<uint64_t>(std::numeric_limits<int>::max())
            ? static_cast<int>(raw_n_block)
            : -static_cast<int>(~raw_n_block + 1);
    row.start = start;
    row.end = end;
    row.qk_wait_cycles = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceQkWaitCycles)];
    row.qk_wgmma_issue_cycles = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceQkWgmmaIssueCycles)];
    row.softmax_cycles = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTraceSoftmaxCycles)];
    row.pv_wait_cycles = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTracePvWaitCycles)];
    row.pv_wgmma_issue_wait_cycles = profile_clock[fa3_profile_trace_offset(
        step, kFlashFwdProfileTracePvWgmmaIssueWaitCycles)];
    trace.push_back(row);
  }
  return trace;
}

inline int fa3_profile_timeline_offset(int actor, int step, int event) {
  return kFlashFwdProfileTimelineBase +
         flash_fwd_profile_timeline_offset(actor, step, event);
}

inline std::vector<Fa3PrefillProfileTimelineEvent>
extract_fa3_profile_timeline(const std::vector<uint64_t> &profile_clock,
                             int k_tiles) {
  std::vector<Fa3PrefillProfileTimelineEvent> timeline;
  for (int actor = 0; actor < kFlashFwdProfileTimelineActorCount; ++actor) {
    for (int step = 0; step < kFlashFwdProfileTimelineMaxSteps; ++step) {
      for (int event = 0; event < kFlashFwdProfileTimelineEventCount;
           ++event) {
        const int offset = fa3_profile_timeline_offset(actor, step, event);
        if (offset >= static_cast<int>(profile_clock.size())) return timeline;
        const uint64_t clock = profile_clock[offset];
        if (clock == 0) continue;
        Fa3PrefillProfileTimelineEvent row;
        row.actor = actor;
        row.step = step;
        row.event = event;
        row.clock = clock;
        if (event == kFlashFwdProfileTimelineProducerTailBegin ||
            event == kFlashFwdProfileTimelineProducerTailEnd) {
          row.n_block = -2;
        } else if (step < k_tiles) {
          row.n_block = k_tiles - 1 - step;
        } else if (step == k_tiles) {
          row.n_block = -1;
        } else {
          row.n_block = -9999;
        }
        timeline.push_back(row);
      }
    }
  }
  return timeline;
}

inline int fa3_profile_reg_timeline_meta_offset(int block_slot, int field) {
  return kFlashFwdProfileRegTimelineBase +
         flash_fwd_profile_reg_timeline_meta_offset(block_slot, field);
}

inline int fa3_profile_reg_timeline_offset(int block_slot, int slot, int thread,
                                           int event) {
  return kFlashFwdProfileRegTimelineBase +
         flash_fwd_profile_reg_timeline_offset(block_slot, slot, thread, event);
}

inline std::vector<Fa3PrefillProfileRegTimelineEvent>
extract_fa3_profile_reg_timeline(const std::vector<uint64_t> &profile_clock,
                                 int k_tiles, int logical_tiles) {
  (void)logical_tiles;
  std::vector<Fa3PrefillProfileRegTimelineEvent> timeline;
  const int first_step = flash_fwd_profile_reg_timeline_first_step(k_tiles);
  const int capture_count =
      flash_fwd_profile_reg_timeline_capture_count(k_tiles);
  if (first_step < 0 || capture_count <= 0) return timeline;
  const int block_count = kFlashFwdProfileRegTimelineMaxBlocks;
  for (int block_slot = 0; block_slot < block_count; ++block_slot) {
    const int block_x_offset = fa3_profile_reg_timeline_meta_offset(block_slot, 0);
    if (block_x_offset + kFlashFwdProfileRegTimelineBlockMetaCount >
        static_cast<int>(profile_clock.size())) {
      return timeline;
    }
    const int block_x = static_cast<int>(profile_clock[block_x_offset + 0]);
    const int block_y = static_cast<int>(profile_clock[block_x_offset + 1]);
    const int block_z = static_cast<int>(profile_clock[block_x_offset + 2]);
    const int smid = static_cast<int>(profile_clock[block_x_offset + 3]);
    for (int slot = 0; slot < capture_count; ++slot) {
      const int step = first_step + slot;
      const int n_block = k_tiles - 1 - step;
      for (int thread = 0; thread < kFlashFwdProfileRegTimelineConsumerThreads;
           ++thread) {
        for (int event = 0; event < kFlashFwdProfileRegTimelineEventCount;
             ++event) {
          const int offset =
              fa3_profile_reg_timeline_offset(block_slot, slot, thread, event);
          if (offset >= static_cast<int>(profile_clock.size())) return timeline;
          const uint64_t clock = profile_clock[offset];
          if (clock == 0) continue;
          Fa3PrefillProfileRegTimelineEvent row;
          row.block_slot = block_slot;
          row.block_x = block_x;
          row.block_y = block_y;
          row.block_z = block_z;
          row.smid = smid;
          row.slot = slot;
          row.step = step;
          row.n_block = n_block;
          row.local_thread = thread;
          row.local_wg = thread / 128;
          row.warp_in_wg = (thread % 128) / 32;
          row.lane = thread % 32;
          row.event = event;
          row.clock = clock;
          timeline.push_back(row);
        }
      }
    }
  }
  return timeline;
}
#endif

inline bool is_valid_fa3_prefill_case(const Fa3PrefillCase &cfg) {
  return is_supported_fa3_prefill_case(cfg) &&
         cfg.batch * cfg.seqlen == 32 * 1024;
}

inline bool is_valid_fa3_prefill_smoke_case(const Fa3PrefillCase &cfg) {
  return is_supported_fa3_prefill_case(cfg) &&
         cfg.batch == 2 && cfg.seqlen == 128;
}

inline bool is_valid_fa3_prefill_tuning_case(const Fa3PrefillCase &cfg) {
  return is_supported_fa3_prefill_case(cfg) && cfg.seqlen % 128 == 0;
}

inline void fill_half(std::vector<cutlass::half_t> &x, float scale) {
  for (size_t i = 0; i < x.size(); ++i) {
    float v = std::sin(float(i % 251) * 0.013f) * scale;
    x[i] = cutlass::half_t(v);
  }
}

inline std::vector<cutlass::half_t> make_fa3_collect_half_data(
    size_t count, uint32_t seed, float scale) {
  std::vector<cutlass::half_t> data(count);
  for (size_t i = 0; i < count; ++i) {
    uint32_t x = uint32_t(i) * 1103515245u + seed * 12345u + 0x9e3779b9u;
    float centered = float(int((x >> 8) & 0x3ffu) - 512) / 512.0f;
    data[i] = cutlass::half_t(centered * scale);
  }
  return data;
}

inline int fa3_round_up(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

template <typename Params>
inline void set_fa3_prefill_base_params(Params &params,
                                        cutlass::half_t *d_q,
                                        cutlass::half_t *d_k,
                                        cutlass::half_t *d_v,
                                        cutlass::half_t *d_o,
                                        float *d_lse,
                                        int batch,
                                        int seqlen_q,
                                        int seqlen_k,
                                        int heads,
                                        int heads_kv,
                                        int head_dim,
                                        int head_dim_v,
                                        int seqlen_q_rounded,
                                        int seqlen_k_rounded,
                                        bool causal) {
  params.q_ptr = d_q;
  params.k_ptr = d_k;
  params.v_ptr = d_v;
  params.o_ptr = d_o;
  params.softmax_lse_ptr = d_lse;

  params.q_batch_stride = seqlen_q * heads * head_dim;
  params.k_batch_stride = seqlen_k * heads_kv * head_dim;
  params.v_batch_stride = seqlen_k * heads_kv * head_dim_v;
  params.o_batch_stride = seqlen_q * heads * head_dim_v;

  params.q_row_stride = heads * head_dim;
  params.k_row_stride = heads_kv * head_dim;
  params.v_row_stride = heads_kv * head_dim_v;
  params.o_row_stride = heads * head_dim_v;

  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim_v;
  params.o_head_stride = head_dim_v;
  params.v_dim_stride = 1;

  params.b = batch;
  params.b_k = batch;
  params.h = heads;
  params.h_k = heads_kv;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_knew = 0;
  params.total_q = batch * seqlen_q;
  params.total_k = batch * seqlen_k;
  params.total_knew = 0;
  params.d = head_dim;
  params.dv = head_dim_v;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d_rounded = head_dim;
  params.dv_rounded = head_dim_v;
  params.rotary_dim = 0;

  params.scale_softmax = 1.0f / std::sqrt(float(head_dim));
  params.softcap = 0.0f;
  params.window_size_left = seqlen_k - 1;
  params.window_size_right = causal ? 0 : seqlen_q - 1;
  params.attention_chunk = 0;

  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;

  params.is_bf16 = false;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = causal;
  params.is_local = false;
  params.is_rotary_interleaved = false;

  params.num_splits = 1;
  params.pack_gqa = false;
  params.tile_count_semaphore_offset = 0;
  params.skip_scheduler_metadata_computation = true;
  params.varlen_sort_batches = true;
  params.head_swizzle = causal;
  params.prepare_varlen_pdl = false;
  params.arch = 90;
  params.num_sm = 0;
}

template <int HeadDim, bool IsCausal>
inline Fa3RunResult run_fa3_prefill_fp16_typed(const Fa3PrefillCase &cfg) {
  constexpr int D = HeadDim;
  constexpr int DV = HeadDim;
  const int B = cfg.batch;
  const int M = cfg.seqlen;
  const int N = cfg.seqlen;
  const int H = cfg.heads;
  const int H_KV = cfg.heads;

  const size_t q_elems = size_t(B) * M * H * D;
  const size_t k_elems = size_t(B) * N * H_KV * D;
  const size_t v_elems = size_t(B) * N * H_KV * DV;
  const size_t o_elems = size_t(B) * M * H * DV;
  const size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;
  int *d_tile_count_semaphore = nullptr;

  Fa3RunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);
    free_if_needed(d_tile_count_semaphore);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  if (!is_supported_fa3_prefill_case(cfg) || cfg.head_dim != HeadDim ||
      cfg.causal != IsCausal) {
    return finish(cudaErrorInvalidValue, "Fa3PrefillCase");
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_tile_count_semaphore, sizeof(int)));
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_q, 0, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_k, 0, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_v, 0, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_tile_count_semaphore, 0, sizeof(int)));
  }

  Flash_fwd_params params = {};
  set_fa3_prefill_base_params(params, d_q, d_k, d_v, d_o, d_lse,
                              B, M, N, H, H_KV, D, DV, M, N, IsCausal);
  params.tile_count_semaphore = d_tile_count_semaphore;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, D, DV, 1, cutlass::half_t, cutlass::half_t,
                IsCausal, false, false, false, false,
                false, false, false, false, false>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  cutlass::half_t output0;
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&output0, d_o, sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&result.lse0, d_lse, sizeof(float),
                                      cudaMemcpyDeviceToHost));
  result.output0 = float(output0);

  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

template <int HeadDim, bool IsCausal>
inline Fa3RunResult run_fa3_prefill_fp16_bwd_typed(const Fa3PrefillCase &cfg) {
  constexpr int D = HeadDim;
  constexpr int DV = HeadDim;
  constexpr int kBlockM = D <= 64 ? 128 : (IsCausal ? 64 : 80);
  constexpr int kBlockN = 128;
  const int B = cfg.batch;
  const int M = cfg.seqlen;
  const int N = cfg.seqlen;
  const int H = cfg.heads;
  const int H_KV = cfg.heads;
  const int M_rounded = fa3_round_up(M, kBlockM);
  const int N_rounded = fa3_round_up(N, kBlockN);

  const size_t q_elems = size_t(B) * M * H * D;
  const size_t k_elems = size_t(B) * N * H_KV * D;
  const size_t v_elems = size_t(B) * N * H_KV * DV;
  const size_t o_elems = size_t(B) * M * H * DV;
  const size_t lse_elems = size_t(B) * H * M;
  const size_t lse_log2_elems = size_t(B) * H * M_rounded;
  const size_t dsoftmax_elems = size_t(B) * H * M_rounded;
  const size_t dq_accum_elems = size_t(B) * H * M_rounded * D;
  const size_t dq_semaphore_elems =
      size_t((M + kBlockM - 1) / kBlockM) * B * H;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  cutlass::half_t *d_do = nullptr;
  cutlass::half_t *d_dq = nullptr;
  cutlass::half_t *d_dk = nullptr;
  cutlass::half_t *d_dv = nullptr;
  float *d_lse = nullptr;
  float *d_lse_log2 = nullptr;
  float *d_dsoftmax = nullptr;
  float *d_dq_accum = nullptr;
  int *d_dq_semaphore = nullptr;

  Fa3RunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_do);
    free_if_needed(d_dq);
    free_if_needed(d_dk);
    free_if_needed(d_dv);
    free_if_needed(d_lse);
    free_if_needed(d_lse_log2);
    free_if_needed(d_dsoftmax);
    free_if_needed(d_dq_accum);
    free_if_needed(d_dq_semaphore);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  if (!is_supported_fa3_prefill_case(cfg) || cfg.head_dim != HeadDim ||
      cfg.causal != IsCausal) {
    return finish(cudaErrorInvalidValue, "Fa3PrefillCase");
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_do, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dq, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dk, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dv, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse_log2,
                                      lse_log2_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dsoftmax,
                                      dsoftmax_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dq_accum,
                                      dq_accum_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_dq_semaphore,
                                      dq_semaphore_elems * sizeof(int)));

  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_q, 0, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_k, 0, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_v, 0, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_do, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dq, 0, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dk, 0, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dv, 0, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse_log2, 0,
                                      lse_log2_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dsoftmax, 0,
                                      dsoftmax_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dq_accum, 0,
                                      dq_accum_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_dq_semaphore, 0,
                                      dq_semaphore_elems * sizeof(int)));

  Flash_bwd_params params = {};
  set_fa3_prefill_base_params(params, d_q, d_k, d_v, d_o, d_lse,
                              B, M, N, H, H_KV, D, DV,
                              M_rounded, N_rounded, IsCausal);

  params.do_ptr = d_do;
  params.dq_ptr = d_dq;
  params.dk_ptr = d_dk;
  params.dv_ptr = d_dv;
  params.do_batch_stride = M * H * DV;
  params.do_row_stride = H * DV;
  params.do_head_stride = DV;
  params.dq_batch_stride = M * H * D;
  params.dk_batch_stride = N * H_KV * D;
  params.dv_batch_stride = N * H_KV * DV;
  params.dq_row_stride = H * D;
  params.dk_row_stride = H_KV * D;
  params.dv_row_stride = H_KV * DV;
  params.dq_head_stride = D;
  params.dk_head_stride = D;
  params.dv_head_stride = DV;
  params.dq_accum_ptr = d_dq_accum;
  params.dsoftmax_sum = d_dsoftmax;
  params.softmax_lse_log2_ptr = d_lse_log2;
  params.dq_semaphore = d_dq_semaphore;
  params.deterministic = false;
  params.dq_accum_split_stride = 0;

  cudaStream_t stream = nullptr;
  if constexpr (D == 64) {
    run_flash_bwd<90, D, 128, 128, cutlass::half_t, IsCausal, false, false,
                  false, false, false, 2, 2, true, false, false, 2, 1, 2, 2,
                  false>(params, stream);
  } else if constexpr (IsCausal) {
    run_flash_bwd<90, D, 64, 128, cutlass::half_t, true, false, false,
                  false, false, false, 2, 2, true, false, false, 2, 1, 2, 1,
                  false>(params, stream);
  } else {
    run_flash_bwd<90, D, 80, 128, cutlass::half_t, false, false, false,
                  false, false, false, 2, 2, true, false, true, 2, 1, 2, 1,
                  false>(params, stream);
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  cutlass::half_t dq0;
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&dq0, d_dq, sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&result.lse0, d_dsoftmax, sizeof(float),
                                      cudaMemcpyDeviceToHost));
  result.output0 = float(dq0);

  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

inline Fa3RunResult run_fa3_prefill_fp16(const Fa3PrefillCase &cfg) {
  if (cfg.head_dim == 64 && !cfg.causal) {
    return run_fa3_prefill_fp16_typed<64, false>(cfg);
  }
  if (cfg.head_dim == 64 && cfg.causal) {
    return run_fa3_prefill_fp16_typed<64, true>(cfg);
  }
  if (cfg.head_dim == 128 && !cfg.causal) {
    return run_fa3_prefill_fp16_typed<128, false>(cfg);
  }
  if (cfg.head_dim == 128 && cfg.causal) {
    return run_fa3_prefill_fp16_typed<128, true>(cfg);
  }

  Fa3RunResult result;
  result.error = cudaErrorInvalidValue;
  result.where = "Fa3PrefillCase";
  return result;
}

inline Fa3RunResult run_fa3_prefill_fp16_bwd(const Fa3PrefillCase &cfg) {
  if (cfg.head_dim == 64 && !cfg.causal) {
    return run_fa3_prefill_fp16_bwd_typed<64, false>(cfg);
  }
  if (cfg.head_dim == 64 && cfg.causal) {
    return run_fa3_prefill_fp16_bwd_typed<64, true>(cfg);
  }
  if (cfg.head_dim == 128 && !cfg.causal) {
    return run_fa3_prefill_fp16_bwd_typed<128, false>(cfg);
  }
  if (cfg.head_dim == 128 && cfg.causal) {
    return run_fa3_prefill_fp16_bwd_typed<128, true>(cfg);
  }

  Fa3RunResult result;
  result.error = cudaErrorInvalidValue;
  result.where = "Fa3PrefillCase";
  return result;
}

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)
template <int HeadDim, bool IsCausal>
inline Fa3PrefillProfileResult run_fa3_prefill_profile_fp16_typed(
    const Fa3PrefillCase &cfg) {
  constexpr int D = HeadDim;
  constexpr int DV = HeadDim;
  constexpr int kBlockM = HeadDim <= 64 ? 192 : 128;
  constexpr int kBlockN = HeadDim <= 64 ? (IsCausal ? 128 : 192)
                                        : (IsCausal ? 128 : 176);
  const int B = cfg.batch;
  const int M = cfg.seqlen;
  const int N = cfg.seqlen;
  const int H = cfg.heads;
  const int H_KV = cfg.heads;

  const size_t q_elems = size_t(B) * M * H * D;
  const size_t k_elems = size_t(B) * N * H_KV * D;
  const size_t v_elems = size_t(B) * N * H_KV * DV;
  const size_t o_elems = size_t(B) * M * H * DV;
  const size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;
  int *d_tile_count_semaphore = nullptr;
  uint64_t *d_profile_clock = nullptr;

  Fa3PrefillProfileResult result;
  result.name = cfg.name;
  result.batch = cfg.batch;
  result.seqlen_q = cfg.seqlen;
  result.seqlen_k = cfg.seqlen;
  result.heads = cfg.heads;
  result.head_dim = cfg.head_dim;
  result.causal = cfg.causal ? 1 : 0;
  result.block_m = kBlockM;
  result.block_n = kBlockN;
  result.m_tiles = fa3_round_up(M, kBlockM) / kBlockM;
  result.k_tiles = fa3_round_up(N, kBlockN) / kBlockN;
  result.logical_tiles = result.batch * result.heads * result.m_tiles;

  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);
    free_if_needed(d_tile_count_semaphore);
    free_if_needed(d_profile_clock);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  if (!is_supported_fa3_prefill_case(cfg) || cfg.head_dim != HeadDim ||
      cfg.causal != IsCausal) {
    return finish(cudaErrorInvalidValue, "Fa3PrefillCase");
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  std::vector<cutlass::half_t> h_q =
      make_fa3_collect_half_data(q_elems, 1, 0.50f);
  std::vector<cutlass::half_t> h_k =
      make_fa3_collect_half_data(k_elems, 2, 0.45f);
  std::vector<cutlass::half_t> h_v =
      make_fa3_collect_half_data(v_elems, 3, 0.40f);
  const bool profile_clock_enabled = fa3_prefill_profile_clock_enabled();
  std::vector<uint64_t> h_profile_clock = make_fa3_profile_clock_init();

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_tile_count_semaphore, sizeof(int)));
  }
  if (profile_clock_enabled) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(
        &d_profile_clock, h_profile_clock.size() * sizeof(uint64_t)));
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(),
                                      q_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(),
                                      k_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(),
                                      v_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_tile_count_semaphore, 0, sizeof(int)));
  }
  if (profile_clock_enabled) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(
        d_profile_clock, h_profile_clock.data(),
        h_profile_clock.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
  }

  Flash_fwd_params params = {};
  set_fa3_prefill_base_params(params, d_q, d_k, d_v, d_o, d_lse,
                              B, M, N, H, H_KV, D, DV, M, N, IsCausal);
  params.tile_count_semaphore = d_tile_count_semaphore;
  params.profile_clock_ptr = d_profile_clock;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, D, DV, 1, cutlass::half_t, cutlass::half_t,
                IsCausal, false, false, false, false,
                false, false, false, false, false>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  cutlass::half_t output0;
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&output0, d_o, sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&result.lse0, d_lse, sizeof(float),
                                      cudaMemcpyDeviceToHost));
  if (profile_clock_enabled) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(
        h_profile_clock.data(), d_profile_clock,
        h_profile_clock.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  }

  result.clock_start = h_profile_clock[kFlashFwdProfileKernelStart];
  result.clock_end = h_profile_clock[kFlashFwdProfileKernelEnd];
  result.clock_delta =
      fa3_profile_clock_delta(result.clock_start, result.clock_end);
  result.mainloop_start = h_profile_clock[kFlashFwdProfileMainloopStart];
  result.mainloop_end = h_profile_clock[kFlashFwdProfileMainloopEnd];
  result.mainloop_delta =
      fa3_profile_clock_delta(result.mainloop_start, result.mainloop_end);
  result.epilogue_start = h_profile_clock[kFlashFwdProfileEpilogueStart];
  result.epilogue_end = h_profile_clock[kFlashFwdProfileEpilogueEnd];
  result.epilogue_delta =
      fa3_profile_clock_delta(result.epilogue_start, result.epilogue_end);
  result.qk_wait_cycles = h_profile_clock[kFlashFwdProfileQkWaitCycles];
  result.qk_wgmma_issue_cycles =
      h_profile_clock[kFlashFwdProfileQkWgmmaIssueCycles];
  result.softmax_cycles = h_profile_clock[kFlashFwdProfileSoftmaxCycles];
  result.pv_wait_cycles = h_profile_clock[kFlashFwdProfilePvWaitCycles];
  result.pv_wgmma_issue_wait_cycles =
      h_profile_clock[kFlashFwdProfilePvWgmmaIssueWaitCycles];
  result.mainloop_iterations =
      h_profile_clock[kFlashFwdProfileMainloopIterations];
  result.trace_steps = extract_fa3_profile_trace(h_profile_clock);
  result.timeline_events =
      extract_fa3_profile_timeline(h_profile_clock, result.k_tiles);
  result.reg_timeline_events =
      extract_fa3_profile_reg_timeline(h_profile_clock, result.k_tiles,
                                       result.logical_tiles);
  result.output0 = float(output0);

  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

inline Fa3PrefillProfileResult run_fa3_prefill_profile_fp16(
    const Fa3PrefillCase &cfg) {
  if (cfg.head_dim == 64 && !cfg.causal) {
    return run_fa3_prefill_profile_fp16_typed<64, false>(cfg);
  }
  if (cfg.head_dim == 64 && cfg.causal) {
    return run_fa3_prefill_profile_fp16_typed<64, true>(cfg);
  }
  if (cfg.head_dim == 128 && !cfg.causal) {
    return run_fa3_prefill_profile_fp16_typed<128, false>(cfg);
  }
  if (cfg.head_dim == 128 && cfg.causal) {
    return run_fa3_prefill_profile_fp16_typed<128, true>(cfg);
  }

  Fa3PrefillProfileResult result;
  result.error = cudaErrorInvalidValue;
  result.where = "Fa3PrefillCase";
  return result;
}
#endif

inline Fa3RunResult run_fa3_fwd_hdim128_fp16() {
  constexpr int B = 9;
  constexpr int M = 64;
  constexpr int N = 128;
  constexpr int H = 6;
  constexpr int H_KV = 6;
  constexpr int D = 128;
  constexpr int DV = 128;

  constexpr size_t q_elems = size_t(B) * M * H * D;
  constexpr size_t k_elems = size_t(B) * N * H_KV * D;
  constexpr size_t v_elems = size_t(B) * N * H_KV * DV;
  constexpr size_t o_elems = size_t(B) * M * H * DV;
  constexpr size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;

  Fa3RunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  std::vector<cutlass::half_t> h_q(q_elems);
  std::vector<cutlass::half_t> h_k(k_elems);
  std::vector<cutlass::half_t> h_v(v_elems);
  std::vector<cutlass::half_t> h_o(o_elems);
  std::vector<float> h_lse(lse_elems);

  fill_half(h_q, 0.25f);
  fill_half(h_k, 0.20f);
  fill_half(h_v, 0.30f);

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));

  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(),
                                      q_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(),
                                      k_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(),
                                      v_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

  Flash_fwd_params params = {};
  params.q_ptr = d_q;
  params.k_ptr = d_k;
  params.v_ptr = d_v;
  params.o_ptr = d_o;
  params.softmax_lse_ptr = d_lse;

  params.q_batch_stride = M * H * D;
  params.k_batch_stride = N * H_KV * D;
  params.v_batch_stride = N * H_KV * DV;
  params.o_batch_stride = M * H * DV;

  params.q_row_stride = H * D;
  params.k_row_stride = H_KV * D;
  params.v_row_stride = H_KV * DV;
  params.o_row_stride = H * DV;

  params.q_head_stride = D;
  params.k_head_stride = D;
  params.v_head_stride = DV;
  params.o_head_stride = DV;
  params.v_dim_stride = 1;

  params.b = B;
  params.b_k = B;
  params.h = H;
  params.h_k = H_KV;
  params.seqlen_q = M;
  params.seqlen_k = N;
  params.seqlen_knew = 0;
  params.total_q = B * M;
  params.total_k = B * N;
  params.total_knew = 0;
  params.d = D;
  params.dv = DV;
  params.seqlen_q_rounded = M;
  params.seqlen_k_rounded = N;
  params.d_rounded = D;
  params.dv_rounded = DV;
  params.rotary_dim = 0;

  params.scale_softmax = 1.0f / std::sqrt(float(D));
  params.softcap = 0.0f;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.attention_chunk = 0;

  params.is_bf16 = false;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = false;
  params.is_local = false;
  params.is_rotary_interleaved = false;

  params.num_splits = 1;
  params.pack_gqa = false;
  params.skip_scheduler_metadata_computation = true;
  params.varlen_sort_batches = false;
  params.head_swizzle = false;
  params.prepare_varlen_pdl = false;
  params.arch = 90;
  params.num_sm = 0;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, 128, 128, 1, cutlass::half_t, cutlass::half_t,
                false, false, false, false, false,
                false, false, false, false, false>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_o.data(), d_o,
                                      o_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_lse.data(), d_lse,
                                      lse_elems * sizeof(float),
                                      cudaMemcpyDeviceToHost));

  result.output0 = float(h_o[0]);
  result.lse0 = h_lse[0];
  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)
inline Fa3SingleTileProfileResult run_fa3_single_tile_hdim128_fp16_full(
    int seqlen_k) {
  constexpr int B = 1;
  constexpr int M = 128;
  constexpr int H = 1;
  constexpr int H_KV = 1;
  constexpr int D = 128;
  constexpr int DV = 128;
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 176;

  const int N = seqlen_k;
  const size_t q_elems = size_t(B) * M * H * D;
  const size_t k_elems = size_t(B) * N * H_KV * D;
  const size_t v_elems = size_t(B) * N * H_KV * DV;
  const size_t o_elems = size_t(B) * M * H * DV;
  const size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;
  uint64_t *d_profile_clock = nullptr;

  Fa3SingleTileProfileResult result;
  result.seqlen_k = seqlen_k;
  result.block_m = kBlockM;
  result.block_n = kBlockN;
  result.k_tiles = fa3_round_up(seqlen_k, kBlockN) / kBlockN;

  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);
    free_if_needed(d_profile_clock);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  if (seqlen_k <= 0) {
    return finish(cudaErrorInvalidValue, "seqlen_k");
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  std::vector<cutlass::half_t> h_q(q_elems);
  std::vector<cutlass::half_t> h_k(k_elems);
  std::vector<cutlass::half_t> h_v(v_elems);
  std::vector<cutlass::half_t> h_o(o_elems);
  std::vector<float> h_lse(lse_elems);
  std::vector<uint64_t> h_profile_clock = make_fa3_profile_clock_init();

  fill_half(h_q, 0.25f);
  fill_half(h_k, 0.20f);
  fill_half(h_v, 0.30f);

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(
      &d_profile_clock, h_profile_clock.size() * sizeof(uint64_t)));

  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(),
                                      q_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(),
                                      k_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(),
                                      v_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      d_profile_clock, h_profile_clock.data(),
      h_profile_clock.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

  Flash_fwd_params params = {};
  set_fa3_prefill_base_params(params, d_q, d_k, d_v, d_o, d_lse,
                              B, M, N, H, H_KV, D, DV, M, N, false);
  params.profile_clock_ptr = d_profile_clock;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, D, DV, 1, cutlass::half_t, cutlass::half_t,
                false, false, false, false, false,
                false, false, false, false, false, true>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_o.data(), d_o,
                                      o_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_lse.data(), d_lse,
                                      lse_elems * sizeof(float),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      h_profile_clock.data(), d_profile_clock,
      h_profile_clock.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  result.clock_start = h_profile_clock[kFlashFwdProfileKernelStart];
  result.clock_end = h_profile_clock[kFlashFwdProfileKernelEnd];
  result.clock_delta =
      fa3_profile_clock_delta(result.clock_start, result.clock_end);
  result.mainloop_start = h_profile_clock[kFlashFwdProfileMainloopStart];
  result.mainloop_end = h_profile_clock[kFlashFwdProfileMainloopEnd];
  result.mainloop_delta =
      fa3_profile_clock_delta(result.mainloop_start, result.mainloop_end);
  result.epilogue_start = h_profile_clock[kFlashFwdProfileEpilogueStart];
  result.epilogue_end = h_profile_clock[kFlashFwdProfileEpilogueEnd];
  result.epilogue_delta =
      fa3_profile_clock_delta(result.epilogue_start, result.epilogue_end);
  result.qk_wait_cycles = h_profile_clock[kFlashFwdProfileQkWaitCycles];
  result.qk_wgmma_issue_cycles =
      h_profile_clock[kFlashFwdProfileQkWgmmaIssueCycles];
  result.softmax_cycles = h_profile_clock[kFlashFwdProfileSoftmaxCycles];
  result.pv_wait_cycles = h_profile_clock[kFlashFwdProfilePvWaitCycles];
  result.pv_wgmma_issue_wait_cycles =
      h_profile_clock[kFlashFwdProfilePvWgmmaIssueWaitCycles];
  result.mainloop_iterations =
      h_profile_clock[kFlashFwdProfileMainloopIterations];
  result.output0 = float(h_o[0]);
  result.lse0 = h_lse[0];
  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}
#endif

}  // namespace fa3_hopper_test
