#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fa3_fwd_hdim128_fp16_case.cuh"

namespace fa3_hopper_test {
namespace {

class Fa3H1D128ProfileTest : public ::testing::Test {};

constexpr Fa3PrefillCase kD128FullProfileCases[] = {
    // Original H16/D128/full cases.
    {"H16D128FullB64S512", 64, 512, 16, 128, false},
    {"H16D128FullB32S1024", 32, 1024, 16, 128, false},
    {"H16D128FullB16S2048", 16, 2048, 16, 128, false},
    {"H16D128FullB8S4096", 8, 4096, 16, 128, false},
    {"H16D128FullB4S8192", 4, 8192, 16, 128, false},
    {"H16D128CausalB64S512", 64, 512, 16, 128, true},

    // Reduced-H cases.
    {"H4D128FullB64S512", 64, 512, 4, 128, false},
    {"H4D128FullB32S1024", 32, 1024, 4, 128, false},
    {"H4D128FullB16S2048", 16, 2048, 4, 128, false},
    {"H4D128FullB8S4096", 8, 4096, 4, 128, false},
    {"H4D128FullB4S8192", 4, 8192, 4, 128, false},

    // Reduced-B cases.
    {"H16D128FullB16S512", 16, 512, 16, 128, false},
    {"H16D128FullB8S1024", 8, 1024, 16, 128, false},
    {"H16D128FullB4S2048", 4, 2048, 16, 128, false},
    {"H16D128FullB2S4096", 2, 4096, 16, 128, false},
    {"H16D128FullB1S8192", 1, 8192, 16, 128, false},

    // Minimal-H/B long-S cases.
    {"H1D128FullB1S128", 1, 128, 1, 128, false},
    {"H1D128FullB1S256", 1, 256, 1, 128, false},
    {"H1D128FullB1S512", 1, 512, 1, 128, false},
    {"H1D128FullB1S1024", 1, 1024, 1, 128, false},
    {"H1D128FullB1S2048", 1, 2048, 1, 128, false},
    {"H1D128FullB1S4096", 1, 4096, 1, 128, false},
    {"H1D128FullB1S8192", 1, 8192, 1, 128, false},

    // Minimal-H/B causal repro cases.
    {"H1D128CausalB1S512", 1, 512, 1, 128, true},

};

std::vector<std::string> ParseCaseNames() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_CASE_LIST");
  std::string list = env == nullptr || std::string(env).empty()
                         ? "H1D128FullB1S4096"
                         : env;
  for (char &ch : list) {
    if (ch == ',') ch = ' ';
  }
  std::istringstream is(list);
  std::vector<std::string> names;
  std::string value;
  while (is >> value) {
    names.push_back(value);
  }
  if (names.empty()) names.push_back("H1D128FullB1S4096");
  return names;
}

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)

std::string ProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_OUT");
  return env == nullptr || std::string(env).empty()
             ? "fa3_h1d128_profile.csv"
             : std::string(env);
}

std::string IterProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_ITER_OUT");
  return env == nullptr ? std::string() : std::string(env);
}

std::string TimelineProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_TIMELINE_OUT");
  return env == nullptr ? std::string() : std::string(env);
}

std::string RegTimelineProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_REG_TIMELINE_OUT");
  return env == nullptr ? std::string() : std::string(env);
}

std::string TaskProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_TASK_OUT");
  return env == nullptr ? std::string() : std::string(env);
}

#endif

bool LookupCase(const std::string &name, Fa3PrefillCase *out) {
  for (const Fa3PrefillCase &cfg : kD128FullProfileCases) {
    if (name == cfg.name) {
      *out = cfg;
      return true;
    }
  }
  return false;
}

std::vector<Fa3PrefillCase> SelectedCases() {
  std::vector<std::string> names = ParseCaseNames();
  std::vector<Fa3PrefillCase> cases;
  for (const std::string &name : names) {
    if (name == "all") {
      for (const Fa3PrefillCase &cfg : kD128FullProfileCases) {
        cases.push_back(cfg);
      }
      continue;
    }
    Fa3PrefillCase cfg{};
    if (LookupCase(name, &cfg)) cases.push_back(cfg);
  }
  return cases;
}

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)

const char *TraceKindName(int kind) {
  switch (kind) {
    case kFlashFwdProfileTraceInitialQk:
      return "initial_qk";
    case kFlashFwdProfileTraceSteady:
      return "steady";
    case kFlashFwdProfileTraceFinalPv:
      return "final_pv";
    default:
      return "unused";
  }
}

const char *TimelineActorName(int actor) {
  switch (actor) {
    case kFlashFwdProfileTimelineProducer:
      return "producer";
    case kFlashFwdProfileTimelineConsumerLow:
      return "consumer_low";
    case kFlashFwdProfileTimelineConsumerHigh:
      return "consumer_high";
    default:
      return "unknown";
  }
}

const char *TimelineEventName(int event) {
  switch (event) {
    case kFlashFwdProfileTimelineTileBegin:
      return "tile_begin";
    case kFlashFwdProfileTimelineProducerLoadKBegin:
      return "producer_load_k_begin";
    case kFlashFwdProfileTimelineProducerLoadKEnd:
      return "producer_load_k_end";
    case kFlashFwdProfileTimelineProducerLoadVBegin:
      return "producer_load_v_begin";
    case kFlashFwdProfileTimelineProducerLoadVEnd:
      return "producer_load_v_end";
    case kFlashFwdProfileTimelineProducerBarrierOBegin:
      return "producer_barrier_o_begin";
    case kFlashFwdProfileTimelineProducerBarrierOEnd:
      return "producer_barrier_o_end";
    case kFlashFwdProfileTimelineProducerTailBegin:
      return "producer_tail_begin";
    case kFlashFwdProfileTimelineProducerTailEnd:
      return "producer_tail_end";
    case kFlashFwdProfileTimelineConsumerStepBegin:
      return "consumer_step_begin";
    case kFlashFwdProfileTimelineConsumerKWaitBegin:
      return "consumer_k_wait_begin";
    case kFlashFwdProfileTimelineConsumerKWaitEnd:
      return "consumer_k_wait_end";
    case kFlashFwdProfileTimelineConsumerSchedulerWaitBegin:
      return "consumer_scheduler_wait_begin";
    case kFlashFwdProfileTimelineConsumerSchedulerWaitEnd:
      return "consumer_scheduler_wait_end";
    case kFlashFwdProfileTimelineConsumerQkIssueBegin:
      return "consumer_qk_issue_begin";
    case kFlashFwdProfileTimelineConsumerQkIssueEnd:
      return "consumer_qk_issue_end";
    case kFlashFwdProfileTimelineConsumerPvWaitBegin:
      return "consumer_pv_wait_begin";
    case kFlashFwdProfileTimelineConsumerPvWaitEnd:
      return "consumer_pv_wait_end";
    case kFlashFwdProfileTimelineConsumerPvIssueBegin:
      return "consumer_pv_issue_begin";
    case kFlashFwdProfileTimelineConsumerSchedulerArrive:
      return "consumer_scheduler_arrive";
    case kFlashFwdProfileTimelineConsumerWait1End:
      return "consumer_wait1_end";
    case kFlashFwdProfileTimelineConsumerKRelease:
      return "consumer_k_release";
    case kFlashFwdProfileTimelineConsumerWait0Begin:
      return "consumer_wait0_begin";
    case kFlashFwdProfileTimelineConsumerWait0End:
      return "consumer_wait0_end";
    case kFlashFwdProfileTimelineConsumerVRelease:
      return "consumer_v_release";
    case kFlashFwdProfileTimelineConsumerStepEnd:
      return "consumer_step_end";
    default:
      return "unknown";
  }
}

const char *RegTimelineEventName(int event) {
  switch (event) {
    case kFlashFwdProfileRegTimelineSchedulerWaitBefore:
      return "scheduler_wait_before";
    case kFlashFwdProfileRegTimelineSchedulerWaitAfter:
      return "scheduler_wait_after";
    case kFlashFwdProfileRegTimelineQkIssueBegin:
      return "qk_issue_begin";
    case kFlashFwdProfileRegTimelineQkIssueEnd:
      return "qk_issue_end";
    case kFlashFwdProfileRegTimelinePvIssueBegin:
      return "pv_issue_begin";
    case kFlashFwdProfileRegTimelinePvIssueEnd:
      return "pv_issue_end";
    case kFlashFwdProfileRegTimelineSchedulerArriveBefore:
      return "scheduler_arrive_before";
    case kFlashFwdProfileRegTimelineSchedulerArriveAfter:
      return "scheduler_arrive_after";
    case kFlashFwdProfileRegTimelineWait1Before:
      return "wait1_before";
    case kFlashFwdProfileRegTimelineWait1After:
      return "wait1_after";
    case kFlashFwdProfileRegTimelineWait0Before:
      return "wait0_before";
    case kFlashFwdProfileRegTimelineWait0After:
      return "wait0_after";
    default:
      return "unknown";
  }
}

void WriteProfileCsv(const std::string &path,
                     const std::vector<Fa3PrefillProfileResult> &results) {
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,"
         "block_n,m_tiles,k_tiles,logical_tiles,clock_start,clock_end,"
         "clock_delta,mainloop_start,mainloop_end,mainloop_delta,"
         "epilogue_start,epilogue_end,epilogue_delta,qk_wait_cycles,"
         "qk_wgmma_issue_cycles,softmax_cycles,pv_wait_cycles,"
         "pv_wgmma_issue_wait_cycles,mainloop_iterations,output0,lse0\n";
  for (const auto &result : results) {
    out << result.name << ","
        << result.batch << ","
        << result.seqlen_q << ","
        << result.seqlen_k << ","
        << result.heads << ","
        << result.head_dim << ","
        << result.causal << ","
        << result.block_m << ","
        << result.block_n << ","
        << result.m_tiles << ","
        << result.k_tiles << ","
        << result.logical_tiles << ","
        << result.clock_start << ","
        << result.clock_end << ","
        << result.clock_delta << ","
        << result.mainloop_start << ","
        << result.mainloop_end << ","
        << result.mainloop_delta << ","
        << result.epilogue_start << ","
        << result.epilogue_end << ","
        << result.epilogue_delta << ","
        << result.qk_wait_cycles << ","
        << result.qk_wgmma_issue_cycles << ","
        << result.softmax_cycles << ","
        << result.pv_wait_cycles << ","
        << result.pv_wgmma_issue_wait_cycles << ","
        << result.mainloop_iterations << ","
        << result.output0 << ","
        << result.lse0 << "\n";
  }
}

void WriteIterationProfileCsv(
    const std::string &path,
    const std::vector<Fa3PrefillProfileResult> &results) {
  if (path.empty()) return;
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,"
         "block_n,step,step_kind,n_block,step_start,step_end,step_delta,"
         "qk_wait_cycles,qk_wgmma_issue_cycles,softmax_cycles,"
         "pv_wait_cycles,pv_wgmma_issue_wait_cycles,residual_cycles\n";
  for (const auto &result : results) {
    for (const auto &step : result.trace_steps) {
      const uint64_t step_delta =
          fa3_profile_clock_delta(step.start, step.end);
      const uint64_t bucket_sum =
          step.qk_wait_cycles + step.qk_wgmma_issue_cycles +
          step.softmax_cycles + step.pv_wait_cycles +
          step.pv_wgmma_issue_wait_cycles;
      const uint64_t residual =
          step_delta >= bucket_sum ? step_delta - bucket_sum : 0;
      out << result.name << ","
          << result.batch << ","
          << result.seqlen_q << ","
          << result.seqlen_k << ","
          << result.heads << ","
          << result.head_dim << ","
          << result.causal << ","
          << result.block_m << ","
          << result.block_n << ","
          << step.step << ","
          << TraceKindName(step.kind) << ","
          << step.n_block << ","
          << step.start << ","
          << step.end << ","
          << step_delta << ","
          << step.qk_wait_cycles << ","
          << step.qk_wgmma_issue_cycles << ","
          << step.softmax_cycles << ","
          << step.pv_wait_cycles << ","
          << step.pv_wgmma_issue_wait_cycles << ","
          << residual << "\n";
    }
  }
}

void WriteTimelineProfileCsv(
    const std::string &path,
    const std::vector<Fa3PrefillProfileResult> &results) {
  if (path.empty()) return;
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,"
         "block_n,actor,actor_id,step,n_block,event,event_id,clock,"
         "clock_from_kernel_start\n";
  for (const auto &result : results) {
    for (const auto &event : result.timeline_events) {
      const uint64_t clock_from_kernel_start =
          result.clock_start != std::numeric_limits<uint64_t>::max() &&
                  event.clock >= result.clock_start
              ? event.clock - result.clock_start
              : 0;
      out << result.name << ","
          << result.batch << ","
          << result.seqlen_q << ","
          << result.seqlen_k << ","
          << result.heads << ","
          << result.head_dim << ","
          << result.causal << ","
          << result.block_m << ","
          << result.block_n << ","
          << TimelineActorName(event.actor) << ","
          << event.actor << ","
          << event.step << ","
          << event.n_block << ","
          << TimelineEventName(event.event) << ","
          << event.event << ","
          << event.clock << ","
          << clock_from_kernel_start << "\n";
    }
  }
}

void WriteRegTimelineProfileCsv(
    const std::string &path,
    const std::vector<Fa3PrefillProfileResult> &results) {
  if (path.empty()) return;
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,"
         "block_n,block_slot,block_x,block_y,block_z,smid,slot,step,n_block,"
         "local_thread,local_wg,warp_in_wg,lane,event,event_id,clock,"
         "clock_from_kernel_start\n";
  for (const auto &result : results) {
    for (const auto &event : result.reg_timeline_events) {
      const uint64_t clock_from_kernel_start =
          result.clock_start != std::numeric_limits<uint64_t>::max() &&
                  event.clock >= result.clock_start
              ? event.clock - result.clock_start
              : 0;
      out << result.name << ","
          << result.batch << ","
          << result.seqlen_q << ","
          << result.seqlen_k << ","
          << result.heads << ","
          << result.head_dim << ","
          << result.causal << ","
          << result.block_m << ","
          << result.block_n << ","
          << event.block_slot << ","
          << event.block_x << ","
          << event.block_y << ","
          << event.block_z << ","
          << event.smid << ","
          << event.slot << ","
          << event.step << ","
          << event.n_block << ","
          << event.local_thread << ","
          << event.local_wg << ","
          << event.warp_in_wg << ","
          << event.lane << ","
          << RegTimelineEventName(event.event) << ","
          << event.event << ","
          << event.clock << ","
          << clock_from_kernel_start << "\n";
    }
  }
}

void WriteTaskProfileCsv(
    const std::string &path,
    const std::vector<Fa3PrefillProfileResult> &results) {
  if (path.empty()) return;
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,"
         "block_n,logical_tiles,slot,m_block,bidh,bidb,split_idx,cta_x,"
         "smid,tile_valid,mainloop_start,mainloop_end,mainloop_delta,"
         "epilogue_start,epilogue_end,epilogue_delta,task_delta,"
         "mainloop_start_from_kernel,epilogue_end_from_kernel\n";
  for (const auto &result : results) {
    for (const auto &task : result.task_traces) {
      const uint64_t mainloop_delta =
          fa3_profile_clock_delta(task.mainloop_start, task.mainloop_end);
      const uint64_t epilogue_delta =
          fa3_profile_clock_delta(task.epilogue_start, task.epilogue_end);
      const uint64_t task_delta =
          fa3_profile_clock_delta(task.mainloop_start, task.epilogue_end);
      const uint64_t mainloop_start_from_kernel =
          result.clock_start != std::numeric_limits<uint64_t>::max() &&
                  task.mainloop_start >= result.clock_start
              ? task.mainloop_start - result.clock_start
              : 0;
      const uint64_t epilogue_end_from_kernel =
          result.clock_start != std::numeric_limits<uint64_t>::max() &&
                  task.epilogue_end >= result.clock_start
              ? task.epilogue_end - result.clock_start
              : 0;
      out << result.name << ","
          << result.batch << ","
          << result.seqlen_q << ","
          << result.seqlen_k << ","
          << result.heads << ","
          << result.head_dim << ","
          << result.causal << ","
          << result.block_m << ","
          << result.block_n << ","
          << result.logical_tiles << ","
          << task.slot << ","
          << task.m_block << ","
          << task.bidh << ","
          << task.bidb << ","
          << task.split_idx << ","
          << task.cta_x << ","
          << task.smid << ","
          << task.tile_valid << ","
          << task.mainloop_start << ","
          << task.mainloop_end << ","
          << mainloop_delta << ","
          << task.epilogue_start << ","
          << task.epilogue_end << ","
          << epilogue_delta << ","
          << task_delta << ","
          << mainloop_start_from_kernel << ","
          << epilogue_end_from_kernel << "\n";
    }
  }
}

#else

std::string NoProfileOutPath() {
  const char *env = std::getenv("FA3_H1D128_PROFILE_OUT");
  return env == nullptr || std::string(env).empty()
             ? "fa3_h1d128_noprofile.csv"
             : std::string(env);
}

void WriteNoProfileCsv(const std::string &path,
                       const std::vector<Fa3PrefillCase> &cases,
                       const std::vector<Fa3RunResult> &results) {
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,batch,seqlen_q,seqlen_k,heads,head_dim,causal,output0,lse0\n";
  ASSERT_EQ(cases.size(), results.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    const Fa3PrefillCase &cfg = cases[i];
    const Fa3RunResult &result = results[i];
    out << cfg.name << ","
        << cfg.batch << ","
        << cfg.seqlen << ","
        << cfg.seqlen << ","
        << cfg.heads << ","
        << cfg.head_dim << ","
        << (cfg.causal ? 1 : 0) << ","
        << result.output0 << ","
        << result.lse0 << "\n";
  }
}

#endif

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK)

TEST_F(Fa3H1D128ProfileTest, SelectedD128FullCases) {
  std::vector<Fa3PrefillCase> cases = SelectedCases();
  ASSERT_FALSE(cases.empty())
      << "No cases matched FA3_H1D128_PROFILE_CASE_LIST";

  std::vector<Fa3PrefillProfileResult> results;
  for (const Fa3PrefillCase &cfg : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "case=" << cfg.name
                 << " batch=" << cfg.batch
                 << " seqlen=" << cfg.seqlen
                 << " heads=" << cfg.heads);
    ASSERT_TRUE(is_supported_fa3_prefill_case(cfg));
    ASSERT_EQ(cfg.head_dim, 128);

    Fa3PrefillProfileResult result =
        run_fa3_prefill_profile_fp16(cfg);
    ASSERT_EQ(result.error, cudaSuccess)
        << result.where << " failed: " << cudaGetErrorString(result.error);
    if (fa3_prefill_profile_clock_enabled()) {
      ASSERT_GT(result.clock_delta, uint64_t{0})
          << "clock64 timestamps were not written";
    }
    results.push_back(result);
  }
  WriteProfileCsv(ProfileOutPath(), results);
  WriteIterationProfileCsv(IterProfileOutPath(), results);
  WriteTimelineProfileCsv(TimelineProfileOutPath(), results);
  WriteRegTimelineProfileCsv(RegTimelineProfileOutPath(), results);
  WriteTaskProfileCsv(TaskProfileOutPath(), results);
}

#else

TEST_F(Fa3H1D128ProfileTest, SelectedD128FullCases) {
  std::vector<Fa3PrefillCase> cases = SelectedCases();
  ASSERT_FALSE(cases.empty())
      << "No cases matched FA3_H1D128_PROFILE_CASE_LIST";

  std::vector<Fa3RunResult> results;
  for (const Fa3PrefillCase &cfg : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "case=" << cfg.name
                 << " batch=" << cfg.batch
                 << " seqlen=" << cfg.seqlen
                 << " heads=" << cfg.heads);
    ASSERT_TRUE(is_supported_fa3_prefill_case(cfg));
    ASSERT_EQ(cfg.head_dim, 128);

    Fa3RunResult result = run_fa3_prefill_fp16(cfg);
    ASSERT_EQ(result.error, cudaSuccess)
        << result.where << " failed: " << cudaGetErrorString(result.error);
    results.push_back(result);
  }
  WriteNoProfileCsv(NoProfileOutPath(), cases, results);
}

#endif

}  // namespace
}  // namespace fa3_hopper_test
