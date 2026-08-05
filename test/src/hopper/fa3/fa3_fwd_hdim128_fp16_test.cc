#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fa3_fwd_hdim128_fp16_case.cuh"

#if defined(FA3_STANDARD_FORWARD_TU) || \
    defined(FA3_STANDARD_BACKWARD_TU)
#if !defined(FA3_STANDARD_HEAD_DIM) || !defined(FA3_STANDARD_CAUSAL)
#error "FA3 standard split TU requires HEAD_DIM and CAUSAL selectors"
#endif
#if FA3_STANDARD_CAUSAL != 0 && FA3_STANDARD_CAUSAL != 1
#error "FA3_STANDARD_CAUSAL must be 0 or 1"
#endif
#if FA3_STANDARD_HEAD_DIM == 64 && FA3_STANDARD_CAUSAL == 0
#define FA3_STANDARD_PREFILL_CASE_LIST(X) \
  FA3_PREFILL_D64_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMOKE_CASE_LIST(X) \
  FA3_PREFILL_SMOKE_D64_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMALL_CASE_LIST(X) \
  FA3_PREFILL_SMALL_D64_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_MEDIUM_CASE_LIST(X) \
  FA3_PREFILL_MEDIUM_D64_NONCAUSAL_CASE_LIST(X)
#elif FA3_STANDARD_HEAD_DIM == 64 && FA3_STANDARD_CAUSAL == 1
#define FA3_STANDARD_PREFILL_CASE_LIST(X) \
  FA3_PREFILL_D64_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMOKE_CASE_LIST(X) \
  FA3_PREFILL_SMOKE_D64_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMALL_CASE_LIST(X) \
  FA3_PREFILL_SMALL_D64_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_MEDIUM_CASE_LIST(X) \
  FA3_PREFILL_MEDIUM_D64_CAUSAL_CASE_LIST(X)
#elif FA3_STANDARD_HEAD_DIM == 128 && FA3_STANDARD_CAUSAL == 0
#define FA3_STANDARD_PREFILL_CASE_LIST(X) \
  FA3_PREFILL_D128_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMOKE_CASE_LIST(X) \
  FA3_PREFILL_SMOKE_D128_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMALL_CASE_LIST(X) \
  FA3_PREFILL_SMALL_D128_NONCAUSAL_CASE_LIST(X)
#define FA3_STANDARD_MEDIUM_CASE_LIST(X) \
  FA3_PREFILL_MEDIUM_D128_NONCAUSAL_CASE_LIST(X)
#elif FA3_STANDARD_HEAD_DIM == 128 && FA3_STANDARD_CAUSAL == 1
#define FA3_STANDARD_PREFILL_CASE_LIST(X) \
  FA3_PREFILL_D128_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMOKE_CASE_LIST(X) \
  FA3_PREFILL_SMOKE_D128_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_SMALL_CASE_LIST(X) \
  FA3_PREFILL_SMALL_D128_CAUSAL_CASE_LIST(X)
#define FA3_STANDARD_MEDIUM_CASE_LIST(X) \
  FA3_PREFILL_MEDIUM_D128_CAUSAL_CASE_LIST(X)
#else
#error "FA3_STANDARD_HEAD_DIM must be 64 or 128"
#endif
#else
#define FA3_STANDARD_PREFILL_CASE_LIST(X) FA3_PREFILL_CASE_LIST(X)
#define FA3_STANDARD_SMOKE_CASE_LIST(X) FA3_PREFILL_SMOKE_CASE_LIST(X)
#define FA3_STANDARD_SMALL_CASE_LIST(X) FA3_PREFILL_SMALL_CASE_LIST(X)
#define FA3_STANDARD_MEDIUM_CASE_LIST(X) FA3_PREFILL_MEDIUM_CASE_LIST(X)
#define FA3_STANDARD_SHAPE_TESTS
#define FA3_STANDARD_FIXED_TESTS
#endif

namespace fa3_hopper_test {

#if !defined(FA3_STANDARD_BACKWARD_TU)
class Fa3PrefillFp16IntegrationTest : public ::testing::Test {};
class Fa3PrefillFp16SmokeTest : public ::testing::Test {};
class Fa3PrefillFp16SmallTest : public ::testing::Test {};
class Fa3PrefillFp16MediumTest : public ::testing::Test {};
class Fa3FwdHdim128Fp16IntegrationTest : public ::testing::Test {};
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
class Fa3PrefillFp16BackwardIntegrationTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardSmokeTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardSmallTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardMediumTest : public ::testing::Test {};
#endif

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK) && \
    !defined(FA3_STANDARD_BACKWARD_TU)
class Fa3SingleTileProfileTest : public ::testing::Test {};
class Fa3PrefillProfileTest : public ::testing::Test {};

inline std::vector<int> ParseSingleTileSkList() {
  const char *env = std::getenv("FA3_SINGLE_TILE_SK_LIST");
  std::string list = env == nullptr || std::string(env).empty() ? "512" : env;
  for (char &ch : list) {
    if (ch == ',') ch = ' ';
  }
  std::istringstream is(list);
  std::vector<int> out;
  int value = 0;
  while (is >> value) {
    out.push_back(value);
  }
  if (out.empty()) out.push_back(512);
  return out;
}

inline std::string SingleTileProfileOutPath() {
  const char *env = std::getenv("FA3_SINGLE_TILE_PROFILE_OUT");
  return env == nullptr || std::string(env).empty()
             ? "fa3_single_tile_profile.csv"
             : std::string(env);
}

inline std::vector<int> ParsePrefillProfileSeqlenList() {
  const char *env = std::getenv("FA3_PREFILL_PROFILE_S_LIST");
  std::string list =
      env == nullptr || std::string(env).empty()
          ? "512,1024,2048,4096,8192"
          : env;
  for (char &ch : list) {
    if (ch == ',') ch = ' ';
  }
  std::istringstream is(list);
  std::vector<int> out;
  int value = 0;
  while (is >> value) {
    out.push_back(value);
  }
  if (out.empty()) out = {512, 1024, 2048, 4096, 8192};
  return out;
}

inline bool HasPrefillProfileSeqlenOverride() {
  const char *env = std::getenv("FA3_PREFILL_PROFILE_S_LIST");
  return env != nullptr && !std::string(env).empty();
}

inline std::vector<std::string> ParsePrefillProfileCaseList() {
  const char *env = std::getenv("FA3_PREFILL_PROFILE_CASE_LIST");
  std::vector<std::string> cases;
  if (env == nullptr || std::string(env).empty()) return cases;
  std::string list = env;
  for (char &ch : list) {
    if (ch == ',') ch = ' ';
  }
  std::istringstream is(list);
  std::string value;
  while (is >> value) {
    std::string trimmed;
    for (char ch : value) {
      if (!std::isspace(static_cast<unsigned char>(ch))) trimmed.push_back(ch);
    }
    if (!trimmed.empty()) cases.push_back(trimmed);
  }
  return cases;
}

inline std::string PrefillProfileOutPath() {
  const char *env = std::getenv("FA3_PREFILL_PROFILE_OUT");
  return env == nullptr || std::string(env).empty()
             ? "fa3_prefill_h16d128_full_profile.csv"
             : std::string(env);
}

inline Fa3PrefillCase H16D128FullProfileCaseForSeqlen(int seqlen) {
  switch (seqlen) {
    case 512:
      return Fa3PrefillCase{"H16D128FullB64S512", 64, 512, 16, 128, false};
    case 1024:
      return Fa3PrefillCase{"H16D128FullB32S1024", 32, 1024, 16, 128, false};
    case 2048:
      return Fa3PrefillCase{"H16D128FullB16S2048", 16, 2048, 16, 128, false};
    case 4096:
      return Fa3PrefillCase{"H16D128FullB8S4096", 8, 4096, 16, 128, false};
    case 8192:
      return Fa3PrefillCase{"H16D128FullB4S8192", 4, 8192, 16, 128, false};
    default:
      return Fa3PrefillCase{"H16D128FullCustom", 32768 / seqlen, seqlen, 16,
                            128, false};
  }
}

inline std::vector<Fa3PrefillCase> DefaultPrefillProfileCases() {
  return {
      // Original H16D128 full-prefill cases with B*S fixed at 32768.
      {"H16D128FullB64S512", 64, 512, 16, 128, false},
      {"H16D128FullB32S1024", 32, 1024, 16, 128, false},
      {"H16D128FullB16S2048", 16, 2048, 16, 128, false},
      {"H16D128FullB8S4096", 8, 4096, 16, 128, false},
      {"H16D128FullB4S8192", 4, 8192, 16, 128, false},

      // Reduce H only while keeping the original B/S points.
      {"H4D128FullB64S512", 64, 512, 4, 128, false},
      {"H4D128FullB32S1024", 32, 1024, 4, 128, false},
      {"H4D128FullB16S2048", 16, 2048, 4, 128, false},
      {"H4D128FullB8S4096", 8, 4096, 4, 128, false},
      {"H4D128FullB4S8192", 4, 8192, 4, 128, false},

      // Reduce B only while keeping H=16 and the same S points.
      {"H16D128FullB16S512", 16, 512, 16, 128, false},
      {"H16D128FullB8S1024", 8, 1024, 16, 128, false},
      {"H16D128FullB4S2048", 4, 2048, 16, 128, false},
      {"H16D128FullB2S4096", 2, 4096, 16, 128, false},
      {"H16D128FullB1S8192", 1, 8192, 16, 128, false},

      // Minimal-H/B long-S cases for later simulator repro attempts.
      {"H1D128FullB1S512", 1, 512, 1, 128, false},
      {"H1D128FullB1S1024", 1, 1024, 1, 128, false},
      {"H1D128FullB1S2048", 1, 2048, 1, 128, false},
      {"H1D128FullB1S4096", 1, 4096, 1, 128, false},
      {"H1D128FullB1S8192", 1, 8192, 1, 128, false},

      // S-reduced cases bracketing the smallest original S=512 point.
      {"H16D128FullB64S128", 64, 128, 16, 128, false},
      {"H16D128FullB64S256", 64, 256, 16, 128, false},
      {"H1D128FullB1S128", 1, 128, 1, 128, false},
      {"H1D128FullB1S256", 1, 256, 1, 128, false},
  };
}

inline std::vector<Fa3PrefillCase> PrefillProfileCases() {
  std::vector<Fa3PrefillCase> cases;
  if (!HasPrefillProfileSeqlenOverride()) {
    cases = DefaultPrefillProfileCases();
  } else {
    for (int seqlen : ParsePrefillProfileSeqlenList()) {
      cases.push_back(H16D128FullProfileCaseForSeqlen(seqlen));
    }
  }

  std::vector<std::string> selected_names = ParsePrefillProfileCaseList();
  if (selected_names.empty()) return cases;

  std::vector<Fa3PrefillCase> selected_cases;
  for (const Fa3PrefillCase &cfg : cases) {
    for (const std::string &name : selected_names) {
      if (cfg.name == name) selected_cases.push_back(cfg);
    }
  }
  return selected_cases;
}

inline void WritePrefillProfileCsv(
    const std::string &path,
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

inline void WriteSingleTileProfileCsv(
    const std::string &path,
    const std::vector<Fa3SingleTileProfileResult> &results) {
  std::ofstream out(path);
  ASSERT_TRUE(out) << "failed to open " << path;
  out << "case,seqlen_q,seqlen_k,heads,head_dim,causal,block_m,block_n,"
         "k_tiles,clock_start,clock_end,clock_delta,mainloop_start,"
         "mainloop_end,mainloop_delta,epilogue_start,epilogue_end,"
         "epilogue_delta,qk_wait_cycles,qk_wgmma_issue_cycles,"
         "softmax_cycles,pv_wait_cycles,pv_wgmma_issue_wait_cycles,"
         "mainloop_iterations,output0,lse0\n";
  for (const auto &result : results) {
    out << "H16D128FullSq128Sk" << result.seqlen_k << ","
        << result.seqlen_q << ","
        << result.seqlen_k << ","
        << result.heads << ","
        << result.head_dim << ","
        << 0 << ","
        << result.block_m << ","
        << result.block_n << ","
        << result.k_tiles << ","
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
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
static Fa3RunResult RunFa3ForwardKernel(const Fa3PrefillCase &cfg) {
#if defined(FA3_STANDARD_FORWARD_TU)
  return run_fa3_prefill_fp16_typed<FA3_STANDARD_HEAD_DIM,
                                    FA3_STANDARD_CAUSAL != 0>(cfg);
#else
  return run_fa3_prefill_fp16(cfg);
#endif
}

static void RunFa3PrefillCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_case(cfg));

  Fa3RunResult result = RunFa3ForwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

static void RunFa3PrefillSmokeCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_smoke_case(cfg));

  Fa3RunResult result = RunFa3ForwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
static Fa3RunResult RunFa3BackwardKernel(const Fa3PrefillCase &cfg) {
#if defined(FA3_STANDARD_BACKWARD_TU)
  return run_fa3_prefill_fp16_bwd_typed<FA3_STANDARD_HEAD_DIM,
                                        FA3_STANDARD_CAUSAL != 0>(cfg);
#else
  return run_fa3_prefill_fp16_bwd(cfg);
#endif
}

static void RunFa3PrefillBackwardCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_case(cfg));

  Fa3RunResult result = RunFa3BackwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

static void RunFa3PrefillBackwardSmokeCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_smoke_case(cfg));

  Fa3RunResult result = RunFa3BackwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
static void RunFa3PrefillTuningCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_tuning_case(cfg));

  Fa3RunResult result = RunFa3ForwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
static void RunFa3PrefillBackwardTuningCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_tuning_case(cfg));

  Fa3RunResult result = RunFa3BackwardKernel(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16IntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]),
            size_t{kFa3PrefillCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill case";
  }
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16BackwardIntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]),
            size_t{kFa3PrefillCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill backward case";
  }
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16SmokeTest, ShapeTableHas4SmokeCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmokeCases) /
                sizeof(kFa3PrefillSmokeCases[0]),
            size_t{kFa3PrefillSmokeCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmokeCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_smoke_case(cfg))
        << cfg.name << " is not a valid FA3 smoke case";
  }
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16BackwardSmokeTest, ShapeTableHas4SmokeCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmokeCases) /
                sizeof(kFa3PrefillSmokeCases[0]),
            size_t{kFa3PrefillSmokeCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmokeCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_smoke_case(cfg))
        << cfg.name << " is not a valid FA3 backward smoke case";
  }
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16SmallTest, ShapeTableHas4SmallCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmallCases) / sizeof(kFa3PrefillSmallCases[0]),
            size_t{kFa3PrefillSmallCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmallCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 small case";
  }
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16BackwardSmallTest, ShapeTableHas4SmallCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmallCases) / sizeof(kFa3PrefillSmallCases[0]),
            size_t{kFa3PrefillSmallCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmallCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 backward small case";
  }
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16MediumTest, ShapeTableHas4MediumCases) {
  ASSERT_EQ(sizeof(kFa3PrefillMediumCases) /
                sizeof(kFa3PrefillMediumCases[0]),
            size_t{kFa3PrefillMediumCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillMediumCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 medium case";
  }
}
#endif

#if !defined(FA3_STANDARD_FORWARD_TU) && \
    defined(FA3_STANDARD_SHAPE_TESTS)
TEST_F(Fa3PrefillFp16BackwardMediumTest, ShapeTableHas4MediumCases) {
  ASSERT_EQ(sizeof(kFa3PrefillMediumCases) /
                sizeof(kFa3PrefillMediumCases[0]),
            size_t{kFa3PrefillMediumCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillMediumCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 backward medium case";
  }
}
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
#define FA3_PREFILL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16IntegrationTest, name) {                        \
    RunFa3PrefillCase(                                                 \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal}); \
  }

FA3_STANDARD_PREFILL_CASE_LIST(FA3_PREFILL_TEST)

#undef FA3_PREFILL_TEST
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
#define FA3_PREFILL_BWD_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16BackwardIntegrationTest, name) {                    \
    RunFa3PrefillBackwardCase(                                             \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});     \
  }

FA3_STANDARD_PREFILL_CASE_LIST(FA3_PREFILL_BWD_TEST)

#undef FA3_PREFILL_BWD_TEST
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
#define FA3_PREFILL_SMOKE_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16SmokeTest, name) {                                    \
    RunFa3PrefillSmokeCase(                                                  \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

FA3_STANDARD_SMOKE_CASE_LIST(FA3_PREFILL_SMOKE_TEST)

#undef FA3_PREFILL_SMOKE_TEST
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
#define FA3_PREFILL_BWD_SMOKE_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16BackwardSmokeTest, name) {                                \
    RunFa3PrefillBackwardSmokeCase(                                              \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});           \
  }

FA3_STANDARD_SMOKE_CASE_LIST(FA3_PREFILL_BWD_SMOKE_TEST)

#undef FA3_PREFILL_BWD_SMOKE_TEST
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
#define FA3_PREFILL_SMALL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16SmallTest, name) {                                    \
    RunFa3PrefillTuningCase(                                                 \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

FA3_STANDARD_SMALL_CASE_LIST(FA3_PREFILL_SMALL_TEST)

#undef FA3_PREFILL_SMALL_TEST
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
#define FA3_PREFILL_BWD_SMALL_TEST(name, batch, seqlen, heads, head_dim, \
                                   causal)                              \
  TEST_F(Fa3PrefillFp16BackwardSmallTest, name) {                       \
    RunFa3PrefillBackwardTuningCase(                                    \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});  \
  }

FA3_STANDARD_SMALL_CASE_LIST(FA3_PREFILL_BWD_SMALL_TEST)

#undef FA3_PREFILL_BWD_SMALL_TEST
#endif

#if !defined(FA3_STANDARD_BACKWARD_TU)
#define FA3_PREFILL_MEDIUM_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16MediumTest, name) {                                    \
    RunFa3PrefillTuningCase(                                                  \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});        \
  }

FA3_STANDARD_MEDIUM_CASE_LIST(FA3_PREFILL_MEDIUM_TEST)

#undef FA3_PREFILL_MEDIUM_TEST
#endif

#if !defined(FA3_STANDARD_FORWARD_TU)
#define FA3_PREFILL_BWD_MEDIUM_TEST(name, batch, seqlen, heads, head_dim, \
                                    causal)                              \
  TEST_F(Fa3PrefillFp16BackwardMediumTest, name) {                       \
    RunFa3PrefillBackwardTuningCase(                                     \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});   \
  }

FA3_STANDARD_MEDIUM_CASE_LIST(FA3_PREFILL_BWD_MEDIUM_TEST)

#undef FA3_PREFILL_BWD_MEDIUM_TEST
#endif

#if defined(FA3_STANDARD_FIXED_TESTS)
TEST_F(Fa3FwdHdim128Fp16IntegrationTest, FixedForwardCase) {
  Fa3RunResult result = run_fa3_fwd_hdim128_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if defined(FLASH_FWD_ENABLE_PROFILE_CLOCK) && \
    !defined(FA3_STANDARD_BACKWARD_TU)
TEST_F(Fa3SingleTileProfileTest, H16D128FullSq128Sweep) {
  std::vector<Fa3SingleTileProfileResult> results;
  for (int seqlen_k : ParseSingleTileSkList()) {
    SCOPED_TRACE(::testing::Message() << "seqlen_k=" << seqlen_k);
    Fa3SingleTileProfileResult result =
        run_fa3_single_tile_hdim128_fp16_full(seqlen_k);
    ASSERT_EQ(result.error, cudaSuccess)
        << result.where << " failed: " << cudaGetErrorString(result.error);
    ASSERT_GT(result.clock_delta, uint64_t{0})
        << "clock64 timestamps were not written";
    results.push_back(result);
  }
  WriteSingleTileProfileCsv(SingleTileProfileOutPath(), results);
}

TEST_F(Fa3PrefillProfileTest, H16D128FullSqSweep) {
  std::vector<Fa3PrefillProfileResult> results;
  for (const Fa3PrefillCase &cfg : PrefillProfileCases()) {
    SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
    ASSERT_TRUE(is_supported_fa3_prefill_case(cfg));
    if (HasPrefillProfileSeqlenOverride()) {
      ASSERT_EQ(cfg.batch * cfg.seqlen, 32768)
          << "FA3_PREFILL_PROFILE_S_LIST keeps B*S fixed at 32768";
    }
    Fa3PrefillProfileResult result = run_fa3_prefill_profile_fp16(cfg);
    ASSERT_EQ(result.error, cudaSuccess)
        << result.where << " failed: " << cudaGetErrorString(result.error);
    if (fa3_prefill_profile_clock_enabled()) {
      ASSERT_GT(result.clock_delta, uint64_t{0})
          << "clock64 timestamps were not written";
    }
    results.push_back(result);
  }
  WritePrefillProfileCsv(PrefillProfileOutPath(), results);
}
#endif

}  // namespace fa3_hopper_test
