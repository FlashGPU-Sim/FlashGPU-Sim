#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "fa2_fwd_fp16_case.cuh"

namespace fa2_hopper_test {

#if !defined(FA2_PREFILL_GROUP_SMOKE) && \
    !defined(FA2_PREFILL_GROUP_SMALL) && \
    !defined(FA2_PREFILL_GROUP_MEDIUM) && \
    !defined(FA2_PREFILL_GROUP_LARGE) && \
    !defined(FA2_PREFILL_GROUP_BREAKDOWN) && \
    !defined(FA2_PREFILL_GROUP_SCALING) && \
    !defined(FA2_PREFILL_GROUP_CONCURRENCY)
#define FA2_PREFILL_GROUP_SMOKE
#define FA2_PREFILL_GROUP_SMALL
#define FA2_PREFILL_GROUP_MEDIUM
#define FA2_PREFILL_GROUP_LARGE
#endif

class Fa2PrefillFp16IntegrationTest : public ::testing::Test {};
class Fa2PrefillFp16SmokeTest : public ::testing::Test {};
class Fa2PrefillFp16SmallTest : public ::testing::Test {};
class Fa2PrefillFp16MediumTest : public ::testing::Test {};
class Fa2PrefillFp16BreakdownTest : public ::testing::Test {};
class Fa2PrefillFp16ScalingTest : public ::testing::Test {};
class Fa2PrefillFp16ConcurrencyTest : public ::testing::Test {};
class Fa2FwdFp16SmokeIntegrationTest : public ::testing::Test {};

inline bool run_32ki_fa2_cases_enabled() {
  const char *value = std::getenv("FA2_RUN_32KI");
  return value != nullptr && std::string(value) == "1";
}

inline void ExpectFa2ReferenceMatch(const Fa2RunResult &result) {
  constexpr float kOutputAbsTolerance = 5.0e-2f;
  constexpr float kLseAbsTolerance = 5.0e-2f;
  ASSERT_TRUE(result.reference_checked);
  EXPECT_LE(result.max_output_abs_error, kOutputAbsTolerance)
      << "max output abs error at linear index "
      << result.max_output_abs_error_index << ", output0=" << result.output0
      << ", output0_ref=" << result.output0_ref;
  EXPECT_LE(result.max_lse_abs_error, kLseAbsTolerance)
      << "max LSE abs error at linear index " << result.max_lse_abs_error_index
      << ", lse0=" << result.lse0 << ", lse0_ref=" << result.lse0_ref;
}

inline void RunFa2PrefillCase(const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_case(cfg));

  if (!run_32ki_fa2_cases_enabled()) {
    GTEST_SKIP() << "32Ki FA2 launch cases are opt-in; set FA2_RUN_32KI=1 "
                    "to run this simulator path.";
  }

  Fa2RunResult result = run_fa2_prefill_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa2PrefillSmokeCase(const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_smoke_case(cfg));

  Fa2RunResult result =
      run_fa2_prefill_fp16(cfg, /*initialize_inputs=*/true,
                           /*validate_reference=*/true);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
  ExpectFa2ReferenceMatch(result);
}

inline void RunFa2PrefillTuningCase(const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_tuning_case(cfg));

  Fa2RunResult result = run_fa2_prefill_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

#if defined(FA2_PREFILL_GROUP_BREAKDOWN)
inline void RunFa2PrefillBreakdownCase(const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_breakdown_case(cfg));

  Fa2RunResult result = run_fa2_breakdown_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if defined(FA2_PREFILL_GROUP_SCALING)
inline void RunFa2PrefillScalingH1D128Case(const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_scaling_h1d128_case(cfg));

  Fa2RunResult result = run_fa2_scaling_h1d128_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if defined(FA2_PREFILL_GROUP_CONCURRENCY)
inline void RunFa2PrefillConcurrencyD128FullCase(
    const Fa2PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa2_prefill_concurrency_d128_full_case(cfg));

  if (!run_32ki_fa2_cases_enabled()) {
    GTEST_SKIP() << "32Ki FA2 concurrency launch cases are opt-in; set "
                    "FA2_RUN_32KI=1 to run this simulator path.";
  }

  Fa2RunResult result = run_fa2_concurrency_d128_full_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}
#endif

#if defined(FA2_PREFILL_GROUP_LARGE)
TEST_F(Fa2PrefillFp16IntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa2PrefillCases) / sizeof(kFa2PrefillCases[0]),
            size_t{kFa2PrefillCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_SMOKE)
TEST_F(Fa2PrefillFp16SmokeTest, ShapeTableHas4SmokeCases) {
  ASSERT_EQ(sizeof(kFa2PrefillSmokeCases) /
                sizeof(kFa2PrefillSmokeCases[0]),
            size_t{kFa2PrefillSmokeCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillSmokeCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_smoke_case(cfg))
        << cfg.name << " is not a valid FA2 smoke case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_SMALL)
TEST_F(Fa2PrefillFp16SmallTest, ShapeTableHas4SmallCases) {
  ASSERT_EQ(sizeof(kFa2PrefillSmallCases) / sizeof(kFa2PrefillSmallCases[0]),
            size_t{kFa2PrefillSmallCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillSmallCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA2 small case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_MEDIUM)
TEST_F(Fa2PrefillFp16MediumTest, ShapeTableHas4MediumCases) {
  ASSERT_EQ(sizeof(kFa2PrefillMediumCases) /
                sizeof(kFa2PrefillMediumCases[0]),
            size_t{kFa2PrefillMediumCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillMediumCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA2 medium case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_BREAKDOWN)
TEST_F(Fa2PrefillFp16BreakdownTest, ShapeTableHas1BreakdownCase) {
  ASSERT_EQ(sizeof(kFa2PrefillBreakdownCases) /
                sizeof(kFa2PrefillBreakdownCases[0]),
            size_t{kFa2PrefillBreakdownCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillBreakdownCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_breakdown_case(cfg))
        << cfg.name << " is not a valid FA2 breakdown case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_SCALING)
TEST_F(Fa2PrefillFp16ScalingTest, ShapeTableHas10ScalingCases) {
  ASSERT_EQ(sizeof(kFa2PrefillScalingH1D128Cases) /
                sizeof(kFa2PrefillScalingH1D128Cases[0]),
            size_t{kFa2PrefillScalingH1D128CaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillScalingH1D128Cases) {
    EXPECT_TRUE(is_valid_fa2_prefill_scaling_h1d128_case(cfg))
        << cfg.name << " is not a valid FA2 H1D128 scaling case";
  }
}
#endif

#if defined(FA2_PREFILL_GROUP_CONCURRENCY)
TEST_F(Fa2PrefillFp16ConcurrencyTest, ShapeTableHas1ConcurrencyCase) {
  ASSERT_EQ(sizeof(kFa2PrefillConcurrencyD128FullCases) /
                sizeof(kFa2PrefillConcurrencyD128FullCases[0]),
            size_t{kFa2PrefillConcurrencyD128FullCaseCount});

  for (const Fa2PrefillCase &cfg :
       kFa2PrefillConcurrencyD128FullCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_concurrency_d128_full_case(cfg))
        << cfg.name << " is not a valid FA2 D128 concurrency case";
  }
}
#endif

#define FA2_PREFILL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16IntegrationTest, name) {                        \
    RunFa2PrefillCase(                                                 \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal}); \
  }

#if defined(FA2_PREFILL_GROUP_LARGE)
#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
FA2_PREFILL_H32D64_FULL_CASE_LIST(FA2_PREFILL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL)
FA2_PREFILL_H32D64_CAUSAL_CASE_LIST(FA2_PREFILL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_FULL)
FA2_PREFILL_H16D128_FULL_CASE_LIST(FA2_PREFILL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
FA2_PREFILL_H16D128_CAUSAL_CASE_LIST(FA2_PREFILL_TEST)
#endif
#endif

#undef FA2_PREFILL_TEST

#define FA2_PREFILL_SMOKE_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16SmokeTest, name) {                                    \
    RunFa2PrefillSmokeCase(                                                  \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

#if defined(FA2_PREFILL_GROUP_SMOKE)
#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
FA2_PREFILL_SMOKE_H32D64_FULL_CASE_LIST(FA2_PREFILL_SMOKE_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL)
FA2_PREFILL_SMOKE_H32D64_CAUSAL_CASE_LIST(FA2_PREFILL_SMOKE_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_FULL)
FA2_PREFILL_SMOKE_H16D128_FULL_CASE_LIST(FA2_PREFILL_SMOKE_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
FA2_PREFILL_SMOKE_H16D128_CAUSAL_CASE_LIST(FA2_PREFILL_SMOKE_TEST)
#endif
#endif

#undef FA2_PREFILL_SMOKE_TEST

#define FA2_PREFILL_SMALL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16SmallTest, name) {                                    \
    RunFa2PrefillTuningCase(                                                 \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

#if defined(FA2_PREFILL_GROUP_SMALL)
#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
FA2_PREFILL_SMALL_H32D64_FULL_CASE_LIST(FA2_PREFILL_SMALL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL)
FA2_PREFILL_SMALL_H32D64_CAUSAL_CASE_LIST(FA2_PREFILL_SMALL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_FULL)
FA2_PREFILL_SMALL_H16D128_FULL_CASE_LIST(FA2_PREFILL_SMALL_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
FA2_PREFILL_SMALL_H16D128_CAUSAL_CASE_LIST(FA2_PREFILL_SMALL_TEST)
#endif
#endif

#undef FA2_PREFILL_SMALL_TEST

#define FA2_PREFILL_MEDIUM_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16MediumTest, name) {                                    \
    RunFa2PrefillTuningCase(                                                  \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});        \
  }

#if defined(FA2_PREFILL_GROUP_MEDIUM)
#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
FA2_PREFILL_MEDIUM_H32D64_FULL_CASE_LIST(FA2_PREFILL_MEDIUM_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL)
FA2_PREFILL_MEDIUM_H32D64_CAUSAL_CASE_LIST(FA2_PREFILL_MEDIUM_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_FULL)
FA2_PREFILL_MEDIUM_H16D128_FULL_CASE_LIST(FA2_PREFILL_MEDIUM_TEST)
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
FA2_PREFILL_MEDIUM_H16D128_CAUSAL_CASE_LIST(FA2_PREFILL_MEDIUM_TEST)
#endif
#endif

#undef FA2_PREFILL_MEDIUM_TEST

#define FA2_PREFILL_BREAKDOWN_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16BreakdownTest, name) {                                      \
    RunFa2PrefillBreakdownCase(                                                    \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});            \
  }

#if defined(FA2_PREFILL_GROUP_BREAKDOWN)
FA2_PREFILL_BREAKDOWN_CASE_LIST(FA2_PREFILL_BREAKDOWN_TEST)
#endif

#undef FA2_PREFILL_BREAKDOWN_TEST

#define FA2_PREFILL_SCALING_H1D128_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16ScalingTest, name) {                                                \
    RunFa2PrefillScalingH1D128Case(                                                        \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});                    \
  }

#if defined(FA2_PREFILL_GROUP_SCALING)
FA2_PREFILL_SCALING_H1D128_CASE_LIST(FA2_PREFILL_SCALING_H1D128_TEST)
#endif

#undef FA2_PREFILL_SCALING_H1D128_TEST

#define FA2_PREFILL_CONCURRENCY_D128_FULL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16ConcurrencyTest, name) {                                                    \
    RunFa2PrefillConcurrencyD128FullCase(                                                          \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal});                            \
  }

#if defined(FA2_PREFILL_GROUP_CONCURRENCY)
FA2_PREFILL_CONCURRENCY_D128_FULL_CASE_LIST(
    FA2_PREFILL_CONCURRENCY_D128_FULL_TEST)
#endif

#undef FA2_PREFILL_CONCURRENCY_D128_FULL_TEST

#if defined(FA2_PREFILL_GROUP_SMOKE) && \
    defined(FA2_PREFILL_ENABLE_H32D64_FULL)
TEST_F(Fa2FwdFp16SmokeIntegrationTest, SmallForwardCase) {
  Fa2RunResult result = run_fa2_fwd_smoke_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
  ExpectFa2ReferenceMatch(result);
}
#endif

}  // namespace fa2_hopper_test
