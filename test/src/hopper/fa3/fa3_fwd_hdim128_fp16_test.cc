#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "fa3_fwd_hdim128_fp16_case.cuh"

namespace fa3_hopper_test {

class Fa3PrefillFp16IntegrationTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardIntegrationTest : public ::testing::Test {};
class Fa3PrefillFp16SmokeTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardSmokeTest : public ::testing::Test {};
class Fa3PrefillFp16SmallTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardSmallTest : public ::testing::Test {};
class Fa3PrefillFp16MediumTest : public ::testing::Test {};
class Fa3PrefillFp16BackwardMediumTest : public ::testing::Test {};
class Fa3FwdHdim128Fp16IntegrationTest : public ::testing::Test {};

inline void RunFa3PrefillCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa3PrefillSmokeCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_smoke_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa3PrefillBackwardCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16_bwd(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa3PrefillBackwardSmokeCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_smoke_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16_bwd(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa3PrefillTuningCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_tuning_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

inline void RunFa3PrefillBackwardTuningCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_tuning_case(cfg));

  Fa3RunResult result = run_fa3_prefill_fp16_bwd(cfg);
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

TEST_F(Fa3PrefillFp16IntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]),
            size_t{kFa3PrefillCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill case";
  }
}

TEST_F(Fa3PrefillFp16BackwardIntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]),
            size_t{kFa3PrefillCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill backward case";
  }
}

TEST_F(Fa3PrefillFp16SmokeTest, ShapeTableHas4SmokeCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmokeCases) /
                sizeof(kFa3PrefillSmokeCases[0]),
            size_t{kFa3PrefillSmokeCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmokeCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_smoke_case(cfg))
        << cfg.name << " is not a valid FA3 smoke case";
  }
}

TEST_F(Fa3PrefillFp16BackwardSmokeTest, ShapeTableHas4SmokeCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmokeCases) /
                sizeof(kFa3PrefillSmokeCases[0]),
            size_t{kFa3PrefillSmokeCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmokeCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_smoke_case(cfg))
        << cfg.name << " is not a valid FA3 backward smoke case";
  }
}

TEST_F(Fa3PrefillFp16SmallTest, ShapeTableHas4SmallCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmallCases) / sizeof(kFa3PrefillSmallCases[0]),
            size_t{kFa3PrefillSmallCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmallCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 small case";
  }
}

TEST_F(Fa3PrefillFp16BackwardSmallTest, ShapeTableHas4SmallCases) {
  ASSERT_EQ(sizeof(kFa3PrefillSmallCases) / sizeof(kFa3PrefillSmallCases[0]),
            size_t{kFa3PrefillSmallCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillSmallCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 backward small case";
  }
}

TEST_F(Fa3PrefillFp16MediumTest, ShapeTableHas4MediumCases) {
  ASSERT_EQ(sizeof(kFa3PrefillMediumCases) /
                sizeof(kFa3PrefillMediumCases[0]),
            size_t{kFa3PrefillMediumCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillMediumCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 medium case";
  }
}

TEST_F(Fa3PrefillFp16BackwardMediumTest, ShapeTableHas4MediumCases) {
  ASSERT_EQ(sizeof(kFa3PrefillMediumCases) /
                sizeof(kFa3PrefillMediumCases[0]),
            size_t{kFa3PrefillMediumCaseCount});

  for (const Fa3PrefillCase &cfg : kFa3PrefillMediumCases) {
    EXPECT_TRUE(is_valid_fa3_prefill_tuning_case(cfg))
        << cfg.name << " is not a valid FA3 backward medium case";
  }
}

#define FA3_PREFILL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16IntegrationTest, name) {                        \
    RunFa3PrefillCase(                                                 \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal}); \
  }

FA3_PREFILL_CASE_LIST(FA3_PREFILL_TEST)

#undef FA3_PREFILL_TEST

#define FA3_PREFILL_BWD_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16BackwardIntegrationTest, name) {                    \
    RunFa3PrefillBackwardCase(                                             \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});     \
  }

FA3_PREFILL_CASE_LIST(FA3_PREFILL_BWD_TEST)

#undef FA3_PREFILL_BWD_TEST

#define FA3_PREFILL_SMOKE_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16SmokeTest, name) {                                    \
    RunFa3PrefillSmokeCase(                                                  \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

FA3_PREFILL_SMOKE_CASE_LIST(FA3_PREFILL_SMOKE_TEST)

#undef FA3_PREFILL_SMOKE_TEST

#define FA3_PREFILL_BWD_SMOKE_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16BackwardSmokeTest, name) {                                \
    RunFa3PrefillBackwardSmokeCase(                                              \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});           \
  }

FA3_PREFILL_SMOKE_CASE_LIST(FA3_PREFILL_BWD_SMOKE_TEST)

#undef FA3_PREFILL_BWD_SMOKE_TEST

#define FA3_PREFILL_SMALL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16SmallTest, name) {                                    \
    RunFa3PrefillTuningCase(                                                 \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});       \
  }

FA3_PREFILL_SMALL_CASE_LIST(FA3_PREFILL_SMALL_TEST)

#undef FA3_PREFILL_SMALL_TEST

#define FA3_PREFILL_BWD_SMALL_TEST(name, batch, seqlen, heads, head_dim, \
                                   causal)                              \
  TEST_F(Fa3PrefillFp16BackwardSmallTest, name) {                       \
    RunFa3PrefillBackwardTuningCase(                                    \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});  \
  }

FA3_PREFILL_SMALL_CASE_LIST(FA3_PREFILL_BWD_SMALL_TEST)

#undef FA3_PREFILL_BWD_SMALL_TEST

#define FA3_PREFILL_MEDIUM_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16MediumTest, name) {                                    \
    RunFa3PrefillTuningCase(                                                  \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});        \
  }

FA3_PREFILL_MEDIUM_CASE_LIST(FA3_PREFILL_MEDIUM_TEST)

#undef FA3_PREFILL_MEDIUM_TEST

#define FA3_PREFILL_BWD_MEDIUM_TEST(name, batch, seqlen, heads, head_dim, \
                                    causal)                              \
  TEST_F(Fa3PrefillFp16BackwardMediumTest, name) {                       \
    RunFa3PrefillBackwardTuningCase(                                     \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal});   \
  }

FA3_PREFILL_MEDIUM_CASE_LIST(FA3_PREFILL_BWD_MEDIUM_TEST)

#undef FA3_PREFILL_BWD_MEDIUM_TEST

TEST_F(Fa3FwdHdim128Fp16IntegrationTest, FixedForwardCase) {
  Fa3RunResult result = run_fa3_fwd_hdim128_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

}  // namespace fa3_hopper_test
