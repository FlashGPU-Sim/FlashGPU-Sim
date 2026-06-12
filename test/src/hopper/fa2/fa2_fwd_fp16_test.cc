#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "fa2_fwd_fp16_case.cuh"

namespace fa2_hopper_test {

class Fa2PrefillFp16IntegrationTest : public ::testing::Test {};
class Fa2FwdFp16SmokeIntegrationTest : public ::testing::Test {};

inline bool run_32ki_fa2_cases_enabled() {
  const char *value = std::getenv("FA2_RUN_32KI");
  return value != nullptr && std::string(value) == "1";
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

TEST_F(Fa2PrefillFp16IntegrationTest, ShapeTableHas20PrefillCases) {
  ASSERT_EQ(sizeof(kFa2PrefillCases) / sizeof(kFa2PrefillCases[0]),
            size_t{kFa2PrefillCaseCount});

  for (const Fa2PrefillCase &cfg : kFa2PrefillCases) {
    EXPECT_TRUE(is_valid_fa2_prefill_case(cfg))
        << cfg.name << " is not a valid 32Ki-token prefill case";
  }
}

#define FA2_PREFILL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa2PrefillFp16IntegrationTest, name) {                        \
    RunFa2PrefillCase(                                                 \
        Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal}); \
  }

FA2_PREFILL_CASE_LIST(FA2_PREFILL_TEST)

#undef FA2_PREFILL_TEST

TEST_F(Fa2FwdFp16SmokeIntegrationTest, SmallForwardCase) {
  Fa2RunResult result = run_fa2_fwd_smoke_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

}  // namespace fa2_hopper_test
