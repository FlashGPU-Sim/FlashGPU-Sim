#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "fa3_fwd_hdim128_fp16_case.cuh"

namespace fa3_hopper_test {

class Fa3PrefillFp16IntegrationTest : public ::testing::Test {};
class Fa3FwdHdim128Fp16IntegrationTest : public ::testing::Test {};

inline bool run_32ki_fa3_cases_enabled() {
  const char *value = std::getenv("FA3_RUN_32KI");
  return value != nullptr && std::string(value) == "1";
}

inline void RunFa3PrefillCase(const Fa3PrefillCase &cfg) {
  SCOPED_TRACE(::testing::Message()
               << "case=" << cfg.name
               << " batch=" << cfg.batch
               << " seqlen=" << cfg.seqlen
               << " heads=" << cfg.heads
               << " head_dim=" << cfg.head_dim
               << " causal=" << cfg.causal);

  ASSERT_TRUE(is_valid_fa3_prefill_case(cfg));

  if (!run_32ki_fa3_cases_enabled()) {
    GTEST_SKIP() << "32Ki FA3 launch cases are opt-in; set FA3_RUN_32KI=1 "
                    "to run this simulator path.";
  }

  Fa3RunResult result = run_fa3_prefill_fp16(cfg);
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

#define FA3_PREFILL_TEST(name, batch, seqlen, heads, head_dim, causal) \
  TEST_F(Fa3PrefillFp16IntegrationTest, name) {                        \
    RunFa3PrefillCase(                                                 \
        Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal}); \
  }

FA3_PREFILL_CASE_LIST(FA3_PREFILL_TEST)

#undef FA3_PREFILL_TEST

TEST_F(Fa3FwdHdim128Fp16IntegrationTest, FixedForwardCase) {
  Fa3RunResult result = run_fa3_fwd_hdim128_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

}  // namespace fa3_hopper_test
