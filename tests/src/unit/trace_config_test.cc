#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "trace.h"

namespace {

class TraceConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = Trace::config_str;
    std::copy_n(Trace::trace_streams_enabled, Trace::NUM_TRACE_STREAMS,
                components_.begin());
    std::fill_n(Trace::trace_streams_enabled, Trace::NUM_TRACE_STREAMS, false);
  }

  void TearDown() override {
    Trace::config_str = config_;
    std::copy(components_.begin(), components_.end(),
              Trace::trace_streams_enabled);
  }

  const char *config_ = nullptr;
  std::array<bool, Trace::NUM_TRACE_STREAMS> components_{};
};

TEST_F(TraceConfigTest, SelectsWgmmaWithoutEnablingMma) {
  Trace::config_str = "WGMMA_RF_TRAFFIC";
  Trace::init();
  EXPECT_TRUE(Trace::trace_streams_enabled[Trace::WGMMA_RF_TRAFFIC]);
  EXPECT_FALSE(Trace::trace_streams_enabled[Trace::MMA]);
  EXPECT_EQ(1, std::count(std::begin(Trace::trace_streams_enabled),
                          std::end(Trace::trace_streams_enabled), true));
}

TEST_F(TraceConfigTest, AcceptsCommaSeparatedNamesAndIgnoresPartialMatches) {
  Trace::config_str = " MMA , PTX_IR , NOT_MBAR ,";
  Trace::init();
  EXPECT_TRUE(Trace::trace_streams_enabled[Trace::MMA]);
  EXPECT_TRUE(Trace::trace_streams_enabled[Trace::PTX_IR]);
  EXPECT_FALSE(Trace::trace_streams_enabled[Trace::MBAR]);
  EXPECT_EQ(2, std::count(std::begin(Trace::trace_streams_enabled),
                          std::end(Trace::trace_streams_enabled), true));
}

TEST_F(TraceConfigTest, ReinitializationReplacesComponentSelection) {
  Trace::config_str = "MMA,TCGEN05";
  Trace::init();
  Trace::config_str = "TMA";
  Trace::init();
  EXPECT_TRUE(Trace::trace_streams_enabled[Trace::TMA]);
  EXPECT_FALSE(Trace::trace_streams_enabled[Trace::MMA]);
  EXPECT_FALSE(Trace::trace_streams_enabled[Trace::TCGEN05]);
  Trace::config_str = nullptr;
  Trace::init();
  EXPECT_EQ(0, std::count(std::begin(Trace::trace_streams_enabled),
                          std::end(Trace::trace_streams_enabled), true));
}

}  // namespace
