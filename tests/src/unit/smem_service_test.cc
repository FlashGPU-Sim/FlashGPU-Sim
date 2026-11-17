#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/smem_service.h"

TEST(SmemService, TwoPhaseGrantNextCycle) {
  shared_memory_service_t s(128);
  s.expose(shared_memory_service_t::LSU, 64);
  EXPECT_EQ(s.take(shared_memory_service_t::LSU, 64), 0u);
  s.cycle();
  EXPECT_EQ(s.take(shared_memory_service_t::LSU, 64), 64u);
}

TEST(SmemService, LocalIdleDsmDoesNotStarve) {
  shared_memory_service_t s(32);
  for (int i = 0; i < 8; i++) {
    s.expose(shared_memory_service_t::LSU, 32);
    s.cycle();
    EXPECT_EQ(s.take(shared_memory_service_t::LSU, 32), 32u)
        << "LSU starved at step " << i << " with DSM idle";
  }
}

TEST(SmemService, IndependentSms) {
  shared_memory_service_t a(16);
  shared_memory_service_t b(16);
  a.expose(shared_memory_service_t::DSM, 16);
  b.expose(shared_memory_service_t::LSU, 16);
  a.cycle();
  b.cycle();
  EXPECT_EQ(a.take(shared_memory_service_t::DSM, 16), 16u);
  EXPECT_EQ(b.take(shared_memory_service_t::LSU, 16), 16u);
  EXPECT_EQ(a.take(shared_memory_service_t::LSU, 16), 0u);
}

TEST(SmemService, ShareBudgetLsuAndDsm) {
  shared_memory_service_t s(32);
  unsigned lsu = 0, dsm = 0;
  for (int i = 0; i < 4; i++) {
    s.expose(shared_memory_service_t::LSU, 32);
    s.expose(shared_memory_service_t::DSM, 32);
    s.cycle();
    lsu += s.take(shared_memory_service_t::LSU, 32);
    dsm += s.take(shared_memory_service_t::DSM, 32);
  }
  EXPECT_EQ(lsu + dsm, 128u);
  EXPECT_GT(lsu, 0u);
  EXPECT_GT(dsm, 0u);
}

TEST(SmemService, UnlimitedStillDelaysOneCycle) {
  shared_memory_service_t s(0);
  s.expose(shared_memory_service_t::TMA, 256);
  EXPECT_EQ(s.take(shared_memory_service_t::TMA, 256), 0u);
  s.cycle();
  EXPECT_EQ(s.take(shared_memory_service_t::TMA, 256), 256u);
}
