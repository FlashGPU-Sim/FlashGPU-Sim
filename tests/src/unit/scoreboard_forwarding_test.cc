#include <gtest/gtest.h>

#include "gpgpu-sim/scoreboard.h"

namespace {
class TestInst : public warp_inst_t {
 public:
  TestInst(unsigned uid, unsigned dest = 0, unsigned source = 0) {
    m_uid = uid;
    op = SP_OP;
    for (auto &reg : out) reg = 0;
    for (auto &reg : in) reg = 0;
    out[0] = dest;
    in[0] = source;
    outcount = dest != 0;
    incount = source != 0;
    pred = ar1 = ar2 = 0;
  }
};
}  // namespace

TEST(ScoreboardForwardingTest, ReadyResultRemainsPendingUntilWriteback) {
  Scoreboard scoreboard(0, 1, nullptr);
  TestInst write(1, 7), read(2, 0, 7);
  scoreboard.reserveRegistersForWarp(&write, 0);
  EXPECT_TRUE(scoreboard.checkCollision(0, &read));
  EXPECT_EQ(scoreboard.getCollisionType(0, &read), PROD_SP_INT);
  scoreboard.markRegistersReadyForWarp(0, write.get_uid(), write.out);
  EXPECT_FALSE(scoreboard.checkCollision(0, &read));
  EXPECT_EQ(scoreboard.getCollisionType(0, &read), PROD_OTHER);
  EXPECT_TRUE(scoreboard.pendingWrites(0));
  scoreboard.releaseRegistersForWarp(&write, 0);
  EXPECT_FALSE(scoreboard.pendingWrites(0));
}

TEST(ScoreboardForwardingTest, OlderEventsCannotReleaseYoungerWriter) {
  Scoreboard scoreboard(0, 1, nullptr);
  TestInst older(1, 7), younger(2, 7), read(3, 0, 7);
  scoreboard.reserveRegistersForWarp(&older, 0);
  EXPECT_TRUE(scoreboard.checkCollision(0, &younger));
  scoreboard.markRegistersReadyForWarp(0, older.get_uid(), older.out);
  EXPECT_FALSE(scoreboard.checkCollision(0, &younger));
  scoreboard.reserveRegistersForWarp(&younger, 0);
  scoreboard.releaseRegistersForWarp(&older, 0);
  scoreboard.markRegistersReadyForWarp(0, older.get_uid(), older.out);
  EXPECT_TRUE(scoreboard.checkCollision(0, &read));
  EXPECT_TRUE(scoreboard.pendingWrites(0));
  scoreboard.markRegistersReadyForWarp(0, younger.get_uid(), younger.out);
  EXPECT_FALSE(scoreboard.checkCollision(0, &read));
  scoreboard.releaseRegistersForWarp(&younger, 0);
  EXPECT_FALSE(scoreboard.pendingWrites(0));
}

TEST(ScoreboardForwardingTest, CompletedEventCannotAffectReusedWarpRegisters) {
  Scoreboard scoreboard(0, 1, nullptr);
  TestInst older(1, 7), newer(2, 7), read(3, 0, 7);
  scoreboard.reserveRegistersForWarp(&older, 0);
  scoreboard.releaseRegistersForWarp(&older, 0);
  scoreboard.reserveRegistersForWarp(&newer, 0);
  scoreboard.markRegistersReadyForWarp(0, older.get_uid(), older.out);
  scoreboard.releaseRegistersForWarp(&older, 0);
  EXPECT_TRUE(scoreboard.checkCollision(0, &read));
}

TEST(ScoreboardForwardingTest, PredicateAddressAndMultipleOutputsAreTracked) {
  Scoreboard scoreboard(0, 2, nullptr);
  TestInst write(1, 7), pred_read(2), address_read(3);
  write.out[1] = 8;
  write.outcount = 2;
  pred_read.pred = 7;
  address_read.ar1 = 8;
  scoreboard.reserveRegistersForWarp(&write, 0);
  EXPECT_TRUE(scoreboard.checkCollision(0, &pred_read));
  EXPECT_TRUE(scoreboard.checkCollision(0, &address_read));
  EXPECT_FALSE(scoreboard.checkCollision(1, &pred_read));
  scoreboard.markRegistersReadyForWarp(0, write.get_uid(), write.out);
  EXPECT_FALSE(scoreboard.checkCollision(0, &pred_read));
  EXPECT_FALSE(scoreboard.checkCollision(0, &address_read));
  scoreboard.releaseRegistersForWarp(&write, 0);
  EXPECT_FALSE(scoreboard.pendingWrites(0));
}

TEST(ScoreboardForwardingTest, WithoutForwardEventLegacyHazardsRemain) {
  Scoreboard scoreboard(0, 1, nullptr);
  TestInst load(1, 7), read(2, 0, 7);
  load.op = LOAD_OP;
  load.space = global_space;
  scoreboard.reserveRegistersForWarp(&load, 0);
  EXPECT_TRUE(scoreboard.islongop(0, 7));
  EXPECT_TRUE(scoreboard.checkCollision(0, &read));
  scoreboard.releaseRegister(0, 7);
  EXPECT_FALSE(scoreboard.checkCollision(0, &read));
  EXPECT_FALSE(scoreboard.islongop(0, 7));
  EXPECT_FALSE(scoreboard.pendingWrites(0));
}

TEST(ScoreboardForwardingTest, NominalAndDelayedExecutionDetermineReadyCycle) {
  EXPECT_EQ(scoreboard_forward_ready_cycle(100, 102, 4), 104ULL);
  EXPECT_EQ(scoreboard_forward_ready_cycle(100, 102, 18), 118ULL);
  EXPECT_EQ(scoreboard_forward_ready_cycle(100, 120, 18), 136ULL);
  EXPECT_EQ(scoreboard_forward_ready_cycle(100, 102, 1), 102ULL);
}
