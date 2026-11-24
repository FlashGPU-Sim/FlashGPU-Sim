#include <gtest/gtest.h>
#include <set>
#include "gpgpu-sim/flash/clc_queue.h"
TEST(ClcQueueTest, CancelsCompletePairsAcrossRowsAndPlanesExactlyOnce) {
  using Coord = std::array<uint32_t, 3>;
  const Coord grid{{4, 2, 2}}, shape{{2, 1, 1}};
  Coord next{{0, 0, 0}}, result{};
  for (unsigned z = 0; z < 2; ++z)
    for (unsigned y = 0; y < 2; ++y)
      for (unsigned x = 0; x < 4; x += 2) {
        ASSERT_TRUE(flash_gpgpu_sim::cancel_pending_clc_cluster(grid, shape,
                                                                next, result));
        EXPECT_EQ(result, (Coord{{x, y, z}}));
      }
  const auto terminal = next;
  result = {{99, 98, 97}};
  EXPECT_FALSE(
      flash_gpgpu_sim::cancel_pending_clc_cluster(grid, shape, next, result));
  EXPECT_EQ(next, terminal);
  EXPECT_EQ(result, (Coord{{99, 98, 97}}));
}
TEST(ClcQueueTest,
     RejectsPartialClusterAndUnrepresentableResponseWithoutMutation) {
  using Coord = std::array<uint32_t, 3>;
  const Coord shape{{2, 1, 1}};
  for (Coord next : {Coord{{1, 0, 0}}, Coord{{3, 1, 0}}, Coord{{0, 65536, 0}},
                     Coord{{0, 0, 65536}}}) {
    const auto original = next;
    Coord result{{99, 98, 97}};
    EXPECT_FALSE(flash_gpgpu_sim::cancel_pending_clc_cluster(
        {{4, 65537, 65537}}, shape, next, result));
    EXPECT_EQ(next, original);
    EXPECT_EQ(result, (Coord{{99, 98, 97}}));
  }
  Coord next{{0, 0, 0}}, result{{99, 98, 97}};
  EXPECT_FALSE(flash_gpgpu_sim::cancel_pending_clc_cluster({{3, 1, 1}}, shape,
                                                           next, result));
  EXPECT_EQ(next, (Coord{{0, 0, 0}}));
  EXPECT_EQ(result, (Coord{{99, 98, 97}}));
  EXPECT_TRUE(flash_gpgpu_sim::cancel_pending_clc_cluster(
      {{3, 1, 1}}, {{1, 1, 1}}, next, result));
  EXPECT_EQ(next, (Coord{{1, 0, 0}}));
  EXPECT_EQ(result, (Coord{{0, 0, 0}}));
}

TEST(ClcQueueTest, MixedAdmissionsAndCancellationsNeverLoseOrRepeatWork) {
  using Coord = std::array<uint32_t, 3>;
  Coord next{{0, 0, 0}}, result{};
  const Coord grid{{8, 3, 2}}, shape{{2, 1, 1}};
  std::set<unsigned> visited;
  while (next[2] < grid[2]) {
    // Normal admission of one whole cluster, interleaved with cancellation.
    for (unsigned rank = 0; rank < 2; ++rank) {
      EXPECT_TRUE(
          visited
              .insert(next[0] + rank + grid[0] * (next[1] + grid[1] * next[2]))
              .second);
    }
    auto next_after_admission = next;
    ASSERT_TRUE(flash_gpgpu_sim::cancel_pending_clc_cluster(
        grid, shape, next_after_admission, result));
    next = next_after_admission;
    if (flash_gpgpu_sim::cancel_pending_clc_cluster(grid, shape, next, result))
      for (unsigned rank = 0; rank < 2; ++rank)
        EXPECT_TRUE(visited
                        .insert(result[0] + rank +
                                grid[0] * (result[1] + grid[1] * result[2]))
                        .second);
  }
  EXPECT_EQ(visited.size(), 48u);
}

#include "gpgpu-sim/flash/clc.h"
TEST(ClcQueueTest, NativeResponsePackingKeepsAllCoordinateBits) {
  EXPECT_EQ(flash_gpgpu_sim::clc_response(true, 0xfedcba98, 0x1234, 0xabcd),
            (std::array<uint32_t, 4>{{0xfedcba98, 0xabcd1234, 1, 0}}));
  EXPECT_EQ(flash_gpgpu_sim::clc_response(false, 42, 99, 11),
            (std::array<uint32_t, 4>{{0, 0, 0, 0}}));
}
