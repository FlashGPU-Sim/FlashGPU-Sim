#include <gtest/gtest.h>

#include <vector>

#include "../../../src/kernel_info.h"

namespace {

struct Cta {
  unsigned x, y, z;
};

std::vector<Cta> walk(dim3 grid, dim3 cluster, bool cluster_launch) {
  std::vector<Cta> order;
  dim3 cta;
  cta.x = cta.y = cta.z = 0;
  for (int n = 0; n < 64; n++) {
    if (cta.x >= grid.x || cta.y >= grid.y || cta.z >= grid.z) break;
    order.push_back({cta.x, cta.y, cta.z});
    step_next_cta(cta, grid, cluster, cluster_launch);
  }
  return order;
}

void expect_order(const std::vector<Cta> &got, const std::vector<Cta> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < want.size(); i++) {
    EXPECT_EQ(got[i].x, want[i].x) << i;
    EXPECT_EQ(got[i].y, want[i].y) << i;
    EXPECT_EQ(got[i].z, want[i].z) << i;
  }
}

}  // namespace

TEST(ClusterCtaOrder, Grid4x2Cluster2x2IsClusterMajor) {
  dim3 grid, cluster;
  grid.x = 4;
  grid.y = 2;
  grid.z = 1;
  cluster.x = 2;
  cluster.y = 2;
  cluster.z = 1;
  expect_order(walk(grid, cluster, true),
               {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {2, 0, 0}, {3, 0, 0}, {2, 1, 0}, {3, 1, 0}});
}

TEST(ClusterCtaOrder, LinearXClusterMatchesXThenY) {
  dim3 grid, cluster;
  grid.x = 4;
  grid.y = 1;
  grid.z = 1;
  cluster.x = 2;
  cluster.y = 1;
  cluster.z = 1;
  expect_order(walk(grid, cluster, true),
               {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}});
}

TEST(ClusterCtaOrder, ZClusterWithUnitXY) {
  dim3 grid, cluster;
  grid.x = 1;
  grid.y = 1;
  grid.z = 4;
  cluster.x = 1;
  cluster.y = 1;
  cluster.z = 2;
  expect_order(walk(grid, cluster, true),
               {{0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3}});
}

TEST(ClusterCtaOrder, OrdinaryLaunchStaysXThenYThenZ) {
  dim3 grid, cluster;
  grid.x = 4;
  grid.y = 2;
  grid.z = 1;
  cluster.x = 2;
  cluster.y = 2;
  cluster.z = 1;
  expect_order(walk(grid, cluster, false),
               {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0},
                {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}});
}

TEST(ClusterCtaOrder, NonMultipleGridFallsBackToLinear) {
  dim3 grid, cluster;
  grid.x = 3;
  grid.y = 2;
  grid.z = 1;
  cluster.x = 2;
  cluster.y = 2;
  cluster.z = 1;
  expect_order(walk(grid, cluster, true),
               {{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                {0, 1, 0}, {1, 1, 0}, {2, 1, 0}});
}
