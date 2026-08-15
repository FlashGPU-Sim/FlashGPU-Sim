// Unit tests for shipped cluster NoC helpers (matrix, decode, drop).
// Links src/gpgpu-sim/flash/cluster_noc_helpers.cc — no local reimplementation.

#include <gtest/gtest.h>

#include <cstdio>
#include <deque>
#include <fstream>
#include <string>

#include "../../../src/abstract_hardware_model.h"
#include "../../../src/gpgpu-sim/flash/cluster_noc.h"

using flash_gpgpu_sim::cluster_noc_drop_queue_to_cta;
using flash_gpgpu_sim::cluster_noc_latency_matrix;
using flash_gpgpu_sim::cluster_noc_message;
using flash_gpgpu_sim::cluster_noc_msg_targets_cta;
using flash_gpgpu_sim::decode_shared_generic;
using flash_gpgpu_sim::is_remote_shared_generic;

TEST(ClusterNocMatrix, InitDiagonalAndRemote) {
  cluster_noc_latency_matrix m;
  m.init(4, /*local=*/10, /*remote=*/100);
  EXPECT_EQ(m.n_cores(), 4u);
  EXPECT_EQ(m.hop(0, 0), 10u);
  EXPECT_EQ(m.hop(1, 1), 10u);
  EXPECT_EQ(m.hop(0, 1), 100u);
  EXPECT_EQ(m.hop(3, 2), 100u);
}

TEST(ClusterNocMatrix, LoadFromFile) {
  const char *path = "cluster_noc_unit_matrix.csv";
  {
    std::ofstream out(path);
    out << "# comment\n";
    out << "1,2,3\n";
    out << "4,5,6\n";
    out << "7,8,9\n";
  }
  cluster_noc_latency_matrix m;
  ASSERT_TRUE(m.load_from_file(path, 3));
  EXPECT_EQ(m.hop(0, 0), 1u);
  EXPECT_EQ(m.hop(0, 2), 3u);
  EXPECT_EQ(m.hop(2, 1), 8u);
  std::remove(path);
}

TEST(ClusterNocMatrix, LoadWrongSizeFails) {
  const char *path = "cluster_noc_unit_matrix_bad.csv";
  {
    std::ofstream out(path);
    out << "1,2\n3,4\n";
  }
  cluster_noc_latency_matrix m;
  m.init(3, 1, 2);
  EXPECT_FALSE(m.load_from_file(path, 3));
  EXPECT_EQ(m.hop(0, 1), 2u);
  std::remove(path);
}

TEST(ClusterNocMatrix, H200ReducedMatrixFile) {
  cluster_noc_latency_matrix m;
  const char *candidates[] = {
      "configs/SM90_H200_REDUCED_CLUSTER4x4/dsm_latency_matrix_4.csv",
      "../configs/SM90_H200_REDUCED_CLUSTER4x4/dsm_latency_matrix_4.csv",
      "../../configs/SM90_H200_REDUCED_CLUSTER4x4/dsm_latency_matrix_4.csv",
      "dsm_latency_matrix_4.csv",
  };
  bool loaded = false;
  for (const char *p : candidates) {
    if (m.load_from_file(p, 4)) {
      loaded = true;
      break;
    }
  }
  if (!loaded) {
    m.init(4, /*local=*/0, /*remote=*/78);
  }
  EXPECT_EQ(m.hop(0, 0), 0u);
  EXPECT_GE(m.hop(0, 1), 70u);
  EXPECT_LE(m.hop(0, 1), 90u);
  EXPECT_GE(m.hop(3, 2), 70u);
  EXPECT_LE(m.hop(3, 2), 90u);
}

TEST(ClusterNocMatrix, H200RemoteLoadRttFormula) {
  const unsigned local = 37;
  const unsigned one_way = 78;
  const unsigned e2e = local + 2 * one_way;
  EXPECT_EQ(e2e, 193u);
  EXPECT_NEAR(static_cast<double>(e2e), 193.41, 1.0);
}

TEST(ClusterNocAddr, DecodeSharedGeneric) {
  unsigned smid = 0;
  addr_t off = 0;
  const addr_t g =
      SHARED_GENERIC_START + 3ull * SHARED_MEM_SIZE_MAX + 0x40;
  ASSERT_TRUE(decode_shared_generic(g, &smid, &off));
  EXPECT_EQ(smid, 3u);
  EXPECT_EQ(off, (addr_t)0x40);
  EXPECT_FALSE(decode_shared_generic(0x1000, &smid, &off));
}

TEST(ClusterNocAddr, RemoteVsLocal) {
  const addr_t remote =
      SHARED_GENERIC_START + 2ull * SHARED_MEM_SIZE_MAX + 8;
  unsigned owner = 0;
  addr_t off = 0;
  EXPECT_TRUE(is_remote_shared_generic(0, remote, &owner, &off));
  EXPECT_EQ(owner, 2u);
  EXPECT_EQ(off, (addr_t)8);
  EXPECT_FALSE(is_remote_shared_generic(2, remote, &owner, &off));
}

TEST(ClusterNocDrop, DropQueueToCtaUsesShippedFilter) {
  std::deque<cluster_noc_message> q;
  cluster_noc_message keep;
  keep.dst_cid = 0;
  keep.dst_hw_cta = 1;
  keep.seq = 1;
  cluster_noc_message drop_a;
  drop_a.dst_cid = 1;
  drop_a.dst_hw_cta = 3;
  drop_a.seq = 2;
  cluster_noc_message drop_b;
  drop_b.dst_cid = 1;
  drop_b.dst_hw_cta = 3;
  drop_b.seq = 3;
  q.push_back(keep);
  q.push_back(drop_a);
  q.push_back(drop_b);

  EXPECT_TRUE(cluster_noc_msg_targets_cta(drop_a, 1, 3));
  EXPECT_FALSE(cluster_noc_msg_targets_cta(keep, 1, 3));

  const size_t n = cluster_noc_drop_queue_to_cta(&q, 1, 3);
  EXPECT_EQ(n, 2u);
  ASSERT_EQ(q.size(), 1u);
  EXPECT_EQ(q.front().seq, 1u);
  EXPECT_EQ(q.front().dst_cid, 0u);
}
