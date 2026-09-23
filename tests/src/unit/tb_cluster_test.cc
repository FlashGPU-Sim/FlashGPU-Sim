// Unit tests for generic shared-window decode (inline in tb_cluster.h).

#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/flash/tb_cluster.h"

using flash_gpgpu_sim::decode_shared_generic;
using flash_gpgpu_sim::is_remote_shared_generic;

TEST(TbClusterAddr, DecodeSharedGeneric) {
  unsigned smid = 0;
  addr_t off = 0;
  const addr_t g =
      SHARED_GENERIC_START + 3ull * SHARED_MEM_SIZE_MAX + 0x40;
  ASSERT_TRUE(decode_shared_generic(g, &smid, &off));
  EXPECT_EQ(smid, 3u);
  EXPECT_EQ(off, (addr_t)0x40);
  EXPECT_FALSE(decode_shared_generic(0x1000, &smid, &off));
}

TEST(TbClusterAddr, RemoteVsLocal) {
  const addr_t remote =
      SHARED_GENERIC_START + 2ull * SHARED_MEM_SIZE_MAX + 8;
  unsigned owner = 0;
  addr_t off = 0;
  EXPECT_TRUE(is_remote_shared_generic(0, remote, &owner, &off));
  EXPECT_EQ(owner, 2u);
  EXPECT_EQ(off, (addr_t)8);
  EXPECT_FALSE(is_remote_shared_generic(2, remote, &owner, &off));
}
