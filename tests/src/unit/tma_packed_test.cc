#include <gtest/gtest.h>
#include <set>
#include "gpgpu-sim/flash/tma_agu.h"
#include "gpgpu-sim/flash/tma_driver_tensormap.h"
#include "gpgpu-sim/flash/tma_packed.h"
namespace {
using namespace flash_gpgpu_sim;
tensormap_descriptor_t padded_map(bool expanded) {
  tensormap_descriptor_t tm{};
  auto &f = tm.fields;
  f.globalAddress = 0x10000;
  f.tensorRank = 3;
  f.tensorDataType =
      expanded ? TMA_DTYPE_16U4_ALIGN16B : TMA_DTYPE_16U4_ALIGN8B;
  f.globalDim[0] = 256;
  f.globalDim[1] = 5;
  f.globalDim[2] = 2;
  f.globalDim[3] = 2;
  f.boxDim[0] = 128;
  f.boxDim[1] = 4;
  f.boxDim[2] = 1;
  f.boxDim[3] = 1;
  f.globalStrides[0] = 160;
  f.globalStrides[1] = 864;
  f.globalStrides[2] = 1792;
  for (unsigned d = 0; d < 4; ++d) f.elementStrides[d] = 1;
  return tm;
}
TEST(TmaPacked4, RetainedDriverDescriptorsUseDifferentMemoryExtents) {
  tensormap_descriptor_t raw{}, decoded{};
  const uint64_t words[] = {0x100000,
                            0xe000266630,
                            0x150000000a8000,
                            0,
                            0xbff00001bff,
                            0x2f00000001,
                            0x7f00000000000000,
                            127,
                            0,
                            0x400};
  std::copy(std::begin(words), std::end(words), raw.raw_u64);
  ASSERT_TRUE(decode_sm100_driver_tensormap(raw, 4, decoded));
  EXPECT_EQ(decoded.fields.tensorDataType, TMA_DTYPE_16U4_ALIGN16B);
  EXPECT_EQ(decoded.fields.globalDim[0], 7168u);
  EXPECT_EQ(decoded.fields.globalDim[3], 48u);
  EXPECT_EQ(decoded.fields.globalStrides[0], 3584u);
  EXPECT_EQ(decoded.get_tile_size_bytes(), 8192u);
  EXPECT_EQ(decoded.get_tile_smem_size_bytes(), 16384u);
  auto malformed = raw;
  malformed.raw_u64[8] = 8192;
  EXPECT_FALSE(decode_sm100_driver_tensormap(malformed, 4, decoded));
  malformed = raw;
  malformed.raw_u64[1] ^= 1ull << 32;
  EXPECT_FALSE(decode_sm100_driver_tensormap(malformed, 4, decoded));
  tensormap_descriptor_t compact{};
  const uint64_t compact_words[] = {
      0x100000, 0x4002645a0,        0x110, 0, 0x3f0000007f,
      0x1fff,   0x7f00000000000000, 0x3f,  0, 0x200};
  std::copy(std::begin(compact_words), std::end(compact_words),
            compact.raw_u64);
  ASSERT_TRUE(decode_sm100_driver_tensormap(compact, 3, decoded));
  EXPECT_EQ(decoded.get_tile_size_bytes(), 4096u);
  EXPECT_EQ(decoded.get_tile_smem_size_bytes(), 4096u);
  EXPECT_EQ(decoded.fields.globalStrides[1], 4352u);
}
TEST(TmaPacked4, PayloadAndAguAgreeAcrossBoundsAndRequestWidths) {
  for (bool expanded : {false, true})
    for (int x : {-32, 0, 224, 256}) {
      const auto tm = padded_map(expanded);
      const int32_t coords[5] = {x, 3, 1, 1, 0};
      std::set<uint64_t> actual_addresses;
      std::vector<uint8_t> payload;
      ASSERT_TRUE(tma_packed4_load(
          tm, coords,
          [&](uint64_t at, void *dst, size_t bytes) {
            auto *p = static_cast<uint8_t *>(dst);
            for (size_t i = 0; i < bytes; ++i) {
              actual_addresses.insert(at + i);
              p[i] = (at + i) * 37;
            }
            return true;
          },
          payload));
      const unsigned stride = expanded ? 128 : 64;
      ASSERT_EQ(payload.size(), 4 * stride);
      std::vector<uint8_t> expected(payload.size(), 0);
      std::set<uint64_t> expected_addresses;
      for (unsigned row = 0; row < 4; ++row)
        for (unsigned byte = 0; byte < 64; ++byte) {
          const int global_x = x / 2 + byte;
          if (row + 3 >= 5 || global_x < 0 || global_x >= 128) continue;
          const uint64_t at = 0x10000 + 1792 + 864 + (row + 3) * 160 + global_x;
          expected_addresses.insert(at);
          expected[row * stride +
                   (expanded ? byte / 8 * 16 + byte % 8 : byte)] = at * 37;
        }
      EXPECT_EQ(payload, expected);
      EXPECT_EQ(actual_addresses, expected_addresses);
      for (unsigned width : {32u, 128u}) {
        tma_agu_unit_t agu;
        tma_agu_state_t state;
        agu.init_tensor(state, tm, coords);
        uint64_t at;
        uint32_t bytes, total = 0;
        std::set<uint64_t> addresses;
        while (agu.gen_next_req(state, at, bytes, width)) {
          total += bytes;
          if (!state.is_fill_request)
            for (unsigned i = 0; i < bytes; ++i) addresses.insert(at + i);
        }
        EXPECT_EQ(total, 256u);
        EXPECT_EQ(addresses, expected_addresses);
      }
    }
}
TEST(TmaPacked4, FailedReadsAndInvalidShapesPreservePayload) {
  auto tm = padded_map(true);
  int32_t coords[5] = {0, 0, 0, 0, 0};
  std::vector<uint8_t> tile{1, 2, 3};
  const auto saved = tile;
  unsigned reads = 0;
  EXPECT_FALSE(tma_packed4_load(
      tm, coords, [&](uint64_t, void *, size_t) { return ++reads < 3; }, tile));
  EXPECT_EQ(tile, saved);
  for (unsigned failure = 0; failure < 5; ++failure) {
    auto bad = tm;
    if (failure == 0) bad.fields.boxDim[0] = 64;
    if (failure == 1) bad.fields.globalDim[0] = 255;
    if (failure == 2) bad.fields.elementStrides[1] = 2;
    if (failure == 3) bad.fields.swizzle = 2;
    if (failure == 4) bad.fields.globalAddress += 16;
    EXPECT_FALSE(tma_packed4_supported(bad));
    EXPECT_FALSE(tma_packed4_load(
        bad, coords,
        [](uint64_t, void *, size_t) {
          ADD_FAILURE();
          return false;
        },
        tile));
    EXPECT_EQ(tile, saved);
  }
  coords[0] = 1;
  EXPECT_FALSE(tma_packed4_load(
      tm, coords,
      [](uint64_t, void *, size_t) {
        ADD_FAILURE();
        return false;
      },
      tile));
}
}  // namespace
