#include "../test_support.h"

namespace {

TEST(SassTensorMapTest, DecodesValidatedDriverAndDeviceTwoDimensionalForms) {
  std::array<uint64_t, 8> words{{
      0x000001000a400000ull,
      0x0000020000206310ull,
      0,
      0,
      0x0000007f00000fffull,
      0,
      0x3f00000000000000ull,
      0x3full,
  }};
  auto decoded = decode_sm120_tensor_map_2d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x000001000a400000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 2u);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 2>{{4096, 128}}));
  EXPECT_EQ(decoded.descriptor.box_dim, (std::array<uint32_t, 2>{{64, 64}}));
  EXPECT_EQ(decoded.descriptor.row_stride_bytes, 8192u);
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 128u);

  // Device-side tensormap.replace construction leaves the optional Driver
  // control field clear while retaining the same hardware layout.
  words[1] &= ~uint64_t{0x200000};
  decoded = decode_sm120_tensor_map_2d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;

  words[1] ^= 0x40;
  decoded = decode_sm120_tensor_map_2d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("2D FP16/F32"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesValidatedDeviceTwoDimensionalF32Form) {
  std::array<uint64_t, 8> words{{
      0x0000000c00000000ull,
      0x0000001000000390ull,
      0,
      0,
      0x0000003f0000003full,
      0,
      0x0f00000000000000ull,
      0x0full,
  }};
  const auto decoded = decode_sm120_tensor_map_2d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim, (std::array<uint32_t, 2>{{64, 64}}));
  EXPECT_EQ(decoded.descriptor.box_dim, (std::array<uint32_t, 2>{{16, 16}}));
  EXPECT_EQ(decoded.descriptor.row_stride_bytes, 256u);
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 0u);
}

TEST(SassTensorMapTest, DecodesHostTwoDimensionalF32AndFtzForms) {
  std::array<uint64_t, 16> words{};
  words[0] = 0x0000000c00000000ull;
  words[1] = 0x0000001000000001ull;
  words[2] = 2;
  words[4] = 0x0000000200000010ull;
  words[6] = 0x0000004000000000ull;
  words[11] = 0x0000000100000000ull;
  words[12] = 1;
  words[14] = 7;

  auto decoded = decode_flashgpu_host_tensor_map_2d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.element_type, tensor_map_element_type::kFloat32);
  EXPECT_EQ(decoded.descriptor.global_dim, (std::array<uint32_t, 2>{{16, 2}}));
  EXPECT_EQ(decoded.descriptor.box_dim, (std::array<uint32_t, 2>{{16, 2}}));
  EXPECT_EQ(decoded.descriptor.row_stride_bytes, 64u);

  words[14] = 10;
  decoded = decode_flashgpu_host_tensor_map_2d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.element_type,
            tensor_map_element_type::kFloat32Ftz);

  words[14] = 2;
  decoded = decode_flashgpu_host_tensor_map_2d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("rank-2 F32"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesValidatedDeviceThreeDimensionalF32Form) {
  std::array<uint64_t, 8> words{{
      0x0000000c00000000ull,
      0x00000004000003a0ull,
      0x40ull,
      0,
      0x0000000f0000000full,
      0x0full,
      0x0f00000000000000ull,
      0x0f0full,
  }};
  auto decoded = decode_sm120_tensor_map_3d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 3>{{16, 16, 16}}));
  EXPECT_EQ(decoded.descriptor.box_dim,
            (std::array<uint32_t, 3>{{16, 16, 16}}));
  EXPECT_EQ(decoded.descriptor.global_stride_bytes,
            (std::array<uint64_t, 3>{{4, 64, 1024}}));
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 0u);

  words[1] ^= 0x10;
  decoded = decode_sm120_tensor_map_3d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("3D F32"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesValidatedDeviceFourDimensionalF32Form) {
  std::array<uint64_t, 8> words{{
      0x0000000c00000000ull,
      0x00000002000003b0ull,
      0x0000008000000010ull,
      0,
      0x0000000700000007ull,
      0x0000000700000007ull,
      0x0700000000000000ull,
      0x0000000000070707ull,
  }};
  auto decoded = decode_sm120_tensor_map_4d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 4>{{8, 8, 8, 8}}));
  EXPECT_EQ(decoded.descriptor.box_dim,
            (std::array<uint32_t, 4>{{8, 8, 8, 8}}));
  EXPECT_EQ(decoded.descriptor.global_stride_bytes,
            (std::array<uint64_t, 4>{{4, 32, 256, 2048}}));
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 0u);

  words[1] ^= 0x10;
  decoded = decode_sm120_tensor_map_4d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("4D F32"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesFlashGpuHostFourDimensionalF16Form) {
  std::array<uint64_t, 16> words{};
  words[0] = 0x0000000c00000000ull;
  words[1] = 0x0000004000000003ull;
  words[2] = 0x0000000200000080ull;
  words[3] = 1;
  words[4] = 0x0000100000000080ull;
  words[5] = 0x0000000200000010ull;
  words[6] = 0x0000008000000000ull;
  words[7] = 0x0000020000000000ull;
  words[8] = 0x0000400000000000ull;
  words[11] = 0x0000000100000000ull;
  words[12] = 0x0000000100000001ull;
  words[13] = 0x0000000000000001ull;
  words[14] = TMA_DTYPE_F16;
  words[15] = TMA_SWIZZLE_128B;

  auto decoded = decode_flashgpu_host_tensor_map_4d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 2u);
  EXPECT_EQ(decoded.descriptor.element_type, tensor_map_element_type::kFloat16);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 4>{{128, 4096, 16, 2}}));
  EXPECT_EQ(decoded.descriptor.box_dim,
            (std::array<uint32_t, 4>{{64, 128, 2, 1}}));
  EXPECT_EQ(decoded.descriptor.global_stride_bytes,
            (std::array<uint64_t, 4>{{2, 128, 512, 16384}}));
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 128u);

  words[14] = TMA_DTYPE_F32;
  decoded = decode_flashgpu_host_tensor_map_4d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("F16"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesValidatedDeviceFiveDimensionalF32Form) {
  std::array<uint64_t, 8> words{{
      0x0000000c00000000ull,
      0x00000001000003c0ull,
      0x0000001000000004ull,
      0x40ull,
      0x0000000300000003ull,
      0x0000000300000003ull,
      0x0300000000000003ull,
      0x0000000003030303ull,
  }};
  auto decoded = decode_sm120_tensor_map_5d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 5>{{4, 4, 4, 4, 4}}));
  EXPECT_EQ(decoded.descriptor.box_dim,
            (std::array<uint32_t, 5>{{4, 4, 4, 4, 4}}));
  EXPECT_EQ(decoded.descriptor.global_stride_bytes,
            (std::array<uint64_t, 5>{{4, 16, 64, 256, 1024}}));
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 0u);

  words[1] ^= 0x10;
  decoded = decode_sm120_tensor_map_5d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("5D F32"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesFlashGpuHostFiveDimensionalF16Form) {
  std::array<uint64_t, 16> words{};
  words[0] = 0x0000000c00000000ull;
  words[1] = 0x0000001000000004ull;
  words[2] = 0x0000000200000008ull;
  words[3] = 0x0000000100000001ull;
  words[4] = 0x0000010000000080ull;
  words[5] = 0x0000000200000004ull;
  words[6] = 0x0000010000000001ull;
  words[7] = 0x0001000000000000ull;
  words[8] = 0x0004000000000000ull;
  // Rank-5 output descriptors may use a broadcast stride for a unit extent.
  words[9] = 0;
  words[11] = 0x0000000100000000ull;
  words[12] = 0x0000000100000001ull;
  words[13] = 0x0000000100000001ull;
  words[14] = TMA_DTYPE_F16;
  words[15] = TMA_SWIZZLE_128B;

  auto decoded = decode_flashgpu_host_tensor_map_5d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x0000000c00000000ull);
  EXPECT_EQ(decoded.descriptor.element_bytes, 2u);
  EXPECT_EQ(decoded.descriptor.element_type, tensor_map_element_type::kFloat16);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 5>{{128, 256, 4, 2, 1}}));
  EXPECT_EQ(decoded.descriptor.box_dim,
            (std::array<uint32_t, 5>{{16, 8, 2, 1, 1}}));
  EXPECT_EQ(decoded.descriptor.global_stride_bytes,
            (std::array<uint64_t, 5>{{2, 256, 65536, 262144, 0}}));
  EXPECT_EQ(decoded.descriptor.swizzle_bytes, 128u);

  words[10] = uint64_t{1} << 32;
  decoded = decode_flashgpu_host_tensor_map_5d(words);
  EXPECT_FALSE(decoded.supported);
  EXPECT_NE(decoded.detail.find("extra global stride"), std::string::npos);
}

TEST(SassTensorMapTest, DecodesValidatedDeviceOneDimensionalFourByteForms) {
  std::array<uint64_t, 8> words{};
  words[0] = 0x123400;
  words[1] = 0x0380;
  words[4] = 15;
  words[6] = uint64_t{15} << 56;

  const auto decoded = decode_sm120_tensor_map_1d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x123400u);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim, (std::array<uint32_t, 2>{{16, 1}}));
  EXPECT_EQ(decoded.descriptor.box_dim, (std::array<uint32_t, 2>{{16, 1}}));
  EXPECT_EQ(decoded.descriptor.element_type, tensor_map_element_type::kFloat32);

  words[1] = 0x0100;
  const auto decoded_u32 = decode_sm120_tensor_map_1d(words);
  ASSERT_TRUE(decoded_u32.supported) << decoded_u32.detail;
  EXPECT_EQ(decoded_u32.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded_u32.descriptor.element_type,
            tensor_map_element_type::kUnsigned32);

  words[1] = 0x0310;
  EXPECT_FALSE(decode_sm120_tensor_map_1d(words).supported);
}

TEST(SassTensorMapTest, DecodesFlashGpuHostOneDimensionalF32Form) {
  std::array<uint64_t, 16> words{};
  words[0] = 0x123400;
  words[1] = uint64_t{256} << 32;
  words[4] = 1024;
  words[11] = uint64_t{1} << 32;
  words[14] = 7;

  const auto decoded = decode_flashgpu_host_tensor_map_1d(words);
  ASSERT_TRUE(decoded.supported) << decoded.detail;
  EXPECT_EQ(decoded.descriptor.global_address, 0x123400u);
  EXPECT_EQ(decoded.descriptor.element_bytes, 4u);
  EXPECT_EQ(decoded.descriptor.global_dim,
            (std::array<uint32_t, 2>{{1024, 1}}));
  EXPECT_EQ(decoded.descriptor.box_dim, (std::array<uint32_t, 2>{{256, 1}}));

  words[14] = 2;
  EXPECT_FALSE(decode_flashgpu_host_tensor_map_1d(words).supported);
}

}  // namespace
