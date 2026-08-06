// Host-side unit coverage for the SM100 opaque TensorMap decoder.
#include <gtest/gtest.h>

#include <cuda.h>

#include "gpgpu-sim/flash/tma_opaque_tensormap.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using flash_gpgpu_sim::decode_sm100_opaque_tensormap;

struct ExpectedDescriptor {
  uint64_t address;
  uint32_t dim0;
  uint32_t dim1;
  uint32_t dim2;
  uint32_t dim3;
  uint64_t stride_dim1;
  uint64_t stride_dim2;
  uint64_t stride_dim3;
};

tensormap_descriptor_t make_encoded(const std::array<uint64_t, 16> &words) {
  tensormap_descriptor_t encoded{};
  for (unsigned i = 0; i < words.size(); ++i)
    encoded.raw_u64[i] = words[i];
  return encoded;
}

void expect_decoded_fields(const tensormap_descriptor_t &decoded,
                           const ExpectedDescriptor &expected) {
  EXPECT_EQ(decoded.fields.globalAddress, expected.address);
  EXPECT_EQ(decoded.fields.tensorRank, 3u);
  EXPECT_EQ(decoded.fields.tensorDataType, TMA_DTYPE_F16);
  EXPECT_EQ(decoded.fields.globalDim[0], expected.dim0);
  EXPECT_EQ(decoded.fields.globalDim[1], expected.dim1);
  EXPECT_EQ(decoded.fields.globalDim[2], expected.dim2);
  EXPECT_EQ(decoded.fields.globalDim[3], expected.dim3);
  EXPECT_EQ(decoded.fields.globalStrides[0], expected.stride_dim1);
  EXPECT_EQ(decoded.fields.globalStrides[1], expected.stride_dim2);
  EXPECT_EQ(decoded.fields.globalStrides[2], expected.stride_dim3);
  EXPECT_EQ(decoded.fields.boxDim[0], 64u);
  EXPECT_EQ(decoded.fields.boxDim[1], 128u);
  EXPECT_EQ(decoded.fields.boxDim[2], 1u);
  EXPECT_EQ(decoded.fields.boxDim[3], 1u);
  EXPECT_EQ(decoded.fields.elementStrides[0], 1u);
  EXPECT_EQ(decoded.fields.elementStrides[1], 1u);
  EXPECT_EQ(decoded.fields.elementStrides[2], 1u);
  EXPECT_EQ(decoded.fields.elementStrides[3], 1u);
  EXPECT_EQ(decoded.fields.interleave, TMA_INTERLEAVE_NONE);
  EXPECT_EQ(decoded.fields.swizzle, TMA_SWIZZLE_128B);
  EXPECT_EQ(decoded.fields.oobFill, TMA_OOB_ZERO);
  EXPECT_TRUE(decoded.is_valid());
}

void expect_decodes_to(const std::array<uint64_t, 16> &raw,
                       const ExpectedDescriptor &expected) {
  tensormap_descriptor_t decoded{};
  ASSERT_TRUE(decode_sm100_opaque_tensormap(make_encoded(raw), decoded));
  expect_decoded_fields(decoded, expected);
}

bool running_with_simulator_cuda() {
  if (std::getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN"))
    return true;

  const char *ld_library_path = std::getenv("LD_LIBRARY_PATH");
  if (!ld_library_path)
    return false;

  const std::string path(ld_library_path);
  const char *gpgpusim_root = std::getenv("GPGPUSIM_ROOT");
  if (gpgpusim_root) {
    const std::string sim_lib_prefix =
        std::string(gpgpusim_root) + "/lib/";
    if (path.find(sim_lib_prefix) != std::string::npos)
      return true;
  }

  return path.find("/flashgpu-sim/lib/") != std::string::npos;
}

std::string cuda_error_string(CUresult status) {
  const char *message = nullptr;
  cuGetErrorString(status, &message);
  return message ? message : "unknown CUDA driver error";
}

struct CudaContextGuard {
  CUcontext context = nullptr;

  ~CudaContextGuard() {
    if (context)
      cuCtxDestroy(context);
  }
};

struct CudaMemoryGuard {
  CUdeviceptr ptr = 0;

  ~CudaMemoryGuard() {
    if (ptr)
      cuMemFree(ptr);
  }
};

CUresult encode_cuda_api_tensormap(const ExpectedDescriptor &expected,
                                   CUtensorMap *map) {
  const cuuint64_t global_dim[4] = {expected.dim0, expected.dim1,
                                    expected.dim2, expected.dim3};
  const cuuint64_t global_strides[3] = {expected.stride_dim1,
                                        expected.stride_dim2,
                                        expected.stride_dim3};
  const cuuint32_t box_dim[4] = {64, 128, 1, 1};
  const cuuint32_t element_strides[4] = {1, 1, 1, 1};

  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4,
      reinterpret_cast<void *>(expected.address), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

bool check_cuda_api_encoded_descriptor(const ExpectedDescriptor &expected) {
  CUtensorMap map{};
  const CUresult status = encode_cuda_api_tensormap(expected, &map);
  if (status != CUDA_SUCCESS) {
    ADD_FAILURE() << "cuTensorMapEncodeTiled failed: "
                  << cuda_error_string(status);
    return false;
  }

  tensormap_descriptor_t encoded{};
  std::memcpy(&encoded, &map, sizeof(encoded));
  if (encoded.is_valid())
    return true;

  tensormap_descriptor_t decoded{};
  if (!decode_sm100_opaque_tensormap(encoded, decoded)) {
    ADD_FAILURE() << "decode_sm100_opaque_tensormap rejected CUDA API output";
    return false;
  }

  expect_decoded_fields(decoded, expected);
  return false;
}

constexpr std::array<uint64_t, 16> kQPrefillSmoke = {
    // Captured from the CUDA 13.1 / CuTeDSL 4.5.2 generated FA4 SM100 launcher
    // by intercepting _cudaLaunchKernelEx and dumping the by-value CUtensorMap
    // kernel parameters. Shape: b=1, seqlen_q=128, seqlen_k=128, h=2, d=64.
    0x0000000000100000ull, 0x0000001000046332ull,
    0x0000080000000008ull, 0x0000000000000000ull,
    0x0000007f0000003full, 0x0000000000000001ull,
    0x3f00000000000000ull, 0x000000000000007full,
    0x0000000500000001ull, 0x0000000000028000ull,
    0x0000000000028000ull, 0x0000000000028000ull,
    0x0000000000187d39ull, 0x0000000000187d39ull,
    0x0000000000001000ull, 0x0000000400000001ull,
};

constexpr std::array<uint64_t, 16> kKPrefillSmoke = {
    // Same launch as kQPrefillSmoke, K descriptor.
    0x0000000000200000ull, 0x0000001000046332ull,
    0x0000080000000008ull, 0x0000000000000000ull,
    0x0000007f0000003full, 0x0000000000000001ull,
    0x3f00000000000000ull, 0x000000000000007full,
    0x0000000000000310ull, 0x0000000000000008ull,
    0x0000000400000003ull, 0x00000000001d67a0ull,
    0x00000000001d67a0ull, 0x00000000001d67a0ull,
    0x000000000000001cull, 0x000000000000001cull,
};

constexpr std::array<uint64_t, 16> kQMixedShape = {
    // Shape: b=2, seqlen_q=64, seqlen_k=256, h=3, d=64.
    0x0000000000100000ull, 0x0000001800046332ull,
    0x0000060000000008ull, 0x0000000000000000ull,
    0x0000003f0000003full, 0x0000000100000002ull,
    0x3f00000000000000ull, 0x000000000000007full,
    0x0000000500000001ull, 0x0000000000028000ull,
    0x0000000000028000ull, 0x0000000000028000ull,
    0x0000000000187d39ull, 0x0000000000187d39ull,
    0x0000000000001000ull, 0x0000000400000001ull,
};

constexpr std::array<uint64_t, 16> kKMixedShape = {
    // Same launch as kQMixedShape, K descriptor. The low word of raw[1] includes
    // the K/V operand flag in addition to the descriptor magic.
    0x0000000000200000ull, 0x0000001800246332ull,
    0x0000180000000008ull, 0x0000000000000000ull,
    0x000000ff0000003full, 0x0000000100000002ull,
    0x3f00000000000000ull, 0x000000000000007full,
    0x0000000000000310ull, 0x0000000000000008ull,
    0x0000000400000003ull, 0x00000000001d67a0ull,
    0x00000000001d67a0ull, 0x00000000001d67a0ull,
    0x000000000000001cull, 0x000000000000001cull,
};

} // namespace

TEST(Fa4OpaqueTensorMapTest, DecodesSmokeQAndKDescriptors) {
  expect_decodes_to(kQPrefillSmoke,
                    {/*address=*/0x100000,
                     /*dim0=*/64,
                     /*dim1=*/128,
                     /*dim2=*/2,
                     /*dim3=*/1,
                     /*stride_dim1=*/256,
                     /*stride_dim2=*/128,
                     /*stride_dim3=*/32768});
  expect_decodes_to(kKPrefillSmoke,
                    {/*address=*/0x200000,
                     /*dim0=*/64,
                     /*dim1=*/128,
                     /*dim2=*/2,
                     /*dim3=*/1,
                     /*stride_dim1=*/256,
                     /*stride_dim2=*/128,
                     /*stride_dim3=*/32768});
}

TEST(Fa4OpaqueTensorMapTest, DecodesMixedShapeAndKvFlag) {
  expect_decodes_to(kQMixedShape,
                    {/*address=*/0x100000,
                     /*dim0=*/64,
                     /*dim1=*/64,
                     /*dim2=*/3,
                     /*dim3=*/2,
                     /*stride_dim1=*/384,
                     /*stride_dim2=*/128,
                     /*stride_dim3=*/24576});
  expect_decodes_to(kKMixedShape,
                    {/*address=*/0x200000,
                     /*dim0=*/64,
                     /*dim1=*/256,
                     /*dim2=*/3,
                     /*dim3=*/2,
                     /*stride_dim1=*/384,
                     /*stride_dim2=*/128,
                     /*stride_dim3=*/98304});
}

TEST(Fa4OpaqueTensorMapTest, PreservesHighAddressBits) {
  auto raw = kQMixedShape;
  raw[0] = 0x0000002000100000ull;

  expect_decodes_to(raw,
                    {/*address=*/0x0000002000100000ull,
                     /*dim0=*/64,
                     /*dim1=*/64,
                     /*dim2=*/3,
                     /*dim3=*/2,
                     /*stride_dim1=*/384,
                     /*stride_dim2=*/128,
                     /*stride_dim3=*/24576});
}

TEST(Fa4OpaqueTensorMapTest, KeepsFa4SwizzleForWideHeadTile) {
  auto raw = kQMixedShape;
  raw[6] = (raw[6] & 0x00ffffffffffffffull) | (0x7full << 56);

  tensormap_descriptor_t decoded{};
  ASSERT_TRUE(decode_sm100_opaque_tensormap(make_encoded(raw), decoded));
  EXPECT_EQ(decoded.fields.boxDim[0], 128u);
  EXPECT_EQ(decoded.fields.swizzle, TMA_SWIZZLE_128B);
  EXPECT_TRUE(decoded.is_valid());
}

TEST(Fa4OpaqueTensorMapTest, RejectsUnknownMagic) {
  auto raw = kQPrefillSmoke;
  raw[1] = 0x0000001000056332ull;

  tensormap_descriptor_t decoded{};
  EXPECT_FALSE(decode_sm100_opaque_tensormap(make_encoded(raw), decoded));
}

TEST(Fa4OpaqueTensorMapTest, RejectsMissingStrideFields) {
  auto raw = kQPrefillSmoke;
  raw[2] = 0;

  tensormap_descriptor_t decoded{};
  EXPECT_FALSE(decode_sm100_opaque_tensormap(make_encoded(raw), decoded));
}

TEST(Fa4OpaqueTensorMapTest, RejectsNullGlobalAddress) {
  auto raw = kQPrefillSmoke;
  raw[0] = 0;

  tensormap_descriptor_t decoded{};
  EXPECT_FALSE(decode_sm100_opaque_tensormap(make_encoded(raw), decoded));
}

TEST(Fa4OpaqueTensorMapTest, DecodesRealCudaApiEncodedDescriptors) {
  if (running_with_simulator_cuda()) {
    GTEST_SKIP() << "requires the real CUDA driver, not simulator libcuda";
  }

  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    GTEST_SKIP() << "cuInit failed: " << cuda_error_string(status);
  }

  CUdevice device = 0;
  status = cuDeviceGet(&device, 0);
  if (status != CUDA_SUCCESS) {
    GTEST_SKIP() << "cuDeviceGet failed: " << cuda_error_string(status);
  }

  CudaContextGuard context;
  status = cuCtxCreate(&context.context, 0, device);
  if (status != CUDA_SUCCESS) {
    GTEST_SKIP() << "cuCtxCreate failed: " << cuda_error_string(status);
  }

  CudaMemoryGuard q_base;
  CudaMemoryGuard k_base;
  status = cuMemAlloc(&q_base.ptr, 1u << 20);
  if (status != CUDA_SUCCESS)
    GTEST_SKIP() << "cuMemAlloc(q) failed: " << cuda_error_string(status);
  status = cuMemAlloc(&k_base.ptr, 1u << 20);
  if (status != CUDA_SUCCESS)
    GTEST_SKIP() << "cuMemAlloc(k) failed: " << cuda_error_string(status);

  if (check_cuda_api_encoded_descriptor(
      {/*address=*/static_cast<uint64_t>(q_base.ptr),
       /*dim0=*/64,
       /*dim1=*/128,
       /*dim2=*/2,
       /*dim3=*/1,
       /*stride_dim1=*/256,
       /*stride_dim2=*/128,
       /*stride_dim3=*/32768})) {
    GTEST_SKIP()
        << "cuTensorMapEncodeTiled produced simulator internal descriptor; "
           "run this test without simulator libcuda interposition";
  }
  if (check_cuda_api_encoded_descriptor(
      {/*address=*/static_cast<uint64_t>(k_base.ptr),
       /*dim0=*/64,
       /*dim1=*/256,
       /*dim2=*/3,
       /*dim3=*/2,
       /*stride_dim1=*/384,
       /*stride_dim2=*/128,
       /*stride_dim3=*/98304})) {
    GTEST_SKIP()
        << "cuTensorMapEncodeTiled produced simulator internal descriptor; "
           "run this test without simulator libcuda interposition";
  }
}
