// CUDA integration tests for Hopper warp-group MMA.
//
// Uniform and random tests share one matrix runner so every case exercises the
// same A/B/C data path and accumulator-to-matrix layout against a CPU
// reference.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace wgmma_test {

__device__ __forceinline__ void wgmma_m64n8k16_f32_f16_f16_rs(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
    float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %9, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "%8, p, %10, %11, %12;\n"
      "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d), "n"(1),
        "n"(1), "n"(0));
}

__global__ void wgmma_m64n8k16_f16_matrix_kernel(float *out, const uint16_t *A,
                                                 const uint16_t *B,
                                                 const float *C,
                                                 int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(16) uint16_t smem_b[kSharedBElements];

  for (int i = threadIdx.x; i < kSharedBElements; i += blockDim.x) {
    smem_b[i] = 0;
  }

  for (int i = threadIdx.x; i < kN * kK; i += blockDim.x) {
    int col = i / kK;
    int k = i % kK;
    smem_b[b_smem_index(col, k)] = B[b_matrix_index(col, k)];
  }
  __syncthreads();

  uint64_t desc_b =
      make_gmma_k_major_desc(smem_b, kBLeadingByteOffset, kBStrideByteOffset);

  int tid_mma_col = threadIdx.x % 4;
  int tid_row = (threadIdx.x / 4) % 8;
  int tid_m_block = threadIdx.x / 32;

  uint32_t a_regs[4];
  for (int reg = 0; reg < 4; ++reg) {
    int reg_row_block = reg & 0x1;
    int reg_k_block = (reg >> 1) & 0x1;
    int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
    int k0 = 2 * tid_mma_col + 8 * reg_k_block;
    a_regs[reg] = pack_f16_pair(A[a_matrix_index(row, k0)],
                                A[a_matrix_index(row, k0 + 1)]);
  }

  float d[4];
  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(threadIdx.x, reg, row, col);
    d[reg] = C[c_matrix_index(row, col)];
  }

  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3]), "+f"(d[0]), "+f"(d[1]), "+f"(d[2]),
                 "+f"(d[3])::"memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  wgmma_m64n8k16_f32_f16_f16_rs(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                                desc_b, d[0], d[1], d[2], d[3], scale_d);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])::"memory");

  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(threadIdx.x, reg, row, col);
    out[c_matrix_index(row, col)] = d[reg];
  }
}

__global__ void wgmma_m64n8k16_f16_three_warpgroups_matrix_kernel(
    float *out, const uint16_t *A, const uint16_t *B, const float *C,
    int32_t scale_d) {
  if (threadIdx.x >= kParallelWarpgroupThreads) return;

  int group_id = threadIdx.x / kWarpgroupThreads;
  int local_thread_id = threadIdx.x % kWarpgroupThreads;

  __shared__ __align__(16)
      uint16_t smem_b[kParallelWgmmaGroups][kSharedBElements];

  const uint16_t *tile_A = A + group_id * kM * kK;
  const uint16_t *tile_B = B + group_id * kN * kK;
  const float *tile_C = C + group_id * kOutputRegs;
  float *tile_out = out + group_id * kOutputRegs;

  for (int i = local_thread_id; i < kSharedBElements; i += kWarpgroupThreads) {
    smem_b[group_id][i] = 0;
  }

  for (int i = local_thread_id; i < kN * kK; i += kWarpgroupThreads) {
    int col = i / kK;
    int k = i % kK;
    smem_b[group_id][b_smem_index(col, k)] = tile_B[b_matrix_index(col, k)];
  }
  __syncthreads();

  uint64_t desc_b = make_gmma_k_major_desc(
      smem_b[group_id], kBLeadingByteOffset, kBStrideByteOffset);

  int tid_mma_col = local_thread_id % 4;
  int tid_row = (local_thread_id / 4) % 8;
  int tid_m_block = local_thread_id / 32;

  uint32_t a_regs[4];
  for (int reg = 0; reg < 4; ++reg) {
    int reg_row_block = reg & 0x1;
    int reg_k_block = (reg >> 1) & 0x1;
    int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
    int k0 = 2 * tid_mma_col + 8 * reg_k_block;
    a_regs[reg] = pack_f16_pair(tile_A[a_matrix_index(row, k0)],
                                tile_A[a_matrix_index(row, k0 + 1)]);
  }

  float d[4];
  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(local_thread_id, reg, row, col);
    d[reg] = tile_C[c_matrix_index(row, col)];
  }

  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3]), "+f"(d[0]), "+f"(d[1]), "+f"(d[2]),
                 "+f"(d[3])::"memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  wgmma_m64n8k16_f32_f16_f16_rs(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                                desc_b, d[0], d[1], d[2], d[3], scale_d);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])::"memory");

  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(local_thread_id, reg, row, col);
    tile_out[c_matrix_index(row, col)] = d[reg];
  }
}

RunResult run_wgmma_matrix_kernel(const std::vector<uint16_t> &A,
                                  const std::vector<uint16_t> &B,
                                  const std::vector<float> &C,
                                  int32_t scale_d) {
  RunResult result;
  result.output.assign(kOutputRegs, 0.0f);

  if (A.size() != static_cast<size_t>(kM * kK) ||
      B.size() != static_cast<size_t>(kN * kK) ||
      C.size() != static_cast<size_t>(kM * kN)) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }

  uint16_t *d_A = nullptr;
  uint16_t *d_B = nullptr;
  float *d_C = nullptr;
  float *d_out = nullptr;

  result.setup_error = cudaMalloc(&d_A, A.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) return result;
  result.setup_error = cudaMalloc(&d_B, B.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    return result;
  }
  result.setup_error = cudaMalloc(&d_C, C.size() * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    return result;
  }
  result.setup_error = cudaMalloc(&d_out, kOutputRegs * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return result;
  }

  result.input_copy_error = cudaMemcpy(
      d_A, A.data(), A.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_B, B.data(), B.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  }
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_C, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  if (result.input_copy_error == cudaSuccess) {
    cudaGetLastError();
    wgmma_m64n8k16_f16_matrix_kernel<<<1, kWarpgroupThreads>>>(d_out, d_A, d_B,
                                                               d_C, scale_d);

    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) {
      result.sync_error = cudaDeviceSynchronize();
    }
  }

  if (result.input_copy_error == cudaSuccess &&
      result.launch_error == cudaSuccess && result.sync_error == cudaSuccess) {
    result.copy_error =
        cudaMemcpy(result.output.data(), d_out, kOutputRegs * sizeof(float),
                   cudaMemcpyDeviceToHost);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_out);
  return result;
}

RunResult run_wgmma_three_warpgroups_matrix_kernel(
    const std::vector<uint16_t> &A, const std::vector<uint16_t> &B,
    const std::vector<float> &C, int32_t scale_d) {
  RunResult result;
  result.output.assign(kParallelOutputRegs, 0.0f);

  if (A.size() != static_cast<size_t>(kParallelWgmmaGroups * kM * kK) ||
      B.size() != static_cast<size_t>(kParallelWgmmaGroups * kN * kK) ||
      C.size() != static_cast<size_t>(kParallelOutputRegs)) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }

  uint16_t *d_A = nullptr;
  uint16_t *d_B = nullptr;
  float *d_C = nullptr;
  float *d_out = nullptr;

  result.setup_error = cudaMalloc(&d_A, A.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) return result;
  result.setup_error = cudaMalloc(&d_B, B.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    return result;
  }
  result.setup_error = cudaMalloc(&d_C, C.size() * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    return result;
  }
  result.setup_error = cudaMalloc(&d_out, kParallelOutputRegs * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return result;
  }

  result.input_copy_error = cudaMemcpy(
      d_A, A.data(), A.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_B, B.data(), B.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  }
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_C, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  if (result.input_copy_error == cudaSuccess) {
    cudaGetLastError();
    wgmma_m64n8k16_f16_three_warpgroups_matrix_kernel<<<
        1, kParallelWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);

    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) {
      result.sync_error = cudaDeviceSynchronize();
    }
  }

  if (result.input_copy_error == cudaSuccess &&
      result.launch_error == cudaSuccess && result.sync_error == cudaSuccess) {
    result.copy_error =
        cudaMemcpy(result.output.data(), d_out,
                   kParallelOutputRegs * sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_out);
  return result;
}

constexpr int kFa3WgmmaN = 64;
constexpr int kFa3WgmmaOutputRegs = kM * kFa3WgmmaN;
constexpr int kFa3WgmmaSharedElements = 4096;
constexpr uint32_t kFa3WgmmaLeadingByteOffset = 16;
constexpr uint32_t kFa3WgmmaStrideByteOffset = 1024;
constexpr uint32_t kFa3WgmmaSwizzleMode128B = 1;

__host__ __device__ __forceinline__ int fa3_wgmma_c_index(int row, int col) {
  return row * kFa3WgmmaN + col;
}

__host__ __device__ __forceinline__ int fa3_wgmma_a_index(int row, int k) {
  return row * kK + k;
}

__host__ __device__ __forceinline__ int fa3_wgmma_b_index(int col, int k) {
  return col * kK + k;
}

__host__ __device__ __forceinline__ uint32_t
fa3_wgmma_apply_swizzle(uint32_t byte_offset) {
  uint32_t y_mask = 0x7u << 7;
  return byte_offset ^ ((byte_offset & y_mask) >> 3);
}

__host__ __device__ __forceinline__ int fa3_wgmma_k_major_smem_index(int row,
                                                                     int k) {
  constexpr uint32_t kElementSize = sizeof(uint16_t);
  constexpr uint32_t kElementsPer16B = 16 / kElementSize;
  uint32_t row_group = static_cast<uint32_t>(row / 8);
  uint32_t row_in_group = static_cast<uint32_t>(row % 8);
  uint32_t k_group = static_cast<uint32_t>(k / kElementsPer16B);
  uint32_t k_in_group = static_cast<uint32_t>(k % kElementsPer16B);
  uint32_t byte_offset =
      row_group * kFa3WgmmaStrideByteOffset + row_in_group * 128 +
      k_group * kFa3WgmmaLeadingByteOffset + k_in_group * kElementSize;
  return static_cast<int>(fa3_wgmma_apply_swizzle(byte_offset) / kElementSize);
}

__host__ __device__ __forceinline__ int fa3_wgmma_mn_major_smem_index(int row,
                                                                      int k) {
  constexpr uint32_t kElementSize = sizeof(uint16_t);
  constexpr uint32_t kRowsPerSwizzle = 128 / kElementSize;
  uint32_t row_group = static_cast<uint32_t>(row / kRowsPerSwizzle);
  uint32_t row_in_group = static_cast<uint32_t>(row % kRowsPerSwizzle);
  uint32_t k_group = static_cast<uint32_t>(k / 8);
  uint32_t k_in_group = static_cast<uint32_t>(k % 8);
  uint32_t byte_offset = row_group * kFa3WgmmaLeadingByteOffset +
                         row_in_group * kElementSize +
                         k_group * kFa3WgmmaStrideByteOffset + k_in_group * 128;
  return static_cast<int>(fa3_wgmma_apply_swizzle(byte_offset) / kElementSize);
}

__device__ __forceinline__ uint64_t
make_gmma_swizzle_desc(void const *ptr, uint32_t leading_byte_offset,
                       uint32_t stride_byte_offset, uint32_t swizzle_mode) {
  uint32_t smem_addr = smem_ptr_to_uint(ptr);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  desc |= static_cast<uint64_t>(swizzle_mode & 0x3) << 62;
  return desc;
}

template <int BMajor>
__device__ __forceinline__ void wgmma_m64n64k16_f32_f16_f16_ss(
    uint64_t desc_a, uint64_t desc_b, float (&d)[32], int32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, %35, %36, %37, %38;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0),
        "n"(BMajor));
}

template <int BMajor>
__global__ void wgmma_m64n64k16_f16_ss_fa3_kernel(float *out, const uint16_t *A,
                                                  const uint16_t *B,
                                                  const float *C,
                                                  int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint16_t smem_a[kFa3WgmmaSharedElements];
  __shared__ __align__(128) uint16_t smem_b[kFa3WgmmaSharedElements];

  for (int i = threadIdx.x; i < kFa3WgmmaSharedElements; i += blockDim.x) {
    smem_a[i] = 0;
    smem_b[i] = 0;
  }

  for (int i = threadIdx.x; i < kM * kK; i += blockDim.x) {
    int row = i / kK;
    int k = i % kK;
    smem_a[fa3_wgmma_k_major_smem_index(row, k)] = A[fa3_wgmma_a_index(row, k)];
  }

  for (int i = threadIdx.x; i < kFa3WgmmaN * kK; i += blockDim.x) {
    int col = i / kK;
    int k = i % kK;
    int smem_index = BMajor == 0 ? fa3_wgmma_k_major_smem_index(col, k)
                                 : fa3_wgmma_mn_major_smem_index(col, k);
    smem_b[smem_index] = B[fa3_wgmma_b_index(col, k)];
  }
  __syncthreads();

  uint64_t desc_a = make_gmma_swizzle_desc(smem_a, kFa3WgmmaLeadingByteOffset,
                                           kFa3WgmmaStrideByteOffset,
                                           kFa3WgmmaSwizzleMode128B);
  uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, BMajor == 0 ? kFa3WgmmaLeadingByteOffset : 0,
      kFa3WgmmaStrideByteOffset, kFa3WgmmaSwizzleMode128B);

  float d[32];
  for (int reg = 0; reg < 32; ++reg) {
    int row = 0;
    int col_in_group = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, col_in_group);
    int col = (reg / 4) * 8 + col_in_group;
    d[reg] = C[fa3_wgmma_c_index(row, col)];
  }

  asm volatile(""
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
                 "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
                 "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                 "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                 "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                 "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
                 "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                 "+f"(d[30]), "+f"(d[31])::"memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  wgmma_m64n64k16_f32_f16_f16_ss<BMajor>(desc_a, desc_b, d, scale_d);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile(""
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
                 "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
                 "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                 "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                 "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                 "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
                 "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                 "+f"(d[30]), "+f"(d[31])::"memory");

  for (int reg = 0; reg < 32; ++reg) {
    int row = 0;
    int col_in_group = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, col_in_group);
    int col = (reg / 4) * 8 + col_in_group;
    out[fa3_wgmma_c_index(row, col)] = d[reg];
  }
}

template <int BMajor>
RunResult run_wgmma_m64n64k16_f16_ss_fa3_kernel(const std::vector<uint16_t> &A,
                                                const std::vector<uint16_t> &B,
                                                const std::vector<float> &C,
                                                int32_t scale_d) {
  RunResult result;
  result.output.assign(kFa3WgmmaOutputRegs, 0.0f);

  if (A.size() != static_cast<size_t>(kM * kK) ||
      B.size() != static_cast<size_t>(kFa3WgmmaN * kK) ||
      C.size() != static_cast<size_t>(kFa3WgmmaOutputRegs)) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }

  uint16_t *d_A = nullptr;
  uint16_t *d_B = nullptr;
  float *d_C = nullptr;
  float *d_out = nullptr;

  result.setup_error = cudaMalloc(&d_A, A.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) return result;
  result.setup_error = cudaMalloc(&d_B, B.size() * sizeof(uint16_t));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    return result;
  }
  result.setup_error = cudaMalloc(&d_C, C.size() * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    return result;
  }
  result.setup_error = cudaMalloc(&d_out, result.output.size() * sizeof(float));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return result;
  }

  result.input_copy_error = cudaMemcpy(
      d_A, A.data(), A.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_B, B.data(), B.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  }
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error = cudaMemcpy(
        d_C, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  if (result.input_copy_error == cudaSuccess) {
    cudaGetLastError();
    wgmma_m64n64k16_f16_ss_fa3_kernel<BMajor>
        <<<1, kWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);

    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) {
      result.sync_error = cudaDeviceSynchronize();
    }
  }

  if (result.input_copy_error == cudaSuccess &&
      result.launch_error == cudaSuccess && result.sync_error == cudaSuccess) {
    result.copy_error = cudaMemcpy(result.output.data(), d_out,
                                   result.output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_out);
  return result;
}

class WgmmaF16M64N8K16IntegrationTest : public ::testing::Test {
 protected:
  void assert_run_success(const RunResult &result,
                          size_t expected_output_size = kOutputRegs) {
    ASSERT_EQ(result.setup_error, cudaSuccess)
        << "CUDA setup error: " << cudaGetErrorString(result.setup_error);
    ASSERT_EQ(result.input_copy_error, cudaSuccess)
        << "CUDA input copy error: "
        << cudaGetErrorString(result.input_copy_error);
    ASSERT_EQ(result.launch_error, cudaSuccess)
        << "CUDA launch error: " << cudaGetErrorString(result.launch_error);
    ASSERT_EQ(result.sync_error, cudaSuccess)
        << "CUDA sync error: " << cudaGetErrorString(result.sync_error);
    ASSERT_EQ(result.copy_error, cudaSuccess)
        << "CUDA copy error: " << cudaGetErrorString(result.copy_error);

    ASSERT_EQ(result.output.size(), expected_output_size);
  }

  void compute_reference(const std::vector<uint16_t> &A,
                         const std::vector<uint16_t> &B,
                         const std::vector<float> &C, std::vector<float> &D_ref,
                         int32_t scale_d) {
    D_ref.assign(kM * kN, 0.0f);

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kN; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < kK; ++k) {
          sum += f16_to_f32(A[a_matrix_index(row, k)]) *
                 f16_to_f32(B[b_matrix_index(col, k)]);
        }
        D_ref[c_matrix_index(row, col)] =
            sum + (scale_d != 0 ? C[c_matrix_index(row, col)] : 0.0f);
      }
    }
  }

  void compute_batched_reference(const std::vector<uint16_t> &A,
                                 const std::vector<uint16_t> &B,
                                 const std::vector<float> &C,
                                 std::vector<float> &D_ref, int groups,
                                 int32_t scale_d) {
    D_ref.assign(groups * kOutputRegs, 0.0f);

    for (int group = 0; group < groups; ++group) {
      const uint16_t *tile_A = A.data() + group * kM * kK;
      const uint16_t *tile_B = B.data() + group * kN * kK;
      const float *tile_C = C.data() + group * kOutputRegs;
      float *tile_D = D_ref.data() + group * kOutputRegs;

      for (int row = 0; row < kM; ++row) {
        for (int col = 0; col < kN; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < kK; ++k) {
            sum += f16_to_f32(tile_A[a_matrix_index(row, k)]) *
                   f16_to_f32(tile_B[b_matrix_index(col, k)]);
          }
          tile_D[c_matrix_index(row, col)] =
              sum + (scale_d != 0 ? tile_C[c_matrix_index(row, col)] : 0.0f);
        }
      }
    }
  }

  void validate_matrix_result(const std::vector<float> &actual,
                              const std::vector<float> &expected,
                              float absolute_tolerance,
                              float relative_tolerance) {
    ASSERT_EQ(actual.size(), static_cast<size_t>(kM * kN));
    ASSERT_EQ(expected.size(), static_cast<size_t>(kM * kN));

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kN; ++col) {
        int index = c_matrix_index(row, col);
        float tolerance = std::max(
            absolute_tolerance, std::abs(expected[index]) * relative_tolerance);
        EXPECT_NEAR(actual[index], expected[index], tolerance)
            << "Mismatch at D[" << row << "][" << col << "]" << " index "
            << index << " (got: " << actual[index]
            << ", expected: " << expected[index] << ")";
      }
    }
  }

  void validate_batched_matrix_result(const std::vector<float> &actual,
                                      const std::vector<float> &expected,
                                      int groups, float absolute_tolerance,
                                      float relative_tolerance) {
    ASSERT_EQ(actual.size(), static_cast<size_t>(groups * kOutputRegs));
    ASSERT_EQ(expected.size(), static_cast<size_t>(groups * kOutputRegs));

    for (int group = 0; group < groups; ++group) {
      for (int row = 0; row < kM; ++row) {
        for (int col = 0; col < kN; ++col) {
          int index = group * kOutputRegs + c_matrix_index(row, col);
          float tolerance =
              std::max(absolute_tolerance,
                       std::abs(expected[index]) * relative_tolerance);
          EXPECT_NEAR(actual[index], expected[index], tolerance)
              << "Mismatch at group " << group << " D[" << row << "][" << col
              << "]" << " index " << index << " (got: " << actual[index]
              << ", expected: " << expected[index] << ")";
        }
      }
    }
  }

  void fill_random_inputs(std::vector<uint16_t> &A, std::vector<uint16_t> &B,
                          std::vector<float> &C, uint32_t seed, float min_value,
                          float max_value) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_value, max_value);

    A.resize(kM * kK);
    B.resize(kN * kK);
    C.resize(kM * kN);

    for (uint16_t &value : A) {
      value = f32_to_f16(dist(gen));
    }
    for (uint16_t &value : B) {
      value = f32_to_f16(dist(gen));
    }
    for (float &value : C) {
      value = dist(gen);
    }
  }

  void validate_uniform_wgmma(uint16_t a_bits, uint16_t b_bits, float initial_c,
                              int32_t scale_d, float tolerance = 1e-3f) {
    std::vector<uint16_t> A(kM * kK, a_bits);
    std::vector<uint16_t> B(kN * kK, b_bits);
    std::vector<float> C(kM * kN, initial_c);
    std::vector<float> D_ref;

    compute_reference(A, B, C, D_ref, scale_d);

    RunResult result = run_wgmma_matrix_kernel(A, B, C, scale_d);
    ASSERT_NO_FATAL_FAILURE(assert_run_success(result));
    validate_matrix_result(result.output, D_ref, tolerance,
                           /*relative_tolerance=*/0.0f);
  }
};

class WgmmaF16M64N64K16SsFa3IntegrationTest : public ::testing::Test {
 protected:
  void fill_inputs(std::vector<uint16_t> &A, std::vector<uint16_t> &B,
                   std::vector<float> &C) {
    A.resize(kM * kK);
    B.resize(kFa3WgmmaN * kK);
    C.resize(kFa3WgmmaOutputRegs);

    for (int row = 0; row < kM; ++row) {
      for (int k = 0; k < kK; ++k) {
        float value = 0.03125f * static_cast<float>((row % 9) - 4) +
                      0.015625f * static_cast<float>((k % 5) - 2);
        A[fa3_wgmma_a_index(row, k)] = f32_to_f16(value);
      }
    }

    for (int col = 0; col < kFa3WgmmaN; ++col) {
      for (int k = 0; k < kK; ++k) {
        float value = 0.046875f * static_cast<float>((col % 11) - 5) -
                      0.015625f * static_cast<float>((k % 7) - 3);
        B[fa3_wgmma_b_index(col, k)] = f32_to_f16(value);
      }
    }

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kFa3WgmmaN; ++col) {
        C[fa3_wgmma_c_index(row, col)] = 0.001f * static_cast<float>(row) -
                                         0.0005f * static_cast<float>(col);
      }
    }
  }

  void compute_reference(const std::vector<uint16_t> &A,
                         const std::vector<uint16_t> &B,
                         const std::vector<float> &C, std::vector<float> &D_ref,
                         int32_t scale_d) {
    D_ref.assign(kFa3WgmmaOutputRegs, 0.0f);

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kFa3WgmmaN; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < kK; ++k) {
          sum += f16_to_f32(A[fa3_wgmma_a_index(row, k)]) *
                 f16_to_f32(B[fa3_wgmma_b_index(col, k)]);
        }
        D_ref[fa3_wgmma_c_index(row, col)] =
            sum + (scale_d != 0 ? C[fa3_wgmma_c_index(row, col)] : 0.0f);
      }
    }
  }

  void validate_result(const std::vector<float> &actual,
                       const std::vector<float> &expected,
                       float absolute_tolerance, float relative_tolerance) {
    ASSERT_EQ(actual.size(), static_cast<size_t>(kFa3WgmmaOutputRegs));
    ASSERT_EQ(expected.size(), static_cast<size_t>(kFa3WgmmaOutputRegs));

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kFa3WgmmaN; ++col) {
        int index = fa3_wgmma_c_index(row, col);
        float tolerance = std::max(
            absolute_tolerance, std::abs(expected[index]) * relative_tolerance);
        EXPECT_NEAR(actual[index], expected[index], tolerance)
            << "Mismatch at D[" << row << "][" << col << "]" << " index "
            << index << " (got: " << actual[index]
            << ", expected: " << expected[index] << ")";
      }
    }
  }

  template <int BMajor>
  void run_case() {
    std::vector<uint16_t> A;
    std::vector<uint16_t> B;
    std::vector<float> C;
    std::vector<float> D_ref;

    fill_inputs(A, B, C);
    compute_reference(A, B, C, D_ref, /*scale_d=*/1);

    RunResult result =
        run_wgmma_m64n64k16_f16_ss_fa3_kernel<BMajor>(A, B, C, /*scale_d=*/1);
    ASSERT_NO_FATAL_FAILURE(
        wgmma_test::assert_run_success(result, kFa3WgmmaOutputRegs));
    validate_result(result.output, D_ref, /*absolute_tolerance=*/0.02f,
                    /*relative_tolerance=*/0.02f);
  }
};

TEST_F(WgmmaF16M64N8K16IntegrationTest, AllOnesTest) {
  validate_uniform_wgmma(f32_to_f16(1.0f), f32_to_f16(1.0f), 0.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, ZeroAWithAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(0.0f), f32_to_f16(1.0f), 3.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, MixedSignTest) {
  validate_uniform_wgmma(f32_to_f16(-1.0f), f32_to_f16(2.0f), 0.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, NonZeroAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(2.0f), f32_to_f16(0.5f), 7.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, ScaleDZeroIgnoresAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(1.0f), f32_to_f16(1.0f), 9.0f,
                         /*scale_d=*/0);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, RandomValuesTest) {
  std::vector<uint16_t> A;
  std::vector<uint16_t> B;
  std::vector<float> C;
  std::vector<float> D_ref;

  fill_random_inputs(A, B, C, /*seed=*/42, /*min_value=*/-1.0f,
                     /*max_value=*/1.0f);
  compute_reference(A, B, C, D_ref, /*scale_d=*/1);

  RunResult result = run_wgmma_matrix_kernel(A, B, C, /*scale_d=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_run_success(result));
  validate_matrix_result(result.output, D_ref, /*absolute_tolerance=*/0.02f,
                         /*relative_tolerance=*/0.01f);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, RandomValuesLargeRangeTest) {
  std::vector<uint16_t> A;
  std::vector<uint16_t> B;
  std::vector<float> C;
  std::vector<float> D_ref;

  fill_random_inputs(A, B, C, /*seed=*/123, /*min_value=*/-4.0f,
                     /*max_value=*/4.0f);
  compute_reference(A, B, C, D_ref, /*scale_d=*/1);

  RunResult result = run_wgmma_matrix_kernel(A, B, C, /*scale_d=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_run_success(result));
  validate_matrix_result(result.output, D_ref, /*absolute_tolerance=*/0.1f,
                         /*relative_tolerance=*/0.02f);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, RandomValuesScaleDZeroTest) {
  std::vector<uint16_t> A;
  std::vector<uint16_t> B;
  std::vector<float> C;
  std::vector<float> D_ref;

  fill_random_inputs(A, B, C, /*seed=*/987, /*min_value=*/-2.0f,
                     /*max_value=*/2.0f);
  compute_reference(A, B, C, D_ref, /*scale_d=*/0);

  RunResult result = run_wgmma_matrix_kernel(A, B, C, /*scale_d=*/0);
  ASSERT_NO_FATAL_FAILURE(assert_run_success(result));
  validate_matrix_result(result.output, D_ref, /*absolute_tolerance=*/0.05f,
                         /*relative_tolerance=*/0.02f);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, ThreeWarpgroupsLaunch12WarpsTest) {
  std::vector<uint16_t> A(kParallelWgmmaGroups * kM * kK);
  std::vector<uint16_t> B(kParallelWgmmaGroups * kN * kK);
  std::vector<float> C(kParallelOutputRegs);
  std::vector<float> D_ref;

  for (int group = 0; group < kParallelWgmmaGroups; ++group) {
    uint16_t *tile_A = A.data() + group * kM * kK;
    uint16_t *tile_B = B.data() + group * kN * kK;
    float *tile_C = C.data() + group * kOutputRegs;

    for (int row = 0; row < kM; ++row) {
      for (int k = 0; k < kK; ++k) {
        float value = 0.25f * static_cast<float>(group + 1) +
                      0.03125f * static_cast<float>((row % 7) - 3) +
                      0.015625f * static_cast<float>(k % 5);
        tile_A[a_matrix_index(row, k)] = f32_to_f16(value);
      }
    }

    for (int col = 0; col < kN; ++col) {
      for (int k = 0; k < kK; ++k) {
        float value = 0.125f * static_cast<float>(group + 2) +
                      0.0625f * static_cast<float>(col - 3) -
                      0.015625f * static_cast<float>(k % 4);
        tile_B[b_matrix_index(col, k)] = f32_to_f16(value);
      }
    }

    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < kN; ++col) {
        tile_C[c_matrix_index(row, col)] = static_cast<float>(group) +
                                           0.01f * static_cast<float>(row) +
                                           0.1f * static_cast<float>(col);
      }
    }
  }

  compute_batched_reference(A, B, C, D_ref, kParallelWgmmaGroups,
                            /*scale_d=*/1);

  RunResult result =
      run_wgmma_three_warpgroups_matrix_kernel(A, B, C, /*scale_d=*/1);
  ASSERT_NO_FATAL_FAILURE(
      assert_run_success(result, static_cast<size_t>(kParallelOutputRegs)));
  validate_batched_matrix_result(result.output, D_ref, kParallelWgmmaGroups,
                                 /*absolute_tolerance=*/0.05f,
                                 /*relative_tolerance=*/0.02f);
}

TEST_F(WgmmaF16M64N64K16SsFa3IntegrationTest, KMajorOperandsMatchReference) {
  run_case<0>();
}

TEST_F(WgmmaF16M64N64K16SsFa3IntegrationTest, MnMajorBOperandMatchesReference) {
  run_case<1>();
}

}  // namespace wgmma_test
