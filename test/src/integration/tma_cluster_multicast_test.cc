#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// Inline mbarrier helpers matching cp_kernels.cuh patterns.
__device__ inline void mbarrier_init_impl(unsigned long long *bar_addr,
                                           unsigned expected_arrivals) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_addr));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(expected_arrivals));
}

__device__ inline void mbarrier_arrive_expect_tx_impl(
    unsigned long long *bar_addr, unsigned tx_bytes) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"l"(bar_s), "r"(tx_bytes));
}

__device__ inline void mbarrier_try_wait_impl(unsigned long long *bar_addr,
                                               int parity) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT_IMPL:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra.uni LAB_WAIT_IMPL;\n"
      "}\n" ::"l"(bar_s), "r"(parity));
}

// TMA bulk load: shared::cta <- global (regular CTA scope).
template <int bytes>
__device__ inline void cp_async_bulk_cta(void *smem_dst,
                                          const void *global_src,
                                          unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %3, [%2];"
      ::"l"(dst_s), "l"(src_g), "l"(bar_s), "n"(bytes));
}

// TMA cluster multicast: shared::cluster <- global.
template <int bytes>
__device__ inline void cp_async_bulk_cluster(void *smem_dst,
                                              const void *global_src,
                                              unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %3, [%2];"
      ::"l"(dst_s), "l"(src_g), "l"(bar_s), "n"(bytes));
}

// TMA load kernel: each block loads the same input chunk via TMA, then
// every thread copies from shared memory to a unique per-thread slot in
// global memory (no race conditions).
template <int CHUNK_BYTES, bool USE_CLUSTER>
__global__ void tmaLoadKernel(const uint8_t *global_src, uint8_t *global_dst,
                               int total_bytes) {
  __shared__ uint8_t smem[CHUNK_BYTES + 64];
  __shared__ unsigned long long bar;
  __shared__ volatile int done;

  uint8_t *data_buf = smem;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // Thread 0 initializes the mbarrier.
  if (tid == 0) {
    done = 0;
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();

  // Thread 0 issues the TMA load.
  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
    if (USE_CLUSTER) {
      cp_async_bulk_cluster<CHUNK_BYTES>(data_buf, global_src, &bar);
    } else {
      cp_async_bulk_cta<CHUNK_BYTES>(data_buf, global_src, &bar);
    }
  }
  __syncthreads();

  // Thread 0 polls the mbarrier until TMA completes.
  if (tid == 0) {
    mbarrier_try_wait_impl(&bar, 0);
    done = 1;
  }
  __syncthreads();

  // All threads copy their portion from shared memory to global memory.
  // Each thread writes to a UNIQUE index to avoid race conditions.
  int offset = bid * CHUNK_BYTES;
  if (offset + CHUNK_BYTES <= total_bytes) {
    for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
      global_dst[offset + i] = data_buf[i];
    }
  }
}

class TMAClusterMulticastTest : public ::testing::Test {
protected:
  static constexpr int CHUNK_BYTES = 256;
  static constexpr int NUM_BLOCKS = 2;
  static constexpr int THREADS_PER_BLOCK = 32;

  void SetUp() override {
    h_input.resize(CHUNK_BYTES);
    for (int i = 0; i < CHUNK_BYTES; i++) {
      h_input[i] = static_cast<uint8_t>(i);
    }
  }

  void runTest(bool use_cluster, const char *name) {
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, CHUNK_BYTES), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), CHUNK_BYTES,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dst, 0xff, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);

    
    if (use_cluster) {
      tmaLoadKernel<CHUNK_BYTES, true>
          <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst,
                                                          CHUNK_BYTES * NUM_BLOCKS);
    } else {
      tmaLoadKernel<CHUNK_BYTES, false>
          <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst,
                                                          CHUNK_BYTES * NUM_BLOCKS);
    }

    ASSERT_EQ(cudaGetLastError(), cudaSuccess)
        << name << ": Kernel launch failed";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
        << name << ": Kernel execution failed";

    std::vector<uint8_t> h_output(CHUNK_BYTES * NUM_BLOCKS);
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_dst, CHUNK_BYTES * NUM_BLOCKS,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    int errors = 0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
      for (int i = 0; i < CHUNK_BYTES; i++) {
        int idx = b * CHUNK_BYTES + i;
        uint8_t expected = h_input[i];
        uint8_t got = h_output[idx];
        if (got != expected) {
          if (errors < 10) {
            printf("%s Block %d, offset %d: expected %u, got %u\n",
                   name, b, i, expected, got);
          }
          errors++;
        }
      }
    }
    EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

    cudaFree(d_src);
    cudaFree(d_dst);
  }

  std::vector<uint8_t> h_input;
};

TEST_F(TMAClusterMulticastTest, CtaScopeTMALoad) {
  runTest(false, "CTA");
}

TEST_F(TMAClusterMulticastTest, ClusterMulticastTMALoad) {
  runTest(true, "Cluster");
}

// ============================================================================
// [Test 1] Tensor (multi-dimensional) TMA cluster multicast tests
// Uses inline PTX tensormap.replace.* to construct tensormap dynamically,
// following the same pattern as cuda_tma_multidim_test.cc, but with
// .shared::cluster destination for the cluster-multicast variant.
// ============================================================================

// --- Tensor TMA helpers (same pattern as cuda_tma_multidim_test.cc) ---
constexpr int TMAP_SIZE = 128;

__device__ inline void tmap_init_smem(void* tmap_smem, int tid) {
    uint32_t* p = reinterpret_cast<uint32_t*>(tmap_smem);
    if (tid < 32) {
        p[tid] = 0;
    }
    asm volatile("bar.warp.sync -1;");
}

__device__ inline void tmap_set_global_address(uint64_t tmap_smem, uint64_t addr) {
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :: "l"(tmap_smem), "l"(addr));
}

__device__ inline void tmap_cp_fenceproxy(uint64_t global_tmap, uint64_t smem_tmap) {
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 0x80;"
                 :: "l"(global_tmap), "l"(smem_tmap));
}

__device__ inline void tmap_fence_acquire(uint64_t global_tmap) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 0x80;\n"
                 "cp.async.bulk.commit_group;\n"
                 "cp.async.bulk.wait_group.read 0;\n"
                 :: "l"(global_tmap));
}

// Tensor 2D TMA load helpers (CTA-scope and CLUSTER-scope)
__device__ inline void cp_async_bulk_tensor_2d_cta(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t c0, int32_t c1, uint32_t mbar_addr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_addr), "l"(tmap_addr), "r"(c0), "r"(c1), "r"(mbar_addr));
}

__device__ inline void cp_async_bulk_tensor_2d_cluster(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t c0, int32_t c1, uint32_t mbar_addr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_addr), "l"(tmap_addr), "r"(c0), "r"(c1), "r"(mbar_addr));
}

// Tensor 3D TMA load helpers
__device__ inline void cp_async_bulk_tensor_3d_cta(
    uint32_t smem_addr, uint64_t tmap_addr,
    int32_t c0, int32_t c1, int32_t c2, uint32_t mbar_addr) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(smem_addr), "l"(tmap_addr), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_addr));
}

__device__ inline void cp_async_bulk_tensor_3d_cluster(
    uint32_t smem_addr, uint64_t tmap_addr,
    int32_t c0, int32_t c1, int32_t c2, uint32_t mbar_addr) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(smem_addr), "l"(tmap_addr), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_addr));
}

// --- 2D Tensor TMA kernel with cluster/CTA scope ---
template <int TILE_SIZE, bool USE_CLUSTER>
__global__ void tma_tensor_2d_kernel(const float *in, float *out,
                                      int rows, int cols,
                                      uint8_t *global_scratch) {
    constexpr int TILE_BYTES = TILE_SIZE * TILE_SIZE * sizeof(float);
    extern __shared__ uint8_t smem[];

    // 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);

    float* data = reinterpret_cast<float*>(tmap_smem + TMAP_SIZE);
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(tmap_smem + TMAP_SIZE + TILE_BYTES);

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tile_row = (bid * TILE_SIZE) / cols;
    int tile_col = (bid * TILE_SIZE) % cols;

    // PTX TMA: coord0 = innermost dim (elem_size stride), coord1 = strides by globalStrides[0]
    int coord0 = tile_col;  // column (innermost dimension)
    int coord1 = tile_row;  // row (strides by globalStrides[0])

    uint8_t* global_tmap = global_scratch + bid * 256;

    // Init tensormap in smem
    tmap_init_smem(tmap_smem, tid);
    __syncthreads();

    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        tmap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));

        // Set rank = 2
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x1;" :: "l"(tmap_s));

        uint32_t box_dims[2] = {static_cast<uint32_t>(TILE_SIZE), static_cast<uint32_t>(TILE_SIZE)};
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box_dims[0]));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box_dims[1]));

        // D0 = innermost (column), D1 = row — matches coord0/coord1 and stride0.
        uint32_t gd0 = static_cast<uint32_t>(cols), gd1 = static_cast<uint32_t>(rows);
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));

        uint64_t global_stride0 = static_cast<uint64_t>(cols) * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(global_stride0));

        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));

        // float type: elemtype = 0x7 (see NVIDIA PTX ISA tensormap encoding)
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    __syncthreads();

    // Copy tensormap to global and fence
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tmap_cp_fenceproxy(tmap_g, tmap_s);
        tmap_fence_acquire(tmap_g);
    }
    __syncthreads();

    // Init mbarrier
    if (tid == 0) {
        mbarrier_init_impl(bar, 1);
        mbarrier_arrive_expect_tx_impl(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // Issue TMA load - elected thread
    uint32_t elected, pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);

    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        if (USE_CLUSTER) {
            cp_async_bulk_tensor_2d_cluster(smem_data, tmap_g, coord0, coord1, mbar_addr);
        } else {
            cp_async_bulk_tensor_2d_cta(smem_data, tmap_g, coord0, coord1, mbar_addr);
        }
    }
    __syncthreads();

    // Wait for TMA to finish
    if (tid == 0) {
        mbarrier_try_wait_impl(bar, 0);
    }
    __syncthreads();

    // Copy to output - each thread writes to unique slot
    int offset = bid * TILE_SIZE * TILE_SIZE;
    int nelem = TILE_SIZE * TILE_SIZE;
    if (offset + nelem <= rows * cols) {
        for (int i = tid; i < nelem; i += blockDim.x) {
            out[offset + i] = data[i];
        }
    }
}

// --- 3D Tensor TMA kernel with cluster/CTA scope ---
template <int TILE_DIM, bool USE_CLUSTER>
__global__ void tma_tensor_3d_kernel(const float *in, float *out,
                                      int d0, int d1, int d2,
                                      uint8_t *global_scratch) {
    constexpr int TILE_BYTES = TILE_DIM * TILE_DIM * TILE_DIM * sizeof(float);
    constexpr int TILE_ELEMS = TILE_DIM * TILE_DIM * TILE_DIM;
    extern __shared__ uint8_t smem[];

    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);

    float* data = reinterpret_cast<float*>(tmap_smem + TMAP_SIZE);
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(tmap_smem + TMAP_SIZE + TILE_BYTES);

    int tid = threadIdx.x;

    // 3D grid: coord0 -> D0 dim, coord1 -> D1 dim, coord2 -> D2 dim
    // (matches reference tensor TMA convention in cuda_tma_multidim_test.cc)
    int coord0 = blockIdx.x * TILE_DIM;
    int coord1 = blockIdx.y * TILE_DIM;
    int coord2 = blockIdx.z * TILE_DIM;

    int linear_bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    uint8_t* global_tmap = global_scratch + linear_bid * 256;

    tmap_init_smem(tmap_smem, tid);
    __syncthreads();

    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        tmap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));

        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x2;" :: "l"(tmap_s));

        uint32_t box_dims[3] = {static_cast<uint32_t>(TILE_DIM), static_cast<uint32_t>(TILE_DIM), static_cast<uint32_t>(TILE_DIM)};
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box_dims[0]));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box_dims[1]));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(box_dims[2]));

        uint32_t gd0 = static_cast<uint32_t>(d0), gd1 = static_cast<uint32_t>(d1), gd2 = static_cast<uint32_t>(d2);
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(gd2));

        // globalStrides[0] = stride for dim1 = d0 * sizeof(float)
        uint64_t gs0 = static_cast<uint64_t>(d0) * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(gs0));
        // globalStrides[1] = stride for dim2 = d0 * d1 * sizeof(float)
        uint64_t gs1 = static_cast<uint64_t>(d0) * static_cast<uint64_t>(d1) * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x1, %1;" :: "l"(tmap_s), "l"(gs1));

        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));

        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    __syncthreads();

    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tmap_cp_fenceproxy(tmap_g, tmap_s);
        tmap_fence_acquire(tmap_g);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_init_impl(bar, 1);
        mbarrier_arrive_expect_tx_impl(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    uint32_t elected, pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);

    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        if (USE_CLUSTER) {
            cp_async_bulk_tensor_3d_cluster(smem_data, tmap_g, coord0, coord1, coord2, mbar_addr);
        } else {
            cp_async_bulk_tensor_3d_cta(smem_data, tmap_g, coord0, coord1, coord2, mbar_addr);
        }
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_try_wait_impl(bar, 0);
    }
    __syncthreads();

    // Write back tile data contiguously in output buffer
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        out[linear_bid * TILE_ELEMS + i] = data[i];
    }
}

// ============================================================================
// [Test 2] Data-type cluster multicast tests
// Uses linear TMA loads with different data types: float, int32, half (fp16), uint8
// ============================================================================

template <int BYTES, bool USE_CLUSTER>
__global__ void tma_datatype_kernel(const void *global_src, void *global_dst,
                                     int total_bytes) {
    __shared__ uint8_t smem[BYTES + 64];
    __shared__ unsigned long long bar;
    __shared__ volatile int done;

    uint8_t *data_buf = smem;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid == 0) {
        done = 0;
        mbarrier_init_impl(&bar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_expect_tx_impl(&bar, BYTES);
        if (USE_CLUSTER) {
            cp_async_bulk_cluster<BYTES>(data_buf, global_src, &bar);
        } else {
            cp_async_bulk_cta<BYTES>(data_buf, global_src, &bar);
        }
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_try_wait_impl(&bar, 0);
        done = 1;
    }
    __syncthreads();

    int offset = bid * BYTES;
    if (offset + BYTES <= total_bytes) {
        for (int i = tid; i < BYTES; i += blockDim.x) {
            reinterpret_cast<uint8_t*>(global_dst)[offset + i] = data_buf[i];
        }
    }
}

// ============================================================================
// [Test 3] OOB (out-of-bounds) cluster multicast tests
// Uses tensor TMA with coordinates that partially exceed tensor bounds,
// verifying fill_mode=0 (zero fill) for OOB elements.
// ============================================================================

template <int TILE_SIZE, bool USE_CLUSTER>
__global__ void tma_oob_kernel(const float *in, float *out,
                                int rows, int cols,
                                uint8_t *global_scratch,
                                int bad_coord0, int bad_coord1) {
    constexpr int TILE_BYTES = TILE_SIZE * TILE_SIZE * sizeof(float);
    extern __shared__ uint8_t smem[];

    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);

    float* data = reinterpret_cast<float*>(tmap_smem + TMAP_SIZE);
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(tmap_smem + TMAP_SIZE + TILE_BYTES);

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uint8_t* global_tmap = global_scratch + bid * 256;

    // Zero out data first so we can detect zeros from OOB fill
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
        data[i] = 0.0f;
    }
    __syncthreads();

    tmap_init_smem(tmap_smem, tid);
    __syncthreads();

    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        tmap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));

        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x1;" :: "l"(tmap_s));

        uint32_t box_dims[2] = {static_cast<uint32_t>(TILE_SIZE), static_cast<uint32_t>(TILE_SIZE)};
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box_dims[0]));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box_dims[1]));

        // D0 = innermost (column), D1 = row
        uint32_t gdims[2] = {static_cast<uint32_t>(cols), static_cast<uint32_t>(rows)};
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gdims[0]));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gdims[1]));

        uint64_t global_stride0 = static_cast<uint64_t>(cols) * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(global_stride0));

        uint32_t estride[2] = {1, 1};
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(estride[0]));
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(estride[1]));

        // fill_mode=0 means zero-fill OOB elements
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    __syncthreads();

    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tmap_cp_fenceproxy(tmap_g, tmap_s);
        tmap_fence_acquire(tmap_g);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_init_impl(bar, 1);
        mbarrier_arrive_expect_tx_impl(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    uint32_t elected, pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);

    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        if (USE_CLUSTER) {
            // PTX: coord0=innermost(col), coord1=strided(row); bad_coord0/1 are (row,col) from host
            cp_async_bulk_tensor_2d_cluster(smem_data, tmap_g, bad_coord1, bad_coord0, mbar_addr);
        } else {
            cp_async_bulk_tensor_2d_cta(smem_data, tmap_g, bad_coord1, bad_coord0, mbar_addr);
        }
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_try_wait_impl(bar, 0);
    }
    __syncthreads();

    int offset = bid * TILE_SIZE * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
        out[offset + i] = data[i];
    }
}


// ============================================================================
// Test fixtures
// ============================================================================

// --- Tensor 2D Cluster Multicast Test Fixture ---
class TMAClusterMulticastTensor2DTest : public ::testing::Test {
protected:
    static constexpr int TILE_SIZE = 4;  // 4x4 tile = 16 floats = 64 bytes
    // Non-square on purpose: catches global_dim D0/D1 (cols/rows) swaps.
    // With COLS=8, block 1 starts at col=4; swapped dims would OOB-clip that tile.
    static constexpr int ROWS = 4;
    static constexpr int COLS = 8;
    static constexpr int THREADS = 32;
    static constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;

    void SetUp() override {
        h_input.resize(ROWS * COLS);
        for (int i = 0; i < ROWS * COLS; i++) {
            h_input[i] = static_cast<float>(i + 1);
        }
    }

    void runTensor2DTest(bool use_cluster) {
        const char* name = use_cluster ? "Tensor2D-Cluster" : "Tensor2D-CTA";
        int total_elems = ROWS * COLS;
        int num_blocks = total_elems / TILE_ELEMS;
        int smem_per_block = TMAP_SIZE + TILE_ELEMS * sizeof(float) + 8 + 128;
        size_t output_bytes = total_elems * sizeof(float);
        size_t scratch_bytes = num_blocks * 256;

        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;

        ASSERT_EQ(cudaMalloc(&d_in, total_elems * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, output_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_scratch, scratch_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), total_elems * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);

        if (use_cluster) {
            // Cluster scope: use 1 block to avoid cross-block SMEM interference.
            // Cluster multicast writes to same SMEM offset on all SMs in the cluster,
            // so multiple blocks would overwrite each other's data.
            tma_tensor_2d_kernel<TILE_SIZE, true>
                <<<1, THREADS, smem_per_block>>>(d_in, d_out, ROWS, COLS, d_scratch);
        } else {
            tma_tensor_2d_kernel<TILE_SIZE, false>
                <<<num_blocks, THREADS, smem_per_block>>>(d_in, d_out, ROWS, COLS, d_scratch);
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << name << ": Launch failed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << name << ": Kernel failed";

        std::vector<float> h_out(total_elems, 0.0f);
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, output_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);

        int errors = 0;
        // Each tile is a contiguous 4x4 block from the 8x8 input
        int check_blocks = use_cluster ? 1 : num_blocks;  // cluster: only 1 block to avoid SMEM cross-talk
        for (int b = 0; b < check_blocks; b++) {
            int tile_row = b * TILE_SIZE / COLS;
            int tile_col = (b * TILE_SIZE) % COLS;
            for (int ti = 0; ti < TILE_SIZE; ti++) {
                for (int tj = 0; tj < TILE_SIZE; tj++) {
                    int input_row = tile_row + ti;
                    int input_col = tile_col + tj;
                    float expected = h_input[input_row * COLS + input_col];
                    float got = h_out[b * TILE_ELEMS + ti * TILE_SIZE + tj];
                    if (got != expected) {
                        if (errors < 10) {
                            printf("%s Block %d (%d,%d): expected %f, got %f\n",
                                   name, b, ti, tj, expected, got);
                        }
                        errors++;
                    }
                }
            }
        }
        EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_scratch);
    }

    std::vector<float> h_input;
};

// --- Tensor 3D Cluster Multicast Test Fixture ---
class TMAClusterMulticastTensor3DTest : public ::testing::Test {
protected:
    static constexpr int TILE_DIM = 2;  // 2x2x2 = 8 floats = 32 bytes
    static constexpr int D0 = 4;
    static constexpr int D1 = 4;
    static constexpr int D2 = 4;
    static constexpr int THREADS = 32;
    static constexpr int TILE_ELEMS = TILE_DIM * TILE_DIM * TILE_DIM;

    void SetUp() override {
        h_input.resize(D0 * D1 * D2);
        for (int i = 0; i < D0 * D1 * D2; i++) {
            h_input[i] = static_cast<float>(i + 1);
        }
    }

    void runTensor3DTest(bool use_cluster) {
        const char* name = use_cluster ? "Tensor3D-Cluster" : "Tensor3D-CTA";
        int total_elems = D0 * D1 * D2;
        dim3 grid((D0 + TILE_DIM - 1) / TILE_DIM,
                   (D1 + TILE_DIM - 1) / TILE_DIM,
                   (D2 + TILE_DIM - 1) / TILE_DIM);
        int num_blocks = grid.x * grid.y * grid.z;
        int smem_per_block = TMAP_SIZE + TILE_ELEMS * sizeof(float) + 8 + 128;

        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;

        ASSERT_EQ(cudaMalloc(&d_in, total_elems * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, total_elems * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_scratch, num_blocks * 256), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), total_elems * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);

        if (use_cluster) {
            // Cluster scope: use 1 block to avoid cross-block SMEM interference
            tma_tensor_3d_kernel<TILE_DIM, true>
                <<<1, THREADS, smem_per_block>>>(d_in, d_out, D0, D1, D2, d_scratch);
        } else {
            tma_tensor_3d_kernel<TILE_DIM, false>
                <<<grid, THREADS, smem_per_block>>>(d_in, d_out, D0, D1, D2, d_scratch);
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<float> h_out(total_elems, 0.0f);
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, total_elems * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        int errors = 0;
        if (use_cluster) {
            // Cluster scope: only 1 block launched at tile (0,0,0)
            int coord0 = 0, coord1 = 0, coord2 = 0;
            int linear_bid = 0;
            for (int i = 0; i < TILE_ELEMS; i++) {
                int lx = i % TILE_DIM;
                int ly = (i / TILE_DIM) % TILE_DIM;
                int lz = i / (TILE_DIM * TILE_DIM);
                int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz;
                int src_idx = gx + gy * D0 + gz * D0 * D1;
                float expected = h_input[src_idx];
                float got = h_out[linear_bid * TILE_ELEMS + i];
                if (got != expected) {
                    if (errors < 10) {
                        printf("%s Tile (0,0,0) idx=%d: expected %f, got %f\n",
                               name, i, expected, got);
                    }
                    errors++;
                }
            }
        } else {
            int tiles_x = (D0 + TILE_DIM - 1) / TILE_DIM;
            int tiles_y = (D1 + TILE_DIM - 1) / TILE_DIM;
            int tiles_z = (D2 + TILE_DIM - 1) / TILE_DIM;
            for (int tz = 0; tz < tiles_z; tz++) {
                for (int ty = 0; ty < tiles_y; ty++) {
                    for (int tx = 0; tx < tiles_x; tx++) {
                        int coord0 = tx * TILE_DIM;
                        int coord1 = ty * TILE_DIM;
                        int coord2 = tz * TILE_DIM;
                        int linear_bid = tx + ty * tiles_x + tz * tiles_x * tiles_y;
                        for (int i = 0; i < TILE_ELEMS; i++) {
                            int lx = i % TILE_DIM;
                            int ly = (i / TILE_DIM) % TILE_DIM;
                            int lz = i / (TILE_DIM * TILE_DIM);
                            int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz;
                            int src_idx = gx + gy * D0 + gz * D0 * D1;
                            float expected = h_input[src_idx];
                            float got = h_out[linear_bid * TILE_ELEMS + i];
                            if (got != expected) {
                                if (errors < 10) {
                                    printf("%s Tile (%d,%d,%d) idx=%d: expected %f, got %f\n",
                                           name, tx, ty, tz, i, expected, got);
                                }
                                errors++;
                            }
                        }
                    }
                }
            }
        }
        EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_scratch);
    }

    std::vector<float> h_input;
};

// --- Data-Type Cluster Multicast Test Fixture ---
// Tests TMA cluster multicast with different load sizes (32B, 64B, 128B, 256B).
// Different byte sizes map to different data widths (e.g., 32B = 8 floats or 4 doubles).
template <int CHUNK_BYTES>
class TMAClusterMulticastDataTypeTest : public ::testing::Test {
protected:
    static constexpr int NUM_BLOCKS = 2;
    static constexpr int THREADS = 32;

    void runDataTypeTest(bool use_cluster, const char* name) {
        std::vector<uint8_t> h_in(CHUNK_BYTES);
        for (int i = 0; i < CHUNK_BYTES; i++) {
            h_in[i] = static_cast<uint8_t>(i);
        }

        uint8_t *d_src = nullptr, *d_dst = nullptr;
        ASSERT_EQ(cudaMalloc(&d_src, CHUNK_BYTES), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_dst, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), CHUNK_BYTES, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_dst, 0xff, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);

        if (use_cluster) {
            tma_datatype_kernel<CHUNK_BYTES, true>
                <<<NUM_BLOCKS, THREADS>>>(d_src, d_dst, CHUNK_BYTES * NUM_BLOCKS);
        } else {
            tma_datatype_kernel<CHUNK_BYTES, false>
                <<<NUM_BLOCKS, THREADS>>>(d_src, d_dst, CHUNK_BYTES * NUM_BLOCKS);
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << name << ": Launch failed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << name << ": Kernel failed";

        std::vector<uint8_t> h_out(CHUNK_BYTES * NUM_BLOCKS);
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_dst, CHUNK_BYTES * NUM_BLOCKS,
                             cudaMemcpyDeviceToHost), cudaSuccess);

        int errors = 0;
        for (int b = 0; b < NUM_BLOCKS; b++) {
            for (int i = 0; i < CHUNK_BYTES; i++) {
                uint8_t expected = h_in[i];
                uint8_t got = h_out[b * CHUNK_BYTES + i];
                if (got != expected) {
                    if (errors < 10) {
                        printf("%s Block %d, offset %d: expected %u, got %u\n",
                               name, b, i, expected, got);
                    }
                    errors++;
                }
            }
        }
        EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

        cudaFree(d_src);
        cudaFree(d_dst);
    }
};

// Typedefs for specific data sizes
using TMAClusterMulticastDataTypeTest_32B = TMAClusterMulticastDataTypeTest<32>;
using TMAClusterMulticastDataTypeTest_64B = TMAClusterMulticastDataTypeTest<64>;
using TMAClusterMulticastDataTypeTest_128B = TMAClusterMulticastDataTypeTest<128>;
using TMAClusterMulticastDataTypeTest_256B = TMAClusterMulticastDataTypeTest<256>;

// --- OOB (Out-of-Bounds) Cluster Multicast Test Fixture ---
class TMAClusterMulticastOOBTest : public ::testing::Test {
protected:
    static constexpr int TILE_SIZE = 4;
    static constexpr int ROWS = 4;  // small tensor to force OOB
    static constexpr int COLS = 4;
    static constexpr int THREADS = 32;
    static constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;

    void SetUp() override {
        h_input.resize(ROWS * COLS);
        for (int i = 0; i < ROWS * COLS; i++) {
            h_input[i] = static_cast<float>(i + 1);
        }
    }

    void runOOBTest(bool use_cluster) {
        const char* name = use_cluster ? "OOB-Cluster" : "OOB-CTA";
        int smem_per_block = TMAP_SIZE + TILE_ELEMS * sizeof(float) + 8 + 128;

        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;

        ASSERT_EQ(cudaMalloc(&d_in, ROWS * COLS * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, TILE_ELEMS * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_scratch, 256), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), ROWS * COLS * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);

        // Request tile at (2, 2) in a 4x4 tensor with TILE_SIZE=4
        // Coords (2,2) to (5,5): rows 2-3 are valid, rows 4-5 are OOB
        // columns 0-1 are valid, columns 2-3 are OOB partially
        if (use_cluster) {
            tma_oob_kernel<TILE_SIZE, true>
                <<<1, THREADS, smem_per_block>>>(d_in, d_out, ROWS, COLS, d_scratch, 2, 2);
        } else {
            tma_oob_kernel<TILE_SIZE, false>
                <<<1, THREADS, smem_per_block>>>(d_in, d_out, ROWS, COLS, d_scratch, 2, 2);
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<float> h_out(TILE_ELEMS, 0.0f);
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, TILE_ELEMS * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        // Validate: elements within bounds should match input
        // Elements out of bounds should be 0 (fill_mode=0)
        int errors = 0;
        int start_row = 2, start_col = 2;
        for (int ti = 0; ti < TILE_SIZE; ti++) {
            for (int tj = 0; tj < TILE_SIZE; tj++) {
                int src_row = start_row + ti;
                int src_col = start_col + tj;
                float expected = 0.0f;  // OOB = zero fill
                if (src_row < ROWS && src_col < COLS) {
                    expected = h_input[src_row * COLS + src_col];
                }
                float got = h_out[ti * TILE_SIZE + tj];
                if (got != expected) {
                    if (errors < 10) {
                        printf("%s (%d,%d): src(%d,%d) expected %f, got %f\n",
                               name, ti, tj, src_row, src_col, expected, got);
                    }
                    errors++;
                }
            }
        }
        EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_scratch);
    }

    std::vector<float> h_input;
};

// Stress kernel: each block TMA-loads its own chunk at bid * CHUNK_BYTES
// (unlike tma_datatype_kernel, which always loads from the base pointer).
template <int CHUNK_BYTES, bool USE_CLUSTER>
__global__ void tma_stress_kernel(const uint8_t *global_src, uint8_t *global_dst,
                                   int total_bytes) {
    __shared__ uint8_t smem[CHUNK_BYTES + 64];
    __shared__ unsigned long long bar;
    __shared__ volatile int done;

    uint8_t *data_buf = smem;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * CHUNK_BYTES;
    if (offset + CHUNK_BYTES > total_bytes) {
        return;
    }
    const uint8_t *src = global_src + offset;

    if (tid == 0) {
        done = 0;
        mbarrier_init_impl(&bar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
        if (USE_CLUSTER) {
            cp_async_bulk_cluster<CHUNK_BYTES>(data_buf, src, &bar);
        } else {
            cp_async_bulk_cta<CHUNK_BYTES>(data_buf, src, &bar);
        }
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_try_wait_impl(&bar, 0);
        done = 1;
    }
    __syncthreads();

    for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
        global_dst[offset + i] = data_buf[i];
    }
}

// One-producer multi-consumer: only block 0 issues .shared::cluster TMA;
// all blocks wait on their local mbarrier (peer complete_tx must wake them)
// and write the same tile to their output region.
template <int CHUNK_BYTES>
__global__ void tma_one_producer_cluster_kernel(const uint8_t *global_src,
                                                 uint8_t *global_dst,
                                                 int *ready_count,
                                                 int num_blocks) {
    __shared__ uint8_t smem[CHUNK_BYTES + 64];
    __shared__ unsigned long long bar;
    uint8_t *data_buf = smem;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid == 0) {
        mbarrier_init_impl(&bar, 1);
        mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
        // Publish that this CTA's mbarrier is armed before any producer issues.
        atomicAdd(ready_count, 1);
        while (atomicAdd(ready_count, 0) < num_blocks) {
            // spin until every CTA has armed expect_tx
        }
        if (bid == 0) {
            cp_async_bulk_cluster<CHUNK_BYTES>(data_buf, global_src, &bar);
        }
        mbarrier_try_wait_impl(&bar, 0);
    }
    __syncthreads();

    int offset = bid * CHUNK_BYTES;
    for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
        global_dst[offset + i] = data_buf[i];
    }
}

// --- Stress / Large-Scale Cluster Multicast Test Fixture ---
class TMAClusterMulticastStressTest : public ::testing::Test {
protected:
    static constexpr int CHUNK_BYTES = 1024;       // 1KB chunks
    static constexpr int NUM_BLOCKS = 16;           // many blocks
    static constexpr int THREADS = 128;             // full warp x4
    static constexpr int TOTAL_BYTES = CHUNK_BYTES * NUM_BLOCKS; // 16KB

    void SetUp() override {
        h_input.resize(TOTAL_BYTES);
        for (int i = 0; i < TOTAL_BYTES; i++) {
            h_input[i] = static_cast<uint8_t>(i * 7 + 13);  // non-trivial pattern
        }
    }

    void runStressTest(bool use_cluster) {
        const char* name = use_cluster ? "Stress-Cluster" : "Stress-CTA";

        uint8_t *d_src = nullptr, *d_dst = nullptr;
        ASSERT_EQ(cudaMalloc(&d_src, TOTAL_BYTES), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_dst, TOTAL_BYTES), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), TOTAL_BYTES,
                             cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_dst, 0xff, TOTAL_BYTES), cudaSuccess);

        // CTA path: each block loads a distinct bid*CHUNK slice (multi-chunk).
        // Cluster multi-issuer path must load the *same* tile: peer multicast
        // writes the same smem offsets on cluster partners, so different chunks
        // would race. Cluster occupancy is still stressed with NUM_BLOCKS CTAs.
        if (use_cluster) {
            tma_datatype_kernel<CHUNK_BYTES, true>
                <<<NUM_BLOCKS, THREADS>>>(d_src, d_dst, TOTAL_BYTES);
        } else {
            tma_stress_kernel<CHUNK_BYTES, false>
                <<<NUM_BLOCKS, THREADS>>>(d_src, d_dst, TOTAL_BYTES);
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << name << ": Launch failed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << name << ": Kernel failed";

        std::vector<uint8_t> h_out(TOTAL_BYTES);
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_dst, TOTAL_BYTES,
                             cudaMemcpyDeviceToHost), cudaSuccess);

        int errors = 0;
        if (use_cluster) {
            // Every block should hold a copy of h_input[0:CHUNK_BYTES].
            for (int b = 0; b < NUM_BLOCKS; b++) {
                for (int i = 0; i < CHUNK_BYTES; i++) {
                    uint8_t expected = h_input[i];
                    uint8_t got = h_out[b * CHUNK_BYTES + i];
                    if (got != expected) {
                        if (errors < 10) {
                            printf("%s block %d offset %d: expected %u, got %u\n",
                                   name, b, i, expected, got);
                        }
                        errors++;
                    }
                }
            }
        } else {
            for (int i = 0; i < TOTAL_BYTES; i++) {
                if (h_out[i] != h_input[i]) {
                    if (errors < 10) {
                        printf("%s offset %d: expected %u, got %u\n",
                               name, i, h_input[i], h_out[i]);
                    }
                    errors++;
                }
            }
        }
        EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

        cudaFree(d_src);
        cudaFree(d_dst);
    }

    std::vector<uint8_t> h_input;
};

// One producer (CTA 0) multicasts; peer CTAs only wait/consume.
class TMAClusterOneProducerTest : public ::testing::Test {
protected:
    static constexpr int CHUNK_BYTES = 256;
    static constexpr int NUM_BLOCKS = 2;
    static constexpr int THREADS = 32;

    void SetUp() override {
        h_input.resize(CHUNK_BYTES);
        for (int i = 0; i < CHUNK_BYTES; i++) {
            h_input[i] = static_cast<uint8_t>(i * 3 + 5);
        }
    }

    std::vector<uint8_t> h_input;
};

TEST_F(TMAClusterOneProducerTest, OneProducerPeerConsumers) {
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    int *d_ready = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, CHUNK_BYTES), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), CHUNK_BYTES,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dst, 0xff, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);

    tma_one_producer_cluster_kernel<CHUNK_BYTES>
        <<<NUM_BLOCKS, THREADS>>>(d_src, d_dst, d_ready, NUM_BLOCKS);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> h_out(CHUNK_BYTES * NUM_BLOCKS);
    ASSERT_EQ(cudaMemcpy(h_out.data(), d_dst, CHUNK_BYTES * NUM_BLOCKS,
                         cudaMemcpyDeviceToHost), cudaSuccess);

    int errors = 0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int i = 0; i < CHUNK_BYTES; i++) {
            uint8_t expected = h_input[i];
            uint8_t got = h_out[b * CHUNK_BYTES + i];
            if (got != expected) {
                if (errors < 10) {
                    printf("OneProducer block %d offset %d: expected %u, got %u\n",
                           b, i, expected, got);
                }
                errors++;
            }
        }
    }
    EXPECT_EQ(errors, 0) << "OneProducer: Total mismatches: " << errors;

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_ready);
};

// ============================================================================
// Test cases
// ============================================================================

// Tensor 2D tests
TEST_F(TMAClusterMulticastTensor2DTest, CtaScopeTensor2DTMALoad) {
    runTensor2DTest(false);
}

TEST_F(TMAClusterMulticastTensor2DTest, ClusterMulticastTensor2DTMALoad) {
    runTensor2DTest(true);
}

// Tensor 3D tests
TEST_F(TMAClusterMulticastTensor3DTest, CtaScopeTensor3DTMALoad) {
    runTensor3DTest(false);
}

TEST_F(TMAClusterMulticastTensor3DTest, ClusterMulticastTensor3DTMALoad) {
    runTensor3DTest(true);
}

// Data type tests (different load sizes map to different data widths)
// 32 bytes = 8 floats or 4 doubles or 8 ints
TEST_F(TMAClusterMulticastDataTypeTest_32B, CtaScopeData32B) {
    runDataTypeTest(false, "Data32B-CTA");
}

TEST_F(TMAClusterMulticastDataTypeTest_32B, ClusterMulticastData32B) {
    runDataTypeTest(true, "Data32B-Cluster");
}

// 64 bytes = 16 floats
TEST_F(TMAClusterMulticastDataTypeTest_64B, CtaScopeData64B) {
    runDataTypeTest(false, "Data64B-CTA");
}

TEST_F(TMAClusterMulticastDataTypeTest_64B, ClusterMulticastData64B) {
    runDataTypeTest(true, "Data64B-Cluster");
}

// 128 bytes = larger chunk
TEST_F(TMAClusterMulticastDataTypeTest_128B, CtaScopeData128B) {
    runDataTypeTest(false, "Data128B-CTA");
}

TEST_F(TMAClusterMulticastDataTypeTest_128B, ClusterMulticastData128B) {
    runDataTypeTest(true, "Data128B-Cluster");
}

// OOB tests
TEST_F(TMAClusterMulticastOOBTest, CtaScopeOOBTMALoad) {
    runOOBTest(false);
}

TEST_F(TMAClusterMulticastOOBTest, ClusterMulticastOOBTMALoad) {
    runOOBTest(true);
}

// Stress tests
TEST_F(TMAClusterMulticastStressTest, CtaScopeStressLoad) {
    runStressTest(false);
}

TEST_F(TMAClusterMulticastStressTest, ClusterMulticastStressLoad) {
    runStressTest(true);
}
