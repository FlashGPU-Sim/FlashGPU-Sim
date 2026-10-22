// Cluster special registers.
//
// Reads the PTX registers directly so the check does not depend on nvcc
// lowering a CUDA builtin:
//   %is_explicit_cluster
//   %cluster_ctarank / %cluster_nctarank
//   %cluster_ctaid.{x,y,z} / %cluster_nctaid.{x,y,z}
//   %clusterid.{x,y,z} / %nclusterid.{x,y,z}
//
// An ordinary launch reports a 1-CTA cluster whose id is the block index.
// An explicit cluster launch reports rank and cluster id from blockIdx and
// the launch cluster dim. Multi-SM cases skip when a GPC has too few SMs.
// SM90_H200_CLUSTER132 can run them.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

namespace {

enum SregField {
  kIsExplicit = 0,
  kRank,
  kNRank,
  kCtaidX,
  kCtaidY,
  kCtaidZ,
  kNctaidX,
  kNctaidY,
  kNctaidZ,
  kClidX,
  kClidY,
  kClidZ,
  kNclidX,
  kNclidY,
  kNclidZ,
  kNFields
};

const char *kFieldName[kNFields] = {
    "is_explicit_cluster", "cluster_ctarank", "cluster_nctarank",
    "cluster_ctaid.x",     "cluster_ctaid.y", "cluster_ctaid.z",
    "cluster_nctaid.x",    "cluster_nctaid.y","cluster_nctaid.z",
    "clusterid.x",         "clusterid.y",     "clusterid.z",
    "nclusterid.x",        "nclusterid.y",    "nclusterid.z",
};

__device__ __forceinline__ unsigned read_u32_sreg_rank() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nrank() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_explicit() {
  // %is_explicit_cluster is a predicate, not a .u32 register.
  unsigned v;
  asm volatile(
      "{\n\t"
      ".reg .pred %%p_explicit;\n\t"
      "mov.pred %%p_explicit, %%is_explicit_cluster;\n\t"
      "selp.u32 %0, 1, 0, %%p_explicit;\n\t"
      "}\n"
      : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_ctaid_x() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_ctaid_y() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_ctaid_z() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nctaid_x() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nctaid_y() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nctaid_z() {
  unsigned v;
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_clid_x() {
  unsigned v;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_clid_y() {
  unsigned v;
  asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_clid_z() {
  unsigned v;
  asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nclid_x() {
  unsigned v;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nclid_y() {
  unsigned v;
  asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(v));
  return v;
}
__device__ __forceinline__ unsigned read_u32_sreg_nclid_z() {
  unsigned v;
  asm volatile("mov.u32 %0, %%nclusterid.z;" : "=r"(v));
  return v;
}

__global__ void cluster_sreg_kernel(int *out, int *mismatch) {
  const unsigned fields[kNFields] = {
      read_u32_sreg_explicit(), read_u32_sreg_rank(),   read_u32_sreg_nrank(),
      read_u32_sreg_ctaid_x(),  read_u32_sreg_ctaid_y(), read_u32_sreg_ctaid_z(),
      read_u32_sreg_nctaid_x(), read_u32_sreg_nctaid_y(),
      read_u32_sreg_nctaid_z(), read_u32_sreg_clid_x(),  read_u32_sreg_clid_y(),
      read_u32_sreg_clid_z(),   read_u32_sreg_nclid_x(), read_u32_sreg_nclid_y(),
      read_u32_sreg_nclid_z(),
  };
  __shared__ unsigned lane0[kNFields];
  if (threadIdx.x == 0) {
    const unsigned linear =
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    for (int i = 0; i < kNFields; ++i) {
      lane0[i] = fields[i];
      out[linear * kNFields + i] = static_cast<int>(fields[i]);
    }
  }
  __syncthreads();
  for (int i = 0; i < kNFields; ++i) {
    if (fields[i] != lane0[i])
      atomicAdd(mismatch, 1);
  }
}

struct SregExpect {
  int v[kNFields];
};

SregExpect expect_ordinary(dim3 grid, dim3 block) {
  SregExpect e{};
  e.v[kIsExplicit] = 0;
  e.v[kRank] = 0;
  e.v[kNRank] = 1;
  e.v[kCtaidX] = e.v[kCtaidY] = e.v[kCtaidZ] = 0;
  e.v[kNctaidX] = e.v[kNctaidY] = e.v[kNctaidZ] = 1;
  e.v[kClidX] = static_cast<int>(block.x);
  e.v[kClidY] = static_cast<int>(block.y);
  e.v[kClidZ] = static_cast<int>(block.z);
  e.v[kNclidX] = static_cast<int>(grid.x);
  e.v[kNclidY] = static_cast<int>(grid.y);
  e.v[kNclidZ] = static_cast<int>(grid.z);
  return e;
}

SregExpect expect_cluster(dim3 grid, dim3 cluster, dim3 block) {
  const unsigned rel_x = block.x % cluster.x;
  const unsigned rel_y = block.y % cluster.y;
  const unsigned rel_z = block.z % cluster.z;
  SregExpect e{};
  e.v[kIsExplicit] = 1;
  e.v[kRank] = static_cast<int>(rel_x + cluster.x * (rel_y + cluster.y * rel_z));
  e.v[kNRank] = static_cast<int>(cluster.x * cluster.y * cluster.z);
  e.v[kCtaidX] = static_cast<int>(rel_x);
  e.v[kCtaidY] = static_cast<int>(rel_y);
  e.v[kCtaidZ] = static_cast<int>(rel_z);
  e.v[kNctaidX] = static_cast<int>(cluster.x);
  e.v[kNctaidY] = static_cast<int>(cluster.y);
  e.v[kNctaidZ] = static_cast<int>(cluster.z);
  e.v[kClidX] = static_cast<int>(block.x / cluster.x);
  e.v[kClidY] = static_cast<int>(block.y / cluster.y);
  e.v[kClidZ] = static_cast<int>(block.z / cluster.z);
  e.v[kNclidX] = static_cast<int>(grid.x / cluster.x);
  e.v[kNclidY] = static_cast<int>(grid.y / cluster.y);
  e.v[kNclidZ] = static_cast<int>(grid.z / cluster.z);
  return e;
}

void check_records(const std::vector<int> &got, dim3 grid, dim3 cluster,
                   bool is_cluster) {
  for (unsigned z = 0; z < grid.z; ++z) {
    for (unsigned y = 0; y < grid.y; ++y) {
      for (unsigned x = 0; x < grid.x; ++x) {
        const unsigned linear = x + grid.x * (y + grid.y * z);
        const SregExpect exp =
            is_cluster ? expect_cluster(grid, cluster, dim3(x, y, z))
                       : expect_ordinary(grid, dim3(x, y, z));
        for (int f = 0; f < kNFields; ++f) {
          EXPECT_EQ(got[linear * kNFields + f], exp.v[f])
              << kFieldName[f] << " at block (" << x << "," << y << "," << z
              << ")";
        }
      }
    }
  }
}

void run_sregs(dim3 grid, dim3 cluster, bool is_cluster) {
  const unsigned ncta = grid.x * grid.y * grid.z;
  const size_t nbytes = ncta * kNFields * sizeof(int);
  int *d_out = nullptr;
  int *d_mismatch = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, nbytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_mismatch, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, nbytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_mismatch, 0, sizeof(int)), cudaSuccess);

  if (is_cluster) {
    void *args[] = {&d_out, &d_mismatch};
    ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                  (const void *)cluster_sreg_kernel, grid, dim3(32), cluster,
                  args),
              cudaSuccess);
  } else {
    cluster_sreg_kernel<<<grid, 32>>>(d_out, d_mismatch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> h(ncta * kNFields, -1);
  int mismatch = -1;
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, nbytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&mismatch, d_mismatch, sizeof(int),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  cudaFree(d_mismatch);

  EXPECT_EQ(mismatch, 0) << "lanes in one CTA disagreed on a cluster register";
  check_records(h, grid, cluster, is_cluster);
}

}  // namespace

TEST(ClusterSregTest, OrdinaryLaunch_UsesBlockIdxAsClusterId) {
  run_sregs(dim3(2, 2, 2), dim3(1, 1, 1), false);
}

TEST(ClusterSregTest, ClusterDim2x1x1_RankAndClusterId) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  run_sregs(dim3(4, 1, 1), dim3(2, 1, 1), true);
}

TEST(ClusterSregTest, ClusterDim2x2x1_RankSpansY) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);
  run_sregs(dim3(4, 2, 1), dim3(2, 2, 1), true);
}

TEST(ClusterSregTest, ClusterDim1x1x2_RankSpansZ) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  run_sregs(dim3(1, 1, 4), dim3(1, 1, 2), true);
}
