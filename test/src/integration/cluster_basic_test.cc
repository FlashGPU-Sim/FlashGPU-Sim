 #include <gtest/gtest.h>
 #include <cuda_runtime.h>
 #include <vector>

 // Each thread writes a deterministic value derived from its block and thread
 // indices. With 2 blocks and a 2-SM-per-cluster reduced config, this exercises
 // round-robin CTA issuance across the two cores in the simt_core_cluster.
 //
 // Topology / scheduling only: plain <<<N, threads>>> launches — not CUDA
 // Thread Block Clusters (no cudaLaunchKernelEx / __cluster_dims__). Prefer
 // SM120_RTX5090_REDUCED_CLUSTER2 when running this test.
 __global__ void clusterBasicKernel(int* output) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   output[idx] = blockIdx.x * 1000 + threadIdx.x;
 }

 class ClusterBasicTest : public ::testing::Test {
  protected:
   static constexpr int kThreadsPerBlock = 32;
   static constexpr int kNumBlocks = 2;
   static constexpr int kNumElements = kThreadsPerBlock * kNumBlocks;
 };

 TEST_F(ClusterBasicTest, TwoBlocksRunCorrectly) {
   std::vector<int> h_output(kNumElements, -1);
   int* d_output = nullptr;

   ASSERT_EQ(cudaMalloc(&d_output, kNumElements * sizeof(int)), cudaSuccess);
   ASSERT_EQ(cudaMemset(d_output, 0xff, kNumElements * sizeof(int)),
             cudaSuccess);

   clusterBasicKernel<<<kNumBlocks, kThreadsPerBlock>>>(d_output);
   ASSERT_EQ(cudaGetLastError(), cudaSuccess);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

   ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, kNumElements * sizeof(int),
                        cudaMemcpyDeviceToHost),
             cudaSuccess);
   cudaFree(d_output);

   for (int b = 0; b < kNumBlocks; ++b) {
     for (int t = 0; t < kThreadsPerBlock; ++t) {
       int idx = b * kThreadsPerBlock + t;
       EXPECT_EQ(h_output[idx], b * 1000 + t)
           << "Mismatch at block " << b << " thread " << t;
     }
   }
 }
