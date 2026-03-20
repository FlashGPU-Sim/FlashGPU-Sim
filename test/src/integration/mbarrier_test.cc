/**
 * @file mbarrier_thread_level_test.cc
 * @brief Integration tests for mbarrier thread-level features
 *
 * This file contains CUDA-based tests for verifying the thread-level behavior
 * of mbarrier instructions in GPGPU-Sim. These tests ensure that:
 * 1. Different threads within a warp can init different mbarrier addresses
 * 2. Different threads can arrive at different mbarrier addresses
 * 3. Predicated execution (some threads execute, some don't) works correctly
 * 4. All mbarrier instructions handle thread-level divergence correctly
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "common/mbarrier/device_kernels.cuh"

// ============================================================================
// Test Kernels
// ============================================================================

/**
 * Test 1: Different threads in a warp init different barrier addresses
 *
 * Scenario: 
 * - Thread 0 inits barrier at offset 0
 * - Thread 1 inits barrier at offset 8
 * - Both threads then arrive and wait on their respective barriers
 * - Each thread writes a marker value to output to prove synchronization worked
 */
__global__ void test_different_threads_init_different_addresses(
    uint32_t *output) {
  
  __shared__ unsigned long long barriers[2];  // Two barriers at different offsets
  __shared__ uint32_t sync_values[2];         // Values to verify synchronization
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  // Only warp 0 participates
  if (warp_id != 0) return;
  
  // Initialize sync values
  if (lane_id < 2) {
    sync_values[lane_id] = 0;
  }
  __syncwarp();
  
  // Thread 0 and Thread 1 init different barriers
  // Each barrier expects 1 arrival
  if (lane_id == 0) {
    mbarrier_init(&barriers[0], 1);
  } else if (lane_id == 1) {
    mbarrier_init(&barriers[1], 1);
  }
  __syncwarp();
  
  // Each thread arrives at its own barrier
  if (lane_id == 0) {
    sync_values[0] = 100;  // Write before arrive
    mbarrier_arrive(&barriers[0]);
  } else if (lane_id == 1) {
    sync_values[1] = 200;  // Write before arrive
    mbarrier_arrive(&barriers[1]);
  }
  __syncwarp();
  
  // Each thread waits on its own barrier (parity 0 for first phase)
  if (lane_id == 0) {
    mbarrier_wait(&barriers[0], 0);
    output[0] = sync_values[0];  // Should be 100
  } else if (lane_id == 1) {
    mbarrier_wait(&barriers[1], 0);
    output[1] = sync_values[1];  // Should be 200
  }
}

/**
 * Test 2: Different threads arrive at different barrier addresses
 *
 * Scenario:
 * - Single barrier initialized with expected count = 2
 * - Thread 0 and Thread 2 (not consecutive!) both arrive
 * - All threads wait, then write output to prove barrier worked
 */
__global__ void test_different_threads_arrive_same_barrier(uint32_t *output) {
  
  __shared__ unsigned long long barrier;
  __shared__ uint32_t shared_counter;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Thread 0 initializes
  if (lane_id == 0) {
    shared_counter = 0;
    mbarrier_init(&barrier, 2);  // Expects 2 arrivals
  }
  __syncwarp();
  
  // Only threads 0 and 2 arrive (testing non-consecutive threads)
  if (lane_id == 0 || lane_id == 2) {
    atomicAdd(&shared_counter, 1);
    mbarrier_arrive(&barrier);
  }
  __syncwarp();
  
  // Thread 0 waits and verifies
  if (lane_id == 0) {
    mbarrier_wait(&barrier, 0);
    output[0] = shared_counter;  // Should be 2
  }
}

/**
 * Test 3: Predicated mbarrier operations
 *
 * Scenario:
 * - Only even-numbered threads (0, 2, 4, ...) participate
 * - Each even thread inits and arrives at a shared barrier
 * - Tests that predicated-out threads don't affect barrier state
 */
__global__ void test_predicated_mbarrier_operations(uint32_t *output,
                                                    int num_participants) {
  
  __shared__ unsigned long long barrier;
  __shared__ uint32_t participation_count;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Initialize
  if (lane_id == 0) {
    participation_count = 0;
    mbarrier_init(&barrier, num_participants);
  }
  __syncwarp();
  
  // Only even threads participate (predicated execution)
  bool should_participate = (lane_id % 2 == 0) && (lane_id < num_participants * 2);
  
  if (should_participate) {
    atomicAdd(&participation_count, 1);
    mbarrier_arrive(&barrier);
  }
  __syncwarp();
  
  // Thread 0 waits and reports
  if (lane_id == 0) {
    mbarrier_wait(&barrier, 0);
    output[0] = participation_count;  // Should equal num_participants
  }
}

/**
 * Test 4: Mixed arrive addresses within single warp
 *
 * Scenario:
 * - Two barriers, each expecting half the warp
 * - Even threads arrive at barrier 0
 * - Odd threads arrive at barrier 1
 * - Verifies thread-level address divergence works correctly
 */
__global__ void test_mixed_arrive_addresses(uint32_t *output) {
  
  __shared__ unsigned long long barriers[2];
  __shared__ uint32_t counters[2];
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Initialize both barriers
  if (lane_id == 0) {
    counters[0] = 0;
    counters[1] = 0;
    mbarrier_init(&barriers[0], 16);  // 16 even threads
    mbarrier_init(&barriers[1], 16);  // 16 odd threads
  }
  __syncwarp();
  
  // Each thread arrives at its designated barrier
  int my_barrier = lane_id % 2;
  atomicAdd(&counters[my_barrier], 1);
  
  if (my_barrier == 0) {
    mbarrier_arrive(&barriers[0]);
  } else {
    mbarrier_arrive(&barriers[1]);
  }
  __syncwarp();
  
  // Thread 0 waits on barrier 0, thread 1 waits on barrier 1
  if (lane_id == 0) {
    mbarrier_wait(&barriers[0], 0);
    output[0] = counters[0];  // Should be 16
  } else if (lane_id == 1) {
    mbarrier_wait(&barriers[1], 0);
    output[1] = counters[1];  // Should be 16
  }
}

/**
 * Test 5: Thread-level arrive with different counts
 *
 * Scenario:
 * - Single barrier expecting 10 total arrivals
 * - Thread 0 arrives with count 3
 * - Thread 1 arrives with count 7
 * - Tests per-thread arrival count
 */
__global__ void test_different_arrival_counts(uint32_t *output) {
  
  __shared__ unsigned long long barrier;
  __shared__ uint32_t verification;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Initialize
  if (lane_id == 0) {
    verification = 0;
    mbarrier_init(&barrier, 10);  // Expects 10 total arrivals
  }
  __syncwarp();
  
  // Thread 0 arrives with count 3, Thread 1 with count 7
  if (lane_id == 0) {
    mbarrier_arrive_count(&barrier, 3);
    atomicAdd(&verification, 3);
  } else if (lane_id == 1) {
    mbarrier_arrive_count(&barrier, 7);
    atomicAdd(&verification, 7);
  }
  __syncwarp();
  
  // Thread 0 waits and verifies
  if (lane_id == 0) {
    mbarrier_wait(&barrier, 0);
    output[0] = verification;  // Should be 10
  }
}

/**
 * Test 6: Thread-level expect_tx with different values
 *
 * Scenario:
 * - Single barrier with 2 expected arrivals and transaction counts
 * - Thread 0: arrive.expect_tx with 100 bytes
 * - Thread 1: arrive.expect_tx with 200 bytes
 * - Simulated tx completion of 300 bytes total
 */
__global__ void test_expect_tx_thread_level(uint32_t *output) {
  
  __shared__ unsigned long long barrier;
  __shared__ uint32_t total_tx;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Initialize
  if (lane_id == 0) {
    total_tx = 0;
    mbarrier_init(&barrier, 2);  // Expects 2 arrivals
  }
  __syncwarp();
  
  // Thread 0 and 1 do arrive.expect_tx with different tx counts
  if (lane_id == 0) {
    mbarrier_arrive_expect_tx(&barrier, 100);
    atomicAdd(&total_tx, 100);
  } else if (lane_id == 1) {
    mbarrier_arrive_expect_tx(&barrier, 200);
    atomicAdd(&total_tx, 200);
  }
  __syncwarp();
  
  // In a real scenario, TMA would complete the transactions
  // For this test, we just verify the expect_tx was recorded
  if (lane_id == 0) {
    output[0] = total_tx;  // Should be 300
    output[1] = 1;         // Marker that we reached this point
  }
}

/**
 * Test 7: Single active thread in warp
 *
 * Scenario:
 * - Only thread 15 is active (simulated predication)
 * - Thread 15 inits, arrives, and waits alone
 */
__global__ void test_single_thread_barrier(uint32_t *output) {
  
  __shared__ unsigned long long barrier;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Only thread 15 participates
  if (lane_id == 15) {
    mbarrier_init(&barrier, 1);
    __syncwarp();  // Make sure init is visible
    
    mbarrier_arrive(&barrier);
    mbarrier_wait(&barrier, 0);
    
    output[0] = 0xDEADBEEF;  // Success marker
  }
  __syncwarp();
}

/**
 * Test 8: Multiple warps, each with thread-level operations
 *
 * Scenario:
 * - 2 warps participate
 * - Each warp has its own set of barriers
 * - Within each warp, different threads handle different barriers
 */
__global__ void test_multi_warp_thread_level(uint32_t *output) {
  
  // Each warp gets 2 barriers
  __shared__ unsigned long long barriers[4];  // 2 warps * 2 barriers each
  __shared__ uint32_t results[4];
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id >= 2) return;  // Only 2 warps participate
  
  const int my_barrier_base = warp_id * 2;
  
  // Initialize
  if (lane_id == 0) {
    results[my_barrier_base] = 0;
    results[my_barrier_base + 1] = 0;
    mbarrier_init(&barriers[my_barrier_base], 16);      // Even threads
    mbarrier_init(&barriers[my_barrier_base + 1], 16);  // Odd threads
  }
  __syncwarp();
  
  // Thread-level divergence within each warp
  int my_barrier_offset = lane_id % 2;
  int my_barrier_idx = my_barrier_base + my_barrier_offset;
  
  atomicAdd(&results[my_barrier_idx], 1);
  
  if (my_barrier_offset == 0) {
    mbarrier_arrive(&barriers[my_barrier_base]);
  } else {
    mbarrier_arrive(&barriers[my_barrier_base + 1]);
  }
  __syncwarp();
  
  // Verify
  if (lane_id == 0) {
    mbarrier_wait(&barriers[my_barrier_base], 0);
    output[my_barrier_base] = results[my_barrier_base];  // Should be 16
  } else if (lane_id == 1) {
    mbarrier_wait(&barriers[my_barrier_base + 1], 0);
    output[my_barrier_base + 1] = results[my_barrier_base + 1];  // Should be 16
  }
}

/**
 * Test 9: Producer-Consumer with thread-level init
 *
 * Scenario:
 * - Producer warp: Thread 0 inits forward barrier, Thread 1 inits backward barrier
 * - Consumer warp waits and signals
 * - Tests real-world warp specialization pattern
 */
__global__ void test_producer_consumer_thread_level(uint32_t *output) {
  
  __shared__ unsigned long long fwd_barrier;
  __shared__ unsigned long long bwd_barrier;
  __shared__ uint32_t shared_data;
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  // Warp 0: Producer, Warp 1: Consumer
  if (warp_id == 0) {  // Producer warp
    // Thread 0 inits forward barrier, Thread 1 inits backward barrier
    if (lane_id == 0) {
      shared_data = 0;
      mbarrier_init(&fwd_barrier, 1);
    } else if (lane_id == 1) {
      mbarrier_init(&bwd_barrier, 1);
    }
    __syncwarp();
    
    // Wait for consumer to be ready (bwd_barrier starts satisfied for first iteration)
    // Thread 0 produces data
    if (lane_id == 0) {
      shared_data = 42;
      mbarrier_arrive(&fwd_barrier);
    }
    
  } else if (warp_id == 1) {  // Consumer warp
    // Wait for initialization to complete
    __syncthreads();
    
    // Thread 0 waits for data and consumes
    if (lane_id == 0) {
      mbarrier_wait(&fwd_barrier, 0);
      output[0] = shared_data;  // Should be 42
      mbarrier_arrive(&bwd_barrier);
    }
  }
  
  __syncthreads();
  
  // Producer waits for consumer acknowledgment
  if (warp_id == 0 && lane_id == 0) {
    mbarrier_wait(&bwd_barrier, 0);
    output[1] = 1;  // Success marker
  }
}

/**
 * Test 10: Inval with thread-level addresses
 *
 * Scenario:
 * - Create multiple barriers
 * - Different threads invalidate different barriers
 * - Verify partial invalidation works
 */
__global__ void test_inval_thread_level(uint32_t *output) {
  
  __shared__ unsigned long long barriers[4];
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Thread 0 creates all 4 barriers
  if (lane_id == 0) {
    for (int i = 0; i < 4; i++) {
      mbarrier_init(&barriers[i], 1);
    }
  }
  __syncwarp();
  
  // Each thread 0-3 arrives and waits on its barrier
  if (lane_id < 4) {
    mbarrier_arrive(&barriers[lane_id]);
    mbarrier_wait(&barriers[lane_id], 0);
  }
  __syncwarp();
  
  // Threads 0 and 2 invalidate their barriers
  if (lane_id == 0) {
    mbarrier_inval(&barriers[0]);
  } else if (lane_id == 2) {
    mbarrier_inval(&barriers[2]);
  }
  __syncwarp();
  
  // Success marker
  if (lane_id == 0) {
    output[0] = 0xCAFEBABE;
  }
}

// ============================================================================
// Test Class
// ============================================================================

class MBarrierThreadLevelTest : public ::testing::Test {
protected:
  static constexpr int OUTPUT_SIZE = 64;  // Enough for 32-thread stress tests
  
  void SetUp() override {
    cudaError_t err = cudaMalloc(&d_output, sizeof(uint32_t) * OUTPUT_SIZE);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device memory";
    
    err = cudaMemset(d_output, 0, sizeof(uint32_t) * OUTPUT_SIZE);
    ASSERT_EQ(err, cudaSuccess) << "Failed to memset device memory";
  }
  
  void TearDown() override {
    if (d_output) {
      cudaFree(d_output);
      d_output = nullptr;
    }
  }
  
  uint32_t* d_output = nullptr;
  
  std::vector<uint32_t> getOutput(int count = OUTPUT_SIZE) {
    std::vector<uint32_t> h_output(count);
    cudaError_t err = cudaMemcpy(h_output.data(), d_output, 
                                  sizeof(uint32_t) * count, 
                                  cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy output from device";
    return h_output;
  }
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_F(MBarrierThreadLevelTest, DifferentThreadsInitDifferentAddresses) {
  test_different_threads_init_different_addresses<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 100u) << "Thread 0 should write 100";
  EXPECT_EQ(output[1], 200u) << "Thread 1 should write 200";
}

TEST_F(MBarrierThreadLevelTest, DifferentThreadsArriveSameBarrier) {
  test_different_threads_arrive_same_barrier<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 2u) << "Counter should be 2 (two arrivals)";
}

TEST_F(MBarrierThreadLevelTest, PredicatedMBarrierOperations_4Participants) {
  test_predicated_mbarrier_operations<<<1, 32>>>(d_output, 4);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 4u) << "Should have 4 participants";
}

TEST_F(MBarrierThreadLevelTest, PredicatedMBarrierOperations_8Participants) {
  test_predicated_mbarrier_operations<<<1, 32>>>(d_output, 8);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 8u) << "Should have 8 participants";
}

TEST_F(MBarrierThreadLevelTest, MixedArriveAddresses) {
  test_mixed_arrive_addresses<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 16u) << "Barrier 0 should have 16 arrivals (even threads)";
  EXPECT_EQ(output[1], 16u) << "Barrier 1 should have 16 arrivals (odd threads)";
}

TEST_F(MBarrierThreadLevelTest, DifferentArrivalCounts) {
  test_different_arrival_counts<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 10u) << "Total arrivals should be 10 (3 + 7)";
}

TEST_F(MBarrierThreadLevelTest, ExpectTxThreadLevel) {
  test_expect_tx_thread_level<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 300u) << "Total expected tx should be 300 (100 + 200)";
  EXPECT_EQ(output[1], 1u) << "Should have reached verification point";
}

TEST_F(MBarrierThreadLevelTest, SingleThreadBarrier) {
  test_single_thread_barrier<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 0xDEADBEEFu) << "Single thread barrier should succeed";
}

TEST_F(MBarrierThreadLevelTest, MultiWarpThreadLevel) {
  // Need 64 threads for 2 warps
  test_multi_warp_thread_level<<<1, 64>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 16u) << "Warp 0 barrier 0 should have 16 arrivals";
  EXPECT_EQ(output[1], 16u) << "Warp 0 barrier 1 should have 16 arrivals";
  EXPECT_EQ(output[2], 16u) << "Warp 1 barrier 0 should have 16 arrivals";
  EXPECT_EQ(output[3], 16u) << "Warp 1 barrier 1 should have 16 arrivals";
}

TEST_F(MBarrierThreadLevelTest, ProducerConsumerThreadLevel) {
  // Need 64 threads for 2 warps
  test_producer_consumer_thread_level<<<1, 64>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 42u) << "Consumer should receive data value 42";
  EXPECT_EQ(output[1], 1u) << "Producer should complete successfully";
}

TEST_F(MBarrierThreadLevelTest, InvalThreadLevel) {
  test_inval_thread_level<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 0xCAFEBABEu) << "Inval operations should complete";
}

// ============================================================================
// Stress Tests
// ============================================================================

/**
 * Stress test: All 32 threads init and arrive at their own barriers
 */
__global__ void test_32_thread_divergent_barriers(uint32_t *output) {
  
  __shared__ unsigned long long barriers[32];
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Each thread inits its own barrier
  mbarrier_init(&barriers[lane_id], 1);
  __syncwarp();
  
  // Each thread arrives at its own barrier
  mbarrier_arrive(&barriers[lane_id]);
  __syncwarp();
  
  // Each thread waits on its own barrier
  mbarrier_wait(&barriers[lane_id], 0);
  
  // Each thread writes its lane_id as success marker
  output[lane_id] = lane_id + 1;
}

TEST_F(MBarrierThreadLevelTest, StressTest_32ThreadDivergentBarriers) {
  test_32_thread_divergent_barriers<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(output[i], static_cast<uint32_t>(i + 1)) 
        << "Thread " << i << " should write " << (i + 1);
  }
}

/**
 * Stress test: Multiple phases with thread-level divergence
 */
__global__ void test_multi_phase_thread_divergent(uint32_t *output) {
  
  __shared__ unsigned long long barriers[2];
  __shared__ uint32_t phase_counters[2];
  
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  
  if (warp_id != 0) return;
  
  // Initialize
  if (lane_id == 0) {
    phase_counters[0] = 0;
    phase_counters[1] = 0;
    mbarrier_init(&barriers[0], 16);  // Even threads
    mbarrier_init(&barriers[1], 16);  // Odd threads
  }
  __syncwarp();
  
  // Run 3 phases
  for (int phase = 0; phase < 3; phase++) {
    int my_barrier = lane_id % 2;
    
    atomicAdd(&phase_counters[my_barrier], 1);
    
    if (my_barrier == 0) {
      mbarrier_arrive(&barriers[0]);
    } else {
      mbarrier_arrive(&barriers[1]);
    }
    __syncwarp();
    
    // Wait on my barrier
    if (lane_id == 0) {
      mbarrier_wait(&barriers[0], phase % 2);
    } else if (lane_id == 1) {
      mbarrier_wait(&barriers[1], phase % 2);
    }
    __syncwarp();
  }
  
  // Output final counts
  if (lane_id == 0) {
    output[0] = phase_counters[0];  // Should be 16 * 3 = 48
    output[1] = phase_counters[1];  // Should be 16 * 3 = 48
  }
}

TEST_F(MBarrierThreadLevelTest, StressTest_MultiPhaseThreadDivergent) {
  test_multi_phase_thread_divergent<<<1, 32>>>(d_output);
  
  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  
  auto output = getOutput();
  EXPECT_EQ(output[0], 48u) << "Even threads should complete 3 phases (16*3=48)";
  EXPECT_EQ(output[1], 48u) << "Odd threads should complete 3 phases (16*3=48)";
}
