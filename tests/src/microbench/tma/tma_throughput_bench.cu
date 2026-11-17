#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kTmaBytes = 8192;
constexpr int kStages = 28;
constexpr int kIssuerWarps = 4;
constexpr int kThreads = kIssuerWarps * 32;
constexpr uint64_t kTransfers = 512;
constexpr uint64_t kMeasuredBytes = kTransfers * kTmaBytes;
constexpr double kCoreClockGhz = 1.930;

struct TmaThroughputResult {
  uint64_t cycles;
  uint64_t checksum;
  uint64_t errors;
};

__host__ __device__ constexpr uint32_t source_pattern(uint64_t word_index) {
  return static_cast<uint32_t>(word_index * 0x9e3779b9ULL) ^ 0xa5a55a5aU;
}

__device__ __forceinline__ uint32_t shared_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void barrier_init(uint64_t *barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
               :
               : "r"(shared_address(barrier))
               : "memory");
}

template <int TmaBytes>
__device__ __forceinline__ void barrier_arrive_expect_tx(uint64_t *barrier) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(shared_address(barrier)), "r"(TmaBytes)
               : "memory");
}

__device__ __forceinline__ void barrier_wait_parity(uint64_t *barrier,
                                                    uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "TMA_THROUGHPUT_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
      "@!complete bra.uni TMA_THROUGHPUT_WAIT;\n"
      "}\n"
      :
      : "r"(shared_address(barrier)), "r"(parity)
      : "memory");
}

__device__ __forceinline__ void barrier_invalidate(uint64_t *barrier) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
               :
               : "r"(shared_address(barrier))
               : "memory");
}

template <int TmaBytes>
__device__ __forceinline__ void issue_tma(uint8_t *shared_destination,
                                          const uint8_t *global_source,
                                          uint64_t *barrier) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "r"(shared_address(shared_destination)),
        "l"(reinterpret_cast<uint64_t>(global_source)), "n"(TmaBytes),
        "r"(shared_address(barrier))
      : "memory");
}

template <int TmaBytes, int Stages, int IssuerWarps>
__global__ void tma_throughput_kernel(const uint8_t *source, uint64_t transfers,
                                      TmaThroughputResult *result) {
  static_assert(TmaBytes == kTmaBytes);
  static_assert(Stages == kStages);
  static_assert(IssuerWarps == kIssuerWarps);
  static_assert(Stages % IssuerWarps == 0);

  extern __shared__ __align__(128) uint8_t shared[];
  uint8_t *buffers = shared;
  uint64_t *barriers = reinterpret_cast<uint64_t *>(
      buffers + static_cast<uint64_t>(Stages) * TmaBytes);
  uint32_t *verify_errors = reinterpret_cast<uint32_t *>(barriers + Stages);
  uint64_t *verify_checksums =
      reinterpret_cast<uint64_t *>(verify_errors + blockDim.x);

  if (threadIdx.x == 0) {
    for (int slot = 0; slot < Stages; ++slot) barrier_init(&barriers[slot]);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  const uint64_t initial = transfers < static_cast<uint64_t>(Stages)
                               ? transfers
                               : static_cast<uint64_t>(Stages);
  uint64_t start = 0;
  if (threadIdx.x == 0) start = read_clock64();
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (warp < IssuerWarps && lane == 0) {
    for (int slot = warp; slot < static_cast<int>(initial);
         slot += IssuerWarps) {
      barrier_arrive_expect_tx<TmaBytes>(&barriers[slot]);
      issue_tma<TmaBytes>(buffers + static_cast<uint64_t>(slot) * TmaBytes,
                          source + static_cast<uint64_t>(slot) * TmaBytes,
                          &barriers[slot]);
    }

    const uint64_t local_total =
        transfers > static_cast<uint64_t>(warp)
            ? (transfers - 1 - static_cast<uint64_t>(warp)) / IssuerWarps + 1
            : 0;
    uint64_t local_completed = 0;
    uint64_t next_transfer = static_cast<uint64_t>(Stages + warp);
    int slot = warp;
    uint32_t parity = 0;

    while (local_completed < local_total) {
      barrier_wait_parity(&barriers[slot], parity);
      ++local_completed;

      if (next_transfer < transfers) {
        barrier_arrive_expect_tx<TmaBytes>(&barriers[slot]);
        issue_tma<TmaBytes>(buffers + static_cast<uint64_t>(slot) * TmaBytes,
                            source + next_transfer * TmaBytes, &barriers[slot]);
      }

      next_transfer += IssuerWarps;
      slot += IssuerWarps;
      if (slot >= Stages) {
        slot = warp;
        parity ^= 1U;
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) result->cycles = read_clock64() - start;

  constexpr int words_per_tma = TmaBytes / sizeof(uint32_t);
  constexpr int verified_words = Stages * words_per_tma;
  constexpr uint64_t complete_rounds = kTransfers / Stages;
  constexpr uint64_t final_partial_round = kTransfers % Stages;
  uint32_t local_errors = 0;
  uint64_t local_checksum = 0;
  for (int index = static_cast<int>(threadIdx.x); index < verified_words;
       index += static_cast<int>(blockDim.x)) {
    const int slot = index / words_per_tma;
    const int word = index % words_per_tma;
    const uint64_t slot_round =
        static_cast<uint64_t>(slot) < final_partial_round ? complete_rounds
                                                          : complete_rounds - 1;
    const uint64_t last_transfer =
        static_cast<uint64_t>(slot) + slot_round * Stages;
    const volatile uint32_t *stage_words =
        reinterpret_cast<volatile uint32_t *>(
            buffers + static_cast<uint64_t>(slot) * TmaBytes);
    const uint32_t actual = stage_words[word];
    const uint32_t expected =
        source_pattern(last_transfer * words_per_tma + word);
    local_checksum += actual;
    if (actual != expected) ++local_errors;
  }
  verify_errors[threadIdx.x] = local_errors;
  verify_checksums[threadIdx.x] = local_checksum;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t errors = 0;
    uint64_t checksum = 0;
    for (int thread = 0; thread < static_cast<int>(blockDim.x); ++thread) {
      errors += verify_errors[thread];
      checksum += verify_checksums[thread];
    }
    for (int slot = 0; slot < Stages; ++slot)
      barrier_invalidate(&barriers[slot]);
    result->errors = errors;
    result->checksum = checksum;
  }
}

bool check_cuda(cudaError_t error, const char *operation) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
  return false;
}

}  // namespace

int main() {
  constexpr std::size_t source_words =
      static_cast<std::size_t>(kMeasuredBytes / sizeof(uint32_t));
  std::vector<uint32_t> host_source(source_words);
  for (std::size_t word = 0; word < host_source.size(); ++word)
    host_source[word] = source_pattern(word);

  uint8_t *device_source = nullptr;
  TmaThroughputResult *device_result = nullptr;
  if (!check_cuda(cudaMalloc(&device_source, kMeasuredBytes),
                  "allocate TMA source") ||
      !check_cuda(cudaMalloc(&device_result, sizeof(TmaThroughputResult)),
                  "allocate TMA result") ||
      !check_cuda(cudaMemcpy(device_source, host_source.data(), kMeasuredBytes,
                             cudaMemcpyHostToDevice),
                  "initialize TMA source") ||
      !check_cuda(cudaMemset(device_result, 0, sizeof(TmaThroughputResult)),
                  "clear TMA result")) {
    return 1;
  }

  constexpr int shared_bytes =
      kStages * kTmaBytes + kStages * sizeof(uint64_t) +
      kThreads * sizeof(uint32_t) + kThreads * sizeof(uint64_t);
  if (!check_cuda(
          cudaFuncSetAttribute(
              tma_throughput_kernel<kTmaBytes, kStages, kIssuerWarps>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes),
          "set TMA dynamic shared-memory size") ||
      !check_cuda(cudaFuncSetAttribute(
                      tma_throughput_kernel<kTmaBytes, kStages, kIssuerWarps>,
                      cudaFuncAttributePreferredSharedMemoryCarveout,
                      cudaSharedmemCarveoutMaxShared),
                  "set TMA shared-memory carveout")) {
    return 1;
  }

  tma_throughput_kernel<kTmaBytes, kStages, kIssuerWarps>
      <<<1, kThreads, shared_bytes>>>(device_source, kTransfers, device_result);
  if (!check_cuda(cudaPeekAtLastError(), "launch TMA throughput kernel") ||
      !check_cuda(cudaDeviceSynchronize(), "run TMA throughput kernel")) {
    return 1;
  }

  TmaThroughputResult result{};
  if (!check_cuda(cudaMemcpy(&result, device_result, sizeof(result),
                             cudaMemcpyDeviceToHost),
                  "copy TMA result")) {
    return 1;
  }

  if (result.cycles == 0) {
    std::fprintf(stderr,
                 "TMA throughput kernel reported zero elapsed cycles\n");
    return 1;
  }
  if (result.errors != 0) {
    std::fprintf(stderr,
                 "TMA throughput verification found %llu mismatched words\n",
                 static_cast<unsigned long long>(result.errors));
    return 1;
  }

  const double bytes_per_cycle =
      static_cast<double>(kMeasuredBytes) / static_cast<double>(result.cycles);
  const double throughput_gbps = bytes_per_cycle * kCoreClockGhz;
  std::printf("tma_benchmark_transfers = %llu\n",
              static_cast<unsigned long long>(kTransfers));
  std::printf("tma_benchmark_bytes = %llu\n",
              static_cast<unsigned long long>(kMeasuredBytes));
  std::printf("tma_benchmark_elapsed_cycles = %llu\n",
              static_cast<unsigned long long>(result.cycles));
  std::printf("tma_benchmark_bytes_per_cycle = %.6f\n", bytes_per_cycle);
  std::printf("tma_benchmark_throughput_gbps = %.6f\n", throughput_gbps);
  std::printf("tma_benchmark_checksum = %llu\n",
              static_cast<unsigned long long>(result.checksum));
  std::printf("tma_benchmark_errors = %llu\n",
              static_cast<unsigned long long>(result.errors));
  std::printf("tma_benchmark_status = PASS\n");

  cudaFree(device_result);
  cudaFree(device_source);
  return 0;
}
