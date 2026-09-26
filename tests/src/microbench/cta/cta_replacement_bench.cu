#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct CtaRecord {
  uint32_t block;
  uint32_t sm;
  uint64_t start;
  uint64_t end;
};

struct Options {
  uint32_t blocks = 296;
  uint32_t threads = 512;
  uint64_t payload_cycles = 2000;
  uint32_t dynamic_smem = 231424;
  std::string csv = "cta_replacement.csv";
};

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint32_t read_smid() {
  uint32_t value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__global__ void cta_replacement_kernel(CtaRecord *records,
                                       uint64_t payload_cycles) {
  extern __shared__ uint8_t shared[];
  __shared__ uint64_t start_cycle;

  if (threadIdx.x == 0) {
    shared[0] = static_cast<uint8_t>(blockIdx.x);
    start_cycle = read_clock64();
    records[blockIdx.x].block = blockIdx.x;
    records[blockIdx.x].sm = read_smid();
    records[blockIdx.x].start = start_cycle;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    while (read_clock64() - start_cycle < payload_cycles) {
      asm volatile("" ::: "memory");
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    records[blockIdx.x].end = read_clock64();
    const uint32_t shared_sink = shared[0];
    asm volatile("" : : "r"(shared_sink) : "memory");
  }
}

bool check_cuda(cudaError_t error, const char *operation) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(error));
  return false;
}

bool parse_u64(const char *text, uint64_t *value) {
  if (text[0] == '\0' || text[0] == '-') return false;
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool parse_options(int argc, char **argv, Options *options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const std::size_t equals = argument.find('=');
    if (equals == std::string::npos) {
      std::fprintf(stderr, "expected --name=value, got %s\n", argv[i]);
      return false;
    }
    const std::string name = argument.substr(0, equals);
    const std::string value_text = argument.substr(equals + 1);
    if (name == "--csv") {
      options->csv = value_text;
      continue;
    }
    uint64_t value = 0;
    if (!parse_u64(value_text.c_str(), &value)) {
      std::fprintf(stderr, "invalid integer in %s\n", argv[i]);
      return false;
    }
    if (name == "--blocks" && value <= std::numeric_limits<uint32_t>::max()) {
      options->blocks = static_cast<uint32_t>(value);
    } else if (name == "--threads" &&
               value <= std::numeric_limits<uint32_t>::max()) {
      options->threads = static_cast<uint32_t>(value);
    } else if (name == "--payload-cycles") {
      options->payload_cycles = value;
    } else if (name == "--dynamic-smem" &&
               value <= std::numeric_limits<uint32_t>::max()) {
      options->dynamic_smem = static_cast<uint32_t>(value);
    } else {
      std::fprintf(stderr, "unknown or out-of-range option %s\n", argv[i]);
      return false;
    }
  }
  return options->blocks != 0 && options->threads != 0 &&
         options->threads <= 1024 && options->payload_cycles != 0 &&
         options->dynamic_smem != 0;
}

double median(const std::vector<uint64_t> &sorted) {
  if (sorted.empty()) return 0.0;
  const std::size_t middle = sorted.size() / 2;
  if ((sorted.size() & 1U) != 0) return sorted[middle];
  return (static_cast<double>(sorted[middle - 1]) + sorted[middle]) / 2.0;
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) return 2;

  CtaRecord *device_records = nullptr;
  const std::size_t records_bytes = options.blocks * sizeof(CtaRecord);
  if (!check_cuda(cudaMalloc(&device_records, records_bytes),
                  "allocate CTA records") ||
      !check_cuda(cudaMemset(device_records, 0, records_bytes),
                  "clear CTA records") ||
      !check_cuda(cudaFuncSetAttribute(
                      cta_replacement_kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(options.dynamic_smem)),
                  "set maximum dynamic shared memory") ||
      !check_cuda(cudaFuncSetAttribute(
                      cta_replacement_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout,
                      cudaSharedmemCarveoutMaxShared),
                  "set shared-memory carveout")) {
    cudaFree(device_records);
    return 1;
  }

  cta_replacement_kernel<<<options.blocks, options.threads,
                           options.dynamic_smem>>>(device_records,
                                                   options.payload_cycles);
  if (!check_cuda(cudaPeekAtLastError(), "launch CTA replacement benchmark") ||
      !check_cuda(cudaDeviceSynchronize(), "run CTA replacement benchmark")) {
    cudaFree(device_records);
    return 1;
  }

  std::vector<CtaRecord> records(options.blocks);
  if (!check_cuda(cudaMemcpy(records.data(), device_records, records_bytes,
                             cudaMemcpyDeviceToHost),
                  "copy CTA records")) {
    cudaFree(device_records);
    return 1;
  }
  cudaFree(device_records);

  std::sort(records.begin(), records.end(), [](const CtaRecord &left,
                                                const CtaRecord &right) {
    if (left.sm != right.sm) return left.sm < right.sm;
    if (left.start != right.start) return left.start < right.start;
    return left.block < right.block;
  });

  std::vector<uint64_t> gaps;
  std::vector<uint32_t> ctas_per_sm;
  uint64_t first_start = std::numeric_limits<uint64_t>::max();
  uint64_t last_end = 0;
  std::ofstream csv(options.csv);
  if (!csv) {
    std::fprintf(stderr, "failed to open CSV %s\n", options.csv.c_str());
    return 1;
  }
  csv << "block,sm,start,end,active_cycles,sequence_on_sm,gap_from_previous\n";

  std::size_t begin = 0;
  while (begin < records.size()) {
    const uint32_t sm = records[begin].sm;
    std::size_t end = begin;
    while (end < records.size() && records[end].sm == sm) ++end;
    ctas_per_sm.push_back(static_cast<uint32_t>(end - begin));
    for (std::size_t index = begin; index < end; ++index) {
      const CtaRecord &record = records[index];
      if (record.end < record.start) {
        std::fprintf(stderr, "non-monotonic CTA record for block %u\n",
                     record.block);
        return 1;
      }
      first_start = std::min(first_start, record.start);
      last_end = std::max(last_end, record.end);
      uint64_t gap = 0;
      if (index != begin) {
        const CtaRecord &previous = records[index - 1];
        if (record.start < previous.end) {
          std::fprintf(stderr, "overlapping CTAs on SM %u\n", sm);
          return 1;
        }
        gap = record.start - previous.end;
        gaps.push_back(gap);
      }
      csv << record.block << ',' << record.sm << ',' << record.start << ','
          << record.end << ',' << (record.end - record.start) << ','
          << (index - begin) << ',' << gap << '\n';
    }
    begin = end;
  }

  std::sort(gaps.begin(), gaps.end());
  const uint64_t gap_sum =
      std::accumulate(gaps.begin(), gaps.end(), uint64_t{0});
  const double gap_mean = gaps.empty()
                              ? 0.0
                              : static_cast<double>(gap_sum) / gaps.size();
  const auto cta_minmax =
      std::minmax_element(ctas_per_sm.begin(), ctas_per_sm.end());

  std::printf("cta_benchmark_blocks = %u\n", options.blocks);
  std::printf("cta_benchmark_threads = %u\n", options.threads);
  std::printf("cta_benchmark_dynamic_smem = %u\n", options.dynamic_smem);
  std::printf("cta_benchmark_payload_cycles = %llu\n",
              static_cast<unsigned long long>(options.payload_cycles));
  std::printf("cta_benchmark_observed_sms = %zu\n", ctas_per_sm.size());
  std::printf("cta_benchmark_ctas_per_sm_min = %u\n", *cta_minmax.first);
  std::printf("cta_benchmark_ctas_per_sm_max = %u\n", *cta_minmax.second);
  std::printf("cta_benchmark_gap_count = %zu\n", gaps.size());
  std::printf("cta_benchmark_gap_min = %llu\n",
              static_cast<unsigned long long>(gaps.empty() ? 0 : gaps.front()));
  std::printf("cta_benchmark_gap_mean = %.3f\n", gap_mean);
  std::printf("cta_benchmark_gap_median = %.3f\n", median(gaps));
  std::printf("cta_benchmark_gap_max = %llu\n",
              static_cast<unsigned long long>(gaps.empty() ? 0 : gaps.back()));
  std::printf("cta_benchmark_active_span = %llu\n",
              static_cast<unsigned long long>(last_end - first_start));
  std::printf("cta_benchmark_csv = %s\n", options.csv.c_str());
  std::printf("cta_benchmark_status = PASS\n");
  return 0;
}
