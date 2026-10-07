#include "suite_main.cuh"

#include <cstring>

namespace {

__global__ void k_clock_overhead(uint64_t *samples, int n) {
  if (threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i) {
    uint64_t t0 = 0, t1 = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    samples[i] = t1 - t0;
  }
}

void measure_clock_overhead(SuiteOptions &opt) {
  uint64_t *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(uint64_t) * opt.samples));
  k_clock_overhead<<<1, 32>>>(d, opt.samples);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> h(opt.samples);
  CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(uint64_t) * opt.samples,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  CycleStats s = summarize_u64(h, 0.0);
  print_metric_line("clock64_overhead", s, "empty clock64 pair");
  opt.clock64_overhead = s.median;
}

void print_csv_file(const MetricSink &sink, const char *path) {
  FILE *f = std::fopen(path, "w");
  if (!f) {
    std::fprintf(stderr, "cannot write CSV %s\n", path);
    return;
  }
  std::fprintf(f, "suite,metric,sim_knob,median,p90,mean,unit,notes\n");
  for (const auto &m : sink.metrics) {
    std::string notes = m.notes;
    for (char &c : notes) {
      if (c == ',') c = ';';
      if (c == '\n') c = ' ';
    }
    std::fprintf(f, "%s,%s,%s,%.6g,%.6g,%.6g,%s,%s\n", m.suite.c_str(),
                 m.name.c_str(), m.sim_knob.c_str(), m.median, m.p90, m.mean,
                 m.unit.c_str(), notes.c_str());
  }
  std::fclose(f);
}

}  // namespace

int run_suite(int argc, char **argv, const char *suite_name,
              void (*run)(const SuiteOptions &, MetricSink &)) {
  SuiteOptions opt;
  bool emit_csv = true;
  std::string csv_file = std::string(suite_name) + ".csv";
  std::string job_id = "local";
  if (const char *e = std::getenv("SLURM_JOB_ID")) job_id = e;

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto need = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(a, "--device")) {
      opt.device = std::atoi(need("--device"));
    } else if (!std::strcmp(a, "--samples")) {
      opt.samples = std::atoi(need("--samples"));
    } else if (!std::strcmp(a, "--warmup")) {
      opt.warmup = std::atoi(need("--warmup"));
    } else if (!std::strcmp(a, "--dsm-cluster")) {
      opt.dsm_cluster_size = std::atoi(need("--dsm-cluster"));
    } else if (!std::strcmp(a, "--csv-file")) {
      csv_file = need("--csv-file");
    } else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
      std::printf("Usage: %s [--device N] [--samples N] [--warmup N] "
                  "[--dsm-cluster N] [--csv-file PATH]\n",
                  argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", a);
      return 2;
    }
  }
  if (opt.samples < 1) opt.samples = 1;
  if (opt.warmup < 0) opt.warmup = 0;

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {
    std::fprintf(stderr, "No CUDA device\n");
    return 1;
  }
  if (opt.device < 0 || opt.device >= ndev) {
    std::fprintf(stderr, "Invalid --device %d\n", opt.device);
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));

  std::printf("=== H200 Latency Suite ===\n");
  std::printf("job_id: %s\n", job_id.c_str());
  std::printf("methodology: Luo et al. arXiv:2501.12084 + FlashGPU-Sim TODO\n");
  std::printf("device_index: %d\n", opt.device);
  std::printf("device_name: %s\n", prop.name);
  std::printf("compute_capability: %d.%d\n", prop.major, prop.minor);
  std::printf("sm_count: %d\n", prop.multiProcessorCount);
  opt.sm_clock_mhz = prop.clockRate / 1000.0;
  std::printf("prop_core_clock_mhz: %.3f\n", opt.sm_clock_mhz);
  std::printf("prop_memory_clock_mhz: %.3f\n", prop.memoryClockRate / 1000.0);
  std::printf("max_threads_per_sm: %d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("total_global_mem_bytes: %llu\n",
              static_cast<unsigned long long>(prop.totalGlobalMem));
  std::printf("samples: %d  warmup: %d\n", opt.samples, opt.warmup);
  std::printf("suite: %s\n", suite_name);
  std::printf("csv_file: %s\n", emit_csv ? csv_file.c_str() : "(disabled)");
  std::fflush(stdout);

  if (prop.major < 9) {
    std::fprintf(stderr,
                 "WARNING: cc %d.%d < 9.0 — Hopper probes need sm_90+.\n",
                 prop.major, prop.minor);
  }

  measure_clock_overhead(opt);
  MetricSink sink;
  run(opt, sink);
  if (emit_csv) {
    print_csv_file(sink, csv_file.c_str());
    std::printf("wrote_csv: %s\n", csv_file.c_str());
  }
  std::printf("\n=== done ===\n");
  return 0;
}
