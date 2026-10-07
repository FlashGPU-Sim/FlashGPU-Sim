#include "suite_main.cuh"

void run_tma_multicast_probes(const SuiteOptions &opt, MetricSink &sink);

int main(int argc, char **argv) {
  return run_suite(argc, argv, "tma_multicast", run_tma_multicast_probes);
}
