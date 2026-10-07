#include "suite_main.cuh"

void run_dsm_calibration_probes(const SuiteOptions &opt, MetricSink &sink);

int main(int argc, char **argv) {
  return run_suite(argc, argv, "dsm_calibration", run_dsm_calibration_probes);
}
