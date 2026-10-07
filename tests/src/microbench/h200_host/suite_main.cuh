#pragma once

#include "common.cuh"

// Same banner and CSV trailer as the H200 latency suite main. Each binary
// runs one suite and prints that suite's original metric lines.
int run_suite(int argc, char **argv, const char *suite_name,
              void (*run)(const SuiteOptions &, MetricSink &));
