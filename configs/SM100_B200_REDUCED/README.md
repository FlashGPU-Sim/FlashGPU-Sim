# SM100_B200_REDUCED Configuration

Lightweight B200/SM100 simulator configuration for TCGen05 functional tests.

## Purpose

This configuration exists to keep TCGen05 tests aligned with the datacenter
Blackwell target used by FA4 B200 PTX. It is intentionally reduced to one SM so
small parser, TMEM, mbarrier, and MMA tests run quickly.

## Current Scope

- Compute capability: 10.0 / `sm_100`
- PTX force-max capability: 100
- SM count: 1
- Shared memory per SM: 228 KB
- Default shared memory per block: 48 KB
- Maximum opt-in shared memory per block: 227 KB

The memory hierarchy and timing values are inherited from the existing reduced
Blackwell config as functional placeholders. They are not performance-calibrated
for B200.

## Usage

```bash
./tests/run_tests.py -c SM100_B200_REDUCED run --arch sm100 --group unit \
  --gtest-filter 'Tcgen05TmemTest.*'
```

For CUDA integration tests that emit `sm_100a` code, point
`TCGEN05_CUDA_INSTALL_PATH` at a CUDA 13 toolkit or equivalent local shim.
