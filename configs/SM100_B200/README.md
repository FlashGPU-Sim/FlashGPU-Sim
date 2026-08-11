# SM100_B200 Configuration

Full-scale functional B200/SM100 simulator configuration for FA4 launch and
TCGen05 coverage.

## Purpose

This configuration targets datacenter Blackwell PTX (`sm_100` / `sm_100a`) and
keeps the B200 shared-memory limits visible to the simulator:

- Compute capability: 10.0 / `sm_100`
- PTX force-max capability: 100
- SM count: 148
- Shared memory per SM: 228 KB
- Default shared memory per block: 48 KB
- Maximum opt-in shared memory per block: 227 KB
- Unified L1/shared capacity: 256 KB
- Approximate L2 capacity: 128 MB

The memory hierarchy and timing values are functional placeholders. They are not
calibrated against B200 hardware counters.

## Usage

```bash
./tests/run_tests.py -c SM100_B200 run --arch sm100 --group unit
./tests/dev/fa4/run_fa4_b200_cases.sh --config SM100_B200 --suite smoke
./tests/dev/fa4/run_fa4_b200_cases.sh --config SM100_B200 --suite smoke --non-causal
./tests/dev/fa4/run_fa4_b200_cases.sh --config SM100_B200 --suite smoke --artifact-head-dim 128
./tests/dev/fa4/run_fa4_b200_suite_matrix.sh --config SM100_B200 --suite smoke
./tests/dev/fa4/run_fa4_b200_suite_matrix.sh --config SM100_B200 --suite medium
```

`tests/dev/fa4/fa4_b200_cases.csv` follows the FA2/FA3 prefill workload groups:
`smoke`, `small`, `medium`, and `large`.  Each generated FA4 artifact is still
specialized by head dimension, dtype, and causal mode, so mismatched rows are
reported as `SKIP`; use `tests/dev/fa4/run_fa4_b200_suite_matrix.sh` to run every
artifact variant in a suite automatically.  The matrix runner also supports
`--direction bwd`; backward defaults to export-only coverage to keep routine
runs short, and `--run-bwd` enables the launch smoke.  Forward keeps the simple
Q=K=0, V=1 numeric output check.  Backward currently checks that the generated
kernel launches, synchronizes, and preserves input buffers; it does not yet
validate gradients against a CPU reference.

For FA4 artifacts that emit PTX 9.1 / `sm_100a`, point
`PTXAS_CUDA_INSTALL_PATH` at a CUDA 13 Blackwell-capable ptxas while keeping
`CUDA_INSTALL_PATH` on a toolkit usable by this GPGPU-Sim build.
