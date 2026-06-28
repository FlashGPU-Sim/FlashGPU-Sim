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
- Shared memory per block: 227 KB
- Unified L1/shared capacity: 256 KB
- Approximate L2 capacity: 128 MB

The memory hierarchy and timing values are functional placeholders. They are not
calibrated against B200 hardware counters.

## Usage

```bash
./test/run_tests.sh -c SM100_B200 test "*Tcgen05*"
./test/run_fa4_b200_cases.sh --config SM100_B200 --suite smoke
```

For FA4 artifacts that emit PTX 9.1 / `sm_100a`, point
`PTXAS_CUDA_INSTALL_PATH` at a CUDA 13 Blackwell-capable ptxas while keeping
`CUDA_INSTALL_PATH` on a toolkit usable by this GPGPU-Sim build.
