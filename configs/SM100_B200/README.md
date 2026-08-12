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
- L2 topology: 16 memory channels x 12 slices = 192 instances
- Modeled L2 capacity: 126 MiB (approximately 128 MiB)

The 12 slices in each channel use consecutive channel indexing plus the
explicit stable-rotation policy (`-gpgpu_non_power2_l2_slice_mapping 1`). This
mapping is independent of the detailed DRAM-bank count and is not an IPOLY
mode. In the absence of public hardware evidence for the B200 L2-slice hash,
the stable rotation is a deterministic modeling assumption.

The L2 uses the optional multi-issue sector-port model with independent
lookup, hit/dirty-eviction data, and fill widths of three 32-byte sector work
packages per L2 instance and L2 tick. These widths are current deterministic
model values, not publicly documented B200 physical-port counts. Legacy
cache-port utilization statistics apply only when the legacy port model is
selected; the multi-issue model reports accepted sector work and width stalls
separately for lookup, data, and fill.

Ready requests leave the fixed-latency ROP-delay queues through a separate
three-sector-per-L2-tick service model. The latency and output width remain
independent: each child keeps its assigned ready cycle, while the width only
controls ready 32-byte sector children entering the bounded L2 input FIFO.
Three is a candidate chosen for sensitivity testing alongside lookup width
three, not a publicly documented B200 ROP-port count.

Across the wider transport path, ordinary LD/ST, TMA, and ordinary
`cp.async` retain separate local producer/consumer limits. The TMA and
`cp.async` configurations use 32-byte requests with width four, so either type
can use the full shared four-sector request or response budget when it runs
alone. Mixed responses still arbitrate within one four-sector cluster-dispatch
budget.

Detailed DRAM timing and physical address-to-bank mapping remain functional
placeholders; those parts are not calibrated against B200 hardware counters.

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
