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
- L2 topology: 16 memory channels x 12 slices = 192 instances
- Modeled L2 capacity: 126 MiB (approximately 128 MiB)

As in the full configuration, the 12 slices per channel use consecutive
channel indexing with `-gpgpu_non_power2_l2_slice_mapping 1`. The stable
rotation is independent of the detailed DRAM-bank count and is not IPOLY. In
the absence of public hardware evidence for the B200 L2-slice hash, it is a
deterministic modeling assumption.

The L2 uses the same optional multi-issue sector-port model as the full
configuration: independent lookup, hit/dirty-eviction data, and fill widths of
three 32-byte sector work packages per L2 instance and L2 tick. These widths
are current deterministic model values, not publicly documented B200
physical-port counts. Legacy cache-port utilization statistics apply only to
the legacy port model; the multi-issue model reports accepted sector work and
width stalls separately.

Detailed DRAM timing and physical address-to-bank mapping are inherited from
the reduced Blackwell configuration and remain functional placeholders; those
parts are not performance-calibrated for B200.

## Usage

```bash
./tests/run_tests.py -c SM100_B200_REDUCED run --arch sm100 --group unit \
  --gtest-filter 'Tcgen05TmemTest.*'
```

For CUDA integration tests that emit `sm_100a` code, point
`TCGEN05_CUDA_INSTALL_PATH` at a CUDA 13 toolkit or equivalent local shim.
