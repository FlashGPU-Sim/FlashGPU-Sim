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

Ready requests leave the fixed-latency ROP-delay queues through the same
separate three-sector-per-L2-tick service candidate as the full model. This
does not alter any child's ready cycle. It only controls ready 32-byte sector
children entering the bounded L2 input FIFO, and it is a deterministic
sensitivity assumption rather than a documented B200 physical-port count.

The reduced model matches the full model's TMA and ordinary `cp.async`
granularity and local request/response widths: 32-byte requests with width four
on both sides. Each type can therefore use the complete shared four-sector
budget when it runs alone, while mixed responses still share one cluster
dispatch budget. Its TMA max-inflight value remains `0` (unlimited), and its
transaction quota remains at the unlimited code default: this one-SM model is
for functional tests and does not inherit the full model's throughput
calibration.

The reduced model matches the full profile's active
`1080:1080:1155:3996` base-clock tuple and keeps the same commented
`1930:1930:1964:3996` high-clock alternative. It also matches the full model's
fixed L1/L2 latency candidates and unvalidated uniform 800-cycle DRAM
candidate. Detailed DRAM timing and physical address-to-bank mapping remain
functional placeholders; those parts are not performance-calibrated for B200.

## Usage

```bash
./tests/run_tests.py -c SM100_B200_REDUCED run --arch sm100 --group unit \
  --gtest-filter 'Tcgen05TmemTest.*'
```

For CUDA integration tests that emit `sm_100a` code, point
`TCGEN05_CUDA_INSTALL_PATH` at a CUDA 13 toolkit or equivalent local shim.
