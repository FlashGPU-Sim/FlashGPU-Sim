# SM90_H100_REDUCED Configuration

Single-SM Hopper/H100 configuration for FA3 and Hopper instruction debugging.

This config keeps the full H100 per-SM resources, including shared memory,
TMA, mbarrier, WGMMA, cache, and latency settings, but reduces the simulated SM
count from 132 to 1. It is intended for debugging hangs and collecting smaller
logs, not for performance or scaling validation.

Key differences from `SM90_H100`:

- `-gpgpu_n_clusters`: 132 -> 1
- `-trace_sampling_core`: -1 -> 0
- Interconnect topology uses the reduced `k = 2` single-SM layout.

Example:

```bash
cd test
FA3_RUN_32KI=1 ./run_tests.sh -c SM90_H100_REDUCED hopper Fa3PrefillFp16IntegrationTest.H16D128FullB64S512
```
