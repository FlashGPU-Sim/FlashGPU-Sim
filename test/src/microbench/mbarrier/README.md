# MBarrier Microbenchmark Suite

This directory contains the `mbarrier` latency and visibility microbenchmarks.
The current suite builds a single bench binary from
`mbarrier_trywait_latency_bench.cc`.

## Supported Tests

Run these tests from `test/` through `microbench/sm120/mbarrier`.

### Sweep Cases

| Test | Description | Output |
|------|-------------|--------|
| `MBarrierLatencyTest.WaitFalse` | False-path `mbarrier.try_wait.parity` dependency chain. | `MBarrierLatencyTest.WaitFalse.csv` |
| `MBarrierLatencyTest.WaitTrue` | True-path `mbarrier.try_wait.parity` dependency chain. | `MBarrierLatencyTest.WaitTrue.csv` |
| `MBarrierLatencyTest.LoadChase` | Pointer-chasing `ld.shared` baseline. | `MBarrierLatencyTest.LoadChase.csv` |
| `MBarrierLatencyTest.PredGate` | Predicated `ld.shared` pointer chase baseline. | `MBarrierLatencyTest.PredGate.csv` |
| `MBarrierLatencyTest.PredChain` | `ld.shared + setp + selp` chain used for latency decomposition. | `MBarrierLatencyTest.PredChain.csv` |

### Summary / Visibility Cases

| Test | Description | Output |
|------|-------------|--------|
| `MBarrierLatencyTest.Breakdown` | Compares sweep slopes and reports the incremental latency contribution of each chain. | `MBarrierLatencyTest.Breakdown.csv` |
| `MBarrierLatencyTest.ArriveLocal` | Same-thread `arrive_expect_tx` followed by `try_wait`. | `MBarrierLatencyTest.ArriveLocal.csv` |
| `MBarrierLatencyTest.ArriveWarp` | Cross-warp arrive-to-observe visibility latency. | `MBarrierLatencyTest.ArriveWarp.csv` |

## Example Commands

```bash
./run_tests.py run --arch sm120 --test-group microbench --profile mbarrier \
  "MBarrierLatencyTest.WaitFalse"
./run_tests.py run --arch sm120 --test-group microbench --profile mbarrier \
  "MBarrierLatencyTest.Arrive*"
./run_tests.py run --arch sm120 --test-group microbench --profile mbarrier
```

## Notes

- These microbenchmarks are intended for native GPU mode. In simulator mode
  they skip because the timing interpretation and some `try_wait` behaviors are
  not reliable there.
- All CSV files are written into the active `test/run/<GPU_CONFIG>/` directory
  because `run_tests.py` executes each bench from inside that config directory.
- Shared helpers live under `test/common/mbarrier/` and are reused by both the
  microbench and integration sanity tests.
