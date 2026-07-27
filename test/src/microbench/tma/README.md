# TMA Latency Microbenchmarks

Device-side SM-cycle measurements around TMA-related command sequences, using
`%clock64` / `clock64()`. Goal: compare **real GPU** vs **FlashGPU-Sim** on
the same sequences to locate cycle undercount.

## Cases

| Test | Sequence timed |
|------|----------------|
| `TMALatencyTest.ClockOverhead` | empty `clock64(); clock64()` |
| `TMALatencyTest.MBarrierOnly` | `arrive.expect_tx(0)` + `try_wait.parity` (no data) |
| `TMALatencyTest.TMALoad{256B,1KB,4KB,16KB}` | `arrive.expect_tx` + `cp.async.bulk` load + `try_wait` |
| `TMALatencyTest.TMALoad4KBWarm` | 8× load after prime (per-transfer median) |
| `TMALatencyTest.TMAStore4KB` | store + `commit_group` + `wait_group(0)` |
| `TMALatencyTest.GmemLoadBaseline4KB` | scalar `ld.global` 4KB (no TMA) |
| `TMALatencyTest.SummaryTable` | one-shot table for log capture |

## Run

```bash
cd test

# --- Real GPU (clean shell, no setup_environment) ---
./run_tests.sh bench "TMALatency*"

# --- Simulator (source env first) ---
source ../setup.sh && source ../setup_environment
./run_tests.sh -c SM120_RTX5090_REDUCED bench "TMALatency*"
# or full config (slower):
./run_tests.sh -c SM120_RTX5090 bench "TMALatency*"
```

CSVs are written under `test/run/<GPU_CONFIG>/` when using `run_tests.sh`.

## How to interpret

```
TMA_data_path ≈ TMALoad(size) − MBarrierOnly
size_scaling  ≈ TMALoad(16KB) − TMALoad(1KB)   # AGU issue + BW
store_path    ≈ TMAStore4KB
baseline      ≈ GmemLoadBaseline4KB vs TMALoad4KB
```

Compare each median **native vs sim**. Large relative gaps on a single case
point to that subsystem (mbarrier fixed latency, TMA RTT, bulk wait_group, …).

Sim config knobs (calibrated from this suite on RTX 5090, 2026-07):

```
-ptx_opcode_latency_tma 32
-gpgpu_mbarrier_arrive_latency 200   # was 29; TMA complete_tx delay
-gpgpu_mbarrier_trywait_latency 120  # was 32; also applied on immediate try_wait success
-gpgpu_l2_rop_latency 260
-dram_latency 254
```

Code fix (minimal, flash-only): `src/gpgpu-sim/flash/mbarrier.cc` applies
`trywait_latency` even when `try_wait` succeeds immediately (already-complete
barrier). Without that, only opcode overhead (~15 cycles) was measured.

## Notes

- Uses **linear** `cp.async.bulk.shared::cta.global` (not tensor map 2D), so
  results isolate bulk TMA + mbarrier without tensormap AGU complexity.
- Single CTA / warp-leader issue, matching how TMA is typically elect'd.
- Simulator `clock64` returns `gpu_sim_cycle` (see `ptx_sim.cc`).
- Fewer iterations under simulator (auto-detected).
