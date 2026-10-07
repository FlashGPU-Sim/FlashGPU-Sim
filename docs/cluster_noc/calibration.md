# H200 CLUSTER132 calibration status

Frozen snapshot for `configs/SM90_H200_CLUSTER132`. Numbers come from H200 job
2119329 where noted. This is not a live agent log: do not treat `/tmp` paths
or PIDs as current work.

Config: mixed GPC map `-gpgpu_gpc_sms 16,16,16,16,16,16,18,18` (132 SMs),
fabric on, CUDA CC 9.0 occupancy 2048 threads / 32 blocks per SM. HBM geometry
is datasheet-derived (94 simulated channels). IPOLY non-power-of-two mapping
uses mode **3** (bijective rotation). H100 stays at mode **2**.

Reproduce with the microbenchmark profiles (slow, not default CI). Build
from the repository root:

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile dsm-bw
./tests/run_tests.py build --arch sm90 --group microbench --profile tma-bw
./tests/run_tests.py build --arch sm90 --group microbench --profile dsm-latency
./tests/run_tests.py build --arch sm90 --group microbench --profile tma-multicast
./tests/run_tests.py build --arch sm90 --group microbench --profile mbarrier-remote
./tests/run_tests.py run --arch sm90 --group microbench --profile tma-latency \
  -c SM90_H200_CLUSTER132 --gtest-filter 'TMALatencyTest.*'
```

`dsm-bw` and `tma-bw` are the `seanzw/random` suites (`4e8c4f91`).
`dsm-latency` is the latency matrix, dependent remote-load round trip,
store-visibility floor, and contention. `tma-multicast` is the size, skew,
and fan-out sweep. `mbarrier-remote` is the remote arrive. The unicast /
multicast / no-TMA GEMM comparison is not part of this runner. Set
`FLASHGPU_ALLOW_CC_MISMATCH=1` when the host is not CC 9.0. Each bench README
under `tests/src/microbench/` names its binary.

Functional cluster / DSM / TMA cases skip on `SM120_RTX5090` (one SM per
GPC, DSM off). This preset can run them. It is slow and is not a default
CI config. See [`tests.md`](tests.md).

---

## What currently matches

| Path | Result vs H200 |
|------|----------------|
| Normal HBM load bandwidth | ~0% error (PASS, &lt;10%) |
| DSM fabric bandwidth / hop shape | Mostly OK vs silicon (supervisor) |
| Cluster launch, `mapa`, remote ld/st/`atom.add`, remote mbarrier | Functionally correct |

Selected latency knobs on CLUSTER132 (arrive, try_wait, TMA multicast floor,
DSM base / store-visibility) were fitted from accepted job-2119329 rows.
Unmeasured fields use the H100 same-Hopper baseline and are labeled as such
in the config comments (`measured` / `inferred`).

---

## Open gaps (not closed by this branch)

| Path | Last measured vs H200 | Notes |
|------|------------------------|-------|
| cp.async HBM bandwidth | ~+12% | FAIL 10% gate |
| TMA unicast HBM bandwidth | ~+12% | FAIL 10% gate |
| TMA cluster multicast timing | Not silicon-quality | Functional fan-out + fixed latency only; no fabric contention |
| GEMM CUDA-event unicast / multicast | Mostly FAIL 10% | Small and K2K cases; `sq_2k` deferred for runtime |
| `red` / `red.async` | Unimplemented | |

TMA multicast is intentionally **not** on the DSM fabric. Do not treat a
fixed `-gpgpu_tma_multicast_latency` as a multicast NoC.

Job 2119329 is only a partial hardware pass (4096³ GEMM timed launch failed).
Do not republish that job as a complete H200 golden set.

---

## Labels

Stamp numbers in configs and comments as **measured**, **patent**,
**inferred**, or **unresolved**. Do not stamp H200 values as Blackwell
hardware fact. See [`evidence.md`](evidence.md).
