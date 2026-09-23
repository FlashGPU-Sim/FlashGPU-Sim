# H200 CLUSTER132 calibration status

Frozen snapshot for `configs/SM90_H200_CLUSTER132`. Numbers come from H200 job
2119329 where noted. This is not a live agent log: do not treat `/tmp` paths
or PIDs as current work.

Config: mixed GPC map `-gpgpu_gpc_sms 16,16,16,16,16,16,18,18` (132 SMs),
fabric on, CUDA CC 9.0 occupancy 2048 threads / 32 blocks per SM. HBM geometry
is datasheet-derived (94 simulated channels). IPOLY non-power-of-two mapping
uses mode **3** (bijective rotation). H100 stays at mode **2**.

Reproduce (slow, not default CI). Kernel snapshots live in git-ignored
`calibration/` (populate with `bash scripts/sync_calibration_kernels.sh`;
sibling `H200_profiling` / `NVIDIA-Hopper-Benchmark` trees are optional):

```bash
source setup.sh && source setup_environment
export OMP_NUM_THREADS=4
python3 scripts/run_cluster_noc_demo.py
# optional comparator: python3 scripts/compare_h200_calibration.py
```

Functional cluster / DSM / TMA tests belong on
`SM120_RTX5090_REDUCED_CLUSTER4x4` (and 2x1 / 2x2), not this preset.

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
