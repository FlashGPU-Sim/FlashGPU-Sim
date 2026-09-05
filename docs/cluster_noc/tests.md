# Tests and calibration

How to **run** what already exists, and what **new** tests each TODO must add. Agents: copy the commands from the TODO you are closing; do not invent a second runner.

Always:

```bash
source setup.sh && source setup_environment
make FLASH=1 -j$(nproc)
```

Use `./test/run_tests.sh`. Do not invoke test binaries by hand.

---

## 1. Existing functional filters

```bash
# Unit (delay-line matrix / generic decode)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"

# Launch + TMA cluster (SM120 reduced)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group integration
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*:*MultiCluster*"

# DSM / remote mbar / OneProducer on H200 reduced (NoC on)
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"

# Hetero GPC cluster-of-2 (odd leftover SM)
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER_HETERO3_2 \
  run test --target sm120 --group integration "*ClusterLaunch*"

# TMA multicast is functional and has no DSM-fabric callsite
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit \
  "GpuTopology.TmaMulticastDoesNotUseDsmFabric"

# Multicast peer data and mbar completion remain functionally ordered; no
# network or SRAM-contention overlay is needed for TMA multicast.
```

SM120 reduced needs a run-dir **overlay** for NoC-on DSM:

```text
-gpgpu_cluster_noc_enable 1
-gpgpu_mbarrier_cluster_enable 1
-gpgpu_dsm_local_latency 37
-gpgpu_dsm_remote_latency 78
```

`run_tests.sh` recopies config; append after copy if you overlay.

### Suites

| Suite | NoC off | NoC on |
|-------|---------|--------|
| `cluster_launch_api_test` | Ex launch / attrs, cluster.sync two-wave | Same; hetero GPC (`HETERO3_2` / CLUSTER132) |
| `tma_cluster_multicast_test` | Immediate peers | Delayed + try_wait |
| `tma_multicast_mask_test` | Pass | Pass |
| `cluster_multicast_multicluster_test` | Isolation | Isolation |
| `dsm_test` | SelfMapa | Remote store/load/atom, Cluster4, drop-on-exit, u64, after-exit abort, delayed-store try_wait |
| `mbarrier_cluster_test` | Local | Remote arrive / try_wait / expect+complete |
| `cluster_real_shaped_test` | TMA accumulate | Same with hop |
| unit `cluster_noc_test` | Matrix / decode | Same |
| unit `cluster_dsm_store_test` | Immediacy | Immediacy |
| `dsm_bw` upstream smoke | `cluster.sync()` + DSM/TMA checksums | full-chip packing (`SM90_H200_CLUSTER132`) |

`barrier.cluster.arrive` / `barrier.cluster.wait` are timing-model operations, not functional no-ops in performance simulation. A cluster barrier must not release before every active warp in every CTA of the reserved TB-cluster group reaches the same phase.

### Topology skips

Helpers: `test/common/gpgpusim_config_topology.h`.

| Macro | Skip when |
|-------|-----------|
| `SKIP_IF_N_CORES_PER_CLUSTER_LT(2)` | m < 2 |
| `SKIP_IF_N_CLUSTERS_LT(2)` | n < 2 |
| `SKIP_IF_NOT_HETERO_GPC()` | uniform `-gpgpu_gpc_sms` / none |

Skips print `WARNING: skipped Suite.Name` before `GTEST_SKIP`. They are not product failures.

---

## 2. Kernel constraints (any new test)

1. mbarrier for cross-rank visibility, not peer-smem spin.
2. No `__syncthreads` + single-thread `try_wait` in one CTA.
3. Producer alive until `mapa` consumers finish.
4. `mapa.u64`.
5. Do not assume instant remote-store visibility.

---

## 3. Tests the fabric must add (from supervisor §16)

Network-only (no full kernel) gtests under `test/src/unit/`. Names should match the TODO id (`DsmFabric_SameDirMix`, …).

### Topology / global NoC (B1–B2)

- `sm_id ↔ (gpc_id, local_sm_id)` round trip, including PG'd holes
- Global SM nodes and L2 nodes do not overlap
- Fixed total SMs, different GPC grouping → **same** shader endpoint count
- Reply returns to **requester SM** node
- Per-SM response FIFO: no cross-SM head-of-line
- L2 hit roofline does not drop when SMs are grouped into GPCs

### Fabric / VC (B3b)

- Request VC alone saturates at the shaped per-SM rate
- Response VC alone saturates at the same physical cap
- Same source request+response **sum** ≤ that cap
- Opposite directions proceed together
- 128 B payload uses **four** 32 B grants
- `read_command` occupies one reverse request-VC flit
- Full request VC does not consume response capacity
- Response makes forward progress (no constructed deadlock)
- Destination VOQ avoids unrelated HOL
- Idle shaper slot is **not** donated
- `skip_mod` and `fixed_tdm` both cap ~21.33 B/cycle; unit-test each policy
- CPC / GPC isolation
- OMP on/off completion cycle repeats

### Endpoint (B3c)

- read_command ↔ read_data correlation
- atomic request ↔ atomic data response
- store/TMA ACK debt; threshold / timeout / idle flush
- Outstanding limit backpressure (LSU stall, not instant write)
- Late ACK blocks CTA/sim exit

### Functional DSM on fabric (B4)

- Local rank / self address
- Remote store/load
- Mixed-lane target join
- Vector access
- Two TB-cluster groups isolated
- Same-SM peer CTA
- Cross-GPC reject
- Stale target CTA abort
- Remote `atom.add`

Existing `DsmTest.*` / `MbarrierClusterTest.*` must still PASS.

### Scoreboard / SMEM (B5)

- Remote load dest regs collide until writeback (**same as local `ld.shared`**)
- RF is **not** valid before `read_data` completion
- Local + DSM at one SM ≤ shared-memory service budget
- Different target SMs have independent SRAM services

### H200 calibration (B6)

Full write-up and pending result contract: [`calibration.md`](calibration.md).

**Config:** `SM90_H200_CLUSTER132` (inferred 6×16 + 2×18 = 132 SMs, fabric on). **Not** the reduced 32-SM cluster. **Not** shipped `SM90_H200` (132×1).
**Threads:** `OMP_NUM_THREADS=4`.  
**Cycle gate:** inner loop ≈ 1e5 `%clock64` cycles; \(|T_{\mathrm{sim}}-T_{\mathrm{H200}}|/T_{\mathrm{H200}} < 10\%\).

The hardware job is `../H200_profiling/run_h200.sbatch`. Its simulator-side
companion is `python3 scripts/run_cluster_noc_demo.py`; it contains no embedded
hardware targets and emits `report.md`, `suite.log`, CSV, and JSON. Use
`--resume` after a fix. The completed representative pass has 33 PASS, four
explicit SKIP, and no FAIL/TIMEOUT/LIMIT results. The skipped cases are the
full MMA instruction sweep, DSM calibration sweep, cycle gate, and GEMM; run
them separately only when that focused calibration is needed.
Use `--profile exhaustive --exclude none` to run all 37 cases.

Size-slope fitting, rather than a single throughput point, is required for DSM
and TMA bandwidth. Check directionality, scaling, and idle-neighbor behavior
without assuming values from a previous run.

Also required: local and remote mbarrier timing, DSM local/remote RTT, isolated
TMA issue, TMA multicast versus unicast, and the Triton G1/G2/G3 GEMM
comparison. Hardware targets are pending from the new exclusive H200 job; do
not use superseded Slurm results as acceptance values.

Exact hash and per-hop credit depth are **not** v1 accept criteria.

Previous H200 latency, bandwidth, and GEMM close-outs are superseded. Repeat
their acceptance checks after the pending result arrives.

Latest local regression (2026-09-05): 27/27 `ClusterNoc*`/`GpuTopology*` and
7/7 `ClusterHangPrevent*` unit tests; 31/31 SM120 cluster integration tests
with three expected topology skips; and 26/26 reduced-H200 `DsmTest.*`,
`MbarrierClusterTest.*`, and `TMAClusterOneProducer*` tests. The latter includes
the two expected watchdog death tests. The 132-SM calibration resume remains
33 PASS and four intentional SKIP results.

The full-chip calibration preset sets `-gpgpu_ptx_register_allocator 0`. With aliasing enabled, `k_tma_issue_pure` can reuse its loop-carried shared destination register and abort on a bogus unaligned address; calibration must use the architectural-register path.

For a bounded Triton correctness check, run `gemm_compare --gemm-smoke` via
`scripts/run_calibration_sim.sh`. Under functional simulation it checks the
one-tile G1 unicast-TMA and G3 no-TMA outputs plus CPU reference samples; G2
multicast is reserved for timing mode because functional CTAs have no live
peer shared memory.

---

## 4. After B-DEPR

After **B-DEPR**, delete all `dsm_latency_matrix_*.csv`, `-gpgpu_dsm_latency_matrix_file`, and `-gpgpu_dsm_remote_latency` as a hop/bandwidth knob. Configs that still set those, or `-gpgpu_dsm_bytes_per_cycle`, as the **bandwidth** model are bugs. Timing residual may remain as `-gpgpu_dsm_base_latency_cycles` if B6e refits it. Update `configs/SM90_H200*`, `configs/SM90_H200_CLUSTER132`, and any overlay comments in this directory.
