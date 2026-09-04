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

# Same-SM TMA multicast applies locally (fabric rejects src==dst)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit \
  "GpuTopology.DsmTma*"

# Multicast peer mbar rides the fabric with the TMA data (mbar-after-data).
# Without it the demo GEMM mcast kernel stalls at 248/256 on the leftover
# 17th SM. Known tradeoff: L10/L11 TMA-mcast latency probes shift because the
# issuer now waits for one peer fabric+SRAM landing.
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

Full write-up, kernel IDs, H200 numbers, and simulator results: [`calibration.md`](calibration.md).

**Config:** `SM90_H200_CLUSTER132` (inferred 4×17 + 4×16 = 132 SMs, fabric on). **Not** the reduced 32-SM cluster. **Not** shipped `SM90_H200` (132×1).  
**Threads:** `OMP_NUM_THREADS=4`.  
**Cycle gate:** inner loop ≈ 1e5 `%clock64` cycles; \(|T_{\mathrm{sim}}-T_{\mathrm{H200}}|/T_{\mathrm{H200}} < 10\%\).

Supervisor-facing full run and H200 comparison: `python3 scripts/run_cluster_noc_demo.py`. See [`calibration.md`](calibration.md) §7 for result files and shorter rehearsal commands.

Size-**slope** (16–96 KiB), not a single 64 KiB point, still required for DSM BW:

- One-way load/store/TMA ≈ 20–21 B/cycle per SM
- Symmetric load loss ~22–23%/dir; store/TMA ~4–6%
- Same-dir mix ≈ one-way ceiling; opposite-dir higher
- TMA 2/4/8/16 SM aggregate ≈ linear
- Idle neighbor does not raise the active SM’s rate

Also required: local mbarrier arrive / try_wait, DSM local/remote RTT, TMA issue (pure 44, not bundle 68), TMA mcast − unicast e2e, and the Triton unicast/multicast GEMM. Job 2111262 supplies correctness-clean GEMM pairs through M=2048; M=4096 still times out and remains in the suite.

Exact hash and per-hop credit depth are **not** v1 accept criteria.

Pre-calibration functional audit (2026-08-31): upstream smoke 8/8; fresh-process BW1–BW12 representatives 17/17; reduced-workload upstream GMEM normal / `cp.async` / TMA 3/3; reduced-config `DsmTest.*` + one-producer/CTA-scope TMA integration filter 18/18. See [`calibration.md`](calibration.md) §5.2.1. These passes do not replace the size-slope or full-workload performance gates.

Latency close-out (2026-09-01): L1–L11 each complete twice on the full-chip preset with <10% error and valid payloads. Fabric unit coverage includes shaped TMA payloads and store/TMA packet-class visibility floors. L12 still requires the H2 hardware result.

Bandwidth close-out (2026-09-01): primary load/store/TMA slopes and 2/4/8/16-SM TMA scaling pass the 10% blog gates. Load+TMA direction sharing passes; ordinary load+store opposite direction retains the documented pre-fabric issue-order limitation. Twenty-repeat cycle gates exceed 1e5 aggregate timed cycles with valid checksums.

Final regressions: 50/50 `DsmEndpoint*`, `DsmFabric*`, and `Transport*` unit tests; 18/18 reduced-config `DsmTest.*` and `TMAClusterMulticastTest.*` integration tests.

The full-chip calibration preset sets `-gpgpu_ptx_register_allocator 0`. With aliasing enabled, `k_tma_issue_pure` can reuse its loop-carried shared destination register and abort on a bogus unaligned address; calibration must use the architectural-register path.

---

## 4. After B-DEPR

After **B-DEPR**, delete all `dsm_latency_matrix_*.csv`, `-gpgpu_dsm_latency_matrix_file`, and `-gpgpu_dsm_remote_latency` as a hop/bandwidth knob. Configs that still set those, or `-gpgpu_dsm_bytes_per_cycle`, as the **bandwidth** model are bugs. Timing residual may remain as `-gpgpu_dsm_base_latency_cycles` if B6e refits it. Update `configs/SM90_H200*`, `configs/SM90_H200_CLUSTER132`, and any overlay comments in this directory.
