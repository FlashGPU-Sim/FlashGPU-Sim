# Living checklist (agent entry point)

**Audience:** a future agent with **no** chat history. Read this file, then only the docs listed under the item you pick.

**Working spec:** [`README.md`](README.md). Do not treat `docs/cluster.md` or the old `docs/cluster_noc.md` stub as design. Supervisor plan v2 is a local reference (not in this tree); ignore its chapter 18.

## How to take a task

1. Pick the **lowest undone ID** whose **Prereqs** are all `[x]`.
2. Read the **Read first** links. Do not skim unrelated chapters.
3. Implement **only** that ID. Do not mix rename + fabric + calibration in one patch.
4. Run the **Verify** commands.
5. Flip `[ ]` → `[x]`, add date + one-line evidence (test names, config).
6. Stop. Leave follow-ons to their own IDs. Do **not** `git commit`. Reply with the list of modified files and a draft commit message. The human reviews and commits.

### ID rules

- Never reuse an ID.
- Do not re-open **A-*** items marked done unless you have new failing evidence.
- New work gets `B-*` or `A-F4-*`, not a revived `L2-1`.
- English only in checkboxes and close-out notes.
- Checklist IDs (`B1`, `Track B`, `A-F1`, …) live **only** in this file. Do not put them in source comments, config comments, or commit messages. Name the change in product terms (GPC rename, per-SM global NoC, delay-line hop, …).

### Suggested order

```text
C1 (docs — this rewrite)
  → B0 baseline
  → B1 naming / topology
  → B2 per-SM global NoC
  → B3a transport primitives
  → B3b dsm_fabric_t
  → B3c endpoint protocol
  → B4 functional DSM on fabric
  → B5 scoreboard + SRAM service   // remote load = local SMEM load
  → B6 H200 slope calibration
  → B8 TMA mcast / remote mbar on fabric
  → B-DEPR deprecate delay-line knobs
  → B7 per-bank SRAM (optional)
  → B9 research extras (not first delivery)
```

`A-F4` (`red`/`red.async`) is independent and may wait until a dump emits those ops.

---

# Track A — Functional baseline (delay-line era)

Shipped on `cluster_cta2_support`. Do not re-implement. Tests: [`tests.md`](tests.md).

- [x] **A-F1** Land delay-line NoC + cluster launch + TMA cluster + DSM + remote mbar. Evidence: ClusterNoc*, DsmTest*, MbarrierClusterTest*, TMAClusterOneProducer* on SM120 reduced and SM90 H200 reduced.
- [x] **A-F2** Used mbarrier ops + `try_wait` timeout dest-pred.
- [x] **A-F3** `barrier.cluster` / CG map **non-goal**.
- [ ] **A-F4** PTX `red` / `red.async`. Still `inst_not_implemented`. nvcc 12.8 `atomicAdd`+`mapa` emits generic `atom.add` (already works). Implement only if a real dump emits `red`.
- [x] **A-F5** `mapa` of inactive rank **aborts**.
- [x] **A-F6** TMA corners: used paths work; 96B swizzle / unused tensormap abort.
- [x] **A-F7** Hang preventers (bare peer spin; mixed `bar.sync`+`try_wait`).
- [x] **A-F8** Integration/unit surface listed in [`tests.md`](tests.md).
- [x] **A-F9** `-gpgpu_dsm_store_immediate` default 0 (write on deliver).

---

# Track B — Target GPC / DSM fabric

## B0 — Baseline and evidence labels

- [x] **B0** Freeze what “no change” means before refactors. Closed 2026-08-25. No C++ / knob change.

**Read first:** [`evidence.md`](evidence.md), [`tests.md`](tests.md).

**Why:** Later phases must prove they did not silently drop L2 bandwidth or break DsmTest.

**Work:**

1. Record current `icnt` shader node count vs `n_clusters` vs total SMs for `SM90_H200_REDUCED_CLUSTER4x4` (later replaced by `CLUSTER16x2`) and one SM120 reduced cluster config.
2. Run the functional filters in [`tests.md`](tests.md) §1; paste PASS lines in the close-out.
3. Pin the `dsm_bw` commit hash from [`evidence.md`](evidence.md) in the close-out (already in that file).
4. In any new comment/preset, tag numbers `measured` / `patent` / `inferred` / `unresolved`.

**Do not:** Change knobs or C++.

**Verify:** The same filters PASS on a clean tree.

**Exit:** A short note under this item with node counts + test commands + date.

**Prereqs:** C1. **Next:** B1.

**Close-out (2026-08-25):** Observed-in-tree topology (config knobs + existing `icnt_create(m_shader_config->n_simt_clusters, …)` in `src/gpgpu-sim/gpu-sim.cc`; shader nodes = `-gpgpu_n_clusters` / `n_simt_clusters`, **not** total SMs). Not H200 hardware fact. No `measured` / `patent` / `inferred` / `unresolved` numbers copied from [`evidence.md`](evidence.md).

| Config | n_clusters | n_cores_per_cluster | total SMs | icnt shader nodes |
|--------|-----------:|--------------------:|----------:|------------------:|
| `SM90_H200_REDUCED_CLUSTER4x4` (superseded by `CLUSTER16x2`) | 4 | 4 | 16 | 4 |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | 1 | 2 | 2 | 1 |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | 4 | 4 | 16 | 4 |

total SMs = `n_clusters * n_cores_per_cluster`. `g_icnt_n_shader` is set from that first `icnt_create` argument.

Pinned `dsm_bw` hash already recorded in [`evidence.md`](evidence.md): `4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd` (`Add standalone H200 DSM bandwidth benchmark`).

[`tests.md`](tests.md) §1 commands (plus a second `ClusterNoc*` unit run):

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group integration
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*:*MultiCluster*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures; skip allowed):

```text
# unit ClusterNoc* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 8 tests.
✓ test/sm120/unit passed!

# SM120 integration (SM120_RTX5090_REDUCED_CLUSTER4x4)
[  PASSED  ] 31 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] ClusterLaunchApiTest.ExLaunch_ClusterLargerThanPhysical_Fails
✓ Tests passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

The SM120 skip is `GTEST_SKIP` because that case needs `n_cores_per_cluster == 1`; 4x4 has m=4. Not a product failure.

---

## B1 — Naming and `gpu_topology_t`

- [x] **B1** Zero-behavior rename: physical cluster → GPC; introduce topology table with PG'd slots. Closed 2026-08-25.

**Read first:** [`architecture.md`](architecture.md) §§2–5.

**Why:** Stop using `tpc`/`sid` as GPC. Enable non-uniform / PG'd SM maps without `sid % n` math.

**Files (expected):**

- New: `src/gpgpu-sim/gpu_topology.{h,cc}` (names may match style)
- `src/gpgpu-sim/gpu-sim.{h,cc}`, `shader.{h,cc}`, `abstract_hardware_model.h`
- `mem_fetch` requester fields
- `shader_core_config::reg_options` aliases
- Unit: `test/src/unit/` topology round-trip + PG map
- Docs: keep using `gpc_t` in comments as you rename

**Work:**

1. Add `gpu_topology_t` / `sm_location_t`. **All** SM↔GPC↔slot maps go through it.
2. CPC = 6 slots; `cpcs_per_gpc` default 3; extra slots **PG'd** when `num_sms_per_gpc` is smaller.
3. Typedef/alias `simt_core_cluster` → `gpc_t` **or** rename with a temporary typedef so callers compile. Prefer real rename if grep-complete.
4. Add `-gpgpu_num_gpcs` / `-gpgpu_num_sms_per_gpc`; keep `-gpgpu_n_clusters` / `-gpgpu_n_cores_per_cluster` as aliases. Conflict → abort.
5. `mem_fetch`: `m_requester_sm_id`; delete redundant `m_tpc` if it duplicates GPC/SM.
6. TB-cluster fields: `tb_cluster_*` prefix where you touch them.
7. Forward old getters with deprecation comments; do not add **new** `sid`/`tpc` APIs.

**Do not:** Change icnt node count (that is B2). Do not change DSM hop behavior. Do not model TPCARB.

**Verify:**

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

Plus new topology unit tests: round trip; PG'd slot has no `shader_core_ctx`; `sm_id %` is not used outside topology + documented shaper.

**Exit:** Baseline tests bit-identical in **functional** results; no interconnect node count change. Close-out lists the alias knobs.

**Prereqs:** B0. **Next:** B2.

**Close-out (2026-08-25):** Physical cluster is `gpc_t` (`typedef simt_core_cluster gpc_t`). SM↔GPC↔CPC-slot maps go through `gpu_topology_t` (CPC = 6 slots; `-gpgpu_dsm_cpcs_per_gpc` default 3; extra slots PG'd). `create_shader_core_ctx` still allocates only enabled local SMs. `icnt_create` first argument remains GPC count (`n_simt_clusters`). DSM delay-line hops unchanged. No TPCARB.

Alias knobs (conflict at start-up aborts):

| Old (kept) | New |
|------------|-----|
| `-gpgpu_n_clusters` | `-gpgpu_num_gpcs` |
| `-gpgpu_n_cores_per_cluster` | `-gpgpu_num_sms_per_gpc` |

`mem_fetch` requester is `m_requester_sm_id` (`get_sid()` / `get_tpc()` deprecated). TB-cluster storage touched here: `m_cta_tb_cluster_group`, `m_cta_tb_cluster_rank`, `m_next_tb_cluster_group_id`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*:GpuTopology*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures):

```text
# unit ClusterNoc* + GpuTopology* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 17 tests.
✓ test/sm120/unit passed!

# GpuTopology* rerun after exclusive-end SM range
[  PASSED  ] 9 tests.
✓ test/sm120/unit passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

---

## B2 — Global NoC endpoint per SM

- [x] **B2** One shader icnt node per **enabled SM**, not per GPC. Closed 2026-08-26.

**Read first:** [`architecture.md`](architecture.md) §4.

**Why:** Grouping SMs into a GPC must not cut L2/global bandwidth by `m`.

**Files:** `gpu-sim.cc` `icnt_create`; cluster/GPC icnt push/pop; `m_response_fifo` → per-SM; shader memory cycle ejection; stats; SST/gem5 adapter **or** a comment that they are unsupported until an adapter exists.

**Work:**

1. `icnt_create(num_sms, num_l2_subpartitions)` (enabled SM count).
2. SM injects from `global_sm_node_id(sm_id)`; L2 replies to that node.
3. GPC walks **each member SM** ejection port.
4. Per-SM response FIFO and ingress/dispatch budgets.
5. Ordinary CTA issue width restored per member SM.
6. Stats: per-SM and per-GPC, not “per cluster node”.

**Do not:** Put DSM packets on this icnt. Do not use `gpc_id` as a node.

**Verify:** Two configs, **same total SMs**, different GPC sizes (e.g. 16 SMs as 4×4 vs 16×1): shader endpoint count equal; a local L2-hit bandwidth microbench (or existing memory test) does not drop. Functional cluster filters still PASS.

**Exit:** Written table: config A/B, `num_sms`, `icnt` shader nodes, note on L2 roofline.

**Prereqs:** B1. **Next:** B3a.

**Close-out (2026-08-26):** `icnt_create` is sized from enabled SMs (`num_shader()`). `global_sm_node_id(sm)` is the SM; `global_l2_node_id` starts at `num_sms`. Each GPC ejects per member SM into a per-SM response FIFO. Ordinary CTA issue may place one CTA on each member SM per cycle. Incoming interconnect stats are counted per SM and summed per GPC. SST and gem5 still index one shader port per GPC and are unsupported until a per-SM adapter exists. DSM stays off this interconnect.

| Config / map | n_gpcs | sms/gpc | num_sms | icnt shader nodes |
|--------------|-------:|--------:|--------:|------------------:|
| `SM90_H200_REDUCED_CLUSTER16x2` | 2 | 16 | 32 | 32 |
| `gpu_topology_t::build(16, 1, 3)` | 16 | 1 | 16 | 16 |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | 1 | 2 | 2 | 2 |

L2 roofline: no dual-config kernel run. Independent SM ports (`GpuTopology.ShaderIcntNodeCountIndependentOfGpcGrouping`, `LocalInterconnectTest.PerSmShaderPortsDoNotShareBuffer`) mean packing SMs into fewer GPCs cannot cut L2 ports by `m`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "GpuTopology*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*:GpuTopology*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures):

```text
# unit GpuTopology* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 12 tests.
✓ test/sm120/unit passed!

# unit LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!

# unit ClusterNoc* + GpuTopology* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 20 tests.
✓ test/sm120/unit passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

---

## B3a — Shared transport primitives

- [x] **B3a** Bounded VOQ, RR arbiter, flit-credit counters, stats — **no** DSM policy. Closed 2026-08-27.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §§6 and 9.

**Why:** DSM fabric needs queues/credits. `xbar_router` assumes request/reply subnets and whole-packet moves.

**Work:**

1. New small headers (plan-v2 names OK): `transport_packet_metadata_t`, `bounded_voq_t`, `round_robin_arbiter_t`, `interconnect_sink_t`, occupancy/stall/latency stats. Occupancy in **payload flits**.
2. Unit tests: occupancy, HOL per VOQ, credit borrow must not cross a second queue.
3. Optionally wrap `LocalInterconnect` **without** changing its cycle behavior.

**Do not:** Instantiate two xbars as VCs. Do not put shader/memory endpoint assumptions in the primitive.

**Verify:** Existing local_interconnect / memory tests unchanged if you wrap them; new unit tests PASS.

**Exit:** Primitives have **no** `REQ_NET`/`REPLY_NET` in their API.

**Prereqs:** B1 (B2 preferred). **Next:** B3b.

**Close-out (2026-08-27):** Header-only primitives in `src/gpgpu-sim/transport.h`: `transport_packet_metadata_t`, `bounded_voq_t`, `flit_credit_counters_t`, `round_robin_arbiter_t`, `interconnect_sink_t`, `interconnect_stats_t`. Occupancy unit = **payload flits**. API has no `REQ_NET`/`REPLY_NET` and no shader/memory endpoint roles. `LocalInterconnect` was not wrapped.

Unit tests (`Transport*`): `OccupancyCountsPayloadFlits`, `FullDestDoesNotHolSibling`, `CreditTakeDoesNotCrossQueue`, `RoundRobinRotates`, `SinkOccupancyIsPayloadFlits`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "Transport*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
```

PASS evidence (0 failures):

```text
# unit Transport* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!

# unit LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!
```

---

## B3b — `dsm_fabric_t`

- [x] **B3b** Network-only DSM transport: VCs, 32 B **payload** flits, shaper, GPCMMU hash, GX count, CPC 6→4, PG slots.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) **all**, [`knobs.md`](knobs.md) §3, [`evidence.md`](evidence.md).

**Why:** This is the physical model. Delay-line hop cannot reproduce directional VC sharing.

**Hard requirements (do not “simplify away”):**

1. **Payload per grant = 32 B** (`-gpgpu_dsm_flit_payload_bytes`, default 32). Header/metadata may exist on a real package; **do not** charge extra lane occupancy.
2. Two VCs: **independent queues and credits**; **shared** physical lane scheduler.
3. `-gpgpu_dsm_gx_planes` default **2**. Never name planes request/response.
4. GPCMMU = **hash** `(addr, src, dst, uid) → (gx_plane, lane)`. Deterministic, replaceable.
5. Shaper configurable. Implement **all three**: `fixed_tdm` (plan-v2 CPC slots `{0,1,2,3}/{2,3,4,5}/{0,1,4,5}`), `skip_mod` (skip 1 of 3, index `sm_id` or `cpc_slot`), `hard_rate_cap`. Idle eligible slots are **wasted**. Do not claim one policy is Hopper silicon.
6. CPC: 6 slots, 4 lanes. PG'd slots never inject, never take eligibility.
7. Destination VOQ so one blocked dest does not HOL another dest on the same VC.
8. `can_inject` false → **LSU stall**, not “write immediately”.
9. Control packets (`read_command`, `write_ack`) occupy **one** payload-flit slot each.
10. 128 B data = **four** grants.

**Files:** new `dsm_fabric.{h,cc}`, config knobs, per-`gpc_t` instance, network-only gtests, stats dump.

**Work (suggested sequence inside this ID):**

1. Packet struct: plan-v2 §8.1 fields (`packet_id`, `transaction_id`, network src/dst, transaction requester/target, `vc`, `packet_class`, `payload_bytes`, `total_flits`, `remaining_flits`, `route_lane`, created/injected/tail cycles).
2. Ingress `[src][vc][dst]`, egress `[dst][vc]`, credits per dest VC.
3. Shaper + CPC lane arbiter + GX select via hash.
4. API: `can_inject` / `inject` / `top` / `pop` / `cycle` / `busy` / `display_state` (plan-v2 §9.1).
5. Isolation: GPC A fabric must not see GPC B queues.
6. Stats in [`dsm_fabric.md`](dsm_fabric.md) §7.1.

**Do not:** Complete RF/scoreboard here (B5). Do not coalesce ACKs here (B3c). Do not hook TMA multicast yet (B8) except injecting unicast packets in unit tests. Do not model TPCARB. Do not hard-code 21.0; let 2/3 × 32 fall out of the shaper.

**Verify (network-only tests — add them):**

- One SM, others idle: ~21.33 B payload/cycle cap (2 grants per 3 cycles × 32 B).
- Idle neighbor does not raise that rate.
- Same-dir request data + response data share the cap.
- Opposite dirs can exceed one-dir cap.
- `read_command` is one reverse flit.
- Request buffer full does not consume response credits.
- `gx_planes=1` reduces routes/bandwidth vs default 2.
- PG'd slot never sends.
- `skip_mod` and `fixed_tdm` both enforce ~2/3 rate; their co-eligible sets **differ**.

**Exit:** Those tests PASS. Stats print eligibility used vs wasted.

**Prereqs:** B3a, B1. **Next:** B3c.

**Close-out (2026-08-27):** `dsm_fabric_t` in `src/gpgpu-sim/dsm_fabric.{h,cc}`. Payload grant 32 B (`-gpgpu_dsm_flit_payload_bytes`, alias `-gpgpu_dsm_flit_bytes`). Request/response VCs have independent VOQs and credits and share one physical-lane scheduler. Destination VOQ, GPCMMU hash `(addr,src,dst,uid)→(gx_plane,lane)`, GX default 2, CPC 6→4, PG slots never eligible. Shapers `skip_mod` / `fixed_tdm` / `hard_rate_cap`. Per-GPC instance on `simt_core_cluster` (delay-line remains the functional SM↔SM path). `can_inject` false refuses inject.

Unit tests (`DsmFabric*`): `OneSmIdleNeighborsCap`, `IdleNeighborDoesNotRaiseRate`, `SameDirRequestResponseShareCap`, `OppositeDirsExceedOneDirCap`, `ReadCommandIsOneReverseFlit`, `RequestFullDoesNotConsumeResponseCredits`, `DestVoqBlockedDoesNotHolSibling`, `Data128BNeedsFourGrants`, `CanInjectFalseRefusesInject`, `GxPlanesOneReducesRoutesVsTwo`, `PgdSlotNeverSends`, `SkipModAndFixedTdmCoEligibleDiffer`, `HardRateCapAveragesTwoThirds`, `TwoFabricsIsolated`, `DisplayStateShowsUsedAndWasted`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "DsmFabric*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "Transport*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
```

PASS evidence (0 failures):

```text
# unit DsmFabric* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 15 tests.
✓ test/sm120/unit passed!

# unit Transport* + LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 10 tests.
✓ test/sm120/unit passed!

# display_state (DsmFabric.DisplayStateShowsUsedAndWasted)
eligibility used=32 wasted=160 slots=192
```

---

## B3c — Endpoint protocol

- [x] **B3c** Outstanding transactions + coalesced **write_ack** + read_command/data correlation.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §§3 and 7.

**Work:**

1. `dsm_endpoint_protocol_t` per enabled SM.
2. Track tx id → packets remaining.
3. ACK debt per original requester; flush on threshold, timeout, or idle response path; one-flit `write_ack` with count.
4. Remote load always pairs `read_command` + `read_data` (no data coalesce).
5. Outstanding cap backpressure via `can_inject`.
6. Dump: plan-v2 §17 transaction fields + ACK debt, coalescing ratio, timeout flush, outstanding count.

**Do not:** Put coalescing inside the lane arbiter. Do not treat outstanding cap as VC credit. Do not skip SRAM service (still a stub until B5; network-only tests may loop back without SRAM).

**Verify:** Finite buffers + heavy stores: no deadlock; ACK count < store packet count (coalesce). Symmetric-store unit scenario shows **far fewer** reverse flits than payload flits.

**Exit:** Tests named in close-out. **Next:** B4.

**Prereqs:** B3b.

**Close-out (2026-08-27):** `dsm_endpoint_protocol_t` in `src/gpgpu-sim/dsm_endpoint.{h,cc}`. Per-SM outstanding window (`-gpgpu_dsm_max_outstanding_per_sm`, default 16) is not a VC credit. ACK debt is per original requester; one-flit `write_ack` on the response VC flushes on threshold (default 4), timeout (default 64 cycles), or idle response path. Remote load always emits `read_command` + `read_data` (no data coalescing). SRAM service is a stub. Delay-line remains the functional SM↔SM path.

Unit tests (`DsmEndpoint*`, topology 2 GPCs × 16 SMs): `ReducedCluster16x2Packing`, `HeavyStoresDrainNoDeadlock`, `AckPacketsFewerThanStores`, `SymmetricStoresReverseFlitsFarFewer`, `RemoteLoadPairsCommandAndData`, `OutstandingFullDoesNotTouchVcCredits`, `AckFromOneTargetDoesNotRetireSiblingStore`, `ThresholdFlushProducesWriteAck`, `TimeoutFlushProducesWriteAck`, `IdleResponseFlushProducesWriteAck`, `DumpShowsOutstandingDebtRatioTimeout`.

Verify:

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 --group unit "DsmEndpoint*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 --group unit "DsmFabric*"
```

PASS evidence (0 failures):

```text
# unit DsmEndpoint* (run 1 and rerun, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 11 tests.
✓ test/sm120/unit passed!

# unit DsmFabric* regress (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 15 tests.
✓ test/sm120/unit passed!
```

---

## B4 — Functional DSM on the fabric

- [x] **B4** Ordinary `.shared::cluster` ld/st/atom use resolver + fabric packets. Existing DsmTest* PASS. Closed 2026-08-27.

**Read first:** [`programming_model.md`](programming_model.md) §2, [`architecture.md`](architecture.md) §7.

**Work:**

1. Unified `resolve_tb_cluster_rank` used by DSM, and called from TMA/mbar paths even if those still inject delay-line messages until B8.
2. Keep logical generic address; strip owner bits for target smem offset.
3. Mixed-target warp: group by `(target_sm, cta_slot)`, join.
4. Remote st → `write_data` packets; remote ld → `read_command` (data path completion in B5 may still be temporary).
5. Illegal cross-GPC / dead rank: abort (same as today).
6. Dual-run: delay-line still available behind a knob until B-DEPR.

**Do not:** Fill RF in `ld_impl` for the fabric path (if you must keep delay-line RF fill, gate it on the old knob only). Do not implement `red`.

**Verify:** `DsmTest.*` `MbarrierClusterTest.*` on H200 reduced. New tests: mixed-lane target; two TB-cluster groups isolated; cross-GPC reject.

**Exit:** Fabric path is default on H200 reduced **or** documented dual-path with fabric-on tests green.

**Prereqs:** B3c. **Next:** B5.

**Close-out (2026-08-27):** One TB-cluster resolver (`resolve_tb_cluster_rank` / `resolve_tb_cluster_owner_sm` in `src/gpgpu-sim/flash/tb_cluster.{h,cc}`) used by DSM, TMA (`for_each_tb_cluster_peer`), and remote mbarrier. Logical generic address kept; owner bits stripped for the target smem offset. Fabric-on remote st injects `write_data`, remote ld injects `read_command`, remote `atom.add` RMW on owner smem. Mixed-target warps group by `(target_sm, cta_slot)` and join. Fabric-path `ld_impl` does not fill dest RF. Cross-GPC / dead rank abort. Fabric default on `SM90_H200_REDUCED_CLUSTER16x2` via `-gpgpu_dsm_enable 1`; delay-line DSM via `0`. TMA multicast and remote mbarrier still use the delay-line.

Verify (run twice, 0 failures):

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration "DsmTest.*:MbarrierClusterTest.*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group unit "DsmEndpoint*:DsmFabric*"
```

PASS evidence:

```text
# integration DsmTest.*:MbarrierClusterTest.* (run 1 and 2, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 23 tests.
# includes DsmTest.MixedLaneTargetJoin, TwoTbClusterGroupsIsolated,
# VectorRemoteLdSt, CrossGpcReject; 9 MbarrierClusterTest.*

# unit DsmEndpoint*:DsmFabric* (run 1 and 2, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 28 tests.
✓ test/sm120/unit passed!
```

---

## B5 — Remote load = local SMEM load + SRAM service

- [x] **B5** Scoreboard and LDST writeback for remote DSM **match local `ld.shared`**. Target SRAM is a real service. Closed 2026-08-27.

**Read first:** [`pipeline.md`](pipeline.md) **all**.

**Why:** Supervisor: remote load must behave almost the same as local SMEM load. Variable fabric delay makes execute-time RF fill **wrong**.

**Work:**

1. Issue remote load: `reserveRegisters` as **shared** (`PROD_MEM_SHARED`), same as local smem.
2. **Do not** write destination regs in `ld_impl` on the fabric path.
3. Instruction stays in the **LD/ST shared** pipe until data is ready (pending shared load / same writeback as local smem).
4. After `read_data` tail **and** target `shared_memory_service_t` grant: RF write + `releaseRegisters` on that writeback path.
5. A following ALU must `checkCollision` until then.
6. Stores: owner write only on SRAM grant; ACK via B3c; CTA cannot exit with outstanding DSM.
7. `shared_memory_service_t`: local LSU, TMA local, DSM ingress share `gpgpu_shmem_bytes_per_cycle` (or named successor). Two-phase grant.
8. Zero-cycle bypass forbidden: arrival, SRAM, response inject take ≥1 cycle each as in [`architecture.md`](architecture.md) §6.

**Do not:** Add a DSM-specific scoreboard class unless `Scoreboard` literally cannot hold dest regs (it can). Do not classify remote DSM as `PROD_MEM_GLOBAL` without new evidence. Do not keep `dispatch_delay = 2×hop` on the fabric path. Do not sample peer smem at execute “for convenience.”

**Verify:**

- Unit/integration: warp issues `ld` remote then `add` using that dest; `add` issues only after response writeback.
- Compare issue/writeback sequencing to a **local** `ld.shared` of the same footprint (same unit, same scoreboard API).
- `DsmTest.RemoteLoad*` still PASS (data correct).
- Local-only smem bandwidth test does not starve forever when DSM is idle.

**Exit:** Written note: file:line for reserve, RF write, release. **Next:** B6 (calibration) and B8 (TMA/mbar packets).

**Prereqs:** B4.

**Close-out (2026-08-27):** Remote cluster loads reserve dest as shared (`Scoreboard::reserveRegisters` + `reclassifyShared` → `PROD_MEM_SHARED`), stay in the LD/ST shared pipe until `read_data` + SRAM grant, then RF write + `releaseRegister` on `ldst_unit::writeback` (same path as local `ld.shared`). Fabric-path `ld_impl` does not fill dest RF. `dispatch_delay = 2×hop` is delay-line only. Per-SM `shared_memory_service_t` (`-gpgpu_shmem_bytes_per_cycle`, 0=unlimited) two-phase grant shared by local LSU, TMA landing, and DSM ingress; arrival / grant / inject each ≥1 cycle. Stores write owner smem only on grant. CTA exit waits for outstanding DSM.

Verify (run twice, 0 failures, `SM90_H200_REDUCED_CLUSTER16x2`):

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration "DsmTest.*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group unit "SmemService*:DsmEndpoint*"
```

PASS evidence:

```text
# integration DsmTest.* (run 1 and 2)
[  PASSED  ] 16 tests.
# includes RemoteLoadFromPeer_TwoCtas, RemoteLoadThenAdd, LocalSharedLoadThenAdd

# unit SmemService*:DsmEndpoint* (run 1 and 2)
[  PASSED  ] 20 tests.
✓ test/sm120/unit passed!
```

Path note: reserve `shader.cc:2191` / `1670`; RF write `shader.cc:4371` + `6876`; release `shader.cc:4388`.

---

## B6 — H200 bandwidth calibration

- [ ] **B6** Fit knobs to `dsm_bw` **slopes**, not a single GB/s point.

**Read first:** [`evidence.md`](evidence.md), [`tests.md`](tests.md) §3 H200.

**Work:** Size sweep 16–96 KiB unique addresses, one TB-cluster, checksum after the timer. Record one-way load/store/TMA, duplex, same vs opposite mix, 2/4/8/16 TMA, idle-neighbor. Tune shaper period, VC depths, ACK threshold/timeout, optional `base_latency`. **Do not** overfit GPCMMU hash to one camping trace.

**Do not:** Reintroduce `-gpgpu_dsm_bytes_per_cycle` as the model.

**Verify:** Core slopes within a slop you document (suggest ±10% on 21.3 B/cycle; duplex loss sign and magnitude: load ≫ store).

**Exit:** Table in this item + H200 preset comments labeled `measured`/`inferred`.

**Prereqs:** B5.

---

## B7 — Per-bank shared memory (optional, after first delivery)

- [ ] **B7** `shared_access_plan_t`: bank + broadcast constraints on top of aggregate bytes.

**Prereqs:** B5. **Do not** start before B6 unless a kernel is bank-bound and mispredicted.

---

## B8 — TMA multicast and remote mbarrier on the fabric

- [x] **B8** Same VC contract. Data before mbar. Source-expanded unicast v1. Closed 2026-08-27.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §8, [`programming_model.md`](programming_model.md) §§3–4.

**Work:** Replace `cluster_noc_t` TMA_MCAST_* / MBAR_REMOTE_OP with `tma_data` / `mbarrier_*` packets. Configurable `source_unicast` vs later `fabric_replicate`. Hang-preventer quiet window uses real outstanding/RTT, not `2×hop` only.

**Verify:** `TMAClusterOneProducer*`, `TmaMulticastMaskTest.*`, `MbarrierClusterTest.*`.

**Prereqs:** B5.

**Close-out (2026-08-27):** With `-gpgpu_dsm_enable 1`, TMA `.shared::cluster` multicast expands at the source into unicast `tma_data` on the request VC (one packet per selected peer; 128 B = four 32 B grants). Peer smem write and `complete_tx` wait for that packet’s tail and that SM’s SRAM grant (`-gpgpu_tma_mcast_mbar_after_data`). Remote mbarrier `arrive` / `expect_tx` / `complete_tx` / `try_wait` use `mbarrier_request` (request VC, one control flit) and `mbarrier_completion` (response VC, one control flit). Fabric-on does not also inject delay-line `TMA_MCAST_*` / `MBAR_REMOTE_OP`. Hang-preventer quiet window stays armed while endpoint outstanding / observed RTT is live; recognized waits still drop the arm. Delay-line TMA/mbar remains when fabric is off. v1 expand mode is source unicast (in-fabric replication not implemented).

Verify (run twice, 0 failures):

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group unit "DsmEndpoint*:ClusterHangPrevent*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "TMAClusterOneProducer*:MbarrierClusterTest.*:DsmTest.*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 \
  run test --target sm120 --group integration "TmaMulticastMaskTest.*"
```

PASS evidence:

```text
# unit DsmEndpoint*:ClusterHangPrevent* on SM90_H200_REDUCED_CLUSTER16x2 (run 1 and 2)
[  PASSED  ] 26 tests.
# includes TmaPutSourceUnicastNPeers, TmaCompleteTxAfterSramApply,
# MbarRequestAndCompletionVcs, QuietWindowUsesOutstandingRtt,
# PeerArmStaysWhileFabricOutstanding

# integration TMAClusterOneProducer*:MbarrierClusterTest.*:DsmTest.*
# on SM90_H200_REDUCED_CLUSTER16x2 (run 1 and 2)
[  PASSED  ] 26 tests.
# 1 TMAClusterOneProducer + 9 MbarrierClusterTest + 16 DsmTest

# integration TmaMulticastMaskTest.* on SM120_RTX5090_REDUCED_CLUSTER4x4
# (run 1 and 2; fabric-off functional regression)
[  PASSED  ] 4 tests.
```

---

## B-DEPR — Remove the delay line

- [ ] **B-DEPR** Once `dsm_fabric_t` is the only SM↔SM path, **deprecate and delete delay-line knobs** and update **all** config files.

**After B-DEPR, delete these (do not leave unused files or knobs in tree):**

- all `dsm_latency_matrix_*.csv`
- `-gpgpu_dsm_latency_matrix_file`
- `-gpgpu_dsm_remote_latency` as a hop / bandwidth knob

The hop-matrix CSV and the two knobs above are delay-line-only. They are **not** the fabric timing model. After this ID, grep for them must be empty.

**Read first:** [`knobs.md`](knobs.md) §§2–3.

**Why:** Two timing models in tree will drift. Agents must not keep `ready_cycle = hop + BPC`.

**Work:**

1. Delete the three items in the After-B-DEPR list above. No leftover CSV, no unused knob registration, no config that still points at a matrix file.
2. Also delete or `#error` unused: `cluster_noc_t` inject/deliver delay-line, `-gpgpu_dsm_bytes_per_cycle`, `-gpgpu_tma_mcast_hop_latency` if B6 replaced it, `-gpgpu_dsm_store_immediate` if fabric stores are deliver-only only.
3. Keep: hang watchdog, mbarrier cluster enable, TMA data-before-mbar, topology knobs, fabric knobs, possibly `-gpgpu_dsm_local_latency` / `base_latency` if still used as SMEM/floor. Do **not** keep a pairwise hop table or a scalar remote hop as the DSM timing model.
4. Grep configs: `configs/SM90_H200*`, `configs/SM120_*CLUSTER*`, test overlays, `FLASH.md`, comments in `gpu-sim.cc` / `shader.h` / `cluster_noc.*`.
5. Update every `gpgpusim.config` that set the deleted knobs.
6. Rewrite or drop unit tests that only parsed the hop CSV (`ClusterNocMatrix.H200ReducedMatrixFile` and similar). Do not keep the CSV as an unused file.
7. Stub `cluster_noc_t` gone: either remove sources or make them thin wrappers around `dsm_fabric_t` with a deprecation comment, then delete in the same ID if compile-clean.

**Do not:** Leave “0 = unlimited BPC” in any shipped config. Do not keep a second inject path that writes smem immediately when `can_inject` is false.

**Verify:** Full cluster integration filters. Grep is empty for `dsm_latency_matrix`, `dsm_latency_matrix_file`, `gpgpu_dsm_remote_latency`, and `dsm_bytes_per_cycle`. `cluster_noc_enable` has a defined migration (renamed to `dsm_enable` or documented alias).

**Exit:** Grep output empty (or aliases listed). Config list in the close-out.

**Prereqs:** B6 and B8 green on fabric; delay line unused.

---

## B9 — Not first delivery

- [ ] **B9a** Multi-hop / escape VC / wormhole proof
- [ ] **B9b** Independent DSM clock domain
- [ ] **B9c** Blackwell preset from **Blackwell** measurements
- [ ] **B9d** Power model
- [ ] **B9e** SST/gem5 adapter complete

---

# Track C — Documentation

- [x] **C1** Rewrite living spec under `docs/cluster_noc/` (this directory).
- [x] **C2** Stub old `docs/cluster*.md`. Memory-xbar argument lives in `dsm_fabric.md` §9 (no `use_xbar_roter.md`).
- [x] **C3** Point `FLASH.md`, `CLAUDE.md`, config READMEs, C++ comments at this directory.
- [x] **C4** Midterm is historical; not a second spec.

Agents doing **code** should not edit C1–C4 except to fix a broken link after a rename.

---

# Retired IDs (do not revive)

`L0`–`L4`, `L2-1`…`L2-6`, `L3-1`…`L3-5`, `L4-1`… as living IDs. Content now lives in **B5** (scoreboard), **B3b** (BW), **B6** (calibration).

Do not model TPC/TPCARB. Do not put DSM on `xbar_router` request/reply subnets. Do not “fit BPC from a multi-cluster GB/s point.”

---

# Review bans

Reject patches that:

- Use `tpc` to mean GPC, SM, or icnt node
- Map GPC with division/modulo **outside** `gpu_topology_t` (shaper `sm_id % period` is allowed and documented)
- Give DSM two physical endpoints (requester vs service)
- Treat request/reply **subnets** as VCs or give them separate physical BW
- Name GX planes as request/response VCs
- Infer response `network_src` from the transaction requester
- Teleport a whole packet after numeric budget
- Treat outstanding-tx limits as link/VC credit
- Reply from target SRAM **without** `shared_memory_service_t` (after B5)
- Fire-and-forget store and allow CTA exit
- Implicit local/TMA/DSM priority via SM walk order
- Nested OpenMP on one GPC fabric
- Label H200 inferred numbers as Blackwell hardware fact
- Mix large rename and timing behavior in one commit
- Charge header bits as extra **payload** occupancy (payload grant is 32 B)
- Fill RF in `ld_impl` on the fabric path
- Map `can_inject == false` to immediate peer write

---

# First delivery (stop and calibrate)

Declare first delivery when **all** are true:

1. Names: `gpc_t`, SM, TB-cluster, global node, DSM endpoint, VC are unambiguous.
2. Each enabled SM has its own global NoC endpoint.
3. No DSM traffic ⇒ L2 steady-state BW does not drop when GPC grouping changes.
4. One host thread advances each GPC fabric.
5. Request/response: independent queues/credits, shared physical 32 B-payload lanes.
6. Same-dir mix shares a ceiling; opposite dirs run together.
7. Shaper is configurable (`skip_mod` default, `sm_id % period`); idle slots wasted; stats exist.
8. Remote load uses the **local SMEM** scoreboard/LDST completion path.
9. Ordinary DSM ld/st/atom: mixed target, response/ACK, pending join.
10. DSM vs local LSU/TMA compete in `shared_memory_service_t`.
11. Finite buffers + ACK coalesce: no constructed deadlock; pending DSM blocks warp/CTA/sim end.
12. OMP on/off functionally equal; fixed config completion cycles repeat.
13. H200 preset explains `dsm_bw` slope, direction sharing, scaling (B6).
14. TMA multicast can use the same resolver/fabric/SRAM without changing the VC contract (B8).
15. Delay-line knobs gone or hard-deprecated (**B-DEPR**).

Then, and only then, B7/B9.
