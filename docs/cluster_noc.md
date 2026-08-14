# Intra-Cluster Network-on-Chip (NoC)

**Status**: Functional + tunable NoC (TMA mcast, DSM, remote mbarrier) — part of the **unified cluster branch/PR**.  
**Branch overview**: [`docs/cluster.md`](cluster.md)  
**Primary NoC configs**: H200 (`SM90_H200*`); SM120 product configs keep NoC **off** by default.  
**Profiling source**: `../H200_profiling/output-2046238-H200Profiling.txt` (job **2046238**; flat DSM).

This document is the **design authority** for FlashGPU-Sim’s SM↔SM fabric (architecture, knobs, H200 calibration, maturity, sim TODO). It is also the **living checklist**:

| Track | IDs | Meaning | Where |
|-------|-----|---------|-------|
| **Functional gaps** | **F1–F9** | Wrong data, hangs, missing PTX, thin tests | **§6.6** (assessment) · **§12.2** (checkboxes) |
| **Cycle fidelity** | **L0–L4** | SM↔SM timing ladder | **§6.5** (assessment) · **§12.3–§12.5** (checkboxes) |

Flip `[ ]` → `[x]` in **§12** when an item is done. Do not re-open **§12.1** without new evidence.

### Start here (reading order)

| Order | Doc | Purpose |
|------:|-----|---------|
| 0 | **`docs/cluster.md`** | Unified branch/PR overview (launch + TMA + DSM + NoC) |
| 1 | **This file** — **§6.4–§6.6** maturity, **§12** checklist, §7 knobs | Functional gaps (F) + cycle backlog (L) |
| 2 | `docs/cluster_cta2_realLaunch.md` | Cluster launch API + co-residency |
| 3 | `FLASH.md` → *Known Limitations* | Short product caveats |
| 4 | `configs/SM90_H200/README.md` | Calibrated numbers |
| 5 | `../H200_profiling/TODO.md` | Profiling-suite microbenchmarks for L2–L4 |
| 6 | `../H200_profiling/output-2046238-H200Profiling.txt` | Raw HW evidence |

---

## 1. Problem statement

Hopper/Blackwell Thread Block Clusters can:

1. **Multicast TMA** loads into peer CTA shared memory (`.shared::cluster`, optional `.multicast::cluster` + `ctaMask`).
2. Access **distributed shared memory (DSM)** of peer CTAs (`mapa` + remote `ld`/`st`).
3. Synchronize with **mbarrier** objects that may live in peer CTA smem.

**Today (with NoC on, H200 reduced):**

| Path | Functional | Timing (tunable) |
|------|------------|------------------|
| TMA cluster multicast | Peer data after NoC deliver (or immediate if NoC off) | Hop (+ optional BW); mbar after data |
| Peer mbarrier `complete_tx` | Via NoC when NoC on | Same hop path as TMA peer mbar |
| DSM / `mapa` + remote ld/st | Real mapa; peer smem via decode + NoC store path | Flat hop ~78 one-way; load RTT ≈ 2×hop |
| Remote mbarrier (mapa bar) | Arrive / try_wait / expect via NoC | Hop; interest-list WAIT_REG/DONE |

The global memory interconnect (`intersim2` / `local_interconnect`) models **cluster ↔ L2/DRAM**, not SM↔SM. Intra-cluster traffic uses a **separate** `cluster_noc_t`.

---

## 2. High-level architecture

```text
                    Global / L2 / DRAM  (EXISTING icnt)
                              ▲
                              │
┌─────────────────────────────┼─────────────────────────────┐
│              simt_core_cluster (physical TB-cluster home) │
│  ┌────────┐   ┌────────┐         ┌────────┐              │
│  │  SM 0  │   │  SM 1  │   ...   │  SM m  │              │
│  │ TMA    │   │ TMA    │         │ TMA    │              │
│  │ mbar   │   │ mbar   │         │ mbar   │              │
│  │ smem[] │   │ smem[] │         │ smem[] │              │
│  └───┬────┘   └───┬────┘         └───┬────┘              │
│      └────────────┼──────────────────┘                   │
│                   │ inject / eject                       │
│         ┌─────────▼──────────┐                           │
│         │  cluster_noc_t     │                           │
│         │  latency matrix L  │                           │
│         │  inflight msgs     │                           │
│         │  cycle() deliver  │                           │
│         └────────────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

### 2.1 Design principles

1. **Separate fabric** from global memory NoC. DSM/TMA-peer traffic never uses intersim2 ports.
2. **Message-passing**: Cross-SM smem and mbarrier state are mutated only on **delivery**, never by a foreign SM’s pipeline stage.
3. **Single owner per physical cluster**: `cluster_noc_t` lives on `simt_core_cluster`. Only that cluster’s simulation thread runs `cluster_noc_cycle()` after `core_cycle()`.
4. **Same TB-cluster domain only**: Messages target CTAs sharing the same physical cluster and `cluster_group` / ranks. No cross-GPC DSM.
5. **Latency-first v1**: Tunable SM×SM matrix + scalars. Optional simple byte/cycle BW. No full mesh/VC Booksim model yet.

### 2.2 Race-freedom invariant

> **Cross-SM shared state is mutated only by `cluster_noc_t::deliver()` (or the receiving SM after eject), never by a foreign SM pipeline stage.**

| Risk | Mitigation |
|------|------------|
| OpenMP across clusters | NoC is per-cluster; no cross-cluster messages |
| Issuer writes peer smem during peer cycle | Forbidden when NoC on; enqueue only |
| Mid-deliver inject invalidates iterators | Deferred inject queue flushed after sweep |
| Concurrent complete on same mbarrier | Same cluster thread; manager not parallel |
| CTA exit with inflight msgs | `drop_messages_to_cta` (hook for CTA cleanup) |

OpenMP in Flash mode parallelizes **physical clusters**, not SMs inside a cluster (`simt_core_cluster::core_cycle` is serial over local cores). The NoC design remains safe if within-cluster parallelism is added later, as long as the invariant holds.

---

## 3. Source files

| File | Role |
|------|------|
| `src/gpgpu-sim/flash/cluster_noc.h` | Message types, matrix, `cluster_noc_t` API, address helpers |
| `src/gpgpu-sim/flash/cluster_noc.cc` | Inject/cycle/deliver implementations |
| `src/gpgpu-sim/shader.{h,cc}` | Config knobs; `cluster_noc_cycle`; remote mbarrier wrappers |
| `src/gpgpu-sim/gpu-sim.cc` | Option registration; call `cluster_noc_cycle` after `core_cycle` |
| `src/gpgpu-sim/flash/tma.cc` | Peer fan-out via NoC when enabled |
| `src/gpgpu-sim/flash/mbarrier.cc` | Remote mbarrier allow-list; remote arrive/expect |
| `src/cuda-sim/instructions.cc` | Real `mapa`; generic shared decode for remote DSM |
| `configs/SM90_H200*/` | H200 knobs + matrix CSVs only |
| `test/src/unit/cluster_noc_test.cc` | Matrix parse + address decode unit tests |
| `docs/cluster_noc.md` | This document |

---

## 4. Message types

| Type | Payload | Injected by | Effect on deliver |
|------|---------|-------------|-------------------|
| `TMA_MCAST_DATA` | Byte snapshot | TMA finalize | Write peer CTA smem |
| `TMA_MCAST_MBAR` | mbar addr + tx count | After data (same stream) | `try_complete_tx_if_pending` on peer |
| `DSM_STORE` | Bytes | Remote `st` (NoC path) | Write peer smem (also dual-path immediate write) |
| `DSM_LOAD_REQ` | Addr/size | Optional timing path | Read peer; inject `DSM_LOAD_RSP` |
| `DSM_LOAD_RSP` | Bytes | Load req deliver | **Scoreboard/RF completion still partial** (load uses issue delay today) |
| `MBAR_REMOTE_OP` | arrive / expect / complete / wait | Remote mbarrier | Owner CTA manager + WAIT_DONE to requester |

### 4.1 Ordering

- Per `(src_cid, dst_cid, stream_key)` FIFO by inject sequence.
- For one TMA transaction: **data before mbar** (`gpgpu_tma_mcast_mbar_after_data=1`): both share `ready_cycle`, data injected first so delivery sweep applies data then mbar.

### 4.2 Timing formula (v1)

```text
ready_cycle = inject_cycle
            + hop(src, dst)
            + max(0, ceil(bytes / bytes_per_cycle) - 1)   // if BPC > 0
```

- **DSM hop**: from latency matrix, else `gpgpu_dsm_remote_latency` / local.
- **TMA hop**: `gpgpu_tma_mcast_hop_latency` unless `gpgpu_tma_mcast_use_dsm_matrix=1`.
- **Remote mbarrier hop**: `gpgpu_mbarrier_remote_hop_latency` if non-zero, else DSM hop.

---

## 5. Cycle loop integration

```text
gpgpu_sim::cycle()  // CORE domain
  for each physical cluster i:          // OpenMP parallel
      cluster[i]->core_cycle()          // all SMs in cluster, serial
      cluster[i]->cluster_noc_cycle()   // deliver ready SM↔SM messages
  gpu_sim_cycle++
```

TMA path when NoC + `gpgpu_tma_mcast_enable_timing`:

1. **Functional**: issuer (or staging) smem filled; **no** immediate peer `copy_mem`.
2. **Timing**: L2/TMA stream for issuer as today; after arrive latency, `schedule_cluster_tma_mcast_noc` snapshots issuer smem and injects per-peer data+mbar.
3. **Deliver**: peer smem visible; peer waiters unblocked via `try_complete`.

When NoC is **disabled** (`-gpgpu_cluster_noc_enable 0`): legacy immediate multicast + peer complete (bit-compat with pre-NoC branch).

---

## 6. Feature status

### 6.1 TMA multicast

| Item | Status |
|------|--------|
| Functional mask / legacy destinations | Unchanged PTX semantics |
| Immediate peer path when NoC off | Yes |
| Delayed peer data+mbar when NoC on | Yes (validated hop=20, 7 key tests) |
| Issuer-not-in-mask staging | Yes (issuer smem as private stage) |
| Separate TMA hop vs DSM matrix | Yes |
| BW term | Knob present; default unlimited |
| CTA exit drops inflight mcast msgs | Yes (`drop_messages_to_cta`) |

### 6.2 DSM

| Item | Status |
|------|--------|
| `mapa` maps to peer generic shared | Yes (`mapa.u64` preferred) |
| `ld`/`st` generic → remote smem | Yes (via `decode_space`) |
| Store via NoC + functional peer write | Yes (dual-path: inject hop + immediate write) |
| Load pre-delivers due stores | Yes (`deliver_ready` before remote load) |
| Timing hop (shared dispatch_delay) | Yes (load RTT=2×hop, store=1×hop; no phantom L1) |
| Flat topology (H200) | Yes (constant/near-constant hop; stride ratio~1.0) |
| Integration tests | Yes (`dsm_test.cc`: SelfMapa, RemoteStore, RemoteLoad) |
| Full scoreboard / DSM_LOAD_RSP→RF | Partial (optional; see §12) |

### 6.3 Cluster mbarrier

| Item | Status |
|------|--------|
| TMA peer `complete_tx` via NoC | Yes (with TMA mcast) |
| Remote arrive / expect API on owner | Yes (`remote_mbarrier_*` + `inject_mbar_remote`) |
| Remote try_wait / mapa’d bar addr | Yes (WAIT_REG interest list + WAIT_DONE) |
| CTA exit drops NoC msgs | Yes |

### 6.4 Functional usefulness assessment

**Bottom line:** FlashGPU-Sim is a **functionally useful** simulator for Hopper-style **cluster + TMA + DSM + mbarrier** kernels that follow real CUDA practice (producer/consumer with mbarrier), **not** a full bit-exact HW clone of every PTX edge case.

| Area | Functionally usable? | Explanation |
|------|----------------------|-------------|
| **Cluster launch** | **Yes** | `cudaLaunchKernelExC` / cluster dims co-schedule CTAs of one Thread Block Cluster onto one physical `simt_core_cluster`, with ranks and `cluster_group` for peer discovery. Ordinary `<<<>>>` does not force co-residency. See `docs/cluster_cta2_realLaunch.md`. |
| **TMA + cluster multicast** | **Yes (common paths)** | `.shared::cluster` and `.multicast::cluster` + `ctaMask` select peer destinations for data and mbarrier `complete_tx`. With NoC off, peers see data immediately (legacy). With NoC on, peers see data after hop deliver and can `try_wait` correctly. Mask edge cases, OOB, and some bulk corners remain idealized (see `FLASH.md`). |
| **DSM (`mapa` + remote ld/st)** | **Yes (tested patterns)** | Real `mapa.u64` maps into peer generic shared windows; remote `ld`/`st` resolve via `decode_space` and (when NoC on) the store NoC path. Integration coverage: self-mapa, two-CTA remote store, two-CTA remote load. Prefer **`mapa.u64`** (u32 truncates large generic windows). |
| **mbarrier (local + remote)** | **Yes (main ops)** | Local init/arrive/try_wait/complete_tx as before. Remote (mapa’d) arrive and try_wait go through NoC when `-gpgpu_mbarrier_cluster_enable 1` and NoC is on. Timeout and some PTX variants are still unimplemented. |
| **Correctness model for multi-CTA** | **mbarrier-ordered kernels** | Kernels that **sync with mbarrier** (or equivalent interest-list wait) match the intended model. Bare spins on peer smem without the peer having issued, or `__syncthreads` mixed with single-thread try_wait, can hang or behave poorly under functional-first PTX sim. Remote DSM **stores** are dual-path (issuer pays hop; peer smem also updated immediately for mbarrier-safe visibility). |

**What “functionally usable” means here**

- If a correct CUDA kernel uses cluster launch, TMA into peer smem, DSM via `mapa`, and mbarrier to wait for data, the simulator should **produce the right data and complete without hanging** (under supported configs / knobs).
- It does **not** mean every latency, throughput, or microarchitectural hazard matches silicon.

**Open functional holes:** **F1–F9** in **§6.6**. Track close-out in **§12.2**.

**Configs**

| Config | NoC default | Use for |
|--------|-------------|---------|
| `SM90_H200_REDUCED_CLUSTER4x4` | **On** | Multi-SM cluster + hop-calibrated H200 path |
| `SM90_H200` (full) | Off (1 SM/cluster) | Product latency knobs; NoC idle unless multi-SM pack |
| `SM120_RTX5090*` | Off | Functional TMA/cluster tests; enable NoC via run-dir overlay if needed |
| `SM120_*_REDUCED_CLUSTER2x1` | Off | Fast two-CTA smoke; overlay NoC for DSM/mbar cluster tests |

### 6.5 Cycle-accuracy maturity levels

SM↔SM / cluster timing is best thought of as a **ladder**, not a binary “accurate / not accurate.”

| Level | Name | Status | What it means |
|------:|------|--------|----------------|
| **L0** | Functional only | **Past** | Peer TMA/DSM effects applied immediately; no hop; free multicast. Pre-NoC default for most configs still behaves like this when NoC is off. |
| **L1** | Tunable flat hop | **Current** | Separate `cluster_noc_t`; flat (constant/near-constant) hop from H200 job **2046238** (one-way ~78; remote load RTT ≈ local + 2×hop ≈ 193; TMA mcast−unicast e2e ~+135). Topology class matches HW stride ratio ~1.0 (not multi-hop tree). |
| **L2** | Pipeline / scoreboard fidelity | **Partial** | Remote **load** still fills RF at functional execute and only *stalls issue* by ~2×hop. Real HW: hop → peer read → hop back → **scoreboard clear / RF write** on response (`DSM_LOAD_RSP`). Store dual-path immediacy also differs from pure delayed visibility. |
| **L3** | BW + contention | **Not fitted** | `-gpgpu_dsm_bytes_per_cycle` / TMA BPC exist but default **0** (unlimited). Multi-cluster pair/ring GB/s from profiling cannot be dropped in as per-link BPC without a single-cluster fit. No credits, ingress queues, or bcast fan-out tax. |
| **L4** | HW-matched e2e cycles | **Far** | Matching wall-clock or cycle counts of real H200 for full kernels needs L2+L3, broader suite-vs-silicon validation, and the usual GPGPU-Sim limits (idealized mbarrier storage, partial TMA corners, OpenMP edge races, etc.). |

**How to read the levels**

- **L1 is intentional and calibrated:** hop numbers and flat topology come from `../H200_profiling/output-2046238-H200Profiling.txt`. Good for *relative* SM↔SM cost studies and for not treating cluster multicast as free when NoC is on (H200 reduced).
- **L1 is not “cycle-perfect”:** absolute GPU cycles for a large app will still differ from silicon for many reasons outside the NoC (caches, DRAM, scheduling, idealized barriers).
- **Moving L1 → L2:** implement items in **§12.3** (DSM_LOAD_RSP scoreboard; optional delayed-only store). Profiling inputs for validation: `../H200_profiling/TODO.md` §2.2–2.3.
- **Moving L2 → L3:** fit non-zero BPC (**§12.4**); needs **single-cluster** BW microbench from profiling suite (`../H200_profiling/TODO.md` §2.1).
- **L4** is a long-term product goal (**§12.5**), not a claim of the current NoC work.

**Practical summary**

| Question | Answer |
|----------|--------|
| Functionally correct cluster / TMA / DSM for real-style kernels? | **Mostly yes**, caveats in §6.4; remaining holes **F1–F9** (§6.6) |
| Cycle-accurate SM↔SM fabric? | **L1: directionally right, hop-tunable, flat** — not L4 |
| Full SM pipeline cycle-accurate vs H200? | **No** — NoC is only one subsystem; rest is still GPGPU-Sim / Flash idealized in places |

### 6.6 Functional correctness gaps (F1–F9)

**F-track vs L-track.** L0–L4 (§6.5) are about *when* peer effects become visible and *how* cycles accumulate. **F1–F9** are about *whether* the sim produces the right data, completes, or even runs the PTX. Close F-items before treating L2–L4 as the critical path.

| ID | Gap | Sev | Status | Close-out |
|----|-----|-----|--------|-----------|
| **F1** | Land + green-gate NoC stack | Crit | **Done** | §12.2 F1 |
| **F2** | mbarrier idealized / incomplete PTX | High | **Open** | §12.2 F2 |
| **F3** | `barrier.cluster` / CG DSM map builtins | Med | **Non-goal** | §12.6 (reopen only if an app needs it) |
| **F4** | DSM atomics overclaimed / broken | Med | **Open** | §12.2 F4 |
| **F5** | `mapa` lifetime: exited producer → local fallback | Med | **Open** | §12.2 F5 |
| **F6** | TMA corners stubbed (swizzle, tensormap, bulk group) | Med | **Open** | §12.2 F6 |
| **F7** | Programming-model fragility (spin / `bar.sync`+`try_wait`) | Med | **Open** | §12.2 F7 |
| **F8** | Thin DSM / remote-mbar / compose tests | High | **Open** | §12.2 F8 |
| **F9** | Dual-path store (immediate peer write + issuer hop) | Med | **Open** | §12.2 F9 · L2-2 |

**How to read statuses**

| Status | Meaning |
|--------|---------|
| **Open** | Known hole; checklist item is `[ ]` |
| **Partial** | Some sub-items `[x]`; remainder still blocks “functionally complete” |
| **Non-goal** | Documented, will not implement unless a target kernel forces it |
| **Done** | All sub-items `[x]`; do not re-open without new evidence |

#### F1 — Land + green-gate

NoC / DSM / remote mbarrier landed as `cluster_noc_t`. **Green-gate is done** (2026-08-15): unit `ClusterNoc*` plus SM120 NoC-off cluster/TMA filters, SM120 NoC overlay, and H200 reduced NoC-on DSM/mbar/OneProducer all recorded PASS in §12.2 F1-5.

#### F2 — Idealized mbarrier

Barriers live in simulator state, not GPU shared-memory contents. `mbarrier.try_wait` **timeout** is unimplemented. Some PTX variants still `assert`. Remote arrive / try_wait work for the tested ops only.

#### F3 — `barrier.cluster` / CG map (non-goal)

Special regs + `mapa` are the supported cluster discovery path. Full `barrier.cluster` and Cooperative Groups DSM map builtins are out of scope (`docs/cluster.md` §4). Kernels that require them will fail.

#### F4 — DSM atomics

`cluster_noc.h` mentions atomics. `atom_impl` still converts generic→shared with the **local** `smid` only, so mapa’d remote atomics are wrong or undefined. Either implement via the same decode/NoC path or reject loudly.

#### F5 — `mapa` lifetime

`mapa` looks up an **active** peer rank in the same `cluster_group`. If the producer CTA has exited, lookup falls back to **local** — silent wrong data, not a hardware behavior. Tests already keep the producer alive; the sim should fail loud (or keep the rank mapping after exit) instead of aliasing local smem.

#### F6 — TMA corners

Known stubs / idealizations that can break real PTX: 96B swizzle, some `tensormap.*` variants, weak `cp.async.bulk.commit_group` / `wait_group` on the functional path, sector / OOB edges. Implement only what target kernels use; otherwise hard-fail.

#### F7 — Programming-model fragility

Functional-first PTX sim hangs or deadlocks on patterns that are legal-ish but unsupported here:

1. Bare spin on peer smem before the peer has issued (no mbarrier interest list).
2. Mixing `__syncthreads` / `bar.sync` with a **single-thread** `try_wait` in the same CTA.

Correct CUDA cluster code usually avoids both. The sim should **assert or document-and-skip**, not hang silently.

#### F8 — Thin tests

Happy-path only today: `DsmTest` (self-mapa, 2-CTA store, 2-CTA load) and `MbarrierClusterTest` (remote arrive, remote try_wait). Missing: clusterDim=4, multi-peer fan-out, drop-on-CTA-exit, `mapa.u32` vs `u64`, ordering/FIFO, mixed TMA+DSM+mbar, and one real-shaped kernel.

#### F9 — Dual-path store

When NoC is on, a remote `st` **injects a hop for the issuer and writes peer smem immediately**. Safe for mbarrier-ordered kernels (the supported model). Incorrect for algorithms that rely on delayed visibility without a barrier. Timing studies need `-gpgpu_dsm_store_immediate 0` (**L2-2**).

---

## 7. Configuration knobs

All registered in `shader_core_config::reg_options` (`gpu-sim.cc`).

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_cluster_noc_enable` | `0` | Master switch |
| `-gpgpu_dsm_local_latency` | `37` | Local DSM latency (H200~37) |
| `-gpgpu_dsm_remote_latency` | `78` | **One-way** hop if matrix missing (H200~78) |
| `-gpgpu_dsm_latency_matrix_file` | `""` | N×N **one-way hop** CSV (cluster-local core ids) |
| `-gpgpu_dsm_bytes_per_cycle` | `0` | DSM BW: extra cycles ceil(bytes/BPC)−1; 0=unlimited |
| `-gpgpu_tma_mcast_enable_timing` | `1` | When NoC on, route TMA peers through NoC |
| `-gpgpu_tma_mcast_hop_latency` | `0` | Fixed TMA peer hop; H200 product sets **135** |
| `-gpgpu_tma_mcast_use_dsm_matrix` | `0` | TMA hop = L[src][dst] (usually off; TMA ≠ full DSM RTT) |
| `-gpgpu_tma_mcast_bytes_per_cycle` | `0` | 0 = unlimited; else simple BW |
| `-gpgpu_tma_mcast_mbar_after_data` | `1` | Data-before-mbar ordering |
| `-gpgpu_mbarrier_remote_hop_latency` | `0` | 0 ⇒ use DSM hop |
| `-gpgpu_mbarrier_cluster_enable` | `0` | Allow remote mbarrier addresses |

**Latency math (important):**

- Matrix / `gpgpu_dsm_remote_latency` store **one-way fabric hop**.
- Remote **load** issue delay ≈ **2 × hop** (request + response).
- Target e2e remote load ≈ local_smem + 2×hop ≈ 37 + 2×78 ≈ **193** (matches H200 pointer-chase).
- Remote **store** NoC delay ≈ **1 × hop** (plus dual-path immediate peer write for functional mbarrier safety).

### 7.1 Matrix file format

- Dense **N×N** integers, N = `gpgpu_n_cores_per_cluster`.
- Whitespace or comma separated; `#` comment lines allowed.
- Index = **cluster-local core id** (0 … N−1), **not** global `%smid`.
- Values are **one-way hops**, not full RTT. Diagonal = 0 (local not via NoC).
- Example (`configs/SM90_H200_REDUCED_CLUSTER4x4/dsm_latency_matrix_4.csv`):

```text
# one-way hop cycles (flat H200)
0,78,78,78
78,0,78,78
78,78,0,78
78,78,78,0
```

### 7.2 H200 mapping (job **2046238**)

Source: `../H200_profiling/output-2046238-H200Profiling.txt`.

| Profile metric | Knob / value |
|----------------|--------------|
| DSM local mean ~37.05 | `gpgpu_dsm_local_latency=37` |
| DSM remote e2e mean ~193.41 | Load path: local + 2×hop |
| Fabric one-way ~78 | Matrix off-diag / `gpgpu_dsm_remote_latency=78` |
| Stride ratio ~**1.001** | **Flat all-to-all** — constant hop is correct (not multi-hop tree) |
| Remote min/max 178–208 | Optional measured 16×16 matrix; ±8% around mean |
| TMA mcast−unicast e2e ~**135** | `gpgpu_tma_mcast_hop_latency=135` on H200 configs |
| Pair/ring BW multi-cluster | `-gpgpu_dsm_bytes_per_cycle` (default 0; fit later) |
| Legacy bare `.shared::cluster` timeout | **Do not calibrate** |

**Topology verdict:** H200 DSM is **flat**. Modeling multi-hop rank-distance would be wrong. See also `../H200_profiling/TODO.md` for remaining knobs.

**Config policy**: only `configs/SM90_H200*` are updated. SM120 keeps free multicast (NoC default off).

---

## 8. `mapa` and address model

PTX: `mapa{.shared::cluster}.type d, a, b` with `b` = target **cluster rank**.

Implementation:

1. Parse local shared offset from `a` (shared-relative or local generic).
2. Find peer CTA with matching rank in the same `cluster_group`.
3. Write `d = shared_to_generic(peer_sid, offset)`.

Generic shared windows:

```text
shared_generic(smid, off) = SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + off
```

`decode_space` for generic→shared:

- Local smid → `thread->m_shared_mem`.
- Remote smid → peer `get_cta_smem` (same cluster_group preferred).

---

## 9. TMA multicast behavior matrix

| NoC enable | `tma_mcast_enable_timing` | Functional peers | Timing peers |
|------------|---------------------------|------------------|--------------|
| 0 | * | Immediate copy | Immediate try_complete |
| 1 | 0 | Immediate copy | Immediate try_complete |
| 1 | 1 | No peer write until deliver | Data+mbar after hop |

Mask semantics (unchanged):

- Bit *i* → TB-cluster rank *i*.
- Issuer receives data iff its rank bit is set.
- Bare `.shared::cluster` without mask: all peers in `cluster_group`.

---

## 10. Tests

### Unit (`test/src/unit/cluster_noc_test.cc`)

- Matrix init diagonal/remote.
- CSV load / wrong size reject.
- H200 4×4 file load (if path reachable).
- Shared generic decode.

### Integration (existing + extensions)

| Suite | NoC off | NoC on |
|-------|---------|--------|
| `tma_cluster_multicast_test` | Pass (legacy) | Pass (delayed visibility + try_wait) |
| `tma_multicast_mask_test` | Pass | Pass |
| `cluster_multicast_multicluster_test` | Pass | Pass (isolation) |
| `dsm_test` | SelfMapa only | SelfMapa + RemoteStore + RemoteLoad |
| `mbarrier_cluster_test` | Local only | RemoteArrive + RemoteTryWait |

### Recommended run

```bash
source setup.sh && source setup_environment
make FLASH=1 -j$(nproc)

# Unit (matrix parse; self-contained)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group unit
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"

# Integration with NoC (SM120 reduced needs run-dir overlay; H200 reduced has NoC on)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group integration
# Then from test/run with NoC knobs appended to gpgpusim.config:
#   -gpgpu_cluster_noc_enable 1
#   -gpgpu_mbarrier_cluster_enable 1
#   -gpgpu_dsm_local_latency 37
#   -gpgpu_dsm_remote_latency 78
../build/bin/sm120/run_integration_tests \
  --gtest_filter='DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*'

# H200 reduced cluster, NoC on (same DSM/mbar/OneProducer filters as overlay).
# sm120 integration binaries vs SM90 sim config: allow CC mismatch.
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER4x4 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

### Programming constraints (tests / kernels)

1. **Coordinate cross-rank visibility with mbarrier**, not bare smem spin-waits (functional-first sim hangs if the peer has not issued yet).
2. **Never mix `__syncthreads` with single-thread `try_wait`** in the same CTA.
3. **Producer CTA must stay alive** until consumers finish `mapa` (mapa looks up active cluster ranks; exited slots fall back to local).
4. Prefer **`mapa.u64`** over u32 (generic shared windows exceed 32-bit on large GPUs).
5. Remote DSM stores use **dual-path**: NoC hop for issuer timing + immediate peer write so mbarrier-ordered consumers see data.

---

## 11. Migration notes (from idealized cluster TMA)

1. **Default remains NoC-off** except H200 reduced cluster config (`SM90_H200_REDUCED_CLUSTER4x4`).
2. Workloads that **read peer smem without waiting on mbarrier** may see ordering surprises; correct CUDA/TMA code always waits.
3. Do not interpret SM120 product-config sim cycles as multicast cost; NoC is off there unless overlaid.
4. TMA hop and DSM matrix are **intentionally separate** (HW mcast is not full DSM RTT).

---

## 12. FlashGPU-Sim TODO (living checklist)

**Audience:** agents and developers working **in this repo** (FlashGPU-Sim).  
**How to use:** this section is the single checklist. Mark `[x]` when done; leave a one-line note under the item if the close-out differs from the original plan. Assessment text lives in **§6.4–§6.6**; do not fork a second TODO elsewhere.

| Track | Goal | Section |
|-------|------|---------|
| Done baseline (L1 land) | Do not re-open | §12.1 |
| **F1–F9** functional | Right data / no hang on supported kernels | §12.2 |
| **L2** pipeline | Scoreboard / delayed store | §12.3 |
| **L3** BW | Fitted BPC / contention | §12.4 |
| **L4** product | Silicon harness / races | §12.5 |
| Non-goals | Do not implement | §12.6 |

**Branch/PR scope:** still the **same** unified cluster PR/branch (`docs/cluster.md`). F-items and L2–L4 are in-branch / follow-on checklist work, not separate “feature PRs.”  
**Profiling / new microbenchmarks:** `../H200_profiling/TODO.md` (separate agent, separate tree).

Suggested order: **F1 green-gate → F8/F2/F4/F5 → F6/F7/F9 → L2 → L3 → L4**.

### 12.1 Done (do not re-open without new evidence)

- [x] Intra-cluster `cluster_noc_t` (inject / cycle / deliver; race rules)
- [x] TMA peer data + mbar via NoC when enabled
- [x] DSM `mapa` + remote ld/st; dual-path store; load issue delay ≈ 2×hop
- [x] Remote mbarrier arrive / try_wait (WAIT_REG / WAIT_DONE)
- [x] CTA exit `drop_messages_to_cta`
- [x] H200 knobs/matrices from **2046238** (local ~37, one-way ~78, flat; TMA mcast hop ~135)
- [x] Knobs: `-gpgpu_dsm_bytes_per_cycle`, TMA BPC (default **0** = unlimited)
- [x] Docs: flat NoC intentional; §6.4 functional / §6.5 L0–L4 / §6.6 F1–F9

### 12.2 Checklist — F1–F9 functional correctness

Assessment: **§6.6**. Exit for the F-track: mbarrier-ordered cluster / TMA / DSM kernels produce the **right data and complete** on supported configs; unsupported PTX **fails loud**.

#### F1 — Land + green-gate  *(Done)*

- [x] Land `cluster_noc_t` + mapa / DSM / remote mbar + H200 knobs/docs/tests
- [x] **F1-1** Unit green: `./test/run_tests.sh … unit "ClusterNoc*"`
- [x] **F1-2** SM120 reduced, NoC **off** (legacy): `*ClusterLaunch*:*TMACluster*:*MultiCluster*`
- [x] **F1-3** SM120 reduced + NoC overlay: `DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*`
- [x] **F1-4** `SM90_H200_REDUCED_CLUSTER4x4` (NoC on): same integration filters as F1-3
- [x] **F1-5** Record the exact commands + pass/fail (below). Observed 2026-08-15 after `source setup.sh && source setup_environment` and `make FLASH=1`.

**F1-5 command / result record** (do not treat “should pass” as done — these were run):

```text
# F1-1  unit ClusterNoc*
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 \
  run test --target sm120 --group unit "ClusterNoc*"
# runner glob → --gtest_filter=*ClusterNoc**
# RESULT: PASS  [  PASSED  ] 6 tests.  (0 failed)

# F1-2  SM120 reduced, NoC off (legacy immediate multicast)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 \
  run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*:*MultiCluster*"
# RESULT: PASS  [  PASSED  ] 28 tests.  [  SKIPPED ] 4
#   skipped (topology, expected on m=2 n=1):
#     MultiClusterClusterTest.ClusterMulticastWithMultipleClusters
#     ClusterLaunchApiTest.ExLaunch_ClusterLargerThanPhysical_Fails
#     ClusterLaunchApiTest.ExLaunch_ClusterDim4_Succeeds
#     ClusterLaunchApiTest.ExLaunch_ClusterDim4_AllCtasMarkReady

# F1-3  SM120 reduced + NoC overlay (docs §10; run_tests.sh recopies config)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 \
  build test --target sm120 --group integration
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 refresh
# append to test/run/SM120_RTX5090_REDUCED_CLUSTER2x1/gpgpusim.config:
#   -gpgpu_cluster_noc_enable 1
#   -gpgpu_mbarrier_cluster_enable 1
#   -gpgpu_dsm_local_latency 37
#   -gpgpu_dsm_remote_latency 78
(cd test/run/SM120_RTX5090_REDUCED_CLUSTER2x1 && \
  ../../build/bin/sm120/run_integration_tests \
  --gtest_filter='DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*')
# RESULT: PASS  [  PASSED  ] 6 tests.  (0 failed / 0 skipped)
#   DsmTest.{SelfMapaLocal,RemoteStoreToPeer_TwoCtas,RemoteLoadFromPeer_TwoCtas}
#   MbarrierClusterTest.{RemoteArriveUnblocksOwner,RemoteTryWaitSeesLocalArrive}
#   TMAClusterOneProducerTest.OneProducerPeerConsumers

# F1-4  H200 reduced, NoC on (product knobs). sm120 tests vs CC 9.0 sim config.
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh \
  -c SM90_H200_REDUCED_CLUSTER4x4 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
# parsed knobs: -gpgpu_cluster_noc_enable 1, -gpgpu_mbarrier_cluster_enable 1,
#               -gpgpu_dsm_remote_latency 78
# RESULT: PASS  [  PASSED  ] 6 tests.  (0 failed)
```

`FLASHGPU_ALLOW_CC_MISMATCH=1` is required for F1-4: `run_tests.sh` otherwise rejects `--target sm120` on an SM90 config. The integration binary still exercises the H200 sim model (NoC on).

#### F2 — Idealized / incomplete mbarrier  *(Open)*

- [ ] **F2-1** Inventory mbarrier PTX used by target kernels (arrive / arrive_drop / expect_tx / complete_tx / try_wait ± timeout)
- [ ] **F2-2** Implement missing **used** variants; keep unused variants as hard-fail (no silent NOP)
- [ ] **F2-3** `try_wait` timeout: implement or reject with a clear error (today: unimplemented)
- [ ] **F2-4** Document remaining “barriers not in smem” idealization in `FLASH.md` (keep honest)

#### F3 — `barrier.cluster` / CG DSM map  *(Non-goal)*

- [x] Explicitly out of scope (`docs/cluster.md` §4). Reopen only if a target kernel requires it.

#### F4 — DSM atomics  *(Open)*

- [ ] **F4-1** Decide: implement remote `atom` via `decode_space` + NoC **or** reject
- [ ] **F4-2** If reject: `atom_impl` must abort on remote generic shared (stop using local `smid`)
- [ ] **F4-3** If implement: functional test (2-CTA remote atomic add/cas) + update `cluster_noc.h` comment
- [ ] **F4-4** Stop claiming atomics in headers/docs until F4-2 or F4-3 is done

#### F5 — `mapa` lifetime / exited producer  *(Open)*

- [ ] **F5-1** If target rank CTA is gone: **fail loud** (do not fall back to local smem)
- [ ] **F5-2** Integration test: consumer `mapa` after producer exit must not return local alias
- [ ] **F5-3** Keep “producer stays alive until consumer `mapa`” as a kernel rule in §10 until F5-1 ships

#### F6 — TMA corners  *(Open)*

- [ ] **F6-1** Audit PTX from target cluster kernels / Triton traces (only implement what they use)
- [ ] **F6-2** 96B swizzle: implement or reject (today: error print)
- [ ] **F6-3** Stubbed `tensormap.*` variants: implement or reject
- [ ] **F6-4** `cp.async.bulk.commit_group` / `wait_group`: real bulk-group wait if kernels use it; else keep NOP **and** list them as allowed stubs in test docs
- [ ] **F6-5** Sector / OOB edges only if a kernel hits them (`FLASH.md` TMA limitations)

#### F7 — Programming-model fragility  *(Open)*

- [ ] **F7-1** Detect / abort (or bounded timeout) on bare peer-smem spin with no mbarrier wait
- [ ] **F7-2** Detect / abort mixing `bar.sync` / `__syncthreads` with single-thread `try_wait` in the same CTA
- [ ] **F7-3** Keep §10 programming constraints until F7-1/F7-2 exist; point tests at the abort message

#### F8 — Test surface  *(Open)*

- [ ] **F8-1** `DsmTest`: clusterDim=4 (needs m≥4)
- [ ] **F8-2** Multi-peer DSM fan-out (one producer, two+ consumers)
- [ ] **F8-3** Drop-on-CTA-exit: inflight NoC msgs to a recycled slot must not corrupt the next occupant
- [ ] **F8-4** `mapa.u32` vs `mapa.u64` (u32 truncation on large generic windows)
- [ ] **F8-5** TMA data-before-mbar with NoC on (ordering / FIFO)
- [ ] **F8-6** One **compose** kernel: cluster launch + TMA mcast **or** DSM + remote mbar in the same test
- [ ] **F8-7** One **real-shaped** path (FA / Triton cluster / one-producer GEMM-style) functional pass

#### F9 — Dual-path store semantic  *(Open; pairs with L2-2)*

- [ ] **F9-1** Document the default (immediate peer write + issuer hop) as the **supported** functional model
- [ ] **F9-2** Implement `-gpgpu_dsm_store_immediate` (default 1). `0` = write only on NoC deliver — **same item as L2-2**
- [ ] **F9-3** Test: with immediacy=0, consumer `try_wait` still sees data (mbar after hop); without wait, must not assume instant visibility

### 12.3 Checklist — L2 pipeline fidelity

Assessment: **§6.5**. Do not start until F1 green-gate is `[x]` (wrong-data bugs will look like timing bugs).

- [ ] **L2-1** **DSM_LOAD_RSP → RF / scoreboard.** On inject of remote load: mark dest regs pending. On `DSM_LOAD_RSP` deliver: write RF + clear scoreboard. Today RF is filled at functional execute; issue delay only approximates RTT.
- [ ] **L2-2** **`-gpgpu_dsm_store_immediate` (0/1).** Default 1 (current dual-path). `0` = peer write only on NoC deliver. Same close-out as **F9-2**.
- [ ] **L2-3** **Load path uses REQ/RSP end-to-end** when NoC is on; complete only on RSP (pairs with L2-1).
- [ ] **L2-4** **Regression.** `DsmTest.*` + `MbarrierClusterTest.*` + TMA cluster filters still green with NoC on. Remote-load microbench ≈ local + 2×hop.

### 12.4 Checklist — L3 BW / contention

- [ ] **L3-1** **Fit non-zero `-gpgpu_dsm_bytes_per_cycle`.** Use **single-cluster** microbench numbers from the profiling suite (not multi-cluster aggregate GB/s). Leave 0 until a fit exists.
- [ ] **L3-2** **Fit non-zero `-gpgpu_tma_mcast_bytes_per_cycle`.** From TMA size-sweep e2e growth if useful.
- [ ] **L3-3** **Optional: credits / ingress queue.** Only if profiling provides queue-depth or multi-issuer contention data.
- [ ] **L3-4** **Optional: bcast fan-out tax.** HW shows bcast GB/s collapse with cluster size; not multi-hop. Model as a separate tax if needed.

### 12.5 Checklist — L4 / product

- [ ] **L4-1** Suite-vs-silicon validation harness for cluster kernels (beyond unit/integration gtests)
- [ ] **L4-2** OpenMP + NoC race audit (optional TSan)
- [ ] **L4-3** Remaining TMA/mbarrier corner cases not claimed by F2/F6 — see `FLASH.md` limitations

### 12.6 Explicit non-goals for the sim NoC

Do **not** add checklist items for these unless new HW evidence appears.

| Idea | Why not |
|------|---------|
| Multi-hop / tree hop by rank distance | H200 stride ratio ≈ 1.0 (flat) |
| Calibrate from bare `.shared::cluster` timeouts | Known timeout artifact |
| Permanent matrix-import script in this repo | Offline gen; commit CSVs + comments only |
| **F3** `barrier.cluster` / full CG DSM map | Beyond specials + `mapa` (`docs/cluster.md` §4) |
| Cross-physical-cluster DSM or TMA mcast | HW TB clusters do not span arbitrary physical clusters |

---

## 13. Related docs

- `docs/cluster.md` — **branch/PR overview** (all cluster pillars)
- `docs/cluster_cta2_realLaunch.md` — launch API + co-residency
- `FLASH.md` — product feature list / limitations
- `configs/SM90_H200/README.md` — H200 calibration (job 2046238)
- `../H200_profiling/README.md` — profiling suite layout
- `../H200_profiling/TODO.md` — profiling microbenchmarks for L2–L4 knobs
- `../H200_profiling/output-2046238-H200Profiling.txt` — HW ground truth
