# Thread Block Cluster, DSM, and intra-GPC fabric

**Living design** for Hopper/Blackwell Thread Block Clusters in FlashGPU-Sim lives **only** in this directory. English only.

Two layers exist at once:

| Layer | What it is | Status |
|-------|------------|--------|
| **In-tree code** | Functional cluster launch + TMA multicast + DSM `mapa` + remote mbarrier + a **typed delay-line** (`cluster_noc_t`) | Shipped. Functional track **A** closed except `red` / `red.async`. |
| **Target architecture** | Per-GPC flit fabric (`dsm_fabric_t`): two VCs, 32 B **payload** per flit, GPCMMU hash, GPCARB 6→4, configurable GX planes, per-SM traffic control, coalesced write ACK, remote load through the **same scoreboard / LDST path as local SMEM** | Not implemented. Checklist: [`todos.md`](todos.md). |

Do not treat the delay line as the end-state performance model. Do not start a second design doc outside this directory.

The English files in this directory are the working spec. Supervisor plan v2 is a local reference only (not in this tree). Chapter 18 of that plan (student split) is ignored; every phase is in-tree work.

---

## Reading order

| Order | File | Read when |
|------:|------|-----------|
| 1 | **This file** | Always |
| 2 | [`todos.md`](todos.md) | Before any code change. Agent entry point. |
| 3 | [`architecture.md`](architecture.md) | GPC vs TB-cluster, topology, two networks, cycle order |
| 4 | [`dsm_fabric.md`](dsm_fabric.md) | Flits, VCs, GPCMMU, GX, shaper, ACK, why not the memory xbar |
| 5 | [`pipeline.md`](pipeline.md) | Remote load = local SMEM load + fabric; scoreboard; shared-memory service |
| 6 | [`programming_model.md`](programming_model.md) | Launch, `mapa`, TMA, mbarrier, hang preventers |
| 7 | [`knobs.md`](knobs.md) | Current delay-line knobs and **target** fabric knobs |
| 8 | [`tests.md`](tests.md) | What to run after a change |
| 9 | [`evidence.md`](evidence.md) | H200 measurements vs patent vs inference |

Physical path (patent US12248788B2 Figs. 21A–21D). **TPC / TPCARB are not modeled.** PG'd SM slots **are** modeled. ASCII path: [`architecture.md`](architecture.md), [`dsm_fabric.md`](dsm_fabric.md).

Legacy delay-line figure (current code, not the target): [`figures/dsm_model.svg`](figures/dsm_model.svg).

---

## What the branch already does (code)

Hopper/Blackwell let several CTAs form a **Thread Block Cluster**, sit on neighboring SMs in one GPC, and:

1. Launch together and discover rank / cluster id.
2. Copy global tiles into **peer** shared memory with TMA (`.shared::cluster`, optional multicast mask).
3. Read and write a peer CTA’s shared memory (**DSM**) via `mapa`.
4. Synchronize with **mbarrier** objects that may live in a peer CTA.

| Area | Functional (today) | Timing (today) |
|------|--------------------|----------------|
| Cluster launch + co-residency | Yes | Idealized: whole TB-cluster on one physical cluster |
| TMA cluster multicast (+ mask) | Yes | Delay-line hop when NoC on; immediate when off |
| Local mbarrier | Yes (used ops) | Calibrated arrive / try_wait knobs |
| Remote mbarrier (`mapa`) | Yes | Delay-line hop |
| DSM `mapa` + remote ld/st/`atom` | Yes; inactive rank **aborts**; `red`/`red.async` unimplemented | Flat hop ~78 one-way; load RF filled at execute + stall `2×hop` |
| Intra-cluster NoC | `cluster_noc_t` messages | Delay line. Optional BPC default **unlimited** |

Supported kernels are **mbarrier-ordered**. Bare peer-smem spins may hang; hang preventers abort two sim anti-patterns ([`programming_model.md`](programming_model.md)).

---

## What we are building (target)

One sentence:

> Global memory NoC stays **per SM**. DSM is a **separate per-GPC fabric**. Request and response are **virtual channels** that share physical 32 B-payload lanes. Remote loads complete like **local shared-memory loads** (scoreboard + LDST writeback) after the fabric returns data.

Physical path (**no TPC, no TPCARB**):

```text
SM → GPCMMU hash → GPCARB ingress (6 in → 4 out)
  → GX0 … GX{n-1} (default n=2, parallel planes, not VCs)
  → GPCARB egress → destination SM shared_memory_service
```

A **CPC** is always **6 SM slots + one GPCMMU + one GPCARB**. A GPC has a configurable number of CPCs (default 3 → 18 slots). Enabled SM count is configurable; leftover slots are **PG'd** (present in the topology table, unused).

Traffic control: **inferred** 2/3 of a 32 B payload lane per SM, **non-work-conserving**. Implement `fixed_tdm` (plan-v2 CPC-slot schedule), `skip_mod` (different SMs skip different cycles), and `hard_rate_cap`. None of those schedules is a silicon dump. Idle slots are not donated.

---

## Config names

`CLUSTERmxn` today: **m** = `-gpgpu_n_cores_per_cluster` (SMs per physical cluster), **n** = `-gpgpu_n_clusters`.

Target names: **m** = enabled SMs per GPC (`-gpgpu_num_sms_per_gpc`), **n** = `-gpgpu_num_gpcs`. Old knobs remain aliases until **B-DEPR**.

TB-cluster **size** is a launch attribute. Rule: `product(clusterDim) ≤` enabled SMs in that GPC.

| Config | Topology today | Typical use |
|--------|----------------|-------------|
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | m=2, n=1 | Fast peer smoke |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | m=2, n=2 | Multi-cluster isolation |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | m=4, n=4 | Primary multi-SM functional |
| `SM120_RTX5090_CLUSTER16x11` | m=16, n=11 | GPC-sized smoke |
| `SM90_H200_REDUCED_CLUSTER16x2` | m=16, n=2 | Delay-line NoC **on** + H200 hop knobs; GPC-shaped |
| `SM90_H200` | m=1, n=132 | Product latencies; NoC idle |

---

## Non-goals

| Non-goal | Reason |
|----------|--------|
| TPC / TPCARB as a unit | Supervisor + this rewrite: SMs attach to GPCMMU / GPCARB |
| `barrier.cluster` / CG DSM map builtins | Specials + `mapa` only |
| Preferred-substitute cluster dims / full occupancy APIs | Stubs only |
| Cross-GPC DSM or TMA multicast | Hardware TB-clusters stay in one GPC |
| Multi-hop / tree hop by rank | H200 stride ratio ≈ 1.0 |
| Dual request/reply **physical** DSM networks | VCs share lanes |
| BookSim as the first DSM fabric | Global state, extra uncalibrated hops |
| Stamping H200 numbers as Blackwell hardware fact | Separate preset later |
| Making bare peer-smem spins work | Use mbarrier |

---

## Code map (today)

| Area | Location |
|------|----------|
| Launch Ex / attrs | `libcuda/cuda_runtime_api.cc`, `src/kernel_info.h` |
| Co-resident issue | `gpgpu_sim::issue_block2core` (`gpu-sim.cc`) |
| Cluster specials | `src/cuda-sim/ptx_sim.cc` |
| `mapa` / generic decode | `instructions.cc` |
| Remote store policy | `src/gpgpu-sim/flash/cluster_dsm_store.h` |
| TMA multicast | `src/gpgpu-sim/flash/tma.cc` |
| mbarrier + remote | `src/gpgpu-sim/flash/mbarrier.cc` |
| Delay-line NoC | `src/gpgpu-sim/flash/cluster_noc.{h,cc}` |
| Hang preventers | `src/gpgpu-sim/flash/cluster_hang_prevent.h` |
| Memory icnt (not DSM) | `local_interconnect` / `intersim2` |
| Tests | `test/src/integration/cluster_*`, `dsm_test.cc`, `mbarrier_cluster_test.cc`, `tma_cluster_*`; unit `cluster_noc_test.cc` |

Target types (`gpc_t`, `dsm_fabric_t`, `gpu_topology_t`) do not exist yet. Introduce them in **B1–B3** ([`todos.md`](todos.md)).

---

## Historical files

These paths are **stubs** that point here:

- `docs/cluster.md`
- `docs/cluster_noc.md`
- `docs/cluster_cta2_realLaunch.md`
- `docs/cluster_cta2_midterm.md` (midterm snapshot; not a living spec)

Why DSM is not the memory xbar: [`dsm_fabric.md`](dsm_fabric.md) §9.
