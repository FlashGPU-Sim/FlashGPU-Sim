# Thread Block Cluster support (branch overview)

**Branch / PR intent:** one **unified** change set that makes FlashGPU-Sim support Hopper/Blackwell **Thread Block Clusters** end-to-end — **functionally correct** for real-style kernels, and **cycle-tunable** toward cycle-accurate SM↔SM timing (maturity **L1**, climbing L2–L4).

This is **not** a stack of separate “launch-only” / “TMA-only” / “NoC-only” PRs. Submit as **one large PR** covering:

| Pillar | What lands |
|--------|------------|
| **Real cluster launch** | `cudaLaunchKernelExC`, cluster attrs, PTX required dims, co-resident issue |
| **TMA `.shared::cluster`** | Multicast + selective `ctaMask`; peer data and mbarrier complete |
| **Remote mbarrier** | mapa’d arrive / try_wait via cluster path |
| **DSM** | Real `mapa.u64`, remote ld/st/`atom`; inactive rank aborts |
| **Intra-cluster NoC** | Hop / flat matrix (H200 job 2046238), race-safe deliver |

---

## 1. Where to read (doc map)

| Doc | Role |
|-----|------|
| **`docs/cluster.md` (this file)** | Branch/PR overview, feature matrix, concurrency model, non-goals |
| **`docs/cluster_noc.md`** | NoC / DSM / remote mbar **design authority**; **§6.4–§6.6** maturity (F1–F9 + L0–L4); **§12** living checklist |
| **`docs/cluster_cta2_realLaunch.md`** | Launch API, co-residency, configs `CLUSTERmxn`, tests |
| **`FLASH.md`** | Product feature list + limitations (short) |
| **`configs/SM90_H200/README.md`** | H200 calibration numbers |
| **`../H200_profiling/TODO.md`** | Profiling-suite agent: microbenchmarks for deeper cycle accuracy |
| **`../H200_profiling/output-2046238-H200Profiling.txt`** | HW ground truth (flat DSM, TMA mcast premium) |

**Removed (deprecated):**

- `docs/cluster_cta2_todo.md` — multi-PR “later follow-up” framing; non-goals folded **here** and into `cluster_noc.md` §12  
- `docs/cluster_cta2_explain.md` — long pre-NoC branch dump (free multicast, frozen commit list); superseded by this map + `cluster_noc.md` + `realLaunch.md`

---

## 2. Feature matrix (this branch)

| Area | Functional | Timing |
|------|------------|--------|
| Cluster launch + co-residency | Yes | Idealized schedule (pack whole TB-cluster on one physical cluster) |
| TMA cluster multicast (+ mask) | Yes | NoC hop when enabled; free/immediate when NoC off |
| Local mbarrier | Yes (used ops); `try_wait` timeout sets dest pred | Calibrated arrive/try_wait knobs |
| Remote mbarrier (mapa) | Yes | Via NoC hop |
| DSM mapa + remote ld/st/`atom` | Yes (tested patterns); inactive rank **aborts**; `red`/`red.async` unimplemented | Flat hop L1; load RTT≈2×hop; store default dual-path, or deliver-only via `-gpgpu_dsm_store_immediate 0` |
| Intra-cluster NoC | Yes | **L1** flat hop (H200); L2–L4 in §12 of `cluster_noc.md` |

**Maturity (detailed tables):** `docs/cluster_noc.md` **§6.4** (functional usefulness), **§6.5** (cycle levels L0–L4), **§6.6** (functional gaps F1–F9). Living TODO: **§12**.

**Target for this PR:** ship **functional correctness** for mbarrier-ordered cluster/TMA/DSM kernels **and** an honest **L1 cycle-tunable** SM↔SM fabric (not “free multicast”). Remaining work is F4 `red`/`red.async` and L2–L4 (except L2-2) in the **in-branch checklist** in `cluster_noc.md` §12, not separate feature PRs.

---

## 3. Cluster concurrency (silicon vs this sim)

On real Hopper/Blackwell, CTAs in one Thread Block Cluster are **co-resident and live at the same time**. They run on different SMs in the same GPC and can talk through DSM / mbarrier while both still exist.

This sim keeps that as **simulated time**, not as host threads.

### 3.1 What is modeled

| Silicon | This sim |
|---------|----------|
| Whole TB cluster on one GPC | Whole TB cluster on one `simt_core_cluster` (`cluster_group` + ranks) |
| Each SM has its own scheduler | Each GPU cycle, **one host thread** runs `SM.cycle()` for every core in that cluster, then `cluster_noc.cycle()` |
| OpenMP / multi-core host | Flash OpenMP is **one thread per physical cluster**, not per SM |

So rank 0 and rank 1 both **tick every GPU cycle**. They are concurrent the way a cycle-accurate CPU sim is concurrent: lockstep interleaving, not two OS threads.

Event-ordered communication is correct because of **state + messages + cycle order**, not because two CTAs run simultaneously on the host:

1. Both CTAs stay allocated until they retire (`mapa` of an inactive rank **aborts**).
2. Cross-SM effects are NoC messages (or the documented default dual-path store). Peer smem / remote mbarrier change on **deliver** after a hop, on the same cluster thread. Remote DSM stores also write peer smem at issue unless `-gpgpu_dsm_store_immediate 0`.
3. `mbarrier.try_wait` **registers interest and parks the warp**. The producer can issue, the hop can complete, then the waiter is released.

That is the supported model: **mbarrier-ordered** cluster / TMA / DSM (same as `cluster_noc.md` §6.4 and §10).

### 3.2 Gap: busy-wait is not a wait

On silicon, this is usually fine if the producer is still alive:

```text
CTA 1:  while (*peer_flag == 0) { }   // load + branch
CTA 0:  *my_flag = 1;                 // eventually issues
```

Both SMs keep issuing. The spin is just traffic. When the store becomes visible, the loop ends. CUDA still **prefers** mbarrier / `cluster.sync()`; a bare spin on live peer smem is legal-ish, not the recommended pattern.

This sim does **not** treat a load-loop as “I am waiting on a peer”:

| Pattern | Real GPU | This sim |
|---------|----------|----------|
| TMA / DSM + **mbarrier** / `try_wait` | Works | **Supported** |
| Peer ld/st/`atom` after that wait | Works | Works (decode + NoC; default immediate store, or deliver-only) |
| Both CTAs live; each getting cycles | Parallel SMs | Yes in **cycle time**, one host thread per GPC |
| Spin on peer smem or a global flag until the peer writes | Usually works | **May hang or miss the write** |
| Two CTAs `atom` the same word in the same clock | HW arbiter | Never simultaneous (serialized on the cluster thread) |

Typical hang: the consumer functionally executes `while (flag == 0)` **before** the producer has been allowed to issue, or the load does not re-observe the store. `try_wait` yields; a spin does not. Functional-only / execute-until-barrier is worse: a spin never hits `bar.sync`, so the peer CTA is never scheduled.

This is **not** caused by “one OpenMP thread per GPC.” That design is fine for event-ordered comm. Spins fail because **wait ≠ load-loop** in this exec model.

### 3.3 What we do about it

- **Supported kernels:** coordinate with mbarrier (or a future `cluster.sync`), not “load until I see it.” Tests in `dsm_test` / TMA cluster follow that rule.
- **Do not** treat a passing contention test as proof of hardware races — intra-cluster atoms are pipeline-serialized on one host thread.
- **Hang preventers** (sim only, not silicon): after a **recent** peer DSM/TMA access, a tight PC loop with no mbarrier interest aborts (§10 rule 1). The arm expires on a recognized wait or after a hop-scale window with no further peer touch. A partial-warp `try_wait` parked next to a `bar.sync` waiter for longer than hop-scale latencies aborts (§10 rule 2). Parked `try_wait` and hop-scale TMA mixes are exempt. Default dwell 8192 cycles (`-gpgpu_cluster_hang_watchdog`).
- **Only make bare spins work** if a target kernel actually needs them (re-read owner smem each iteration; never starve the peer). That is optional fidelity, not a blocker for this PR.

---

## 4. Config quick reference

`CLUSTERmxn` → **m** = SMs per physical cluster, **n** = number of physical clusters.

| Config | Topology | Typical use |
|--------|----------|-------------|
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | m=2, n=1 | Fast peer smoke |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | m=2, n=2 | Multi-cluster isolation |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | m=4, n=4 | Primary multi-SM functional |
| `SM120_RTX5090_CLUSTER16x11` | m=16, n=11 | Full GPC-aligned smoke |
| `SM90_H200_REDUCED_CLUSTER4x4` | m=4, n=4 | **NoC on** + H200 hop calibration; DSM store deliver-only |
| `SM90_H200` | m=1, n=132 | Product latencies; NoC idle |

Rule: TB-cluster size (product of launch cluster dims) **≤ m**.

---

## 5. Non-goals (still out of scope for this PR)

These remain **non-goals** even with a unified cluster PR (not “later PRs for the same pillars”):

| Non-goal | Reason |
|----------|--------|
| Preferred-substitute cluster dims / full occupancy APIs | CUDA runtime surface; stubs only |
| Cross-physical-cluster DSM or TMA mcast | HW TB clusters do not span arbitrary physical clusters |
| Full `barrier.cluster` / CG DSM map builtins | Beyond special regs + mapa path |
| Multi-hop / tree NoC by rank distance | H200 stride ratio ≈ 1.0 → **flat** |
| Perfect L4 silicon match for all apps | Requires L2–L3 + full GPU model fidelity |
| Unrelated suite hangs (e.g. some `MBarrierSanity` cases) | Pre-existing; not cluster scope |

Optional product polish (floorswept GPC sizes, extra launch attributes) can still land later without splitting the core cluster pillars.

---

## 6. Suggested test filters

```bash
source setup.sh && source setup_environment
make FLASH=1 -j$(nproc)

# Launch + TMA cluster (SM120 reduced)
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group integration
# From test/run with NoC overlay if needed for DSM/remote mbar:
#   -gpgpu_cluster_noc_enable 1 -gpgpu_mbarrier_cluster_enable 1
#   -gpgpu_dsm_local_latency 37 -gpgpu_dsm_remote_latency 78
../build/bin/sm120/run_integration_tests \
  --gtest_filter='*ClusterLaunch*:*TMACluster*:*MultiCluster*:DsmTest.*:MbarrierClusterTest.*'

# Unit NoC matrix
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"
```

H200 product path: config `SM90_H200_REDUCED_CLUSTER4x4` (NoC defaults on).

---

## 7. Code map (high level)

| Area | Location |
|------|----------|
| Launch Ex / attrs | `libcuda/cuda_runtime_api.cc`, `kernel_info` |
| Co-resident issue | `gpgpu_sim::issue_block2core` (`gpu-sim.cc`) |
| TMA multicast | `src/gpgpu-sim/flash/tma.cc` |
| mbarrier + remote | `src/gpgpu-sim/flash/mbarrier.cc` |
| DSM mapa / decode | `src/cuda-sim/instructions.cc` |
| Intra-cluster NoC | `src/gpgpu-sim/flash/cluster_noc.{h,cc}` |
| Tests | `test/src/integration/cluster_*`, `tma_cluster_*`, `dsm_test`, `mbarrier_cluster_test` |
