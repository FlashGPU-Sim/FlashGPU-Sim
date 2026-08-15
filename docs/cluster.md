# Thread Block Cluster support (branch overview)

**Branch / PR intent:** one **unified** change set that makes FlashGPU-Sim support Hopper/Blackwell **Thread Block Clusters** end-to-end — **functionally correct** for real-style kernels, and **cycle-tunable** toward cycle-accurate SM↔SM timing (maturity **L1**, climbing L2–L4).

This is **not** a stack of separate “launch-only” / “TMA-only” / “NoC-only” PRs. Submit as **one large PR** covering:

| Pillar | What lands |
|--------|------------|
| **Real cluster launch** | `cudaLaunchKernelExC`, cluster attrs, PTX required dims, co-resident issue |
| **TMA `.shared::cluster`** | Multicast + selective `ctaMask`; peer data and mbarrier complete |
| **Remote mbarrier** | mapa’d arrive / try_wait via cluster path |
| **DSM** | Real `mapa.u64`, remote ld/st; inactive rank aborts |
| **Intra-cluster NoC** | Hop / flat matrix (H200 job 2046238), race-safe deliver |

---

## 1. Where to read (doc map)

| Doc | Role |
|-----|------|
| **`docs/cluster.md` (this file)** | Branch/PR overview, feature matrix, non-goals, reading order |
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
| Local mbarrier | Yes (main ops) | Calibrated arrive/try_wait knobs |
| Remote mbarrier (mapa) | Yes | Via NoC hop |
| DSM mapa + remote ld/st | Yes (tested patterns); inactive rank **aborts** | Flat hop L1; load RTT≈2×hop; dual-path store |
| Intra-cluster NoC | Yes | **L1** flat hop (H200); L2–L4 in §12 of `cluster_noc.md` |

**Maturity (detailed tables):** `docs/cluster_noc.md` **§6.4** (functional usefulness), **§6.5** (cycle levels L0–L4), **§6.6** (functional gaps F1–F9). Living TODO: **§12**.

**Target for this PR:** ship **functional correctness** for mbarrier-ordered cluster/TMA/DSM kernels **and** an honest **L1 cycle-tunable** SM↔SM fabric (not “free multicast”). Remaining F-items and L2–L4 work are the **in-branch checklist** in `cluster_noc.md` §12, not separate feature PRs.

---

## 3. Config quick reference

`CLUSTERmxn` → **m** = SMs per physical cluster, **n** = number of physical clusters.

| Config | Topology | Typical use |
|--------|----------|-------------|
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | m=2, n=1 | Fast peer smoke |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | m=2, n=2 | Multi-cluster isolation |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | m=4, n=4 | Primary multi-SM functional |
| `SM120_RTX5090_CLUSTER16x11` | m=16, n=11 | Full GPC-aligned smoke |
| `SM90_H200_REDUCED_CLUSTER4x4` | m=4, n=4 | **NoC on** + H200 hop calibration |
| `SM90_H200` | m=1, n=132 | Product latencies; NoC idle |

Rule: TB-cluster size (product of launch cluster dims) **≤ m**.

---

## 4. Non-goals (still out of scope for this PR)

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

## 5. Suggested test filters

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

## 6. Code map (high level)

| Area | Location |
|------|----------|
| Launch Ex / attrs | `libcuda/cuda_runtime_api.cc`, `kernel_info` |
| Co-resident issue | `gpgpu_sim::issue_block2core` (`gpu-sim.cc`) |
| TMA multicast | `src/gpgpu-sim/flash/tma.cc` |
| mbarrier + remote | `src/gpgpu-sim/flash/mbarrier.cc` |
| DSM mapa / decode | `src/cuda-sim/instructions.cc` |
| Intra-cluster NoC | `src/gpgpu-sim/flash/cluster_noc.{h,cc}` |
| Tests | `test/src/integration/cluster_*`, `tma_cluster_*`, `dsm_test`, `mbarrier_cluster_test` |
