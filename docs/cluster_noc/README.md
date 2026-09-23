# Thread Block Cluster, DSM, and intra-GPC fabric

Shipped design for Hopper Thread Block Clusters in FlashGPU-Sim. English only.
Do not start a second design doc outside this directory.

| Feature | Functional | Timing |
|---------|------------|--------|
| Cluster launch (`cudaLaunchKernelExC` / required cluster dims) | Yes | Whole TB-cluster co-resident on one GPC |
| TMA `.shared::cluster` + multicast mask | Yes | Fixed `-gpgpu_tma_multicast_latency`; **no** fabric traffic |
| DSM `mapa` + remote ld/st/`atom.add` | Yes; inactive rank **aborts**; `red`/`red.async` unimplemented | Intra-GPC fabric (`dsm_fabric_t`) |
| Local mbarrier | Used ops | Arrive / try_wait knobs |
| Remote mbarrier | Yes when DSM + cluster mbarrier knobs are on | Fabric request/completion packets |
| `barrier.cluster.arrive` / `wait` | Yes | Releases after every active warp in every CTA of the TB-cluster arrives |

Remote DSM and remote mbarrier require `-gpgpu_dsm_enable 1`. Ordinary
non-cluster kernels and DSM-off configs are unchanged. TMA multicast remains
functional fan-out (known timing gap vs silicon).

## Known timing gaps (frozen)

- **TMA cluster multicast**: functional fan-out plus fixed
  `-gpgpu_tma_multicast_latency` (100 cycles on CLUSTER132). No bandwidth,
  route, queue, SRAM-service, or contention model.
- **GEMM CUDA-event times** (unicast and multicast) are mostly **outside** the
  10% H200 gate on CLUSTER132: small multicast ~−20%, K2K multicast ~+28%
  (last measured; see [`calibration.md`](calibration.md)).
- This PR does not close that gap; a dedicated multicast timing model is a
  separate follow-up.

---

## Reading order

| File | Contents |
|------|----------|
| **This file** | Status, configs, code map |
| [`architecture.md`](architecture.md) | GPC vs TB-cluster, two networks, cycle order |
| [`dsm_fabric.md`](dsm_fabric.md) | Flits, VCs, GPCMMU, GX, shaper, ACK |
| [`pipeline.md`](pipeline.md) | Remote load = local SMEM + fabric |
| [`programming_model.md`](programming_model.md) | Launch, `mapa`, TMA, mbarrier, hang preventers |
| [`knobs.md`](knobs.md) | Knobs that still exist |
| [`tests.md`](tests.md) | How to run the functional filters |
| [`evidence.md`](evidence.md) | Measured / patent / inferred labels |
| [`calibration.md`](calibration.md) | Frozen H200 CLUSTER132 status |

Physical path (patent US12248788B2 Figs. 21A–21D). **TPC / TPCARB are not
modeled.** Power-gated SM slots **are**. Why DSM is not the memory xbar:
[`dsm_fabric.md`](dsm_fabric.md) §9.

---

## Configs

`CLUSTERmxn`: **m** = `-gpgpu_n_cores_per_cluster` (also `-gpgpu_num_sms_per_gpc`),
**n** = `-gpgpu_n_clusters` (also `-gpgpu_num_gpcs`). Old and new knobs are
aliases; if both are set and disagree, start-up aborts.

TB-cluster **size** is a launch attribute: `product(clusterDim) ≤` enabled SMs
in that GPC.

| Config | Topology | Use |
|--------|----------|-----|
| `SM120_RTX5090` | 170 × 1 | Default Blackwell; cluster knobs off |
| `SM90_H100` | 132 × 1 | Hopper FA / WGMMA; cluster knobs off |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | 2 × 1 | Fast peer smoke; fabric on |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | 2 × 2 | Multi-cluster isolation; fabric on |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | 4 × 4 | Primary functional cluster / DSM / TMA |
| `SM90_H200_CLUSTER132` | 6×16 + 2×18 = 132 | Only shipped H200 product packing; fabric on |

---

## Non-goals

| Non-goal | Reason |
|----------|--------|
| TPC / TPCARB | SMs attach to GPCMMU / GPCARB |
| Preferred-substitute cluster dims / full occupancy APIs | Stubs only |
| Cross-GPC DSM or TMA multicast | Hardware TB-clusters stay in one GPC |
| Dual **physical** DSM networks | VCs share lanes |
| BookSim as the DSM fabric | Global state |
| Stamping H200 numbers as Blackwell hardware fact | Separate preset later |
| Making bare peer-smem spins work | Use mbarrier |
| TMA multicast contention / bandwidth | Known follow-up; not modeled |

---

## Code map

| Area | Location |
|------|----------|
| Launch Ex / attrs | `libcuda/cuda_runtime_api.cc`, `src/kernel_info.h` |
| Co-resident issue | `gpgpu_sim::issue_block2core` / `simt_core_cluster::issue_block2core` |
| Cluster specials | `src/cuda-sim/ptx_sim.cc` |
| `mapa` / generic decode | `src/cuda-sim/instructions.cc`, `src/gpgpu-sim/flash/tb_cluster.h` |
| TMA multicast | `src/gpgpu-sim/flash/tma.cc` |
| mbarrier + remote | `src/gpgpu-sim/flash/mbarrier.cc` |
| DSM fabric | `src/gpgpu-sim/dsm_fabric.{h,cc}`, `dsm_endpoint.{h,cc}` |
| Hang preventers | `src/gpgpu-sim/flash/cluster_hang_prevent.h` |
| Memory icnt (not DSM) | `local_interconnect` / `intersim2` |
| Tests | `test/src/integration/cluster_*`, `dsm_test.cc`, `mbarrier_cluster_test.cc`, `tma_cluster_*`; unit `tb_cluster_test.cc` |
