# Thread Block Cluster, intra-GPC DSM, and H200 CLUSTER132

FlashGPU-Sim on this branch launches Thread Block Clusters, copies with TMA
`.shared::cluster` (including multicast masks), reads and writes a peer CTA’s
shared memory through `mapa` on an intra-GPC flit fabric, and synchronizes
with local and remote mbarrier plus `barrier.cluster`. The product-scale
Hopper H200 NVL preset is `SM90_H200_CLUSTER132`.

Design detail stays in the other files in this directory.
[`README.md`](README.md) is the index. This file is the feature summary:
what shipped, what was measured, and what remains open.

---

## Why it exists

Hopper and Blackwell let several CTAs form a cluster, sit on neighboring SMs
in one GPC, and share four things:

1. Co-resident launch, plus rank and cluster-id discovery
2. TMA copies into peer shared memory, with an optional multicast mask
3. Direct peer shared-memory `ld` / `st` / `atom.add` through `mapa`
4. mbarrier objects that may live in a peer CTA

`flash` already had TMA and local mbarrier. It did not have cluster launch,
DSM, remote mbarrier, a GPC-shaped SM map, or an H200 product packing.

Cluster and DSM support is opt-in. Configs that already exist on `flash`
keep their topology, address hashing, and opcode latencies. Ordinary
`cudaLaunchKernel` / `<<<>>>` launches do not invent peer CTAs.

---

## Cluster launch

Implemented in `libcuda/cuda_runtime_api.cc`. Tests:
`tests/src/integration/cluster_launch_api_test.cu`, helper
`tests/common/cluster_launch.h`.

| API | Behavior |
|-----|----------|
| `cudaLaunchKernelExC` + `cudaLaunchAttributeClusterDimension` | Packs one TB-cluster onto one GPC |
| `cudaFuncSetAttribute` RequiredCluster* / MustBeSet | Validates and stores metadata |
| Preferred cluster dim / scheduling policy | Stored or warned; no spread model |
| Ordinary `cudaLaunchKernel` / `<<<>>>` | Unchanged; no false peers |

Validation: each dimension is at least 1, the grid is a multiple of the
cluster dimensions, and `product(clusterDim)` is at most the number of
enabled SMs in that GPC.

Special registers (`%cluster_ctarank`, `%clusterid`, `%is_explicit_cluster`,
and the rest of that set) are in `src/cuda-sim/ptx_sim.cc`.

`barrier.cluster.arrive` / `barrier.cluster.wait` release after every active
warp in every CTA of the reserved TB-cluster has arrived. Upstream
`cluster.sync()` bandwidth kernels need that scope.

---

## Topology

`gpu_topology_t` maps enabled SMs to a GPC and a CPC slot. A CPC is 6 SM
slots. Leftover slots are power-gated: they stay in the table and are unused.

`SM90_H200_CLUSTER132` packing is **inferred**, not a measured SMID dump:

```text
-gpgpu_gpc_sms 16,16,16,16,16,16,18,18
```

That is 6×16 + 2×18 = **132 SMs** across 8 GPCs. CUDA compute capability 9.0
occupancy on this preset is 2048 threads and 32 blocks per SM.

Each enabled SM has its own global-interconnect endpoint. DSM is a separate
per-GPC fabric. DSM traffic does not ride the memory crossbar.

`-gpgpu_n_clusters` / `-gpgpu_n_cores_per_cluster` remain aliases of the GPC
knobs, so `SM120_RTX5090` (170×1) and `SM90_H100` (132×1) still mean what
they mean on `flash`. If both names are set and disagree, start-up aborts.

---

## DSM fabric

`dsm_fabric_t` grants 32-byte payloads. Request and response virtual channels
share physical lanes. The fabric has a GPCMMU hash, a GPCARB 6-to-4 stage,
configurable GX planes (default 2), a per-SM shaper (`skip_mod` by default;
idle slots are not donated), a coalesced write ACK, and an outstanding
transaction window.

Remote `ld` / `st` / `atom.add` use the same TB-cluster resolver as launch
(`src/gpgpu-sim/flash/tb_cluster.cc`). A remote load completes on the local
shared-memory scoreboard and LDST writeback path after the fabric returns
data. A store becomes visible on the SRAM grant, not at issuer execute.

CLUSTER132 fabric floors are inferred or fitted, not a per-hop silicon trace:

- `-gpgpu_dsm_base_latency_cycles 78`
- `-gpgpu_dsm_store_visibility_latency_cycles 245`
- shaper period 3, 32-byte payload, VC depth 512, outstanding window 1024

HBM geometry is datasheet-derived: 94 simulated channels from the H200 NVL
6016-bit bus, 188 L2 slices × 320 KiB ≈ 58.75 MiB, clocks 1785 / 3201 MHz.
See [`calibration.md`](calibration.md).

DSM bandwidth and hop shape on this preset are mostly in line with the H200
comparison used for the branch. This cleanup does not retune those knobs.

PTX `red` / `red.async` are unimplemented. `mapa` of an inactive rank aborts.
Hang preventers abort a bare peer-smem spin and a mixed `bar.sync` plus
single-thread `try_wait` pattern. Those checks are simulator guards, not
hardware detectors. They stay off unless DSM or remote mbarrier is enabled.

---

## TMA cluster multicast

Functional behavior that shipped:

- `.shared::cluster` destination
- `.multicast::cluster` plus `ctaMask`
- multi-cluster isolation
- completion tied to mbarrier `complete_tx`

Timing that did not ship:

- Multicast is not a DSM-fabric transaction. Unit test
  `GpuTopology.TmaMulticastDoesNotUseDsmFabric` locks that in.
- Latency is the fixed knob `-gpgpu_tma_multicast_latency` (100 cycles on
  CLUSTER132). There is no bandwidth, route, queue, SRAM-service, or
  contention model.
- The extra CLUSTER132 TMA completion floors
  (`-gpgpu_tma_cluster_load_completion_*`) are fitted floors, not a
  multicast network.

This gap stays open. A later change that routes multicast through the DSM
fabric is a new model, not a cleanup of this one.

---

## mbarrier

Used operations include `try_wait` with a timeout destination predicate.
Remote arrive, try_wait, expect_tx, and complete_tx travel on the fabric
when both `-gpgpu_dsm_enable 1` and `-gpgpu_mbarrier_cluster_enable 1` are
set. mbarrier objects are not smem-backed 64-bit words. Unused variants
hard-fail.

---

## H200 config

The only shipped H200 product config is `configs/SM90_H200_CLUSTER132/`.
It is the full-chip packing for published H200 numbers. Development H200
directories from earlier work are not in the tree.

CLUSTER132 carries Hopper opt-ins that stay off the SM120 defaults, including
`-ptx_int64_add_lowering_factor 2` (the G3 cubin lowers 64-bit address adds
to paired integer ops). The default int64 lowering factor elsewhere is 1.

IPOLY non-power-of-two channel mapping uses mode **3** (bijective rotation)
only on CLUSTER132. `SM90_H100` and `SM120_RTX5090` stay at mode **2**, the
same value as `flash`.

---

## Open items

| Item | Status |
|------|--------|
| TMA multicast contention / bandwidth | Not modeled. Timing is not silicon-quality |
| GEMM unicast and multicast CUDA-event 10% gate | Open. Small multicast about −20%, K2K multicast about +28% versus H200. `sq_2k` was not run |
| `cp.async` and TMA unicast HBM bandwidth versus H200 | Last measured about +12% (outside 10%). Normal HBM load is about 0% (inside 10%) |
| `red` / `red.async` | Unimplemented |
| Preferred cluster scheduling and full occupancy APIs | Stubs |
| Cross-GPC DSM or multicast | Out of scope. A hardware TB-cluster stays in one GPC |
| Blackwell DSM preset | Follow-up |
| Delay-line hop matrix | Removed. The fabric is the only SM-to-SM DSM path |

Job 2119329 is a partial hardware pass. Stamp config numbers as measured,
patent, inferred, or unresolved. Do not republish H200 figures as Blackwell
hardware fact. Detail is in [`evidence.md`](evidence.md) and
[`calibration.md`](calibration.md).

---

## Tests

Functional cluster, DSM, and TMA cases run on the reduced SM120 presets
(fabric on). Default CI uses `SM120_RTX5090`, which has one core per
cluster, so those cases skip there. `SM90_H200_CLUSTER132` is optional and
is not a default CI config.

```bash
source ./setup_environment

./tests/run_tests.py -c SM120_RTX5090_REDUCED_CLUSTER4x4 run --arch sm120 --group integration \
  --gtest-filter '*ClusterLaunch*:*ClusterBasic*:*ClusterReal*:*TMACluster*:*TmaMulticast*:*DsmTest*:*MbarrierCluster*'

./tests/run_tests.py -c SM120_RTX5090_REDUCED_CLUSTER2x2 run --arch sm120 --group integration \
  --gtest-filter '*MultiCluster*'
```

Suite list and skip macros: [`tests.md`](tests.md).

On commit `6e6e10ab` (2026-09-27):

| Config | Result |
|--------|--------|
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | 66 tests, 62 passed, 4 skips |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | 5 passed |

The four skips are expected on that preset:
`ClusterLaunchApiTest.ExLaunch_ClusterLargerThanPhysical_Fails`,
`ClusterLaunchApiTest.ExLaunch_HeteroGpc_ClusterDim2_ManyClustersSync`,
`DsmTest.ScFenceWaitsForRemoteStore`, and
`MbarrierClusterTest.DefaultTryWaitExpiresAndWakesEarly`.
`DsmTest.CrossGpcReject` refuses a cross-GPC `mapa` and passes.
`gpu_sim_cycle` for every kernel in this filter matches the run of the same
filter from before the functional-simulation fix in `6e6e10ab`.

Default regression CI is `tests/ci/run_ci_tests.sh`. Jobs: `sm120-core`,
`sm90-core`, `sm90-fa2`, `sm90-fa3`. A local run of all four on `6e6e10ab`
passed, including Hopper FA3 smoke, the fixed-forward case, backward smoke,
and PackGQA. That FA3 repair is in `src/cuda-sim/cuda-sim.cc` (branch targets
when the same function is assembled twice, and the named-barrier id copied
onto the issuing warp). It is not a DSM timing change.

---

## Effect on existing `flash` configs

`SM120_RTX5090` and `SM90_H100` `gpgpusim.config` files match `origin/flash`
(`d20ce45e`). DSM defaults off. The hang watchdog defaults off. TMA
completion is parked on the SRAM service only for cluster or DSM traffic.
IPOLY mode 3 is inert while the knob is 2.

Cycle comparison versus that `flash`, same kernels, measured 2026-09-24
(before `6e6e10ab`). `gpu_sim_cycle`:

| Kernel | This branch | `flash` | Delta |
|--------|-------------|---------|-------|
| SM120 VectorAdd (2) | 9941 / 9781 | 9966 / 9813 | about −0.3% |
| SM90 FA2 smoke (5) | 13630 / 12478 / 13897 / 22414 / 17203 | 13482 / 12415 / 13796 / 22073 / 16887 | +0.5% to +1.9% |

The FA2 spread is the H100 iSLIP interconnect policy
(`-icnt_use_voq 1` on `SM90_H100`; SM120 leaves VOQ off). It sits inside a
±2% band. The functional CI re-run on 2026-09-27 passed the same old-config
suites; it did not repeat this cycle table. An earlier 14–28% gap was
against a `flash` that still truncated `perf_memcpy_to_gpu` addresses.
Current `flash` already has the 64-bit copy and turns perf-memcpy off on
`SM120_RTX5090`, and that gap is gone.
