# Cluster programming model (launch, DSM, TMA, mbarrier)

How kernels are supposed to use this sim. Fabric internals: [`dsm_fabric.md`](dsm_fabric.md). Pipeline: [`pipeline.md`](pipeline.md).

---

## 1. Launch so peers actually sit together

Implemented in `libcuda/cuda_runtime_api.cc`. Test helper: `test/common/cluster_launch.h`.

| API | Status |
|-----|--------|
| `cudaLaunchKernelExC` + `cudaLaunchConfig_t` | Yes |
| `cudaLaunchAttributeClusterDimension` | Yes — this is what packs the TB-cluster |
| `cudaLaunchAttributeClusterSchedulingPolicyPreference` | Stored; **no** spread/policy model |
| `cudaLaunchAttributePreferredClusterDimension` | **Ignored** (warning) |
| `cudaFuncSetAttribute` RequiredCluster* / NonPortable / MustBeSet / sched policy | Yes (validate + metadata) |
| Ordinary `cudaLaunchKernel` / `<<<>>>` | Unchanged; **does not** force co-residency |

Cluster-dim priority: Ex attribute → else required dims on the function (PTX / `cudaFuncSetAttribute`) → else non-cluster.

Validation: dims ≥ 1; grid multiple of cluster dims; `product(clusterDim) ≤` enabled SMs per GPC; “must be set” without a dim is an error.

| Launch type | Placement |
|-------------|-----------|
| Cluster launch | All CTAs of one TB-cluster on **one** GPC / physical cluster, shared `cluster_group` + ranks |
| Ordinary launch | Global round-robin; **each CTA its own group** (no false peers) |

On a multi-cluster config, `<<<2,32>>>` can put two CTAs on **different** GPCs. Peer TMA/DSM then never happens and `try_wait` hangs. Tests use `flash_test::launch_kernel_with_cluster`.

### Special registers

| PTX / builtin | Meaning |
|---------------|---------|
| `.explicitcluster` / `.reqnctapercluster` / `__cluster_dims__` | Required cluster size |
| `%is_explicit_cluster` | 1 if this kernel is a cluster launch |
| `%cluster_ctarank` / `%cluster_nctarank` | Rank and cluster size |
| `%cluster_ctaid.{x,y,z}` / `%cluster_nctaid.{x,y,z}` | CTA id inside the TB-cluster |
| `%clusterid.{x,y,z}` / `%nclusterid.{x,y,z}` | Which TB-cluster in the grid |

Code: `ptx_thread_info` in `src/cuda-sim/ptx_sim.cc`.

---

## 2. DSM (`mapa` + remote ld/st/atom)

| PTX | Status |
|-----|--------|
| `mapa.u64` / `mapa.shared::cluster` | **Yes.** Preferred |
| `mapa.u32` | Parses; **truncates** large windows. Do not use |
| `ld` / `st` of a `mapa` generic pointer | Yes (`decode_space`) |
| Explicit `ld/st/atom.shared::{cta,cluster}` scope | Yes; both scope suffixes normalize to shared space before local/peer decode |
| `atom.add` (CUDA `atomicAdd` on a `mapa` pointer) | Yes — RMW on owner smem |
| `red` / `red.async` / `red.shared::cluster` | **No** (`inst_not_implemented`) |

`mapa` (`mapa_impl`):

1. Parse shared offset from `a`.
2. Find an **active** CTA with rank `b` in the same `cluster_group`.
3. Write `d = shared_to_generic(peer_sm_id, offset)`.
4. If no such CTA: named error and **abort**. Do not alias the issuer’s window.

Keep the producer CTA allocated until every consumer that will `mapa` that rank has done so.

The `.shared::cta` and `.shared::cluster` scope suffixes are not interchangeable with global generic addressing. The simulator explicitly normalizes both before address decode; this is required by the upstream mixed DSM/TMA checksum path.

Generic window:

```text
shared_generic(sm_id, off) = SHARED_GENERIC_START + sm_id * SHARED_MEM_SIZE_MAX + off
```

Supported pattern:

```text
rank 1:  init local bar; publish ready; try_wait; read local smem
rank 0:  wait ready (global handshake is OK *before* the peer op);
         ptr = mapa.u64(local_smem, /*rank=*/1);
         *ptr = payload;          // remote st
         arrive on mapa’d bar;
         stay alive until consumers finished mapa / the read
```

**Do not** spin `while (*peer_flag == 0)`. Functional issue order can execute the load before the producer issues; silicon often survives that, this sim often hangs.

---

## 3. TMA into peer shared memory

Typical one-producer (`TMAClusterOneProducerTest`):

1. Each rank inits a **local** mbarrier (`expect_tx` = bytes it will receive).
2. Producer issues `cp.async.bulk` / tensor copy with `.shared::cluster` and optional `ctaMask`.
3. **NoC off:** peers filled immediately; `complete_tx` immediate.
4. **NoC on (delay line):** issuer smem is the staging buffer; after TMA arrive, snapshot + per-peer data then mbar. Consumers `try_wait` on **local** bars, then read **local** smem.

Mask bit *i* is TB-cluster **rank** *i*. Issuer gets a local copy only if its own bit is set.

| PTX | Status |
|-----|--------|
| `cp.async.bulk.shared::{cta,cluster}.global.mbarrier::complete_tx` | Yes |
| `cp.async.bulk.tensor.{1-5}d` load + mbarrier complete | Yes |
| `.multicast::cluster` + `ctaMask` | Yes |
| Bare `.shared::cluster` (no mask) | All peers in `cluster_group` |
| `commit_group` / `wait_group` | Yes |
| Used `tensormap.replace.tile.*` + `cp_fenceproxy` | Yes; unknown variants abort |
| Swizzle none / 32B / 64B / 128B | Yes |
| Swizzle 96B | Named abort |

Target fabric: same programming model; packets become `tma_data` + `mbarrier_completion` on `dsm_fabric_t` (**B8**).

---

## 4. mbarrier

| PTX | Status |
|-----|--------|
| `mbarrier.init.shared::cta.b64` | Yes |
| `mbarrier.arrive` (+ optional count) | Yes; remote if mapa’d |
| `arrive.expect_tx` / `expect_tx` / `complete_tx` | Yes; remote via NoC when enabled |
| `try_wait.parity` (3-operand) | Yes — parks until phase; dest pred success |
| `try_wait.parity` + timeout | Yes — pred true if phase done, false if hint expires (**sim cycles**) |
| `mbarrier.inval` | Yes |
| `test_wait` / `pending_count` / try_wait without `.parity` | Not parsed / hard-fail |
| `barrier.cluster.arrive` / `barrier.cluster.wait` (`cluster.sync()`) | Yes; all active warps in every CTA of the launched TB cluster participate |

Objects live in **simulator tables**, not as 64-bit smem contents. Kernels that read barrier **bytes** will not match hardware.

Remote: `remote_mbarrier_*` in `mbarrier.cc`. Delay line: `inject_mbar_remote`. Target: mbarrier request/completion packets on the fabric.

---

## 5. Hang preventers (sim only)

Not hardware detectors. Default 8192 cycles (`-gpgpu_cluster_hang_watchdog`; env `FLASHGPU_CLUSTER_HANG_WATCHDOG`). `0` = off. Warps stalled on an outstanding DSM scoreboard dependency or a cluster barrier are recognized waits and do not trip the watchdog.

**Rule 1 — bare peer spin.** After a **recent** peer DSM/TMA access, a tight PC loop with no mbarrier interest aborts. The arm expires on a recognized wait (`try_wait` park, `bar.sync`, …) or a hop-scale quiet window with no further peer touch. Parked `try_wait` is exempt.

**Rule 2 — mixed `bar.sync` + single-thread `try_wait`.** Partial-warp `try_wait` parked next to a `bar.sync` waiter in the same CTA for longer than hop-scale latencies aborts.

Code: `cluster_hang_prevent.h`, `shader.cc` `poll_hang_preventers`. After the fabric lands, quiet windows must follow **real** RTT (shaper/credits), not only `2×hop`.

---

## 6. Kernel rules (tests)

1. Coordinate cross-rank visibility with **mbarrier**, not a load-loop on peer smem.
2. Never mix `__syncthreads` / `bar.sync` with single-thread `try_wait` in the same CTA.
3. Producer CTA stays alive until consumers finish `mapa`.
4. Prefer `mapa.u64`.
5. Consumers `try_wait` before reading data the producer stored or TMAd.

---

## 7. Gaps vs silicon (honest)

| Area | Silicon | This sim (today) |
|------|---------|------------------|
| TB-cluster home | One GPC | One `simt_core_cluster` / future `gpc_t` |
| SM concurrency | Parallel SMs | Same **cycle**; one host thread per GPC |
| Busy-wait on peer smem | Usually works if producer is alive | May hang or miss; use mbarrier |
| Same-cycle atom on one word | HW arbiter | Serialized on the GPC thread |
| DSM load completion | Hop, remote SRAM, hop, RF | RF at execute + `2×hop` stall (**B5** fixes) |
| DSM bandwidth | ~21 B/cycle shaped | Default unlimited BPC (**B3b** fixes) |
| mbarrier object | 64-bit in smem | Simulator table |
| `try_wait` timeout unit | ns-scale hint | Sim cycles |
