# Real `cudaLaunchKernelEx` / Thread Block Cluster launch

**Status:** Implemented on branch `cluster_cta2_support` (part of the **unified cluster PR** — launch + TMA + DSM + remote mbarrier + NoC).  
**Branch overview:** [`docs/cluster.md`](cluster.md)  
**NoC / DSM / maturity:** [`docs/cluster_noc.md`](cluster_noc.md)

---

## 1. What is implemented

| Piece | Status |
|-------|--------|
| `cudaLaunchKernelExC` + `cudaLaunchAttributeClusterDimension` | Yes |
| `cudaFuncSetAttribute` RequiredCluster* / NonPortable / SchedPolicy / MustBeSet | Yes |
| PTX `.explicitcluster` + `.reqnctapercluster` (`__cluster_dims__`) | Parsed → required cluster on `function_info` |
| `kernel_info_t` cluster metadata + open TB-cluster issue state | Yes |
| Co-resident CTA issue (whole TB cluster on one physical cluster) | Yes |
| Ordinary `<<<>>>` / `cudaLaunchKernel` without cluster dims | Unchanged (global RR) |
| Cluster special regs (`%cluster_ctarank`, `%clusterid`, …) | Yes |
| Peer TMA / DSM / remote mbar | Via same physical cluster + `cluster_group` / ranks; timing via NoC when enabled |

---

## 2. Config naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster` |
| **n** | Number of physical clusters | `-gpgpu_n_clusters` |

| Config directory | Topology | Notes |
|------------------|----------|-------|
| `SM120_RTX5090_CLUSTER16x11` | m=16, n=11 | GPC-aligned full (ideal 176 SMs) |
| `SM120_RTX5090_CLUSTER2x85` | m=2, n=85 | TPC packing; TB size ≤ 2 |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | m=4, n=4 | Primary m>2 multi-cluster functional |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | m=2, n=1 | Fast m=2 peer smoke |
| `SM120_RTX5090_REDUCED_CLUSTER2x2` | m=2, n=2 | Multi-cluster isolation (m=2) |
| `SM90_H200_REDUCED_CLUSTER4x4` | m=4, n=4 | NoC **on** + H200 hop knobs |
| `SM90_H200` | m=1, n=132 | Product latencies; NoC idle |

**TB cluster size is not in the config name** — it comes from the launch. Capacity: `product(clusterDim) ≤ m`.

---

## 3. Launch contract

### Sources of cluster dim (priority)

1. `cudaLaunchAttributeClusterDimension` on `cudaLaunchKernelExC`
2. Else required dims on the function (PTX / `cudaFuncSetAttribute`)
3. Else non-cluster launch

### Validation

- All cluster dims ≥ 1  
- Grid dims multiple of cluster dims  
- `product(clusterDim) ≤ n_cores_per_cluster`  
- `ClusterDimMustBeSet` without a dim → error  

### Scheduler

| Launch type | Placement |
|-------------|-----------|
| Cluster launch | All CTAs of a TB cluster on **one** physical `simt_core_cluster`, shared `cluster_group` + ranks |
| Ordinary launch | Global RR; **each CTA its own `cluster_group`** (no false peers) |

---

## 4. Why Ex launch matters (OneProducer)

On multi-cluster configs (e.g. `REDUCED_CLUSTER2x2`), ordinary `<<<2,32>>>` RR can place two CTAs on **different** physical clusters → peer TMA/complete never runs → consumer hang.

With `cudaLaunchKernelEx(clusterDim=2)` both CTAs co-reside → peer path works.

---

## 5. Tests

| Suite | Role |
|-------|------|
| `cluster_launch_api_test.cc` | Ex launch success/failure, ordinary `<<<>>>`, required attrs |
| `TMAClusterOneProducerTest` | Ex launch + cluster TMA one-producer |
| `MultiCluster*` | Ordinary `<<<>>>` isolation on multi-cluster configs |
| `DsmTest` / `MbarrierClusterTest` | DSM + remote mbar (need m≥2 + NoC overlay or H200 reduced). `mapa` of an inactive rank aborts (`MapaAfterProducerExit_FailsLoud`). |

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*:*MultiCluster*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*"
```

### Topology skips (`SKIP_IF_*`)

Helpers: **`test/common/gpgpusim_config_topology.h`**.

| Macro | Skip when | Purpose |
|-------|-----------|---------|
| `SKIP_IF_N_CORES_PER_CLUSTER_LT(2)` | **m** < 2 | Peer / Ex paths needing ≥2 SMs per physical cluster |
| `SKIP_IF_N_CLUSTERS_LT(2)` | **n** < 2 | Multi-cluster isolation |
| Negative capacity tests | depend on **m** | e.g. clusterDim larger than physical m |

| Config | Peer/Ex (m≥2) | MultiCluster (n≥2) | Notes |
|--------|---------------|--------------------|--------|
| Plain `REDUCED` (m=1) | Skip peers | Skip isolation | Use for m=1 negative capacity if needed |
| `REDUCED_CLUSTER2x1` | Run | Skip n | One skip often: capacity-negative when m≥2 |
| `REDUCED_CLUSTER2x2` / `4x4` | Run | Run | Primary functional |
| `CLUSTER16x11` | Run | Run | Heavy smoke |

Do not treat topology skips as product failures.

---

## 6. Key code locations

| Area | Location |
|------|----------|
| `cudaLaunchKernelExC` | `libcuda/cuda_runtime_api.cc` |
| `kernel_config` cluster fields | `libcuda/cuda_api_object.h` |
| `kernel_info_t` cluster fields | `src/kernel_info.h` |
| Co-resident issue | `gpgpu_sim::issue_block2core` in `gpu-sim.cc` |
| Group / rank allocation | `simt_core_cluster` / `shader.cc` |
| Peer TMA path | `src/gpgpu-sim/flash/tma.cc` |
| NoC | `src/gpgpu-sim/flash/cluster_noc.{h,cc}` |
| Host helper | `test/common/cluster_launch.h` |
| Topology skip helpers | `test/common/gpgpusim_config_topology.h` |

---

## 7. Non-goals (launch surface)

See [`docs/cluster.md`](cluster.md) §4. Launch-specific: preferred-substitute cluster dims, high-fidelity occupancy APIs, and unrelated CUDA launch attributes remain stubs / out of scope.
