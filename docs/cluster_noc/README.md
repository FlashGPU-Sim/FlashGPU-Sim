# Thread Block Cluster and intra-GPC DSM

This branch adds cluster launch, DSM (`mapa` plus remote shared `ld` / `st` / `atom.add`), remote `mbarrier`, and `barrier.cluster`. They stay off unless `-gpgpu_dsm_enable 1`. The only shipped preset that turns them on is `configs/SM90_H200_CLUSTER132`. `SM90_H100`, `SM100_B200`, and `SM120_RTX5090` match `flash`.

## What is modeled

| Feature | Behavior |
|---------|----------|
| Cluster launch (`cudaLaunchKernelExC`, required cluster dims) | Every CTA of one thread-block cluster is co-resident on one GPC |
| `mapa` plus remote `ld` / `st` / `atom.add` | Intra-GPC fabric, `dsm_fabric_t`. A rank that is not resident aborts |
| Remote `mbarrier` | Fabric request and completion. Needs DSM and `-gpgpu_mbarrier_cluster_enable`. `try_wait` on a `mapa` address is a uniform simulator extension; PTX `try_wait` is on the owner, and remote arrive plus local wait is the standard path |
| `barrier.cluster.arrive` / `wait` | Releases after every active warp in every CTA of the cluster has arrived |
| TMA `.shared::cluster` and `ctaMask` | Functional copy to the selected peers, then `-gpgpu_tma_multicast_latency`. No fabric traffic |

Not implemented: `red` / `red.async`, preferred cluster dimensions, and the cluster scheduling policy. DSM and TMA multicast stay inside one GPC. TPC / TPCARB are not in the model. Power-gated SM slots are. Global TMA multicast has no bandwidth, route, queue, or contention model.

`product(clusterDim)` must be at most the number of enabled SMs on the GPC that receives the cluster. On `SM90_H200_CLUSTER132` the smallest GPC has 16 SMs.

A bare spin on peer shared memory aborts (rule 1). Use `mbarrier`. Mixing `bar.sync` / `__syncthreads` with a single-thread `mbarrier.try_wait` in the same CTA also aborts (rule 2). `-gpgpu_cluster_hang_watchdog` applies only when DSM or remote mbarrier is on. `0` disables it. `FLASHGPU_CLUSTER_HANG_WATCHDOG` overrides the knob.

## Code

| Area | Location |
|------|----------|
| Launch and cluster attributes | `libcuda/cuda_runtime_api.cc`, `src/kernel_info.h` |
| Co-resident issue | `gpgpu_sim::issue_block2core`, `simt_core_cluster::issue_block2core` |
| `%cluster_*`, `%is_explicit_cluster` | `src/cuda-sim/ptx_sim.cc` |
| `mapa` and generic shared decode | `src/cuda-sim/instructions.cc`, `src/gpgpu-sim/flash/tb_cluster.h` |
| TMA multicast | `src/gpgpu-sim/flash/tma.cc` |
| `mbarrier`, including remote | `src/gpgpu-sim/flash/mbarrier.cc` |
| DSM fabric and endpoint | `src/gpgpu-sim/dsm_fabric.{h,cc}`, `dsm_endpoint.{h,cc}` |
| Hang watchdog | `src/gpgpu-sim/flash/cluster_hang_prevent.h` |
| Tests | `tests/src/integration/cluster_*.cu`, `dsm_test.cu`, `mbarrier_cluster_test.cu`, `tma_cluster_multicast_test.cu`, `tma_multicast_mask_test.cu` |

The memory crossbar (`local_interconnect`) is a separate network.

## Knobs

Registered in `shader_core_config::reg_options` (`src/gpgpu-sim/gpu-sim.cc`) unless noted. The H200 column is what `configs/SM90_H200_CLUSTER132/gpgpusim.config` sets. Other configs keep the default.

| Knob | Default | H200_CLUSTER132 | Meaning |
|------|---------|------|---------|
| `-gpgpu_dsm_enable` | 0 | 1 | Remote shared `ld` / `st` / `atom` and remote mbarrier use the fabric. `0` aborts those ops |
| `-gpgpu_mbarrier_cluster_enable` | 0 | 1 | Allow `mbarrier` on a `mapa` address |
| `-gpgpu_tma_multicast_latency` | 0 | 100 | Cycles added before a multicast `complete_tx` |
| `-gpgpu_cluster_hang_watchdog` | 8192 | 8192 | Cycles before rule 1 or rule 2 aborts |
| `-gpgpu_n_clusters` | — | 8 | Number of GPCs. Alias `-gpgpu_num_gpcs`. Both set and disagreeing aborts at startup |
| `-gpgpu_n_cores_per_cluster` | — | 18 | Max enabled SMs per GPC. Alias `-gpgpu_num_sms_per_gpc` |
| `-gpgpu_gpc_sms` | empty | `16,16,16,16,16,16,18,18` | Enabled SMs in each GPC. Empty means every GPC uses the scalar count |
| `-gpgpu_dsm_cpcs_per_gpc` | 3 | 3 | CPCs per GPC. Each CPC has 6 SM slots; extra slots are power-gated |
| `-gpgpu_dsm_flit_payload_bytes` | 32 | 32 | Payload bytes per fabric grant. Alias `-gpgpu_dsm_flit_bytes` |
| `-gpgpu_dsm_lanes_per_cpc` | 4 | 4 | GPCARB outputs per CPC |
| `-gpgpu_dsm_gx_planes` | 2 | 2 | GX planes. Routes = planes × lanes per CPC |
| `-gpgpu_dsm_shaper` | `skip_mod` | `skip_mod` | `skip_mod`, `fixed_tdm`, or `hard_rate_cap` |
| `-gpgpu_dsm_shaper_period` | 3 | 3 | `skip_mod` / TDM period |
| `-gpgpu_dsm_shaper_index` | `sm_id` | `sm_id` | `sm_id` or `cpc_slot` |
| `-gpgpu_dsm_request_vc_flits` | 64 | 512 | Request VC depth, in flits |
| `-gpgpu_dsm_response_vc_flits` | 64 | 512 | Response VC depth |
| `-gpgpu_dsm_ejection_vc_flits` | 64 | 512 | Per-destination ejection depth |
| `-gpgpu_dsm_max_outstanding_per_sm` | 16 | 1024 | Endpoint transaction window |
| `-gpgpu_dsm_ack_coalesce_threshold` | 4 | 64 | Store completions per `write_ack` |
| `-gpgpu_dsm_ack_timeout_cycles` | 64 | 64 | Flush remaining ACK debt |
| `-gpgpu_dsm_base_latency_cycles` | 0 | 78 | Pipeline floor on top of flit grants |
| `-gpgpu_dsm_store_visibility_latency_cycles` | 0 | 245 | Extra floor for stores. `0` uses the base floor |
| `-gpgpu_shmem_bytes_per_cycle` | 0 | 0 | Per-SM shared-memory byte budget for local LSU and DSM. `0` is unlimited |

`-gpgpu_dsm_vc_arbiter` defaults to `bounded_response_priority`. `-gpgpu_dsm_route_policy` defaults to `deterministic_hash`, with `-gpgpu_dsm_route_seed` 0.

H200 memory interleave is `-gpgpu_memory_partition_indexing 2` and `-gpgpu_ipoly_non_power2_balanced 3` (94 channels, bijective). H100 stays on mode 2. Registered in `src/gpgpu-sim/addrdec.cc`.

TMA completion floors (`-gpgpu_tma_load_completion_*` and `-gpgpu_tma_cluster_load_completion_*`) are set in the H200 config. They delay `complete_tx` after the memory transaction. Cluster multicast still does not enter the DSM fabric.
