# Configuration knobs

Registered in `shader_core_config::reg_options` (`src/gpgpu-sim/gpu-sim.cc`) unless noted.

## Non-power-of-two memory partition indexing

Registered in `src/gpgpu-sim/addrdec.cc`. For IPOLY indexing with a
non-power-of-two channel count, `-gpgpu_ipoly_non_power2_balanced` keeps
distinct algorithms:

- **0** flash legacy modulo
- **1** flash map IPOLY buckets to channels before subpartitions
- **2** flash hash into a larger virtual space then range-reduce (**H100**)
- **3** bijective IPOLY-derived cyclic rotation (**H200 CLUSTER132**, 94
  channels). Modes 0/1/2 are **not** aliases of 3.

`-gpgpu_ipoly_channel_stable_l2slice=1` retains its existing channel-stable
path and takes precedence. Power-of-two mapping is unchanged. No setting
here establishes NVIDIA's physical hash.

Run the standalone regression documented in `test/check_address_mapping.cc`.
See `calibration.md` for failing-before evidence, build status, and the separate
inherited row-capacity limitation. Recalibrate after rebuilding the decoder;
old timing results are not acceptance evidence for the repaired mapping.

## Bounded local mbarrier wait

`-gpgpu_mbarrier_trywait_default_timeout_ns` defaults to **0** (legacy
immediate polling, including unchanged H100). The H200 132SM candidate is
**4320 ns**, inferred from job 2119329's vendor WaitFalse slope of 7755
cycles/op after accounting for modeled release overhead. This is a bounded
suspension limit, not successful-wait latency. Completion wakes the warp
early; explicit PTX hints override the default and are also in ns. The default
applies only to uniform local barrier/parity waits; nonuniform no-hint waits
remain immediate queries because the current wakeup manager is warp-based.
Mapped remote waits retain their existing protocol. See `calibration.md` for
the expiry/early-wakeup regression and outstanding matched-probe/GEMM gates.

DSM SM↔SM timing is `dsm_fabric_t`. Do not reintroduce a hop CSV or scalar
remote-hop bandwidth model.

If old and new topology knobs are both set and disagree: **abort at start-up**.

---

## 1. Topology (today → target)

| Today | Target | Meaning |
|-------|--------|---------|
| `-gpgpu_n_clusters` | `-gpgpu_num_gpcs` | Number of GPCs |
| `-gpgpu_n_cores_per_cluster` | `-gpgpu_num_sms_per_gpc` | **Enabled** SMs per GPC |

Target extras:

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_dsm_cpcs_per_gpc` | 3 | CPCs in one GPC. Each CPC is 6 slots + GPCMMU + GPCARB |
| `-gpgpu_dsm_clients_per_cpc` | 6 | SM slots per CPC (do not use 2 for “TPC”) |
| `-gpgpu_gpc_slot_map` | empty = first N slots enabled | Documented alias; **implemented as** `-gpgpu_gpc_sms` |
| `-gpgpu_gpc_sms` | empty = uniform | Per-GPC enabled SM counts, comma-separated. H200 product packing **inferred**: `16,16,16,16,16,16,18,18` (132 SMs). If set, scalar `num_sms_per_gpc` must be omitted or equal to the max. |

`product(clusterDim) ≤ min(enabled SMs in any GPC)`, not ≤ 18 slots.

---

## 2. Cluster / mbarrier knobs

Master switch: `-gpgpu_dsm_enable` (default 0; **1** on SM120 reduced cluster
configs and `SM90_H200_CLUSTER132`). Remote DSM ld/st/atom and remote
mbarrier abort if this is 0.

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_mbarrier_cluster_enable` | 0 | Remote mbarrier addresses. **1** on SM120 reduced cluster configs and `SM90_H200_CLUSTER132` |
| `-gpgpu_cluster_hang_watchdog` | 8192 | Abort bare spin / mixed bar+try_wait. `0` = off. Env `FLASHGPU_CLUSTER_HANG_WATCHDOG`. Ignored unless DSM or remote mbarrier is on |

---

## 3. Fabric knobs

GX port formula: `routes = gx_planes * lanes_per_cpc`. GPCARB still grants at most `lanes_per_cpc` per CPC per cycle. Do not hard-code plane count `2` in C++.

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_dsm_enable` | 0 (1 on SM120 reduced cluster configs and `SM90_H200_CLUSTER132`) | Cluster ld/st/atom and remote mbarrier use the intra-GPC fabric. 0 = those remote ops abort. |
| `-gpgpu_shmem_bytes_per_cycle` | 0 (unlimited) | Per-SM SRAM service byte budget for local LSU and DSM ingress (two-phase grant). |
| `-gpgpu_tma_multicast_latency` | **0** | Fixed cycles added before multicast `complete_tx`; functional fan-out only, with no NoC/fabric traffic or contention. |
| `-gpgpu_dsm_flit_payload_bytes` | **32** | Payload bytes per grant. Alias **`-gpgpu_dsm_flit_bytes`**. Header unmodeled |
| `-gpgpu_dsm_lanes_per_cpc` | 4 | GPCARB outputs |
| `-gpgpu_dsm_gx_planes` | **2** | Parallel GX planes, not VCs |
| `-gpgpu_dsm_shaper` | `skip_mod` | Required: `skip_mod` \| `fixed_tdm` \| `hard_rate_cap`. Not a silicon dump |
| `-gpgpu_dsm_shaper_period` | 3 | For `skip_mod` / TDM |
| `-gpgpu_dsm_shaper_index` | `sm_id` | `sm_id` or `cpc_slot`. Skip/TDM phase uses this index |
| `-gpgpu_dsm_request_vc_flits` | **64** (H200 full-chip **512**) | Request VC buffer depth (flits) |
| `-gpgpu_dsm_response_vc_flits` | **64** (H200 full-chip **512**) | Response VC buffer depth |
| `-gpgpu_dsm_ejection_vc_flits` | **64** (H200 full-chip **512**) | Per-dest ejection depth |
| `-gpgpu_dsm_vc_arbiter` | `bounded_response_priority` | VC select |
| `-gpgpu_dsm_route_policy` | `deterministic_hash` | GPCMMU hash |
| `-gpgpu_dsm_route_seed` | 0 | Hash seed |
| `-gpgpu_dsm_base_latency_cycles` | **0** (H200 full-chip preset **78**, inferred) | Pipeline / serializer floor in addition to flit grants. Visible at dest at `max(tail_arrival, injected+floor)`. Does not add flit occupancy. |
| `-gpgpu_dsm_store_visibility_latency_cycles` | **0** (H200 full-chip **245**, inferred) | Store-only visibility floor; 0 inherits the generic fabric floor. |
| `-gpgpu_tma_load_completion_base_cycles` | **0** (H200 full-chip **1314**, fitted) | Non-cluster global-to-shared TMA completion floor from transaction creation; actual memory completion is also required. Zero disables the floor. |
| `-gpgpu_tma_load_completion_cycles_per_kib` | **0** (H200 full-chip **38**, fitted) | Non-cluster floor size term, cycles/KiB; rounded up to a whole cycle. |
| `-gpgpu_tma_cluster_load_completion_base_cycles` | **0** (H200 full-chip **860**, provisional fit) | Cluster global-to-shared TMA floor from transaction creation; also waits for memory completion. Excludes mapped shared-to-shared DSM copies. |
| `-gpgpu_tma_cluster_load_completion_cycles_per_kib` | **0** (H200 full-chip **12**, provisional fit) | Cluster floor size term, cycles/KiB. Together with the base replaces the probe-specific size table. Multicast adds its separate fixed latency; no DSM traffic. |
| `-gpgpu_ptx_register_allocator` | Generic default **1**; H200 calibration preset **0** | Optional virtual-register aliasing. Disabled for calibration because the looped TMA issue probe keeps its shared destination live across iterations. |
| `-gpgpu_dsm_max_outstanding_per_sm` | **16** (H200 full-chip **1024**) | Endpoint tx window (not a VC/link credit) |
| `-gpgpu_dsm_ack_coalesce_threshold` | **4** (H200 full-chip **64**) | Completions per `write_ack` |
| `-gpgpu_dsm_ack_timeout_cycles` | **64** | Flush remaining ACK debt |

H200 **preset** may set shaper period 3, 6/4 CPC, 2 GX, 32 B payload. Generic / Blackwell presets must **not** inherit those values silently.

Shared-memory aggregate bandwidth stays on the **SRAM service** (`gpgpu_shmem_bytes_per_cycle` or successor), not a DSM network knob.

Hang watchdog is independent of fabric hop timing.

---

## 4. Provisional H200 mapping

Cycle-accurate calibration (latency + slopes + GEMM), full-chip GPC packing, and sim tables: [`calibration.md`](calibration.md).

The old H200 Slurm source is intentionally not retained as evidence. Job
2119329 is the current, partially accepted source; its failed 4096^3 GEMM row
is excluded. The vendor `dsm_bw` pin remains documented in
[`evidence.md`](evidence.md); use [`calibration.md`](calibration.md) for the
acceptance details.

| Provisional target | Former hop model | Fabric intent |
|----------------|-----------------|---------------|
| DSM local ~37.05 | `gpgpu_dsm_local_latency=37` | Local SMEM / self-mapa latency |
| Remote e2e ~193.41 | local + 2×hop | Fabric RTT + SRAM, **not** a baked issue stall |
| One-way ~78 | former hop table | `base_latency` + serialization |
| Stride ~1.001 | flat matrix | Hash should not invent multi-hop by rank |
| TMA mcast−unicast robust median ~99 cycles | `gpgpu_tma_multicast_latency=100` | Fixed completion approximation; job 2119329 observed roughly 100–140 cycles normally |
| ~21 B/cycle / SM | BPC unused (0) | Shaper 2/3 × 32 B payload |
| SM120 product | NoC **off** | Keep functional-immediate until a SM120 fabric preset exists |

Policy: only `configs/SM90_H200_CLUSTER132` carries calibrated DSM timing today.

---

## 5. TMA multicast behavior

Global-memory TMA multicast is functionally copied to every selected peer without entering
`dsm_fabric_t`. Peer and selected-issuer `complete_tx` occurs
after the TMA transaction plus `-gpgpu_tma_multicast_latency`; the H200
full-chip preset uses 100 cycles from job 2119329. Zero adds no multicast
delay. No topology, bandwidth, routing, queue, SRAM-service, or contention
model is attached to global-memory multicast. Mapped shared-to-shared TMA
is different: it uses DSM transport and is the positive routing control.
