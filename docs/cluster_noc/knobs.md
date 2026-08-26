# Configuration knobs

Registered in `shader_core_config::reg_options` (`src/gpgpu-sim/gpu-sim.cc`) unless noted.

Two generations:

1. **Delay-line (in tree now).** Keep working until `dsm_fabric_t` is the only path.
2. **Target fabric.** Add in **B3**. After cut-over, **B-DEPR** removes delay-line knobs and updates every config file that sets them.

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
| `-gpgpu_gpc_slot_map` | empty = first N slots enabled | Which of the `6 * cpcs` slots are live; rest **PG'd** |

`product(clusterDim) ≤ num_sms_per_gpc` (enabled), not ≤ 18 slots.

---

## 2. Delay-line knobs (legacy — B-DEPR)

Master switch today: `-gpgpu_cluster_noc_enable` (default 0; **1** on `SM90_H200_REDUCED_CLUSTER16x2`).

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_dsm_local_latency` | 37 | H200 local DSM; today matrix diagonal, not local issue delay |
| `-gpgpu_dsm_remote_latency` | 78 | One-way hop if no matrix |
| `-gpgpu_dsm_latency_matrix_file` | `""` | N×N one-way hop CSV, cluster-local core ids |
| `-gpgpu_dsm_bytes_per_cycle` | 0 | Extra cycles `ceil(bytes/BPC)-1`; **0 = unlimited** |
| `-gpgpu_dsm_store_immediate` | 0 | `0` write peer smem on deliver; `1` also at issue. Env `FLASHGPU_DSM_STORE_IMMEDIATE` |
| `-gpgpu_tma_mcast_enable_timing` | 1 | When NoC on, TMA peers go through NoC |
| `-gpgpu_tma_mcast_hop_latency` | 0 | H200 product **135** |
| `-gpgpu_tma_mcast_use_dsm_matrix` | 0 | Usually off; TMA ≠ full DSM RTT |
| `-gpgpu_tma_mcast_bytes_per_cycle` | 0 | Unlimited |
| `-gpgpu_tma_mcast_mbar_after_data` | 1 | Data before peer `complete_tx` |
| `-gpgpu_mbarrier_remote_hop_latency` | 0 | 0 ⇒ DSM hop |
| `-gpgpu_mbarrier_cluster_enable` | 0 | Remote mbarrier addresses. **1** on H200 reduced |
| `-gpgpu_cluster_hang_watchdog` | 8192 | Abort bare spin / mixed bar+try_wait. `0` = off. Env `FLASHGPU_CLUSTER_HANG_WATCHDOG` |

Hop math (job **2046238**):

```text
one-way hop              ≈ 78
remote load issue stall  ≈ 2 × hop     // delay-line only
target e2e remote load   ≈ 37 + 156 ≈ 193
remote store visibility  ≈ 1 × hop
TMA mcast − unicast e2e  ≈ +135
stride ratio             ≈ 1.0         // flat, not a tree
```

### Matrix file (delay line)

Dense N×N integers, N = cores per cluster. Whitespace or comma; `#` comments. Index = **cluster-local** core id, not global `%smid`. Values are **one-way** hops. Diagonal 0.

```text
# configs/SM90_H200_REDUCED_CLUSTER16x2/dsm_latency_matrix_16.csv
# (same 16×16 one-way file as configs/SM90_H200/; diagonal 0, off-diag ~71–86)
```

---

## 3. Target fabric knobs (add in B3; not in gpu-sim.cc yet)

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_dsm_enable` | (replaces cluster_noc_enable) | Fabric on |
| `-gpgpu_dsm_flit_payload_bytes` | **32** | Payload bytes per grant. Alias **`-gpgpu_dsm_flit_bytes`** (plan-v2 name). Header unmodeled |
| `-gpgpu_dsm_lanes_per_cpc` | 4 | GPCARB outputs |
| `-gpgpu_dsm_gx_planes` | **2** | Parallel GX planes, not VCs |
| `-gpgpu_dsm_shaper` | `skip_mod` | Required: `skip_mod` \| `fixed_tdm` \| `hard_rate_cap`. Not a silicon dump; B6 may switch H200 preset |
| `-gpgpu_dsm_shaper_period` | 3 | For `skip_mod` / TDM |
| `-gpgpu_dsm_shaper_index` | `sm_id` | `sm_id` or `cpc_slot`. Skip/TDM phase uses this index |
| `-gpgpu_dsm_request_vc_flits` | TBD (fit B3) | Request VC buffer depth (flits) |
| `-gpgpu_dsm_response_vc_flits` | TBD | Response VC buffer depth |
| `-gpgpu_dsm_ejection_vc_flits` | TBD | Per-dest ejection depth |
| `-gpgpu_dsm_vc_arbiter` | `bounded_response_priority` | VC select |
| `-gpgpu_dsm_route_policy` | `deterministic_hash` | GPCMMU hash |
| `-gpgpu_dsm_route_seed` | 0 | Hash seed |
| `-gpgpu_dsm_base_latency_cycles` | TBD | Pipeline / serializer floor in addition to flit grants |
| `-gpgpu_dsm_max_outstanding_per_sm` | TBD | Endpoint tx window |
| `-gpgpu_dsm_ack_coalesce_threshold` | TBD | Completions per `write_ack` |
| `-gpgpu_dsm_ack_timeout_cycles` | TBD | Flush ACK debt |
| `-gpgpu_dsm_tma_mcast_expand` | `source_unicast` | v1 source expansion; later `fabric_replicate` |

H200 **preset** may set shaper period 3, 6/4 CPC, 2 GX, 32 B payload. Generic / Blackwell presets must **not** inherit those values silently.

Shared-memory aggregate bandwidth stays on the **SRAM service** (`gpgpu_shmem_bytes_per_cycle` or successor), not a DSM network knob.

Hang watchdog **stays** after B-DEPR (not a delay-line hop knob). `tma_mcast_mbar_after_data` stays as an ordering policy.

---

## 4. H200 mapping (job 2046238)

Source: `../H200_profiling/output-2046238-H200Profiling.txt`. Blog / `dsm_bw` pin: [`evidence.md`](evidence.md).

| Profile metric | Delay-line knob | Fabric intent |
|----------------|-----------------|---------------|
| DSM local ~37.05 | `gpgpu_dsm_local_latency=37` | Local SMEM / self-mapa latency |
| Remote e2e ~193.41 | local + 2×hop | Fabric RTT + SRAM, **not** a baked issue stall |
| One-way ~78 | matrix / remote_latency=78 | `base_latency` + serialization; do not keep a magic 78 after B-DEPR unless re-fit as base |
| Stride ~1.001 | flat matrix | Hash should not invent multi-hop by rank |
| TMA mcast−unicast ~135 | `tma_mcast_hop_latency=135` | Re-fit as fabric + SRAM, not a second delay line |
| ~21 B/cycle / SM | BPC unused (0) | Shaper 2/3 × 32 B payload |
| SM120 product | NoC **off** | Keep functional-immediate until a SM120 fabric preset exists |

Policy: only `configs/SM90_H200*` carry calibrated DSM timing today.

---

## 5. TMA multicast behavior (delay line)

| NoC enable | `tma_mcast_enable_timing` | Peers |
|------------|---------------------------|-------|
| 0 | * | Immediate copy + try_complete |
| 1 | 0 | Immediate |
| 1 | 1 | Data + mbar after hop |

Fabric path (B8) must keep **data before mbar**.
