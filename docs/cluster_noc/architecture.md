# GPC, topology, and two networks

This file is the ownership and naming spec. Packet/flit rules are in [`dsm_fabric.md`](dsm_fabric.md). Cycle-accurate load completion is in [`pipeline.md`](pipeline.md).

---

## 1. Two identities that must not be mixed

| Name | What it is | What it is not |
|------|------------|----------------|
| **GPC** (`gpc_t`) | Physical container: SMs, DSM fabric, OpenMP ownership | A CUDA Thread Block Cluster |
| **TB-cluster** | Programming-model group of CTAs that may use DSM / TMA cluster / remote mbarrier | A physical GPC, TPC, or interconnect node |

PTX `.shared::cluster` stays `.shared::cluster`. Do not rename it `.shared::gpc`.

Hardware structure follows patent US12248788B2 Figs. 21A–21D. Those figures still show TPCARB; **this sim does not model TPC or TPCARB**. SMs connect to GPCMMU / GPCARB.

---

## 2. Physical hierarchy (target)

```text
gpgpu_sim
├── gpu_topology_t
├── global interconnect          // one shader endpoint PER SM
└── gpc_t[gpc_id]                // one host thread; OpenMP across GPCs only
    ├── shader_core_ctx[enabled local_sm]
    │     LSU, TMA, scoreboard, DSM inject/complete, shared_memory_service_t
    ├── cpc_t[cpc_id]            // always 6 SM slots
    │     GPCMMU (hash, not a page table)
    │     GPCARB ingress 6 → 4
    ├── gx_plane_t[0 .. num_gx-1]  // default 2; NOT request/response
    ├── GPCARB egress 4 → 6
    ├── dsm_fabric_t
    ├── dsm_endpoint_protocol_t[local_sm]
    └── tb_cluster_resolver_t
```

### CPC and PG'd SMs

- A **CPC** is **six SM slots**, one GPCMMU, one GPCARB ingress.
- `-gpgpu_dsm_cpcs_per_gpc` (default **3**) → 18 slots on a full GPC.
- `-gpgpu_num_sms_per_gpc` is the **enabled** SM count (CUDA-visible SMs).
- Remaining slots are **PG'd / floorswept**: they occupy a slot index in `gpu_topology_t`, they do not run a `shader_core_ctx`, they do not inject DSM flits, they do not consume traffic-control eligibility.

Example: reduced functional GPC with 4 enabled SMs and 1 CPC → 6 slots, 2 PG'd.

All SM ↔ GPC ↔ slot conversion goes through `gpu_topology_t`. **No** `sid / n_cores` division or modulo outside that class (floorsweeping is not a regular grid).

Shaper index is a **knob**: `sm_id` or `cpc_slot`. After floorsweeping they are not the same. Prefer documenting the map in `gpu_topology_t`; do not scatter `sm_id % n` for GPC lookup (that ban does not forbid the shaper knob).

---

## 3. Naming (target)

| Concept | Type / field | Scope |
|---------|--------------|--------|
| Global SM | `sm_id_t`, `sm_id` | GPU unique |
| Physical GPC | `gpc_id_t`, `gpc_id` | GPU unique |
| SM inside GPC | `local_sm_id_t` | Enabled SMs in that GPC |
| CPC slot | `cpc_id`, `cpc_slot` in `0..5` | Includes PG'd holes |
| Global NoC node | `global_icnt_node_id_t` | SM–L2 interconnect |
| DSM VC | `dsm_vc_t` | `request` or `response` |
| Physical lane | `dsm_lane_id_t` | CPC scheduler resource |
| GX plane | `gx_plane_id_t` | `0 .. num_gx-1` |
| TB-cluster group | `tb_cluster_group_id_t` | Launch / runtime |
| DSM transaction | `dsm_transaction_id_t` | Request through completion |
| DSM packet | `dsm_packet_id_t` | One unidirectional transmission |
| DSM endpoint | `dsm_endpoint_id_t` | Unique inside one GPC (plan-v2 §5.1) |
| DSM CPC | `dsm_cpc_t` / `cpc_id` | Six slots + GPCMMU + GPCARB |

Do not use `tpc`, `cluster_id`, `port_id`, or `sid` as an overloaded id.

### Rename table (B1)

| Old | New |
|-----|-----|
| `simt_core_cluster` | `gpc_t` |
| GPU `m_cluster` | `m_gpcs` |
| SM `m_cluster` | `m_gpc` |
| `m_cluster_id` | `m_gpc_id` |
| `n_simt_clusters` | `num_gpcs` |
| `n_simt_cores_per_cluster` | `num_sms_per_gpc` (enabled) |
| core `sid` / `m_sid` | `sm_id` / `m_sm_id` |
| `cid` | `local_sm_id` |
| `sid_to_cluster()` | `gpc_id_of_sm()` |
| `m_tpc` (core) | delete; obtain from `m_gpc` |
| `m_tpc` (`mem_fetch`) | delete; keep requester SM |
| `m_sid` (`mem_fetch`) | `m_requester_sm_id` |
| `m_response_fifo` | `m_global_response_fifos[local_sm_id]` |
| `m_cluster_cta_seq` | `m_next_tb_cluster_group_id` |

`shader_core_ctx` may keep its class name. New APIs use `sm_id()` and `gpc()` only.

Keep `-gpgpu_n_clusters` / `-gpgpu_n_cores_per_cluster` as **deprecated aliases** until **B-DEPR**. If old and new knobs disagree at start-up, **abort**.

---

## 4. Two networks, on purpose

| Network | Connects | Carries | Clock |
|---------|----------|---------|--------|
| Existing icnt (`local_interconnect` / BookSim) | **SM ↔ L2/DRAM** | Cache-line `mem_fetch` | ICNT domain |
| New `dsm_fabric_t` | **SM ↔ SM inside one GPC** | DSM / TMA-peer / remote mbarrier **packets and 32 B payload flits** | CORE domain |

DSM bytes never become a `mem_fetch`, never allocate L1/L2, never use an icnt port.

For the H200 calibration preset, architectural TMA completion and physical traffic completion are distinct events. Global-memory requests and multicast `tma_data` packets still execute and update shared memory, but the transaction's mbarrier is released on the measured completion curve. This models H200 overlap without serializing global-memory completion and peer delivery; it is disabled when the completion knobs are zero.

### Global NoC stays per SM (B2)

```text
num_global_shader_nodes = total enabled SMs

request source      = global_sm_node_id(requester_sm_id)
L2 destination      = global_l2_node_id(subpartition)
reply destination   = global_sm_node_id(requester_sm_id)
```

Putting several SMs in one `gpc_t` must **not** collapse shader icnt nodes to one per GPC. Today the delay-line era uses one icnt node per `simt_core_cluster`; that is a known bandwidth lie and B2 exists to remove it.

`gpc_id` is never a global NoC node.

---

## 5. `gpu_topology_t` (target)

```cpp
struct sm_location_t {
  sm_id_t sm_id;
  gpc_id_t gpc_id;
  local_sm_id_t local_sm_id;
  unsigned cpc_id;
  unsigned cpc_slot;   // 0..5; slot may be PG'd for unused indices
};

class gpu_topology_t {
 public:
  unsigned num_gpcs() const;
  unsigned num_sms() const;                 // enabled only
  unsigned num_sms_in_gpc(gpc_id_t) const;  // enabled in that GPC
  unsigned num_cpc_slots_in_gpc(gpc_id_t) const; // always 6 * cpcs_per_gpc

  sm_location_t locate_sm(sm_id_t) const;
  bool slot_is_enabled(gpc_id_t, unsigned cpc_id, unsigned cpc_slot) const;
  sm_id_t sm_id_at(gpc_id_t, local_sm_id_t) const;

  global_icnt_node_id_t global_sm_node_id(sm_id_t) const;
  global_icnt_node_id_t global_l2_node_id(unsigned subpartition_id) const;
};
```

---

## 6. Concurrency and cycle order

Silicon: CTAs of one TB-cluster run on different SMs in the same GPC at the same time.

This sim: **same GPU cycle**, one **host thread per GPC**. Flash OpenMP parallelizes **GPCs**, not SMs inside a GPC.

Invariant:

> Cross-SM shared state is mutated only on the owning GPC thread, in `deliver` / target shared-memory service / endpoint protocol — never by a foreign SM pipeline stage.

Target core-clock order inside one `gpc_t`:

```text
1. Drain completed response-VC packets to requester completion (scoreboard / RF)
2. Drain completed request-VC packets to target ingress
3. Arbitrate shared_memory_service_t (local LSU, TMA, DSM)
4. Execute granted target ops; update ACK / outstanding
5. Cycle SM pipelines; expose new DSM transactions
6. Inject ready request/response packets (subject to shaper + credits)
7. Advance dsm_fabric_t exactly once
8. Release completed `barrier.cluster` groups / CTA release (pending DSM blocks exit)
```

Rules:

- Same-cycle visibility is consistent for every SM in the GPC.
- SM walk order must not change packet arrival cycle.
- Fabric advances **once** per GPC per core tick.
- Request arrival, SRAM service, and response inject must not complete in **zero** cycles.
- `barrier.cluster.wait` releases only after all active warps in all CTAs of the reserved TB-cluster group have arrived; a partially issued cluster cannot release early.
- No nested OpenMP inside a GPC.

`gpgpu_sim::active()`, deadlock detect, and state dump must cover fabric queues, partial packets, target service, outstanding transactions, ACK debt, pending batches, and CTA async work.

---

## 7. TB-cluster resolver

All of DSM, remote mbarrier, TMA peer, and future tcgen peer lookup use one resolver:

```cpp
optional<tb_cluster_target_t> resolve_tb_cluster_rank(
    sm_id_t requester_sm_id,
    unsigned requester_cta_slot,
    unsigned target_rank);
```

It must check CTA lifetime, TB-cluster group, rank, **same GPC**, and the shared-memory object. Cross-GPC or dead rank: named error and **abort** (today’s `mapa` policy). Do not alias the issuer’s smem.

---

## 8. What the delay-line code does today

`simt_core_cluster` owns a `cluster_noc_t`. Messages wait `hop(src,dst)` (+ optional BPC) and commit on `deliver_ready()`. OpenMP is already per physical cluster. Global icnt is still **one node per cluster**.

That structure is the **migration starting point**, not the performance model. B1 renames the container; B2 splits icnt endpoints; B3 replaces `cluster_noc_t`.
