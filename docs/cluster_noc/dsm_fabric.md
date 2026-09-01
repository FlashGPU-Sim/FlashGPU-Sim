# Intra-GPC DSM fabric

Design authority for **how bytes move between SMs** in one GPC. Pipeline/scoreboard: [`pipeline.md`](pipeline.md). Knobs: [`knobs.md`](knobs.md). Evidence: [`evidence.md`](evidence.md).

---

## 1. Goal

Replace `cluster_noc_t` (delay line: `ready_cycle = now + hop + optional BPC`) with `dsm_fabric_t`:

- 32 B **payload** per flit grant
- Two virtual channels (**request**, **response**) with **independent queues and credits**
- **One** physical-lane scheduler (VCs are not extra bandwidth)
- Per-SM traffic control (default 2-of-3 cycles)
- GPCMMU as an **address hash**
- Configurable GX plane count (default 2)
- CPC 6 clients → 4 lanes, with PG'd slots
- Coalesced write ACK; extra reverse **read_command** for remote loads

GX0/GX1 are **parallel switch planes**, never `request_plane` / `response_plane`.

---

## 2. Flit and payload

plan-v2: *“32 B flit service, tail arrival then deliver”* so a 128 B read response occupies **four** service slots. Later agreement: that 32 B is **payload**. A real package may also carry a header; **v1 does not charge extra occupancy**.

Each grant moves **32 B of payload** (or a control packet that occupies **one** payload-flit slot). Alias: `-gpgpu_dsm_flit_bytes` = `-gpgpu_dsm_flit_payload_bytes`.

```text
payload_flits = ceil(payload_bytes / 32)     // data-bearing packets
control packets (read_command, write_ack, mbarrier op): 1 flit
```

| Knob | Default | Meaning |
|------|---------|---------|
| `-gpgpu_dsm_flit_payload_bytes` | 32 | Payload bytes charged per grant. Do not “fit” this away. |
| Header / metadata | unmodeled | May exist; encoding unresolved (v2 §3.3) |

Patent short-write-in-request vs long-write command+data is **unresolved**. v1 models stores as `write_data` payload flits on the request VC.

A saturated remote-load response of 128 B payload is **four** grants on the response VC, plus one **read_command** grant on the request VC in the opposite direction.

Do not teleport a whole packet when a byte budget fills. Advance `remaining_flits` one grant at a time. The packet is complete when the **tail** flit arrives.

---

## 3. Transaction vs packet vs VC

A **transaction** is the software/ISA operation (one remote load, store, atomic, TMA put, mbarrier op).

A **packet** is one unidirectional network message. One load transaction is at least `read_command` + `read_data`. Several stores may share one coalesced `write_ack`.

```text
enum class dsm_vc_t { request, response };

enum class dsm_packet_class_t {
  read_command,      // request VC, reverse of the data
  read_data,         // response VC
  write_data,        // request VC
  write_ack,         // response VC, coalesced
  atomic_request,    // request VC
  atomic_response,   // response VC
  tma_data,          // request VC (peer put)
  mbarrier_request,  // request VC
  mbarrier_completion
};
```

| Packet class | VC |
|--------------|----|
| read_command, write_data, atomic_request, tma_data, mbarrier_request | request |
| read_data, write_ack, atomic_response, mbarrier_completion | response |

`packet_class` and `vc` are **different fields**. Physical direction is `network_src_sm_id → network_dst_sm_id`. A read **response** is sent by the **target** SM. Never infer direction from the original requester.

Required packet fields (plan-v2 §8.1): `packet_id`, `transaction_id`, `network_src_sm_id`, `network_dst_sm_id`, `transaction_requester_sm_id`, `transaction_target_sm_id`, `vc`, `packet_class`, `payload_bytes` (v2 `wire_bytes` = payload in v1), `total_flits`, `remaining_flits`, `route_lane`, `created_cycle`, `injected_cycle`, `tail_arrival_cycle`.

Patent: request VC is **blocking**, response VC is **non-blocking**. That is queue/credit policy, not extra physical bandwidth.

Why this split (H200):

```text
SM0 remote-read SM1:
  READ_COMMAND   SM0 → SM1   request VC
  READ_DATA      SM1 → SM0   response VC

SM1 remote-store SM0:
  WRITE_DATA     SM1 → SM0   request VC
```

`READ_DATA` and `WRITE_DATA` are different VCs but the **same physical sender** (SM1). Measured same-direction mix collapses to ~21 B/cycle. Opposite-direction mix is higher. Therefore VCs **share** SM1’s send slots.

---

## 4. Physical path (no TPCARB)

Patent US12248788B2 Figs. 21A–21D still draw TPCARB and 2-SM TPCs. **Ignore those blocks in the model.** Path:

```text
source SM
  → GPCMMU / uTLB   (hash, not virtual-to-physical translation)
  → GPCARB ingress  (6 inputs, 4 outputs; 32 B payload per output)
  → GX planes       (default 2; each a 6×6; configurable)
  → GPCARB egress   (4 aggregate lanes → 6 slots, two 2-lane groups of 3)
  → destination SM shared_memory_service
```

Every **outgoing** packet (original request **or** a response/ACK generated at the target) goes SM → GPCMMU → GPCARB IG → GX. Incoming packets go GX → GPCARB EG → SM and **do not** walk GPCMMU again.

### GPCMMU = hash table

Not a page table. Given `(payload_address, src_sm, dst_sm, packet_uid, vc)` it returns:

```text
(gx_plane_id, gpcarb_output / lane)
```

Requirements:

- Deterministic for a fixed seed/config
- Sequential addresses spread across planes/lanes
- A knob can force **stride camping** for tests
- **Do not** claim it is NVIDIA’s hash

Default: every 32 B payload grant rehashes its current cache-line address across `num_gx * lanes_per_cpc` routes. A multi-flit TMA command is therefore interleaved rather than pinned to one lane.

### GPCARB

Ingress: six CPC slots (PG'd slots never request) compete for four 32 B-payload outputs. Egress: four lanes fan out to six slots in two groups of three (even/odd groups from the figure, **without** calling them TPCs).

### GX planes

`-gpgpu_dsm_gx_planes` default **2**. Firmware-style: `1` keeps connectivity at reduced bandwidth. Names `GX0` … `GXn`. VCs multiplex on the same planes.

Port wiring when `num_gx = 2` (from the figure, CPC `c`): IG occupies ports `{c, c+3}` on **each** plane (four lanes per CPC). If `num_gx` changes, document the port formula next to the knob; do not hard-code `2` in C++.

---

## 5. Traffic control (shaper)

**Silicon fact (measured):** one SM stays near 21 B/cycle even when neighbors are idle → allocation is **non-work-conserving**.

**Inference (not a schedule dump):** 4 lanes × 32 B / 6 clients = **2/3** of a 32 B payload lane per SM.

plan-v2 §9.4 gives a **verifiable** 4-of-6 TDM as a simulation of that rate, and also `hard_rate_cap`. It says this TDM is **not** a confirmed hardware timetable. Later agreement: also support a 2-of-3 skip with **different SMs skipping different cycles**.

All three policies are required. **None is “the Hopper schedule.”** Idle eligible slots are **wasted**. PG'd slots are never eligible.

| Policy | Eligibility |
|--------|-------------|
| `fixed_tdm` | CPC **slot** TDM (v2): phase 0 `{0,1,2,3}`, phase 1 `{2,3,4,5}`, phase 2 `{0,1,4,5}` — four clients per phase |
| `skip_mod` | Skip when `cycle % period == index % period` (default period 3). `index` is `sm_id` or `cpc_slot` (knob). Four of six live clients if ids are 0..5 |
| `hard_rate_cap` | `(lanes/clients)*payload` B/cycle, no idle reuse |

`skip_mod` and `fixed_tdm` have the same **average** rate and **different co-eligible sets**. That can matter for the blog’s adjacent-pair result. B6 may change the H200 preset; until then keep both and do not hard-code 21.0.

---

## 6. Queues, credits, scheduler

Independent **per VC**:

```text
ingress[src_sm][vc][dst_sm]     // destination VOQ, occupancy in payload flits
egress[dst_sm][vc]
credit[dst][vc]                 // free ejection / downstream flit slots
```

A full request VOQ must **not** steal response-VC occupancy (and the reverse).

Pure read-command streams remain FIFO. When payload data is queued in the same request VOQ, a new read command receives head priority so a 16-KiB TMA packet cannot block all reverse commands. Ejection credit is reserved per granted flit; an unfinished large packet does not reserve an entire destination buffer before its first grant.

Each core cycle, per eligible source SM:

1. Shaper: is this SM allowed to send?
2. Pick an eligible **head flit** among request/response VOQs (VC arbiter).
3. GPCMMU hash → plane + lane.
4. CPC lane arbiter among sources that chose that lane.
5. Check destination **that VC’s** ejection credit.
6. Grant: send **one 32 B payload flit** (or one control flit); decrement `remaining_flits`.
7. Tail arrival: packet becomes visible at destination ejection.

VC arbiter v1: **bounded response priority** (response first, cap consecutive response grants) or rotating RR with a reserved response buffer. Response must not starve (protocol deadlock).

Three **different** credits — do not merge them:

| Mechanism | Where | Purpose |
|-----------|-------|---------|
| VC / ejection credit | fabric | Stop downstream overflow |
| Arbiter fairness | fabric | No permanent VC/source starve |
| Outstanding transaction limit | endpoint | Cap in-flight ISA ops; **not** a link credit |

v1 single-stage fabric may derive link credit from remaining destination VC flit slots.

plan-v2 §9.1 interface (keep this contract):

```cpp
bool can_inject(local_sm_id_t physical_source, dsm_vc_t vc, unsigned flits) const;
void inject(std::unique_ptr<dsm_packet_t> packet);
const dsm_packet_t *top(local_sm_id_t physical_destination, dsm_vc_t vc) const;
std::unique_ptr<dsm_packet_t> pop(local_sm_id_t physical_destination, dsm_vc_t vc);
void cycle(unsigned long long cycle);
bool busy() const;
void display_state(FILE *) const;
```

No requester/service port, shader-side, or memory-side in this API. `can_inject` false → **LSU stall**. Do not map it to “write peer smem immediately” (delay-line `inject()==false` meant “NoC off”).

---

## 7. Endpoint protocol (not the arbiter)

`dsm_endpoint_protocol_t` per enabled SM:

- Outstanding transaction table (id, class, requester/target, remaining responses).
- **ACK coalescer** for stores / TMA puts:

```text
acks_owed[original_requester_sm]
oldest_ack_cycle[original_requester_sm]
```

Emit one `write_ack` (response VC, **one flit**) when count ≥ threshold, timeout expires, or the response path is idle. Carries a completion count. This is why symmetric store/TMA lose ~5% instead of ~23%.

Remote **load** cannot coalesce data. It always pays `read_command` (1 flit, request VC, reverse of data) plus `read_data` flits.

---

## 7.1 Stats and transaction dump (plan-v2 §17)

At least:

- Per GPC / CPC / SM: flits, packets, payload bytes
- Split by VC and `packet_class`
- Per-SM eligibility slots, used slots, **wasted** fixed slots
- Per-lane grants, conflicts, utilization
- VC ingress/ejection occupancy high-water
- Stalls: inject, lane, ejection, target queue, SRAM service, response wait
- Outstanding transaction count
- ACK debt, coalescing ratio, timeout flush
- Route-lane / hash histogram
- Per-SM SRAM client requested vs granted bytes
- Stale/invalid target count (must be 0 in normal runs)

Each transaction dump: ids, requester/target sm/gpc/cta/warp/rank, network src/dst, VC, class, route lane, payload bytes, remaining flits, TB-cluster group, op/address/size, lifecycle state, cycle at every stage.

---

## 8. TMA multicast on this fabric

v1: expand multicast at the **source** into unicast `tma_data` packets (source bandwidth scales with peer count). Keep a route descriptor that can later be `unicast | multicast_mask` so fabric replication is possible without changing the VC contract.

Do not claim a hardware replication point. Peer mbarrier complete stays **after** data (same as today).

---

## 9. Why the memory xbar is the wrong wire

The icnt (`local_interconnect` / BookSim) is **cluster-or-SM ↔ L2**, request/reply **separate physical** routers, whole-packet move after numeric budget, no VC dimension, no per-SM DSM port.

Using it for DSM would:

- Give request and response **two** bandwidths (contradicts mixed-traffic measurements)
- Deliver cache lines, not “these bytes into that CTA’s smem”
- Race on peer smem under OpenMP
- Map a full buffer to the delay-line “NoC off → instant write” boolean

Extract generic **bounded VOQ, RR arbiter, credit, stats** if useful (**B3a**). Do **not** instantiate two `xbar_router`s as DSM VCs. Do not inherit shader-side vs memory-side ports.

BookSim has real flits/VCs but global maps, subnet-by-category, and OpenMP-hostile globals. It is a future research option, not v1.

---

## 10. Delay line vs fabric (migration)

| | Today `cluster_noc_t` | Target `dsm_fabric_t` |
|--|----------------------|------------------------|
| Timing | `ready_cycle = hop (+ BPC)` | Flit grants + credits + shaper |
| Load data | Sample smem at execute | After `read_data` tail + SRAM service |
| Store | Byte snapshot, write on deliver | `write_data` flits + coalesced ACK |
| BW | Default unlimited | 32 B payload × shaped slots |
| Topology | Flat hop matrix | GPCMMU hash + GX + GPCARB |

When the fabric is the only path, **B-DEPR** deletes all `dsm_latency_matrix_*.csv`, `-gpgpu_dsm_latency_matrix_file`, and `-gpgpu_dsm_remote_latency` as a hop/bandwidth knob, plus the rest of the delay-line knobs, and updates configs ([`todos.md`](todos.md)).

---

## 11. Review bans (fabric)

Reject patches that:

- Treat request/reply **subnets** as VCs
- Give request and response independent **physical** bandwidth
- Name GX planes as request/response
- Infer response `network_src` from the transaction requester
- Move a whole packet in one cycle after accumulating budget
- Treat outstanding-tx limits as link credit
- Model TPCARB / 2-SM TPC
- Hard-code 21 B/cycle instead of knobs
- Charge header bits as extra payload occupancy in v1 (payload grant is 32 B; header is unmodeled width)
- Claim `skip_mod` or `fixed_tdm` is the Hopper hardware timetable
