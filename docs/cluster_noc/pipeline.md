# DSM instruction pipeline and scoreboard

Remote DSM **loads must behave like local shared-memory loads** on the SM: same LD/ST unit, same scoreboard reservation, same writeback/RF release. The fabric only changes **where the bytes come from** and **how long until they arrive**.

DSM stores are not dest-reg producers; they use the fabric + ACK. TMA multicast
uses functional fan-out plus its fixed completion latency and does not enter the
fabric. Neither operation may let its CTA retire while outstanding.

---

## 1. Local SMEM load (what to copy)

GPGPU-Sim local shared load today:

1. Issue: `Scoreboard::reserveRegisters` on dest regs (`PROD_MEM_SHARED`).
2. Functional decode points at **this** CTA’s smem.
3. Instruction is a **shared** op in the LD/ST unit (`shared_cycle`).
4. After `smem_latency` (and dispatch delay), writeback: operand collector, `releaseRegisters`, `warp_inst_complete`.
5. A later ALU that reads those regs `checkCollision`s until writeback.

Remote DSM load **reuses this**. Do not invent a second scoreboard, a DSM-only writeback port, or “fill RF in `ld_impl` then stall issue for `2×hop`.”

---

## 2. Remote load (target)

```text
issue
  → reserve dest regs (PROD_MEM_SHARED)     // same as local SMEM
  → do NOT write RF in ld_impl
  → group lanes by (target_sm, cta_slot)
  → create DSM transaction
  → inject read_command (request VC, 1 flit) toward the owner SM

owner SM
  → request-VC eject
  → shared_memory_service_t grant (competes with local LSU / TMA)
  → read owner smem
  → inject read_data (response VC, ceil(payload/32) flits)

requester SM
  → response-VC eject (tail)
  → complete like a local shared load: RF write + scoreboard release
     on the existing LDST writeback path
```

Until that writeback, a consumer instruction in the same warp must fail `checkCollision` — **exactly as after a local `ld.shared`**.

Implications:

- `dispatch_delay = 2×hop` is a **delay-line approximation** and must go away on the fabric path.
- `DSM_LOAD_RSP` deliver being a no-op is a bug relative to this spec.
- Completion time is **not** known at issue (shaper, credits, SRAM). Parking in LDST / pending shared-load state is the same idea as waiting on local smem latency.
- Classify as **shared** (`PROD_MEM_SHARED`), not global long-scoreboard, unless measurements later say otherwise. Remote DSM is slower than local smem, but it is still the shared-memory pipe, not L2.

Optional later: apply `-gpgpu_dsm_local_latency` (~37) to **self-mapa / local generic** so e2e remote ≈ local + fabric RTT. That is still the SMEM path, with a different latency knob.

---

## 3. Remote store / TMA put

```text
issue
  → inject write_data / tma_data (request VC, payload flits)
  → dest regs: none (store)

owner SM
  → SRAM service grant
  → write owner smem
  → ACK debt++ for the requester

endpoint coalescer
  → later write_ack (response VC, 1 flit, count)
```

v1: the warp/CTA **outstanding tracker** waits for ACKs so the CTA cannot exit while stores are in flight. CUDA kernels still **publish** with mbarrier; tests must `try_wait` before reading peer data.

Do not write peer smem at issuer execute on the fabric path (today’s `-gpgpu_dsm_store_immediate 1` is delay-line only).

---

## 4. Remote atomic

`atom.add` on a `mapa` pointer: `atomic_request` → owner SRAM **RMW at grant time** (serialized by the target service, not by “whoever’s `ld_impl` ran first”) → `atomic_response` → requester scoreboard/RF like a load if the atom has a dest, otherwise completion-only.

PTX `red` / `red.async` stay unimplemented until **A-F4**.

---

## 5. `shared_memory_service_t`

Network tail arrival ≠ SRAM done ≠ response injected.

Every client of that SM’s SRAM goes through one service:

- Local LSU shared ld/st
- TMA / cp.async local landing
- Same-SM peer CTA
- Remote DSM ingress
- TMA multicast peer data
- Remote atomic / mbarrier state

v1: consume `gpgpu_shmem_bytes_per_cycle` (or a named successor) with rotating RR or oldest-first. Two-phase expose / grant so GPC SM-walk order is not implicit priority.

v2 (B7): per-bank plan. Do not start B7 before B5 is green.

---

## 6. Mixed-target warps

Keep both (plan-v2 §10.2):

- `logical_dsm_address` — generic pointer that still encodes owner/rank
- `target_shared_offset` — owner bits stripped, used on the target CTA smem object

Lanes of one warp may `mapa` different ranks. Group by `(target_sm, cta_slot)`, one transaction per group, **join** before the warp instruction writeback. The scoreboard still holds the dest regs of the **whole** instruction until every group completes — same as a local shared load with multiple accesses.

---

## 7. Delay-line behavior (current code — do not extend)

Today:

1. `decode_space` may `deliver_ready()` then `ld_impl` **writes RF**.
2. `func_exec_inst` sets `dispatch_delay = 2×hop`, forces `shared_space`, empty accessq.
3. Scoreboard releases at ordinary shared writeback after that baked stall.
4. `inject_dsm_load_req` is **never called**; `DSM_LOAD_RSP` deliver is a **no-op**.

That is only consistent with a delay line where a store is due or not at execute. A shaped/queued fabric can still hold the store after `hop` cycles; the load would commit **stale** data and then stall. **B5** exists because of this. Do not add bandwidth shaping on top of execute-time RF fill.
