# Evidence: measured, patent, inferred, unresolved

Label every number in code comments and presets with one of these four words. Do not stamp H200 values as Blackwell hardware fact.

Pinned reverse-engineering tree:

- Commit `4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd` (`Add standalone H200 DSM bandwidth benchmark`)
- [ANALYSIS.md](https://github.com/seanzw/random/blob/4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd/dsm_bw/ANALYSIS.md)
- [benchmark](https://github.com/seanzw/random/tree/4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd/dsm_bw)
- Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

In-tree latency job: `../H200_profiling/output-2046238-H200Profiling.txt` (hop / RTT / TMA extra).

Calibration report (kernel inventory, expected H200 numbers, sim columns): [`calibration.md`](calibration.md).

Patent: US12248788B2, figures 21A–21D. TPCARB in those figures is **not** modeled; PG'd SM slots **are**.

---

## 1. Measured (H200 `dsm_bw`, size-slope)

Method: unique-address tiles 16–96 KiB, affine fit `T = α + βB`, bandwidth `1/β`. One CUDA cluster. 200 KiB dynamic smem so ranks land on different SMs.

| Operation | One-way B/cycle | Two-way / dir | Loss |
|-----------|-----------------|---------------|------|
| Remote load | 19.87 | 15.37 | ~23% |
| Ordinary remote store | 18.87 | 18.12 | ~4% |
| TMA remote put | 21.25 | 20.29 | ~5% |

Mixed traffic (read response + write/TMA payload):

- Same physical direction: aggregate ~21.4 B/cycle
- Opposite directions: ~29.3–29.7 B/cycle

TMA ring put, per-SM rate stays ~21 B/cycle from 2 to 16 SMs. Idle neighbors do **not** donate bandwidth.

Job 2046238 (pointer-chase / latency, not slope BW):

| Metric | Value |
|--------|------:|
| DSM local | ~37 cycles |
| DSM remote e2e | ~193 cycles |
| Implied one-way | ~78 cycles |
| Stride ratio | ~1.001 (flat) |
| TMA mcast − unicast e2e | ~135 cycles |

---

## 2. Patent (structure, not numbers)

Supported:

- SM → (TPCARB in the drawing) → uTLB/GPCMMU → GPCARB → GX planes → destination SM
- Six SM clients, four aggregate connections per CPC (`SM2SM4`)
- GX0/GX1 **parallel planes**, not VCs
- Three CPCs → GPC-wide `SM2SM12`
- Request VC blocking, response VC non-blocking
- Write ACKs may be **coalesced**
- Firmware may disable one GX plane
- Per-SM bandwidth control exists (placement unspecified)

This rewrite **does not model TPCARB**. SMs attach to GPCMMU / GPCARB. PG'd slots in the figure **are** modeled.

---

## 3. Inferred (configurable, not “hardware fact”)

plan-v2 §3.3 and the blog use the same arithmetic. **Do not** promote an implementation choice (TDM vs skip_mod) to a silicon fact.

| Inference | Why | How we implement it |
|-----------|-----|---------------------|
| Lane **payload** service unit 32 B | `4×32/6 = 21.333` matches one-way rates; 128 B response = four grants | `-gpgpu_dsm_flit_payload_bytes=32` (v2 name: `flit_bytes`; alias) |
| Header/metadata may exist | Patent packets have headers; encoding **unresolved** (v2 §3.3). Later agreement: charge **payload only** | Unmodeled occupancy |
| Average **2/3 lane / SM / cycle** | Same 4/6 ratio; idle neighbor does not raise rate | Non-work-conserving shaper. **Not** one true schedule |
| 128 B read response quantum | Duplex-loss table; `ld.shared::cluster.v4.b32` → four 128 B wavefronts. **Not** a 128 B wire | Four payload flits + one reverse `read_command` |
| GPCMMU is an address hash | Patent uTLB for DSM routing, not a page walk | `route_policy=deterministic_hash` |
| Two GX planes | Patent + figure; firmware may drop one | `-gpgpu_dsm_gx_planes=2` |
| Write ACKs coalesce | Patent + 4–6% store duplex loss | Endpoint coalescer, not the arbiter |
| Request VC blocking, response VC non-blocking | Patent packet table | Independent queues/credits; shared lanes |
| Architectural TMA completion overlaps payload traffic | L9–L11 endpoint curves cannot be matched by serial global load then multicast delivery | H200-only completion curve releases mbarrier while memory/fabric traffic remains modeled |
| Store visibility has a distinct floor | L7 is ~573 cycles while L4–L6 already match with the generic 78-cycle floor | Store-class visibility floor; load path unchanged |

Shaper **implementations** (all required, none is “the Hopper schedule”):

| Policy | Source | Notes |
|--------|--------|-------|
| `fixed_tdm` | plan-v2 §9.4 | CPC **slots** `{0,1,2,3}` / `{2,3,4,5}` / `{0,1,4,5}` — four live clients per phase, matches four lanes. v2’s worked example |
| `skip_mod` | later agreement | Skip 1 of 3 cycles, phase = `sm_id % 3` or `cpc_slot % 3`. Same 2/3 **rate**, **different** co-eligible sets than TDM |
| `hard_rate_cap` | plan-v2 §9.4 | Equivalent average, no idle reuse |

`skip_mod` by global `sm_id` and TDM by CPC slot are **not** the same concurrent set. Adjacent-pair grouping in the blog is **measured** under contention; v2 does **not** claim `%smid` is the EG-select bit. B6 may switch H200 preset policy if adjacent-pair tests need it. Until then both exist; do not hard-code one as hardware.

---

## 4. Unresolved (must stay knobs)

From plan-v2 §3.3 and the blog’s “not bit accurate” list:

- Literal TDM vs skip_mod vs token bucket (all allowed)
- Exact NVIDIA address hash and GX plane select
- Packet header bitfields; short-write-in-request vs long-write command+data (patent). v1 uses `write_data` payload flits only
- Per-hop credit depth (v1: ejection occupancy)
- Router pipeline / hop stages; `base_latency_cycles` is a floor, not a claimed pipeline
- Independent DSM fabric clock (blog: **not** needed to explain 21 B/cycle in the SM clock)
- Blackwell parameters
- TMA multicast replication point
- Whether EG even/odd grouping is `%smid` parity

---

## 5. Alignment with plan-v2 (intentional deltas)

The English spec follows plan-v2 §§1–17, 19–20. Chapter 18 is ignored. These deltas are **explicit**, not accidents:

| Topic | plan-v2 | This spec | Why |
|-------|---------|-----------|-----|
| TPC / TPCARB | In the patent path | **Not modeled**; SM → GPCMMU | Agreed simplification |
| 32 B unit | “32 B flit service” | 32 B **payload** per grant; extra header width unmodeled | Later agreement; same 128 B = 4 grants |
| Shaper default | Example `fixed_tdm` | Both TDM and `skip_mod`; `skip_mod` phase offset required | Configurable 2/3 cap |
| GX count | Two planes in the patent | Knob, default 2 | Agreed |
| PG'd SM slots | Topology table; extra detail in Phase 9 | **B1** (pulled forward) | Need it for `num_sms_per_gpc` ≠ 6×CPCs |
| Remote load completion | Data response + pending join (Phase 5) | Same **SMEM** scoreboard / LDST writeback | Agreed: remote load ≈ local `ld.shared` |
| `tb_cluster_barriers` object | In the ownership tree | Timing state is held by the physical cluster and keyed by TB-cluster group | Required by upstream `cluster.sync()` bandwidth kernels; no longer a non-goal |

Do **not** add further deltas without updating this table.

Implementation evidence from the 2026-08-31 bandwidth audit: PTX scope suffixes on ordinary memory instructions require explicit normalization of both `.shared::cta` and `.shared::cluster`; otherwise the parser's generic-space representation can incorrectly route a valid shared address to global memory. The upstream mixed load+TMA checksum exposed this distinction.

Implementation evidence from the 2026-09-01 slope audit: multi-flit payloads must hash each 32-B cache-line grant rather than pin a whole TMA command to one route; otherwise per-SM TMA rate falls from about 21 B/cycle at two SMs to about 15 B/cycle at sixteen. Reverse read commands need head priority over queued request data, while request/response VCs still share the shaped physical sender. Cluster-barrier release must also wait for outstanding DSM transactions in every CTA in the TB-cluster group.

---

## 6. Hypotheses that failed (do not revive)

- Separate ~1 GHz DSM clock required to explain 21 B/cycle
- One half-duplex SM port (would punish writes like reads)
- 64 B physical payload flit (too much symmetric-read loss)
- TMA-only bypass (ordinary stores match TMA duplex)
- Destination SRAM / LSU stall as the 21 B cap
- Large token bucket bursting above ~21 B/cycle
- TPC-local DSM bypass of GXBAR
