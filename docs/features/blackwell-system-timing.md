# Optional Blackwell system timing

This layer preserves the B200 models needed alongside TCGen05/TMEM. It does
not migrate ICC/GCC, Rubin, 3D placement, or the private NoC experiments. New
options default to the upstream behavior; no production preset is changed.
Functional instruction coverage and calibrated performance qualification are
separate acceptance requirements.

## Added support

All option names below have the `-gpgpu_` prefix.

| Model | Options and defaults | Backend contract |
|---|---|---|
| Ready L2 MSHR release | `l2_mshr_release_requests_per_cycle=1` | Requests, not MSHR entries, per subpartition per L2 tick. Recheck response FIFO and shared data-array credits after every release. Does not increase downstream bandwidth. |
| Two-path simple DRAM reads | `simple_dram_local_read_latency=0`, `simple_dram_remote_read_latency=0`, `simple_dram_locality_stripe_bytes=128` | Remote value is **additional** core cycles; zero disables. Local zero inherits `dram_latency`. Only shader-originated reads reaching simple DRAM use the model. Writes/non-shader requests retain uniform latency. Stable ready-time ordering permits local requests to overtake remote requests without extra return ports. |
| CTA admission | `cta_dispatch_width=0`, `cta_dispatch_per_gpc=0`, `cta_dispatch_interval=0` | Global/per-GPC admissions per core tick and minimum interval between admissions to one SM. Zero is unrestricted. Ordinary and clustered launches share budgets; an incomplete cluster continues on later ticks. |
| Native scalar resource assignment | `native_imad_on_sp=0`, `native_fp32_aux_paths=0` | Optional IMAD SP/FMA routing preserves integer operands. Qualified BF16 pack/FP32 compare use ALU; S32/U32 conversion uses SFU/MIO. Functional arithmetic is unchanged. |
| Qualified wide IMAD pacing | `native_imad_wide_initiation=0`, `native_imad_wide_min_issue_interval=0`, `native_imad_wide_result_latency=0` | Applies only to `IMAD.WIDE.U32` with register multipliers and zero-register addend. Shared dependency tracking blocks dependent reads and younger writes, without blocking independent instructions. Immediate/address forms do not inherit this calibration. |
| Implicit PTX TMA expansion | `tma_instruction_issue_hold=0`, `tma_ptx_issue_hold_latency=0`, `tma_ptx_issue_hold_interval=0` | Distinct per-warp readiness for another TMA versus another instruction; zero overrides inherit opcode latency/interval. Backend TMA service throughput is separate. Explicit SASS, paired tensor copies, and verified single-lane elected regions do not acquire a synthetic expansion delay. |
| Channel hash alternative | Existing `non_power2_l2_channel_indexing`: add `1` for high-bit XOR; default remains `0`, existing IPOLY remains `2` | For non-power-of-two local slice counts and power-of-two channels. Fold channel-local high bits into the decoded channel, preserving DRAM coordinates and local slice mapping. This is a permutation, not an address remapping that aliases storage. |

The HBM locality domains are synthetic: SM parity versus address-stripe parity.
They are **not** a recovered physical Blackwell die/address map. The historical
285 local + 345 remote-extra setting means 630 remote cycles, not 345. Those
values came from a particular clock/latency experiment; this migration does not
enable them or establish frequency-independent calibration.

Scalar routing is a supported resource-contention hypothesis, not proof of a
uniquely identified physical pipe assignment. Keep its qualifiers and the
original measurement context when selecting settings.

The PTX TMA election pass only annotates instructions. It reuses the scheduler's
register-use/definition collector; it does not remove elections, fold constants,
reorder instructions, or import the private compiler optimizer. External edges
into an elected region truncate the safe prefix. Active-lane count is checked
at issue. Both frontends use the same backend readiness primitive; only the PTX
adapter supplies implicit-expansion metadata. Existing
`tma_issue_to_next_instruction_latency` remains independent: do not configure
two overlapping helper penalties unintentionally.

## Existing support reused

- `gpgpu_gpc_sms` and `gpu_topology_t` already describe nonuniform GPC sizes.
  The private `cluster_core_counts` parser is not duplicated.
- Cluster return ingress/dispatch queues and budgets are already per SM. The
  private `cluster_response_scales_with_cores` switch is unnecessary.
- Typed MMA/paired capacity, measured shape recurrence, TMEM timing, shared
  MMA/LDS reads, TMA soft quota, CLC and the optional regional write model were
  already extracted. Preserve their existing options and tests.
- GTO and scalar initiation tables already exist. Selecting the calibrated
  scheduler/rates is a configuration task, not an additional implementation.
- Source review found native local-memory translation still using
  `clusters * maximum_cores_per_cluster`; use `num_shader()` for uneven GPCs.
  This is a separate adapter correction, not a timing parameter.

## Provenance

Ported and adapted from the private B200 implementation, rather than fitted to
new DSV4 outputs:

- `ec14b99bd`: simple DRAM locality; historical report
  `docs/b200-hbm-locality-model-20260928.md` in the private branch.
- `c58bfd702`: MSHR release; shared-source TMA matrix report
  `docs/b200-shared-source-tma-sim-20260930.md` in the private branch.
- `b3bfbcb5a`, `416c427b3`: native scalar routing/pacing; hardware archive
  `data/hardware/b200/dsv4-int-occupancy-20261004/capture/` in the workspace.
- `ad4b6a65d` and its prerequisite private implementation: CTA cadence and
  PTX/SASS TMA hold separation. Global limits now also bound the inner cluster
  admission loop, rather than permitting a whole cluster to overshoot.

Raw captures and old run configurations are retained unchanged.

## Acceptance and remaining qualification

Tests exercise the production L2/DRAM pipeline (merged miss waiters, response
backpressure, data-array bandwidth, local-read overtaking, disabled FIFO), actual
ordinary/partial-cluster/interval CTA admission, channel distribution and
non-aliasing, native scalar projection, dependency visibility, default option
isolation, and PTX election regions including external-entry counterexamples.

The full core suite additionally found an old atomic fixture that started the
adapter without binding its kernel. The fixture is corrected separately; the
production atomic path is unchanged.

Local evidence: `/home/wzr/flashgpu-b200-system-migration-20261127/`.
This support migration is not a claim that the old private B configuration and
the feature preset are performance-identical. Full timing qualification still
requires a frozen current config, matching measured clocks, cold/warm policy,
topology, original cubin/input and NCU cycles/traffic. The former DSV4 functional
fixture disabled already-extracted typed MMA timing and must not be reused as
a calibrated B200 baseline. ICC/GCC remains intentionally excluded.
