# Blackwell timing qualification (private integration)

Follow-up: [mask/TF32 coverage and corrected TMEM timing](blackwell-timing-gap-closure.md)
supersedes the mask rejection and fixed five-cycle gap below. These original
measurements remain recorded as the pre-fix baseline.

The integration branch `feature/blackwell-timing-qualification` adds timing to
previously extracted functional TCGen05 support. PTX and native SASS publish the
same typed operation metadata and use `tcgen05/timing.*`. This is not a claim
that all Blackwell instruction forms or all kernel timing are calibrated.

## Shared implementation and configuration

* Single-thread MMA/CP/commit/shift operations retain their actual issuing lane.
  Warp-collective allocation/deallocation still use a representative lane.
* Metadata includes datatype, M/N/K, CTA group, A storage, collector reuse, and
  the actual shared operand footprint. Both adapters use the same packed-byte
  accounting helper, including swizzles and the two 128 KiB address regions.
* Typed compute rates are per SM per **core cycle**. A paired operation reserves
  both peer engines and divides the work over their combined compute capacity.
  Peer lookup uses resident cluster rank, not adjacent physical SM numbers.
* The 886-entry B200 recurrence CSV preserves hardware case IDs and T128/T256.
  `tools/calibration/generate_tcgen05_recurrence.py --check` validates the
  generated exact lookup. It does not interpolate missing shapes or treat
  unmeasured FP16 as measured BF16. Transpose, sparse, alternate accumulator
  types and collector reuse do not silently borrow the default profile.
* Recurrence is an effective owner completion envelope, overlapped with compute
  and operand service. It is not an extra latency added after all compute, nor
  evidence of physical accumulator storage-port traffic. Independent owners
  retain a shared SM compute budget. Queue/issue limits remain configurable.
* Ordinary 32dp TMEM reads use the retained latency/service tables, with
  initiation capacity per scheduler partition. A wait tracks its warp's own
  operations. Other transfer forms retain their explicitly uncalibrated timing.
* The optional shared model reserves TC read service before LSU service, per
  address half. Compute and operand supply overlap. Wide LDS dispatch and byte
  service advance together; charging them sequentially double-counts data
  wavefronts. This flag currently integrates **read-side TC/LDS competition**;
  STS/TMA writes retain the existing service path. It is not the complete
  private two-level shared write model.
* CLC retains asynchronous delivery and the historical 224-cycle setting.
  It does not impose a global request initiation interval.

All new options default off/zero, preserving the legacy mode. Only the
`SM100_B200` preset explicitly enables this qualification model; other presets
are unchanged. The effective SM100 rates are half 8192, TF32 4096, FP8/mixed
16384, and compact FP4 32768 FLOP/cycle. These are peak service capacities;
recurrence and supply can lower achieved utilization. The WS TMEM-A service
term is the retained effective 12-cycle model, not a general TMEM port model.

Statistics expose `recurrence_table_ops`, `recurrence_fallback_ops`,
`shared_read_bytes`, `shared_read_cycles`, and `lsu_read_denied_bytes`.
The last is per-attempt LSU service denied by the read-port limit, including
prior LSU consumption; it is neither unique traffic nor a stall-cycle count.
`backend_busy_cycles` counts reserved compute slots, not owner waiting time.

## Qualification evidence

Workspace evidence: `runs/tcgen-timing-fixes-20261125/` (outside this worktree).
Each run records frozen runtime/configuration and executable provenance.

* Lane-1 CP/commit and lane-7 MMA pass through both frontends, replacing the
  reproduced PTX lane-1 hang.
* Initial integration: 50 numerical tests / 91 launches per frontend pass
  (TMEM 14/18, MMA 25/52, paired CTA 5/15, collectors 6/6).
* 172 focused host component tests pass, including 19 timing tests. They cover
  exact type lookup, unknown-profile fallback, owner
  admission, peer reservation, overlap, partition-local TMEM, packed-byte
  accounting and shared-half contention.
* Archived mixed FP4 x FP8 and compact FP4 probes reproduce the measured
  M128/N64 recurrence of 48 cycles: differences between 128- and 256-MMA
  bursts are exactly 6144 cycles. All 8192 outputs per run pass the oracle.
  Rebuilding the host with shared cudart is required for replay; `.text`,
  `.nv.info` and `.nv.constant` sections of every kernel match the archived
  binaries exactly. Whole ELF hashes differ because of build metadata.
* Ordinary TMEM dependent-loop slopes for x1/x8/x32/x64 are 39/42/54/70,
  versus hardware 44/47/59/75. Width-dependent increments agree; a fixed
  five-cycle dependent-loop difference remains. The old fit measures execution
  pipeline latency, while this engine schedules asynchronous completion from
  frontend issue; the timing origins need further isolation before refitting.
* CLC successful response is 277 cycles versus hardware 289.9 (-4.45%);
  zero added delay gives 64 cycles. This is not the old private 289-cycle
  result. A controlled override of the old `gpgpu_mbarrier_trywait_latency=44`
  (current preset: 32) restores **289 cycles**, locating the 12-cycle difference
  in barrier/control timing rather than the CLC delivery setting. No unrelated
  barrier default is changed. Failed/concurrent responses remain uncalibrated.
* All 12 shared experiments pass numerics after fixing duplicated LDS service.
  Mixed LDS worker durations (sim/native): QK same-half 49191/49370,
  QK opposite-half 32838/33044, PV same-half 90161/90394, PV opposite-half
  49224/49412; errors range from -0.62% to -0.26%. Pure MMA QK/PV still differs by roughly
  -15%/-7%; successful read-port calibration does not remove that separate gap.
  The final-head numerical gates are recorded separately in the evidence report.

## Explicit limits

The archived BF16/TF32 sweep binaries use a seven-operand native MMA with an
explicit output-mask register group, whereas this extracted feature supports
the six-operand form; shared-A TF32 is also outside the extracted subset.
Those probe runs correctly reject the unsupported forms and are **not** counted
as successful hardware timing replays. Exact table/backend tests are a weaker
qualification than running those binaries.

Ordinary 16dp, packed/reduction TMEM loads, TMEM stores/copies, and implicit MMA
versus LDTM/STTM contention do not acquire invented calibration from the 32dp
read table. x128's read timing is extrapolated. The retained hardware PV/STTM
interaction (~8.7%) remains unmodeled. Shared physical placement between resident
CTAs is not reconstructed; region selection uses CTA-local addresses.

PTX and SASS share the new engine; they do not have identical instruction mixes
or identical total kernel cycles. This integration has not repeated all 117
DeepInfer acceptance entries or made a full-workload accuracy claim. Feature
owner branches and public PRs must be updated only after this qualification.

The original capture acceptance was subsequently closed under its recorded
frozen per-entry runtimes; see [resource preparation and acceptance closeout](tcgen05-resource-contention.md).
That closeout is separate from a full timing-accuracy sweep on this latest head.
