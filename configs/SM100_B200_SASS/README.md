# B200 SASS calibrated profile

Status: qualified as the current B200 SASS production baseline on 2026-11-28, with the timing limitations listed below. All 59 workloads completed; numerical acceptance is documented in `qualification.json`. This is not a claim that all outputs are bit-exact or every kernel is within 15% timing error.

Use all three files together in the simulation working directory: `gpgpusim.config`, `config_ampere_islip.icnt`, `sass_primary_hints.rules`. SASS is explicitly experimental. This opt-in profile does not change the public PTX/B200 default preset.

## Fixed choices

- 148 SM across 8 GPCs (20,20,18,18,18,18,18,18); four CPCs per GPC for the shared backend.
- Core/NoC/L2/DRAM clocks: 1080/1080/1155/3996 MHz. Report both elapsed cycles and time against NCU; native effective core clocks vary by workload.
- Perfect instruction cache; cold data cache, no memcpy L2 prewarming.
- GTO scheduler; VOQ, TMA request multicast, TMA response multicast and extra L2 TMA request coalescing all disabled.
- **L2 MSHR release width 2**, retained from the private B200 shared-source TMA calibration. This is ready waiter requests per subpartition per L2 tick, not MSHR capacity. Ordinary MSHR merging, output FIFO capacity and downstream NoC limits still apply.
- Typed MMA throughput and measured B200 shape recurrence enabled; measured TMEM load/transfer models and shared operand reads enabled. Regional shared-write contention disabled.
- Other public-derived memory port/latency settings retained. No ICC/GCC, synthetic local/remote HBM latency change, ideal global memory, Rubin or 3D settings.
- No simulation cycle cutoff in the profile; the qualification harness retains a 12-hour per-workload timeout.

## Why release width 2 is explicit

The common-backend migration added the option with legacy default 1, correctly preserving upstream behavior, but the private calibrated value 2 was omitted from the candidate profile. With all four transport options disabled and GTO fixed, the omitted setting substantially inflated the compression-projection cycle counts. Restoring only this parameter reproduces most of the historical private timing; opening unrelated merge/multicast switches would mask the regression.

The original 48-case native shared-source TMA matrix and 96 width-1/2 simulations support this choice as an effective calibration, not proof that hardware has exactly two physical completion ports. Public default remains 1.

## Qualification and limits

Frozen simulator source: f34fd18c6c277ad64170b75dd2572f68d33e5037.
Runtime SHA256: 0f2e4cba6d2e0b2855bcb766ff07b8573b700f50c3dca434e4a499efac3a5377.
Workspace experiment: runs/b200-production-config-qualification-r2-20261128/.
Attribution: runs/b200-private-projection-alignment-r2-20261128/.

Scope is the 59 original DSV4 DeepInfer B200 SASS workloads, global B4096 / local B512, original cubins and captured dynamic inputs. Numerical acceptance distinguishes strict matches, previously reviewed residuals and independently checked MoE routing permutations. This does not qualify every batch, PTX performance, or every instruction shape. Unknown MMA shapes retain fallback timing; TMEM 32dp x128 is extrapolated and CP 4x256b is unqualified. CLC failed/concurrent grants and shared write contention remain outside the calibrated claim.

## Confirmed regression attribution (2026-11-28)

Restoring only ready-MSHR release width 2 on the frozen feature runtime changes the C4/C128/Index compression projection cycle errors from +78.11/+75.91/+69.74% to +17.74/+17.24/+14.62%; QKV-A changes from +34.60% to -2.89%. Historical private errors were +18.51/+16.90/+12.89% and -2.08%. All four complete sparse residual sets are identical to their reviewed references. This explains approximately 97–102% of the added private-to-feature cycles without enabling transport optimizations. The complete profile qualification is now finished; see the results below.

## Final 59-workload qualification

All 59 completed without runtime failure. 34 are strict GPU matches, 23 have complete sparse residuals identical to reviewed references, and two MegaMoEs pass output/intermediate checks after accounting for routing permutations. Absolute NCU elapsed-cycle error median: 9.39%; 48/59 are within 15%. Use elapsed-cycle and time errors separately because NCU core frequencies differ from the configured 1080 MHz.

MegaMoE L2/L3 cycle errors are -10.20%/-9.39% after restoring release width 2. These are not the +4.83%/+4.91% values of the prior width-1 candidate. Historical private cycle errors were +3.05%/+3.48%, so private timing has not been reproduced exactly across all kernels. The large compression-projection regressions have been resolved to their prior accuracy range.

Known cycle-error exceptions:

| Kernel | Launch | NCU time (us) | Cycle error |
|---|---:|---:|---:|
| pro-hash-routing | 1 | 144.352 | -29.02% |
| pro-projection-compress-c4-fp32 | 0 | 33.088 | +17.74% |
| pro-projection-compress-c128-fp32 | 0 | 28.704 | +17.24% |
| mega-gate-b512 | 0 | 28.288 | -16.50% |
| wo-b-input-fp8 | 0 | 9.824 | +30.64% |
| first-input-norm-fp8 | 0 | 11.552 | +21.28% |
| pro-hash-routing | 2 | 10.880 | +44.54% |
| indexer-topk-b512 | 0 | 8.640 | +56.75% |
| swa-kv-norm-rope-store | 0 | 8.320 | +20.66% |
| pro-compressed-norm-rope-store-c4-d128 | 1 | 6.400 | +16.21% |
| indexer-schedule | 0 | 10.368 | -15.23% |

These are retained limitations, not accepted as accurate hardware predictions. In particular, some small-kernel errors remain worse than the historical private branch. No kernel-specific switch overrides are used. Public PTX defaults and the legacy width-1 default remain unchanged.
