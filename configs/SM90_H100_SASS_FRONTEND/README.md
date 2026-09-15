# H100 SASS-frontend validation configuration

Parameter definitions, code defaults and selected values are in the
[SASS parameter reference](../../src/gpgpu-sim/flash/sass/PARAMETERS.md).

Current reference: `tests/baselines/fa3-sass-ccd2-20260914/`. Hardware timing
parameters and interconnect config remain identical to that frozen snapshot.
Memcpy L2 prefill is disabled (`-gpgpu_perf_sim_memcpy 0`) to match the retained
NCU collectors' `--cache-control all`. After rebasing onto `d20ce45e`, upstream's
64-bit memcpy-address fix makes prefill effective at the real input addresses;
the old snapshot's enabled prefill truncated those addresses. Keep the fix and
select cold-cache validation explicitly rather than restoring the truncation.
The frozen snapshot itself is unchanged.
CCD=2 supersedes the historical CCD=1 experiment. Use four medium forward
cases to check the current audit's maximum 0.5% cycle drift; rerun all twenty
large forward cases before the PR. Core clock is 1500 MHz.

This working config sets `-gpgpu_execution_frontend sass-timing`: selecting
it enables SASS timing and overrides mode environment variables. Decoder
tool settings remain required; the test runner supplies them with
`--sass-timing`. See the
[frontend usage](../../src/gpgpu-sim/flash/sass/README.md#cuda-runtime-launch-adapter).
The archived config under `tests/baselines/` is unchanged and still uses
environment selection. For historical byte-identity regression, pass that
archived config directory to the packaged runner and use
`tests/scripts/baselines/check_fa3_medium.py`. Current numerical CI uses this
working config and `tests/ci/check_sass_medium.py`; do not replace either
gate's frozen cycle reference.

This configuration is intentionally narrow in scope: it is the stable H100
baseline for validating execution-driven FA-3 SASS functionality and timing.
It retains the calibrated SASS core/shared-memory settings and the memory/SoC
settings below, including CCD=2. It is **not** identical to `SM90_H100` with
only an older SoC: subsequent core calibration also changed this config.

Explicit core/shared-memory option differences (a dash means not explicitly
set in `SM90_H100`, not necessarily a zero effective value):

| Option | SASS validation | `SM90_H100` |
| --- | ---: | ---: |
| `-gpgpu_mio_queue_depth` | `1` | — |
| `-gpgpu_mio_ldsm_issue_interval` | `4` | — |
| `-gpgpu_async_proxy_shared_store_visibility_latency` | `32` | — |
| `-gpgpu_async_proxy_shared_store_initiation_interval` | `4` | — |
| `-gpgpu_instruction_backedge_redirect_latency` | `12` | — |
| `-gpgpu_mbarrier_trywait_latency` | `43` | `32` |
| `-gpgpu_mbarrier_trywait_predicate_latency` | `36` | — |
| `-gpgpu_mbarrier_issue_interval` | `2` | — |
| `-gpgpu_cta_barrier_release_latency` | `30` | `15` |
| `-ptx_opcode_initiation_fp` | `1,2,1,1,2` | `1,1,1,1,2` |
| `-gpgpu_wgmma_admission_queue_depth_ss` | `6` | — |
| `-gpgpu_wgmma_admission_queue_depth_rs` | `5` | — |
| `-gpgpu_shmem_data_wavefronts_per_cycle` | `4` | — |
| `-gpgpu_shmem_matrix_store_data_wavefronts_per_cycle` | `1` | — |
| `-gpgpu_shmem_mio_load_initiation_interval` | `1` | — |

Relative to `../SM90_H100/gpgpusim.config`, the SoC differences are:

| Option | SASS validation value | `SM90_H100` value |
| --- | ---: | ---: |
| `-gpgpu_l2_partition_count` | `1` | `2` |
| `-gpgpu_l2_partition_extra_latency` | `0` | `150` |
| `dramid@` mapping bit | `8` | `9` |
| DRAM `CCD` | `2` | `2` |
| `-gpgpu_ipoly_non_power2_balanced` | `0` | `2` |
| `-gpgpu_tma_response_width` | `2` | — |
| `-icnt_reply_output_grants_per_cycle` | `2` | — |

Both configurations keep 80 memory channels, two subpartitions per channel,
the same cache geometry, interconnect topology and clocks. The reply-grant
setting above is separate from topology. TMA OOB traffic remains disabled
because the validation baseline must
never turn an out-of-bounds fill into a memory access.

Each of the 160 L2 subpartitions has an explicit 32 B/cycle data port.  At the
1.5 GHz core clock used for validation this represents 7.68 TB/s of aggregate
L2 data bandwidth; do not reinterpret the two subpartitions in one memory
channel as one 64 B/cycle subpartition.

Do not use this configuration to infer H100 SoC organization. Changes made to
improve the SASS core model need their own microbenchmark evidence and
regression; do not silently synchronize either config or refresh the frozen
baseline during an audit. Memory/SoC experiments belong outside the frozen
validation config.

## Calibration boundaries

`CCD=2` enforces the existing DRAM bus budget, not an added FA3 latency.
With 80 controllers, an 8-byte bus, BL=4 and data/command ratio 2, each
32-byte burst occupies two command cycles. At the configured 2617 MHz the
aggregate limit is 3.34976 TB/s. The old CCD=1 run allowed 48 channels to
report 1.185 bus utilization; CCD=2 reduced the maximum to 0.9298.

The retained TMA benchmark issues two 16-KiB loads and waits for completion,
with 8 warmup and 64 measured iterations at SM=1500 MHz. At 128 CTAs, the
streaming window changed from 1574.63 to 2089.52 simulator cycles, versus
2274.70 hardware cycles; the hot window stayed near 1299 versus 1280.50.
The 32-CTA streaming error still remained -19.7%: this is not complete
miss-path calibration. Per-CTA clock64 windows must not be presented as
chip-wide bandwidth or mixed with whole-kernel NCU durations.

Raw evidence is retained in `tests/run/tma_cold_bw_20260913.tar.gz` and
`tests/run/TMA_COLD_BW_CCD2_SIM_20260913/`; the original binary SHA-256 is
`0dee09d9d1a5c1cac49bed33ae70b39c0891f70eff9983fbab0f444c99b36898`.
The frozen medium/large baseline remains the end-to-end timing reference.
Removed intermediate task/phase reports are recoverable from Git, e.g.
`git show e7552567:docs/experiments/sass/TMA_CCD2_ABLATION_20260913.md`.
