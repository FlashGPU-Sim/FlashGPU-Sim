# SASS-related configuration parameters

This is the branch's added/changed option-registration inventory relative to
`flash`, not a replacement for the full architecture config. Defaults come
from [gpu-sim.cc](../../gpu-sim.cc), [runtime_adapter.cc](runtime/runtime_adapter.cc)
and [icnt_wrapper.cc](../../icnt_wrapper.cc). The H100 column resolves omitted
options to those defaults in [SM90_H100_SASS_FRONTEND](../../../../configs/SM90_H100_SASS_FRONTEND/README.md).
An explicit zero is **not** universally “instantaneous”: each row defines it.

Cycles below are shader-core cycles unless explicitly identified as local
interconnect cycles; byte rates are per cycle. Queue depths count entries or
operations, not bytes, unless the option says bytes. These are simulator
controls, not claims about independently measured physical queue depths.

Native RF/control models require the corresponding frontend metadata and
enable switches. WGMMA controls apply to the SM90 warpgroup path; classic MMA
scheduler backpressure is distinct. Shared resource controls can also affect
PTX workloads. Do not copy all H100 values into SM120 or vice versa.

## Frontend selection

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_execution_frontend` | `environment` | `sass-timing` | Execution frontend: environment, ptx, sass-functional, sass-timing; explicit selection overrides SASS mode environment variables |

## Native register-file access

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_native_fixed_latency_rf` | `0` | `0` | Use atomic fixed-window register reads for native fixed-latency instructions instead of operand collection |
| `-gpgpu_native_rf_reuse_cache` | `1` | `1` | Honor native per-source .reuse retention in the register-file cache |
| `-gpgpu_native_rf_banks_per_subcore` | `2` | `2` | Regular register-file banks owned by each native subcore |
| `-gpgpu_native_rf_read_ports_per_bank` | `1` | `1` | Regular register-file read ports per bank in each native subcore |
| `-gpgpu_native_rf_write_ports_per_bank` | `1` | `1` | Regular register-file write ports per bank in each native subcore |
| `-gpgpu_native_rf_read_window` | `3` | `3` | Future cycles atomically reserved by native fixed-latency reads |
| `-gpgpu_native_rf_result_queue_depth` | `8` | `8` | Per-subcore result queue entries reserved by native fixed-latency instructions |
| `-gpgpu_native_rf_result_queue_max_pops` | `1` | `1` | Maximum fixed-latency result queue pops per subcore and cycle |

## Shared memory, MIO and classic MMA

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_shmem_data_wavefronts_per_cycle` | `1` | `4` | Number of independent shared-memory data wavefronts serviced per shader-core cycle. Bank-conflict wavefronts from the same access phase remain serialized. (default 1) |
| `-gpgpu_shmem_matrix_store_data_wavefronts_per_cycle` | `0` | `1` | Number of independent shared-memory data wavefronts serviced per shader-core cycle for matrix stores. 0 inherits gpgpu_shmem_data_wavefronts_per_cycle. (default 0) |
| `-gpgpu_shmem_mio_load_initiation_interval` | `1` | `1` | Minimum SM-wide service cycles for a shared-memory load warp instruction, independent of its data-wavefront count. (default 1) |
| `-gpgpu_tensor_core_scheduler_backpressure` | `0` | `0` | Block classic warp-level MMA at scheduler issue until that scheduler's tensor admission lane is ready. |
| `-gpgpu_mio_queue_depth` | `0` | `1` | Per-scheduler MIO admission depth for frontend-marked instructions. 0 disables the queue. |
| `-gpgpu_mio_ldsm_queue_depth` | `0` | `0` | SM-wide queue depth between MIO LDSM admission and serialized shared-data service. 0 disables the queue. |
| `-gpgpu_mio_ldsm_issue_interval` | `0` | `4` | Minimum cycles between matrix shared-load issues from one scheduler. The SM-wide shared-data path is modeled separately. 0 disables the per-scheduler limit. |
| `-gpgpu_mio_read_barrier_latency` | `0` | `0` | Cycles from MIO service start until an explicit read barrier is visible as released. 0 releases after ordinary operand collection. |

## Barriers and async proxy

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_cta_barrier_issue_interval` | `0` | `2` | Minimum cycles between CTA barrier arrivals across the whole SM. 0 disables the shared admission limit. |
| `-gpgpu_cta_barrier_release_latency` | `0` | `30` | Cycles from the final CTA barrier arrival until participating warps are released. 0 releases immediately. |
| `-gpgpu_mbarrier_issue_interval` | `0` | `2` | Minimum cycles between mbarrier instructions across the whole SM. 0 disables the shared admission limit. |
| `-gpgpu_async_proxy_fence_extra_stall` | `0` | `0` | Completion-token stall for a clean native async proxy fence. |
| `-gpgpu_async_proxy_fence_dirty_extra_stall` | `0` | `0` | Additional completion-token stall after a CTA TMA load has dirtied the async proxy epoch. |
| `-gpgpu_async_proxy_fence_initiation_stall` | `0` | `0` | Frontend occupancy for a clean tokenless native async proxy fence. |
| `-gpgpu_async_proxy_fence_dirty_initiation_stall` | `0` | `0` | Frontend occupancy for a dirty tokenless native async proxy fence. |
| `-gpgpu_async_proxy_shared_store_visibility_latency` | `0` | `32` | Cycles from matrix-store shared-data service until async-proxy visibility. 0 makes visibility immediate. |
| `-gpgpu_async_proxy_shared_store_initiation_interval` | `0` | `4` | Per-scheduler admission interval for matrix stores entering the async-proxy visibility path. 0 disables admission serialization. |
| `-gpgpu_mbarrier_trywait_predicate_latency` | `0` | `36` | Issue-to-predicate visibility latency for native mbarrier.try_wait write barriers (default=0) |

## Instruction flow and forwarding

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_sfu_to_sp_forwarding_latency` | `0` | `0` | Cycles from SFU service start until an explicit write barrier is visible to an SP consumer. 0 waits for ordinary completion. |
| `-gpgpu_instruction_backedge_redirect_latency` | `0` | `12` | Additional fetch/decode redirect cycles for a taken physical-ISA backedge. |
| `-gpgpu_instruction_loop_buffer_bytes` | `0` | `0` | Physical instruction bytes retained across a taken backedge. 0 disables execution-driven loop refill timing. |
| `-gpgpu_instruction_loop_refill_granularity` | `256` | `256` | Physical instruction bytes represented by one loop refill level. |
| `-gpgpu_instruction_loop_refill_latency` | `0` | `0` | Fetch bubble cycles per physical instruction loop refill level. |
| `-gpgpu_instruction_loop_refill_max_latency` | `0` | `0` | Maximum physical instruction refill cycles charged per backedge. |

## TMA service

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_tma_read_barrier_latency` | `0` | `0` | Cycles from TMA service start until an explicit read barrier is visible as released. 0 releases after ordinary operand collection. |
| `-gpgpu_tma_max_inflight_bytes` | `0` | `0` | Max in-flight TMA bytes per SM (default=0, 0=unlimited) |
| `-gpgpu_tma_tx_quota_bytes` | `0` | `0` | Base in-flight byte quota per TMA transaction (default=0; when nonzero, overrides gpgpu_tma_tx_quota) |
| `-gpgpu_tma_store_source_bytes_per_cycle` | `0` | `0` | Shared-memory source consumption rate for TMA global stores. 0 waits for destination memory acknowledgement (legacy behavior) |
| `-gpgpu_tma_store_source_fixed_latency` | `0` | `0` | Setup latency before a TMA global store begins consuming its shared source (default=0) |

## WGMMA admission and dispatch

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-gpgpu_wgmma_admission_queue_depth_ss` | `0` | `6` | Maximum per-SM SS WGMMA operations admitted but not yet consumed by the tensor backend; 0 disables admission backpressure (default=0) |
| `-gpgpu_wgmma_admission_queue_depth_rs` | `0` | `5` | Maximum per-SM RS WGMMA operations admitted but not yet consumed by the tensor backend; 0 disables admission backpressure (default=0) |
| `-gpgpu_wgmma_accumulator_queue_depth` | `0` | `8` | Maximum WGMMA writers in flight per warpgroup and accumulator base; 0 disables accumulator-queue backpressure (default=0) |
| `-gpgpu_wgmma_warp_arrival_model` | `0` | `1` | Latch WGMMA arrivals per warp and dispatch the collective operation when the fourth warp arrives (default=0) |
| `-gpgpu_wgmma_dispatch_pressure_period` | `0` | `32` | Period of asynchronous WGMMA result-side dispatch reservations; 0 disables the model (default=0) |
| `-gpgpu_wgmma_dispatch_pressure_tail_multiplier` | `0` | `5` | Number of completion-tail intervals retained by WGMMA result-side dispatch pressure after tensor compute (default=0) |
| `-gpgpu_wgmma_sfu_dispatch_pressure_percent` | `100` | `150` | Percentage of base WGMMA result-side pressure seen by SFU/MUFU instructions (default=100) |

## Interconnect replies

| Option | Default | H100 SASS | Meaning / unit |
| --- | --- | --- | --- |
| `-icnt_reply_output_grants_per_cycle` | `1` | `2` | Maximum reply packets granted to one output per local-interconnect cycle |

## Calibration and disabled options

The hardware calibration experiments are maintained separately from this
frontend. Their parameter rationale is summarized below; use the
[FA-3 regression workflow](../../../../tests/scripts/README.md) for validation:

- MIO/SFU and LDSM/HMMA use the SFU and MMA microbenchmarks; queue admission,
  bank conflicts and SM-wide data service are separate constraints.
- WGMMA was measured with capacity, warp-arrival, RF and mixed-softmax probes;
  a passing isolated probe does not establish
  accuracy for every multi-WG composition.
- Async-proxy, named-barrier and mbarrier parameters use their dedicated
  microbenchmarks. Deferred-barrier **semantic ordering** is in the
  [timing contract](TIMING_CONTRACT.md), not an extra fitted latency.
- TMA/NoC and CCD rationale and remaining miss-path limits are documented with
  the H100 config. The existing `-gpgpu_tma_response_width=2` is two 32-byte
  response sectors, distinct from local-interconnect output grants and the
  32-byte L2 data port.
- The H100 config leaves byte quotas, store-source timing and loop refill
  disabled, but the tracked SM120 config uses nonzero settings for those
  paths. Do not delete them merely because H100 omits them. Other unselected
  alternatives are not automatically calibrated. Nonzero defaults in a
  disabled parent model likewise do not mean that model is active.

Keep the frozen baseline unchanged. Validate config changes using all four
medium forward numerical checks and the 0.5% cycle gate; inspect any drift
before changing a reference. Large forward remains the final PR regression.
