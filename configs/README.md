# FlashGPU-Sim Configuration Guide

FlashGPU-Sim loads its GPU model from `gpgpusim.config` in the workload's
working directory. The configurations in this directory are the supported
starting points for simulation and architecture studies.

## Available Configurations

### SM90_H100

[`SM90_H100`](SM90_H100/gpgpusim.config) models an H100 GPU with:

- **Compute capability:** 9.0
- **Resources:** 132 SMs, 80 memory channels, and 160 L2/memory subpartitions
- **Timing models:** TMA, ordinary `cp.async`, `mbarrier`, MMA, and WGMMA
- **PTX transformation:** Register allocation and conservative instruction
  reordering
- **Clock domains (MHz):** `1500:1700:1700:2617`
  (core:interconnect:L2:DRAM)

### SM120_RTX5090

[`SM120_RTX5090`](SM120_RTX5090/gpgpusim.config) models an RTX 5090 GPU with:

- **Compute capability:** 12.0
- **Resources:** 170 SMs, 16 memory channels, and 128 L2/memory subpartitions
- **Timing models:** TMA, ordinary `cp.async`, `mbarrier`, MMA, and TensorMap
- **PTX transformation:** Register allocation enabled; instruction reordering
  disabled
- **Clock domains (MHz):** `2580:2580:2580:14001`
  (core:interconnect:L2:DRAM)

Both configuration directories include `sass_primary_hints.rules` for
experiments that explicitly enable SASS-guided PTX reordering.

### Legacy Configurations

`tested-cfgs/` contains configurations validated by earlier GPGPU-Sim
releases. `deprecated-cfgs/` contains older configurations retained for
reference. They are not maintained as FlashGPU-Sim release configurations.

## Common Parameters

The options below are inherited from GPGPU-Sim and are common starting points
for architecture studies. Parameters that describe topology, caches, address
mapping, and interconnects must be changed consistently.

### GPU Topology and Clock Domains

| Option | Meaning |
| --- | --- |
| `-gpgpu_compute_capability_major`, `-gpgpu_compute_capability_minor` | Compute capability exposed by the simulated device |
| `-gpgpu_n_clusters` | Number of processing clusters |
| `-gpgpu_n_cores_per_cluster` | SMs per cluster; total SMs are the product of this value and `-gpgpu_n_clusters` |
| `-gpgpu_clock_domains` | Core, interconnect, L2, and DRAM clocks in MHz |

Changing the compute capability also requires compiling the workload for the
matching SM architecture.

### Execution Resources

| Option | Meaning |
| --- | --- |
| `-gpgpu_shader_core_pipeline` | Maximum threads per SM and warp size, encoded as `<threads>:<warp_size>` |
| `-gpgpu_shader_cta` | Maximum resident CTAs per SM |
| `-gpgpu_num_sched_per_core` | Warp schedulers per SM |
| `-gpgpu_num_sp_units`, `-gpgpu_num_dp_units`, `-gpgpu_num_int_units`, `-gpgpu_num_sfu_units` | Scalar execution units per SM |
| `-gpgpu_num_tensor_core_units` | Tensor-core execution units per SM |
| `-gpgpu_pipeline_widths` | Widths of the issue and execution paths, including the FlashGPU-Sim TMA, `cp.async`, and TensorMap paths when present |

The order of `-gpgpu_pipeline_widths` is documented immediately above the
option in each `gpgpusim.config`.

### Cache and Shared Memory

| Option | Meaning |
| --- | --- |
| `-gpgpu_unified_l1d_size` | Unified L1 data cache and shared-memory capacity in KiB per SM |
| `-gpgpu_cache:dl1` | L1 data-cache geometry, policy, MSHRs, queues, and data-port width |
| `-gpgpu_cache:dl2` | L2 cache geometry and policy per subpartition |
| `-gpgpu_shmem_size` | Shared-memory capacity in bytes per SM |
| `-gpgpu_shmem_per_block` | Default shared-memory limit in bytes per CTA |
| `-gpgpu_adaptive_cache_config` | Enable runtime selection among supported shared-memory/L1 carveouts |

Cache configuration strings encode several mutually dependent fields. Start
from a supported configuration rather than constructing one field at a time.

### Memory System

| Option | Meaning |
| --- | --- |
| `-gpgpu_n_mem` | Number of memory channels |
| `-gpgpu_n_sub_partition_per_mchannel` | L2/memory subpartitions per memory channel |
| `-gpgpu_mem_addr_mapping` | Mapping from physical-address bits to channels, banks, rows, columns, and bursts |
| `-gpgpu_memory_partition_indexing` | Memory-partition indexing policy |
| `-gpgpu_dram_buswidth`, `-gpgpu_dram_burst_length` | DRAM interface width and burst length |
| `-gpgpu_dram_timing_opt` | DRAM bank, row, column, and bus timing |
| `-network_mode`, `-inter_config_file` | Interconnect backend and its optional configuration file |

The number of channels, subpartitions, L2 geometry, address mapping, and
interconnect endpoints form one model and should not be scaled independently.

## FlashGPU-Sim-Specific Parameters

`Code default` is the fallback used when an option is absent. It is not a
calibrated GPU configuration. In the tables below, `default` means that the
configuration omits the option and therefore uses the code default.

The meaning of zero is option-specific: it can mean disabled, unlimited, or
inherit another limit.

### Data Movement and Synchronization

| Option | Code default | SM90_H100 | SM120_RTX5090 | Meaning |
| --- | ---: | ---: | ---: | --- |
| `-gpgpu_num_tma_units` | `0` | `1` | `1` | TMA execution units per SM; `0` disables the TMA pipeline |
| `-gpgpu_tma_max_inflight` | `0` | `384` | `384` | Maximum in-flight TMA memory requests per SM; `0` is unlimited |
| `-gpgpu_tma_tx_quota` | `0` | `48` | `48` | Base in-flight request quota per TMA transaction; `0` is unlimited |
| `-gpgpu_tma_quota_segment_bytes` | `0` | default | `8192` | Scale the transaction quota by `ceil(transaction_bytes / segment_bytes)`; `0` disables scaling |
| `-gpgpu_tma_request_granularity` | `32` | `128` | `32` | Bytes represented by one TMA memory request |
| `-gpgpu_tma_request_width` | `1` | default | default | TMA memory requests issued per TMA unit per cycle |
| `-gpgpu_tma_response_width` | `1` | default | default | TMA response tokens accepted per SM per cycle |
| `-gpgpu_tma_oob_l2_traffic` | `1` | `1` | `1` | Route out-of-bounds TMA fill traffic through L2 |
| `-ptx_opcode_latency_tma` | `33` | `32` | `32` | TMA instruction latency in SM cycles |
| `-ptx_opcode_initiation_tma` | `33` | `32` | `32` | Minimum TMA issue interval in SM cycles |
| `-gpgpu_num_cp_async_units` | `0` | `1` | `1` | Ordinary `cp.async` execution units per SM; `0` disables this pipeline |
| `-gpgpu_cp_async_max_inflight` | `0` | `256` | `192` | Maximum in-flight ordinary `cp.async` requests per SM; `0` is unlimited |
| `-gpgpu_cp_async_request_width` | `1` | `4` | `4` | Ordinary `cp.async` requests issued per SM per cycle |
| `-gpgpu_cp_async_response_width` | `1` | `4` | `4` | Ordinary `cp.async` responses accepted per SM per cycle |
| `-gpgpu_cp_async_request_granularity` | `32` | `128` | default | Bytes represented by one ordinary `cp.async` request |
| `-gpgpu_cp_async_wait_release_latency` | `5` | `5` | `5` | Warp-release latency after a `cp.async.wait_group` condition is satisfied |
| `-ptx_opcode_latency_cp_async`, `-ptx_opcode_initiation_cp_async` | `7` | `7` | `7` | Ordinary `cp.async` latency and issue interval |
| `-ptx_opcode_latency_cp_async_commit`, `-ptx_opcode_initiation_cp_async_commit` | `7` | `7` | `7` | `cp.async.commit_group` latency and issue interval |
| `-ptx_opcode_latency_cp_async_wait`, `-ptx_opcode_initiation_cp_async_wait` | `5` | `5` | `5` | `cp.async.wait_group` and `wait_all` latency and issue interval |
| `-gpgpu_num_tensormap_units` | `0` | default | `1` | TensorMap descriptor execution units per SM |
| `-ptx_opcode_latency_tensormap`, `-ptx_opcode_initiation_tensormap` | `1,1,1` | default | `1,1,1` | Latency and issue interval for replace, `cp_fenceproxy`, and TensorMap fence operations |
| `-gpgpu_mbarrier_arrive_latency` | `0` | `29` | `29` | Delay before an arrive operation updates the barrier |
| `-gpgpu_mbarrier_trywait_latency` | `0` | `32` | `32` | Warp-release latency for `mbarrier.try_wait` |
| `-gpgpu_shmem_per_block_optin` | `0` | default | `101376` | Opt-in shared-memory limit per CTA; `0` inherits `-gpgpu_shmem_per_block` |
| `-gpgpu_max_dynamic_smem_prefer_occupancy_carveout` | `0` | `1` | default | Model the driver selecting an occupancy-oriented shared-memory/L1 carveout when no explicit preference is supplied |

### Matrix Execution

> [!NOTE]
> The comma-separated MMA latency and initiation lists follow this shape/type
> order: `m16n8k16.f16`, `m16n8k8.tf32`, `m16n8k32.int8`, `m16n8k8.f16`,
> `m16n8k8.bf16`, `m16n8k4.tf32`, and `m16n8k16.int8`.

| Option | Code default | SM90_H100 | SM120_RTX5090 | Meaning |
| --- | ---: | ---: | ---: | --- |
| `-ptx_opcode_latency_tensor` | `64` | `22,32,19,32,32,32,19` | `34,32,16,32,32,32,16` | MMA result latency by shape and type |
| `-ptx_opcode_initiation_tensor` | `64` | `6,32,19,32,32,32,19` | `34,32,16,32,32,32,16` | MMA issue interval by shape and type |
| `-gpgpu_cta_load_balance` | `0` | `1` | `1` | Cap CTAs per SM for uniform kernels using `ceil(total_ctas / total_sms)` |

The public SM90 configuration also models asynchronous WGMMA execution. SS
uses shared-memory operands for A and B; RS uses registers for A and shared
memory for B. SM120 does not set the WGMMA-specific options.

| Option | Code default | SM90_H100 | Meaning |
| --- | ---: | ---: | --- |
| `-ptx_opcode_latency_wgmma_ss`, `-ptx_opcode_initiation_wgmma_ss` | `4,4,4,4` | `4,4,4,4` | Non-overlappable SS tensor-pipe latency and issue interval for N = 8, 16, 32, and 64 |
| `-ptx_opcode_latency_wgmma_rs`, `-ptx_opcode_initiation_wgmma_rs` | `12,12,12,12` | `3,3,3,3` | Non-overlappable RS tensor-pipe latency and issue interval |
| `-ptx_opcode_completion_wgmma_ss` | `66,66,66,66` | `66,66,66,66` | Overlappable SS completion tail |
| `-ptx_opcode_completion_wgmma_rs` | `64,65,64,64` | `64,65,64,64` | Overlappable RS completion tail |
| `-ptx_opcode_completion_wgmma_int_ss` | `64,64,64,64` | `64,64,64,64` | Overlappable INT/B1 SS completion tail |
| `-ptx_opcode_completion_wgmma_int_rs` | `62,62,62,61` | `62,62,62,61` | Overlappable INT/B1 RS completion tail |
| `-ptx_opcode_compute_throughput_wgmma` | `4096,2048,8192,8192,65536` | `4096,2048,8192,8192,65536` | Per-SM work/cycle for FP16/BF16, TF32, FP8, INT8, and B1 |
| `-gpgpu_wgmma_issue_chain_ss` | `0,0,0,0,64` | `7,0,4,20,64` | SS issue-chain throttle encoded as `depth,startup_gap,fast_gap,slow_gap,reset_gap`; depth `0` disables it |
| `-gpgpu_wgmma_issue_chain_rs` | `0,0,0,0,64` | `7,0,3,13,64` | RS issue-chain throttle using the same encoding |

WGMMA register-file pressure is represented by pending traffic tokens sharing
the normal operand-collector read budget:

| Option | Code default | SM90_H100 | Meaning |
| --- | ---: | ---: | --- |
| `-gpgpu_reg_file_read_bytes_per_cycle` | `0` | `1024` | Total register-file read budget per SM cycle; `0` is unlimited |
| `-gpgpu_wgmma_rf_traffic_enable` | `0` | `1` | Enable WGMMA register-file traffic tokens |
| `-gpgpu_wgmma_rf_traffic_bytes_per_cycle` | `0` | `512` | Maximum WGMMA traffic drained per SM cycle |
| `-gpgpu_wgmma_rf_traffic_share_read_budget` | `0` | `1` | Charge WGMMA traffic against the normal register-file read budget |
| `-gpgpu_wgmma_rf_traffic_assume_accumulate` | `1` | `0` | Include accumulator reads in the WGMMA traffic tokens |
| `-gpgpu_wgmma_rf_traffic_include_rs_a` | `0` | `0` | Include register-A operand reads for RS WGMMA |

### PTX Transformation

| Option | Code default | SM90_H100 | SM120_RTX5090 | Meaning |
| --- | ---: | ---: | ---: | --- |
| `-gpgpu_ptx_register_allocator` | `0` | `1` | `1` | Enable conservative PTX virtual-register aliasing |
| `-gpgpu_ptx_register_allocator_stats` | `0` | `0` | `0` | Print register-allocation statistics |
| `-gpgpu_ptx_reorder` | `0` | `1` | `1` | Enable conservative PTX instruction reordering |
| `-gpgpu_ptx_reorder_sass_guided` | `0` | `0` | `0` | Guide PTX reordering with auto-extracted SASS/PTX-line anchors |
| `-gpgpu_alu_result_forwarding` | `0` | `0` | `1` | Make ALU dependencies ready after the configured operation latency instead of ALU writeback |

When SASS-guided reordering is enabled, FlashGPU-Sim loads the single
`*.rules` file in the run directory. The supplied configurations include
`sass_primary_hints.rules`, but the feature remains opt-in.

### Memory-System Calibration

These controls describe architecture-specific memory locality and local
interconnect behavior. They should be changed together with the corresponding
topology and address mapping.

| Option | Code default | SM90_H100 | SM120_RTX5090 | Meaning |
| --- | ---: | ---: | ---: | --- |
| `-gpgpu_ipoly_non_power2_balanced` | `0` | `2` | default | Balance IPOLY mapping for non-power-of-two memory-channel counts |
| `-gpgpu_ipoly_channel_stable_l2slice` | `0` | `0` | default | Keep the decoded DRAM channel stable while hashing the L2 slice |
| `-gpgpu_l2_partition_count` | `1` | `2` | default | Coarse L2/locality partitions; `1` disables remote-partition detection |
| `-gpgpu_l2_partition_extra_latency` | `0` | `150` | default | Extra cycles for an access to a remote coarse L2 partition |
| `-icnt_use_voq` | `0` | `1` | default | Use virtual output queues in the local crossbar |
| `-icnt_multi_grant_request` | `0` | default | `1` | Permit one request-network input to grant multiple outputs per cycle |
| `-icnt_multi_grant_reply` | `0` | default | `1` | Permit one reply-network input to grant multiple outputs per cycle |

### Experimental Controls

The supported configurations leave the following sensitivity controls at
their code defaults. They are useful for isolating bottlenecks, but do not
represent the calibrated default models.

| Option | Code default | Meaning |
| --- | ---: | --- |
| `-gpgpu_tma_idealized_memory` | `0` | Complete TMA memory requests immediately |
| `-gpgpu_cp_async_idealized_memory` | `0` | Complete ordinary `cp.async` requests immediately |
| `-gpgpu_tensor_core_issue_queue_depth` | `0` | Add an ideal pre-functional-unit tensor-core queue; `0` disables it |
| `-gpgpu_tensor_core_skip_writeback` | `0` | Complete tensor-core instructions without the register-file writeback path |
| `-gpgpu_tensor_core_units_per_sub_partition` | `1` | Tensor issue units sharing each ideal queue subpartition |
| `-gpgpu_tma_request_bytes_per_cycle` | `0` | Apply a TMA request-side byte budget; `0` disables the budget |
| `-gpgpu_dram_frfcfs_rowhit_first` | `0` | Prefer row-hit banks during FR-FCFS bank assignment |

## Custom Configurations

Create a custom configuration by copying the closest supported model as a
complete directory:

```bash
cp -a configs/SM120_RTX5090 configs/MY_CUSTOM_CONFIG
$EDITOR configs/MY_CUSTOM_CONFIG/gpgpusim.config
```

The directory must contain `gpgpusim.config`. A configuration may also include
an interconnect file used by the selected network backend and a
`sass_primary_hints.rules` file for SASS-guided PTX reordering. Keeping these
files together makes the directory self-contained when it is copied into a
workload's run directory.

When changing topology, update the SM count, memory channels, subpartitions,
L2 geometry, address mapping, and interconnect endpoints as one consistent
model. Do not infer an interconnect topology from the SM count alone.

After building FlashGPU-Sim and sourcing `setup_environment`, verify discovery
from the repository root and run a matching integration test:

```bash
./tests/run_tests.py list-configs
./tests/run_tests.py -c MY_CUSTOM_CONFIG run \
  --arch sm120 --group integration CudaVectorAdd
```

Use `--arch sm90` and an SM90 test group for a Hopper configuration. See the
[test guide](../tests/README.md) for the supported architecture/test-group hierarchy.

When troubleshooting a custom configuration:

- **Configuration not discovered:** Confirm that it is a direct child of
  `configs/` and contains a readable `gpgpusim.config`.
- **Simulation fails at startup:** Check compute-capability compatibility,
  then verify the memory topology, cache geometry, address mapping, and
  interconnect as one consistent model.
- **Unexpected performance:** Compare the complete run directory with its base
  configuration and confirm that an automation script has not recopied the
  bundled default configuration over local edits.
