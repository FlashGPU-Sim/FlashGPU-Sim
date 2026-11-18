# TMA Support

FlashGPU-Sim models asynchronous bulk and tensor transfers between global
and shared memory, tensor-map updates, and completion synchronization through
CTA memory barriers or bulk groups.

## Operations

- Linear `cp.async.bulk` global/shared transfers.
- `cp.async.bulk.tensor` tile transfers with one through five dimensions.
- `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group`: commit bulk
  operations to groups and suspend the issuing warp until the requested
  outstanding-group condition is satisfied.
- `tensormap.replace`: update supported descriptor fields.
- `tensormap.cp_fenceproxy`: copy descriptor bytes from shared to global memory.
- `cp.reduce.async.bulk.tensor` tile reduction stores.

Functional simulation performs the data transfer or reduction. Timing
simulation generates memory-system requests and tracks their completion,
including mbarrier transaction-byte accounting and bulk-group waits. The
functional copy alone does not determine when the issuing warp may proceed.

Tensor tiles use descriptor dimensions, strides, element sizes and swizzle
settings. Integration coverage includes multidimensional address mapping,
swizzles, tensor-map updates and bulk-group synchronization. Support for a
particular descriptor or instruction variant is bounded by the implemented
validation and the tested cases.

## Tensor Reduction Stores

The functional `.tile` path reduces each shared-memory element into its
corresponding global-memory destination:

| Operation | Supported tensor-map element types |
| --- | --- |
| `add` | `u32`, `s32`, `u64`, `f16`, `bf16`, `f32` |
| `min`, `max` | `u32`, `s32`, `u64`, `s64`, `f16`, `bf16` |
| `inc`, `dec` | `u32` |
| `and`, `or`, `xor` | 32-bit and 64-bit integer types |

`FLOAT32` preserves subnormal inputs/results; `FLOAT32_FTZ` flushes them to
signed zero.

Reduction-store timing reuses normal tensor-store requests and completion.
It does not separately model the atomic destination read, same-address
serialization or a dedicated reduction resource, so this timing path is
uncalibrated.

## Model Limits and Controls

- Some tensor-map manipulation options are not fully validated.
- Non-cacheline-aligned bulk-copy sizes have limited corner-case coverage.
- The implemented descriptor copy does not establish a complete proxy-memory
  ordering model.
- Tensor reduction stores currently support `.tile` addressing only.

See [data movement and synchronization controls](../../../configs/README.md#data-movement-and-synchronization)
for TMA latency, request granularity/width, inflight limits and response width.
Enable the `TMA` trace component to inspect transactions and memory events;
see [runtime tracing](../../../docs/tracing.md).

## Validate

From the repository root:

```bash
./tests/run_tests.py list-cases --arch sm120 --group tma
./tests/run_tests.py run --arch sm120 --group tma
./tests/run_tests.py run --arch sm100 --group tma
```

See [`tma_test.cu`](../../../tests/src/tma/tma_test.cu) and
[`tma_multidim_test.cu`](../../../tests/src/tma/tma_multidim_test.cu) for the
selected functional and synchronization checks, and the
[mbarrier model](mbarrier.md) for phase-completion semantics.

Global-to-shared loads using `.shared::cluster` with local addresses are
unicast unless `.multicast::cluster` is present. Only explicit multicast
replicates data and mbarrier completion to the `ctaMask` destinations.
Cluster launch dimensions alone do not enable multicast. The
completion floor uses the cluster setting for multi-CTA cluster launches or
explicit multicast, and the non-cluster setting otherwise. Only explicit
multicast adds the multicast latency to global-to-shared loads.

The `TMAClusterUnicastTest` integration cases check bulk and tensor loads with
ordinary launches and cluster sizes one and two, including repeated barrier
phases and distinct per-CTA data. Run the size-two cases on `SM90_H200`.

```bash
./tests/run_tests.py run --arch sm90 --group integration -c SM90_H200 \
  --gtest-filter 'TMAClusterUnicastTest.*:TmaMulticastMaskTest.*'
```

This follows the explicit multicast and `ctaMask` semantics in the
[PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).

Cluster multicast is a functional fan-out plus a fixed latency knob, not a
bandwidth or contention model. See `docs/cluster_noc/README.md`.

