# PTX Addressing Requirements

FlashGPU-Sim uses 64-bit generic addresses to distinguish global, shared and
thread-local memory. Workload PTX must declare:

```ptx
.address_size 64
```

A `.address_size 32` declaration is rejected during parsing. Use a CUDA
Toolkit compatible with the workload's PTX version and target architecture.

## Address Spaces and Limits

Generic shared addresses identify a shared-memory region within an SM;
generic local addresses identify a thread-local region. Global addresses
occupy a separate range. Explicit shared-memory operands use offsets within
shared memory, while global pointers and generic pointers retain 64-bit values.

The current generic address layout reserves space for:

| Resource | Address-layout capacity |
| --- | ---: |
| SMs | 1,024 |
| Shared memory per SM | 1 MiB |
| Local memory per thread | 16 KiB |

These capacities bound the virtual address layout. Actual kernel occupancy,
shared-memory availability and per-block opt-in limits are determined by the
selected GPU configuration and CUDA runtime settings.

Configuration validation rejects an SM count above the address-layout capacity.
Kernel launch validation rejects a shared-memory requirement above its reserved
region. Keep topology and memory requirements within both the address-layout
limits and the modeled device limits.

See the [configuration guide](../configs/README.md) for topology and memory
settings, and the [integration tests](../tests/src/integration/README.md) for
64-bit address operands, shared-memory opt-in and memory-access validation.
