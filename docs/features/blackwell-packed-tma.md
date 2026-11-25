# Blackwell packed U4 tiled loads (private draft)

Ordinary tiled TMA loads support `16U4_ALIGN8B` and `16U4_ALIGN16B` through
one shared payload implementation in `flash/tma_packed.h`. PTX and SASS call
that implementation, then use the existing swizzle, transaction and completion
paths. This is separate from Rubin instruction or research memory changes.

The compact format transfers eight bytes per sixteen values. The expanded
format places each such group in a sixteen-byte shared container. A 128 by 128
tile therefore transfers 8192 global bytes and occupies either 8192 or 16384
shared bytes. Address generation and barrier completion count the packed global
payload. Multicast copies the complete shared image. Model padding is zero;
the tests do not establish hardware-defined values for padding gaps.

The native decoder recognizes retained SM100 Driver controls `0x64590` and
`0x264590` for compact 2D maps, `0x2645a0` for compact 3D maps, and `0x266630`
for expanded 4D maps. Opaque
descriptor decoding remains an observed encoding subset, not a portable CUDA
ABI. The simulator's explicit Driver descriptor also reaches the same backend
through both frontends.

Qualification includes packed row addressing, higher-dimensional strides,
negative and out-of-range coordinates, 64B/128B swizzle with a nonzero shared
base phase, adjacent guards, and byte completion. Component tests compare the
functional read addresses with the timing AGU at both 32B and 128B granularity.
The `sm100-packed-tma` job runs seven complete numerical launch tests through
each frontend, including both opaque and simulator-generated descriptors.
The ordinary Driver TMA and gather4 suites remain regression gates.

This extraction qualifies aligned sixteen-value origins, unit element strides,
no interleave, zero OOB fill, and the ordinary swizzle modes accepted by the
format. Packed stores, ordinary U6 transfers, other opaque controls and
additional swizzle variants remain rejected. The existing gather4 coverage is
unchanged. Small launch tests do not replace retained DeepInfer acceptance or
hardware throughput calibration.

Reference: [NVIDIA CUDA Tensor Map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html).
