# SM100 strict-MXFP4 support

This directory contains the real CUTLASS integration smoke for the simulator's
SM100 `tcgen05.mma.kind::mxf4.block_scale.block32` path. It uses CUTLASS's
block-scaled collective builder and testbed; it is not a hand-written traffic
proxy.

## Modeled semantics

The functional model follows the PTX ISA strict-MXFP4 contract:

- packed E2M1 matrix A and B operands in K-major shared-memory layouts;
- one UE8M0 scale for each block of 32 K elements;
- FP32 accumulation, optional input D accumulation, and A/B negate bits;
- scale-factor IDs 0 and 2 and the PTX TMEM subpartition/byte layout; and
- `tcgen05.cp.32x128b.warpx4` row multicast into four TMEM subpartitions.

The descriptor and compute models cover every dense shape in the PTX table:

- `cta_group::1`: M=128, K=64, N=8..256 in steps of 8;
- `cta_group::2`: M=128 or 256, K=64, N=16..256 in steps of 16; and
- `cta_group::2`: M=256, K=96, N=16..256 in steps of 16.

The end-to-end CUDA execution path currently coordinates `cta_group::1`.
Two-CTA distributed shared-memory/TMEM coordination for `cta_group::2` remains
separate work; its legal descriptors, numerical compute model, and theoretical
latencies are covered by host unit tests.

## Timing model

Dense MXFP4 work is counted as `2*M*N*K` FLOPs and instruction latency is
`ceil(2*M*N*K / throughput_per_sm_cycle)`. The checked-in B200 configurations
use 56,306 FLOP/cycle/SM:

```
56,306 * 148 SM * 1.08 GHz = 8.99995 PFLOP/s
```

This is a theoretical dense-throughput anchor derived from the published B200
9 PFLOP/s per-GPU FP4 rate, not a measured instruction-latency calibration.
All scheduler-facing Tensor Core pipes reserve one shared per-SM TCGen05
backend, so the configured chip throughput is not multiplied by the four
front-end pipes.

## Run the CUTLASS smoke

Prerequisites are a CUDA toolkit with `sm_100a` and strict-MXFP4 PTX support, a
CUTLASS checkout containing the SM100 block-scaled tests, and a built simulator.

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export CUTLASS_ROOT=/path/to/cutlass
tests/dev/tcgen05/run_cutlass_mxfp4_smoke.sh
```

The script builds one shared-CUDART GoogleTest binary, verifies that its PTX
contains strict `tcgen05.mma` instructions, and runs numerical checks for
M128xN64xK256, M128xN128xK256, and M128xN256xK256 CUTLASS tiles with the
one-SM B200 simulator configuration. Each K=256 tile issues four K=64 MXFP4
instructions.

Use `GTEST_FILTER` to select one case and `SIM_TIMEOUT_SECONDS` to change the
default 600-second timeout.

## References

- [NVIDIA PTX ISA: `tcgen05` matrix multiply-accumulate instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-multiply-accumulate-instructions)
- [NVIDIA CUTLASS: Blackwell SM100 GEMMs and scale-factor layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [NVIDIA DGX B200 specifications](https://www.nvidia.com/en-us/data-center/dgx-b200/)
