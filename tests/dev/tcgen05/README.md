# SM100 strict-MXFP4 and mixed MXF8/F6/F4 support

This directory contains the real CUTLASS integration smoke for the simulator's
SM100 `tcgen05.mma.kind::mxf4.block_scale.block32` path, plus the
`kind::mxf8f6f4.block_scale` path used by DeepGEMM W4A8. The strict-MXFP4
CUDA smoke uses CUTLASS's block-scaled collective builder and testbed; it is
not a hand-written traffic proxy. Host numerical and PTX parser tests cover
the mixed path, while the SM100 TMA integration tests exercise its real
sub-byte global-to-shared layout. The downstream Kimi-K3 workload provides the
full DeepGEMM W4A8 integration smoke.

## Modeled semantics

The functional model follows the PTX ISA strict-MXFP4 contract:

- packed E2M1 matrix A and B operands in K-major shared-memory layouts;
- one UE8M0 scale for each block of 32 K elements;
- FP32 accumulation, optional input D accumulation, and A/B negate bits;
- scale-factor IDs 0 and 2 and the PTX TMEM subpartition/byte layout; and
- `tcgen05.cp.32x128b.warpx4` row multicast into four TMEM subpartitions.

The mixed MXF8/F6/F4 model follows the separate PTX contract:

- A and B independently select E4M3, E5M2, E2M3, E3M2, or E2M1, for 25
  ordered input combinations;
- every 16 logical values use one 16-byte shared-memory container: FP8 is
  16-byte payload, FP6 is 12-byte payload plus 4-byte padding, and FP4 is
  8-byte payload plus 8-byte padding;
- one UE8M0 scale per row and 32 K elements, with K=32 per instruction; and
- FP32 accumulation, optional input D, and A/B negate bits.

DeepGEMM's Kimi K3 path selects E4M3 A and E2M1 B: W4A8. Strict W4A4 remains
on `kind::mxf4`, whose packed storage and K=64 instruction shape are distinct.

The descriptor and compute models cover every dense shape in the PTX table:

- `cta_group::1`: M=128, K=64, N=8..256 in steps of 8;
- `cta_group::2`: M=128 or 256, K=64, N=16..256 in steps of 16; and
- `cta_group::2`: M=256, K=96, N=16..256 in steps of 16.

The end-to-end CUDA execution path currently coordinates `cta_group::1`.
Two-CTA distributed shared-memory/TMEM coordination for `cta_group::2` remains
separate work; its legal descriptors, numerical compute model, and theoretical
latencies are covered by host unit tests.

For dense `kind::mxf8f6f4`, descriptors cover M=128, K=32, N=8..256 for
`cta_group::1`, and M=128 or 256, K=32, N=16..256 for `cta_group::2`.
End-to-end execution currently remains K-major shared/shared and
`cta_group::1`.

## Timing model

Dense MXFP4 work is counted as `2*M*N*K` FLOPs and instruction latency is
`ceil(2*M*N*K / throughput_per_sm_cycle)`. The checked-in B200 configurations
use 30,947 FLOP/cycle/SM, derived at the device-reported 1.965 GHz maximum SM
clock:

```
30,947 * 148 SM * 1.965 GHz = 9.00001 PFLOP/s
30,947 * 148 SM * 1.08 GHz  = 4.94657 PFLOP/s
```

This is a theoretical dense-throughput anchor derived from the published B200
9 PFLOP/s per-GPU FP4 rate at the peak-clock operating point, not a measured
instruction-latency calibration. The simulator keeps the per-cycle hardware
throughput fixed when the configured 1.08 GHz B200 workload-calibration clock
is lower; it does not preserve peak FLOP/s by inflating per-cycle throughput.
All scheduler-facing Tensor Core pipes reserve one shared per-SM TCGen05 backend,
so the configured chip throughput is not multiplied by the four front-end
pipes.

The checked-in mixed MXF8/F6/F4 rate is 15,474 FLOP/cycle/SM, or about 4.5
PFLOP/s at the same 148-SM, 1.965-GHz peak-clock point. This is half the strict
MXFP4 rate because NVIDIA documents `mxf8f6f4` at 2x Hopper FP8 Tensor Core
throughput and strict `mxf4` at 4x. It is a theoretical roofline anchor; exact
instruction latency, issue interval, shape efficiency, and tails still need
native Blackwell calibration.

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
- [B200 device-reported 1.965 GHz maximum SM clock](https://github.com/ProjectPhysX/OpenCL-Benchmark#example-results)
