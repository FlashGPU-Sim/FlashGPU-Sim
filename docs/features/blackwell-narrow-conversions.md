# Packed low-precision conversions

This private child of `feature/blackwell-memory-forms` extracts conversion
support from the retained DeepInfer stack. `narrow_conversion.h` is shared by
PTX and native adapters; it replaces duplicate native FP8/FP4 packing and unpack
arithmetic. The PTX adapter is a separate small source file, invoked before the
ordinary BF16/scalar conversion path.

Qualified forms:

- RN + satfinite FP32 pair / BF16x2 to E4M3x2 or E2M1x2.
- RP + satfinite FP32 pair to UE8M0x2.
- RN E4M3x2 to FP16x2 or BF16x2; UE8M0x2 to BF16x2.
- Existing native FP8 scaled BF16 unpack reuses the same numeric helper.

Other roundings, ReLU/saturation combinations, newer packed types and source
formats are not silently treated as these forms: the PTX adapter rejects them.
This is not a claim of complete CUDA 13.3 conversion ISA coverage.

`sm100-narrow-conversion` CI uses CUDA host reference functions, exhaustive
BF16 encodings, all byte encodings/scales, random FP32 bit patterns, midpoint
neighbors, infinities, signed zero and NaNs. Six full numerical launches run the
same frozen cubin/PTX executable through both frontends. Existing native
conversion tests cover merge aliasing, inactive lanes and immediate selectors.
Timing behavior remains the existing scalar conversion path; no hardware
latency recalibration is implied.

A separate parser fix admits mixed literal/register `mov.b16/b32` packs using
the existing vector operand representation and execution path. Its standalone
`mov_pack` fixture needs none of the new conversion instructions. It is a
candidate for separate public-main qualification; no public PR is opened here.

Evidence and version receipts: `runs/narrow-conversions-20261125` and
`runs/CURRENT.json` outside this git worktree. Real C4 PTX now parses and starts
execution but reaches an unsupported `mul.bf16x2` in `mul_impl`, confirmed
with the clean runtime and a debugger at PTX line 9906. The previous FP32x2
diagnosis was incorrect; FP32x2 dispatch already exists. Full C4 numerical acceptance and
full DeepInfer acceptance remain pending. MegaMoE still needs `redux.sync`.

Reference: [NVIDIA PTX cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt).
