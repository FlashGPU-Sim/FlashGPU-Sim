# Captured Blackwell Driver tensor maps (private draft)

This layer enables retained SM100 CUDA Driver tensor-map descriptors to feed
both PTX and native TMA through the existing canonical `tensormap_descriptor_t`.
It follows the TCGen05/CLC stack for operator acceptance, but descriptor decoding
is a TMA prerequisite, not MMA math or launch cancellation.

## Backend and ISA boundary

`flash/tma_driver_tensormap.h` contains the shared decoder. It returns a canonical
map only after matching a retained control value and checking rank, dimensions,
strides, unit element strides, tile-byte count, swizzle metadata and reserved
tail fields. Failure leaves the destination unchanged. It has no SASS dependency.

The PTX TMA reader reaches it through the existing opaque-map decoder. The native
load/store adapters call the same decoder and translate the canonical fields to
their existing view; no second Driver geometry decoder is introduced. Existing
FA4 and compact-device descriptor paths remain available. Dense byte, 32/64-bit
integer, BF16 and F32 transfers preserve the original payload bits.

The accepted encoding controls come from retained B200 descriptors and the
previous private implementation. CUDA does not specify their byte layout.
This is deliberately not a general CUDA Driver ABI claim. The subsequent
[floating-load extension](blackwell-tma-floating-loads.md) qualifies TF32
conversion for observed 2D/3D controls. Packed FP4 controls remain rejected;
simply treating them as dense bytes would be incorrect.

## Separate prerequisites

SM100/103 compiler-generated loads take the by-value parameter pointer from
`c[0][0x348]`. The adapter now supplies the existing indirect parameter window,
backed by launch parameter memory without copying. SM120's reserved slot remains
zero. A focused test checks the pointer, shared backing after updates, read-only
behavior, final in-range byte and the following ordinary global address.

The dual-frontend runner also creates versioned library aliases pointing at the
requested immutable runtime. Without `libcudart.so.13`, an unversioned-only
snapshot could fall through to the mutable build directory. This is a test
harness correction, independent of simulated ISA behavior.

## Initial extraction evidence (historical)

The local `sm100-driver-tma` job selects 15 component tests and five numerical
launch tests (twelve launches per frontend). The CUDA fixture passes opaque
maps by value with `__grid_constant__`, exercises 2D/3D/4D/5D loads and 2D/4D/5D
store roundtrips, and checks all output bytes including untouched store regions.
It constructs descriptors and expected offsets independently of simulator code.
It is simulator qualification; no new GPU measurement is claimed.

The retained-map scan recognizes 99 of 108 blocks. The nine remaining blocks
are three TF32 and six packed-FP4 descriptors. Recognition alone does not prove
that a complete real operator executes correctly.

On the assembled branch, `projection-index-q` now passes the initial ordinary
TMA load and reaches the separately missing `UTMALDG.2D.2CTA` form. Its recovered
PTX also needs packed conversion syntax (`cvt.rn.bf16x2.f32`). A fresh static scan
passes projection for 44/59 retained entries; the others expose ordinary memory,
control and reduction forms not yet extracted. The complete 59-entry SASS /
58-entry PTX DeepInfer acceptance remains pending, with original captures and
strict checkers preserved.

Evidence and exact runtime/binary hashes are indexed by workspace
`runs/CURRENT.json`, under `runs/tcgen-deepinfer-extraction-20261125`.
