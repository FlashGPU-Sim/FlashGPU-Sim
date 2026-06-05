# FA3 Fixed Forward Case

This directory contains a standalone FlashAttention-3 Hopper forward test case
for GPGPU-Sim bring-up.

It does not require files from the original `flash-attention/hopper` directory.
FA3 headers are provided by the GPGPU-Sim submodule at `flash-attention`.
CUTLASS/CuTe is provided by that submodule's nested `csrc/cutlass` submodule.

## Case

- `B = 9`
- `seqlen_q = 64`
- `seqlen_k = 128`
- `nheads = 6`
- `nheads_kv = 6`
- `head_dim = 128`
- `dtype = fp16`
- `causal = false`
- `cluster_dims = (1, 1, 1)`

## Files

- `fa3_fwd_hdim128_fp16_sm90_driver.cu` - standalone host driver using local FA3/CUTLASS headers
- `fa3_fwd_hdim128_fp16_sm90_driver.ptx` - pre-generated SM90a PTX
- `flash-attention/` - flash-attention submodule pinned to `d80a77103021c4e980f8cbbf85774f6a19e6474a`
- `gpgpusim_fa3_ptx_gap.md` - instruction support notes for this PTX

## Build

From the GPGPU-Sim repository root:

```bash
git submodule update --init test/src/integration/fa3/flash-attention
git -C test/src/integration/fa3/flash-attention submodule update --init csrc/cutlass
```

Then from this directory:

```bash
make ptx
make bin
```

The generated kernel targets `sm_90a`. It is for GPGPU-Sim/PTX bring-up and
Hopper inspection; it is not expected to run on non-Hopper hardware.

## Notes

This fixed workload intentionally uses `ClusterM = 1`, so GPGPU-Sim bring-up can
treat cluster-scope spellings as the local-CTA degenerate case.
