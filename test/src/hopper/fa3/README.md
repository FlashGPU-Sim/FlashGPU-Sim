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

- `fa3_fwd_hdim128_fp16_test.cc` - Google Test wrapper for the fixed FA3 case
- `fa3_fwd_hdim128_fp16_case.cuh` - shared CUDA workload implementation
- `flash-attention/` - flash-attention submodule pinned to `d80a77103021c4e980f8cbbf85774f6a19e6474a`

## Build

From the GPGPU-Sim repository root:

```bash
git submodule update --init test/src/hopper/fa3/flash-attention
git -C test/src/hopper/fa3/flash-attention submodule update --init csrc/cutlass
```

Then from `test/`:

```bash
./run_tests.sh build hopper
./run_tests.sh hopper Fa3FwdHdim128Fp16IntegrationTest
./run_tests.sh hopper Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase
./run_tests.sh hopper Fa3PrefillFp16SmokeTest.H32D64FullB2S128
./run_tests.sh hopper Fa3PrefillFp16IntegrationTest.H16D128FullB64S512
```

The generated kernel targets `sm_90a`. It is for GPGPU-Sim/PTX bring-up and
Hopper inspection; it is not expected to run on non-Hopper hardware.

The generated FA3 prefill launch cases are excluded from the default Hopper
suite because they run the simulator path. Run an individual case by passing
its gtest name to `run_tests.sh hopper`; no extra environment variable is
required.

## Notes

This fixed workload intentionally uses `ClusterM = 1`, so GPGPU-Sim bring-up can
treat cluster-scope spellings as the local-CTA degenerate case.
