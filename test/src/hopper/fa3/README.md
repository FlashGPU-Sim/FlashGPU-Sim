# FA3 Fixed Forward Case

This directory contains a standalone FlashAttention-3 Hopper forward test case
for GPGPU-Sim bring-up.

The test wrapper is local, while FA3 kernel headers come from a generated
`flash-attention/` checkout built from upstream FlashAttention plus the patches
in `patches/`. CUTLASS/CuTe is provided by that checkout's nested
`csrc/cutlass` submodule.

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
- `prepare_flash_attention.sh` - clones FlashAttention, checks out the pinned
  upstream commit, initializes CUTLASS, and applies local patches
- `patches/flash-attention-fa2-fa3-hooks.patch` - FA2/FA3 profiling hooks,
  sensitivity macro hooks, and task/globaltimer tracing used by isolated debug
  targets

## Build

From the GPGPU-Sim repository root:

```bash
cd test
make prepare-fa3-flash-attention
```

This prepares:

- FlashAttention base commit: `d80a77103021c4e980f8cbbf85774f6a19e6474a`
- CUTLASS commit: `7127592069c2fe01b041e174ba4345ef9b279671`

Then from `test/`:

```bash
./run_tests.sh run test --target sm90 --group fa3-smoke
./run_tests.sh run test --target sm90 --group fa3-smoke \
  Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase
./run_tests.sh run analysis --target fa3 --group large \
  Fa3PrefillFp16IntegrationTest.H16D128FullB64S512
./run_tests.sh run analysis --target fa3 --group breakdown --mode baseline
./run_tests.sh build analysis --target fa3 --group concurrency --mode all
```

The generated kernel targets `sm_90a`. It is for GPGPU-Sim/PTX bring-up and
Hopper inspection; it is not expected to run on non-Hopper hardware.

The smoke cases live under `test/sm90`; small, medium, large, and sensitivity
workloads live under `analysis/fa3`. A filter can select an individual gtest
case without escaping the selected group.

## Notes

This fixed workload intentionally uses `ClusterM = 1`, so GPGPU-Sim bring-up can
treat cluster-scope spellings as the local-CTA degenerate case.
