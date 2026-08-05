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

- `fa3_fwd_hdim128_fp16_test.cu` - shared Google Test registration and workload
  wrapper
- `fa3_{fwd,bwd}_d{64,128}_{noncausal,causal}_test.cu` - thin, uniquely named
  standard-build wrappers for one kernel specialization each
- `fa3_fwd_hdim128_fp16_case.cuh` - shared CUDA workload implementation
- `fa3_fwd_packgqa_{case.cuh,test.cu}` - one-tile GQA forward case that
  validates the default and `.noinc` `cp.async.mbarrier.arrive` forms
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
./run_tests.py run --arch sm90 --test-group fa3 --profile smoke
./run_tests.py run --arch sm90 --test-group fa3 --profile packgqa
./run_tests.py run --arch sm90 --test-group fa3 --profile smoke \
  Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase
./run_tests.py run --arch sm90 --test-group fa3 --profile large \
  Fa3PrefillFp16IntegrationTest.H16D128FullB64S512
./run_tests.py run --arch sm90 --test-group fa3 --profile breakdown --mode baseline
./run_tests.py build --arch sm90 --test-group fa3 --profile concurrency --mode all
```

For native H100 runs with Nsight Compute collection, use the manifest-backed
hardware entry point from the repository root:

```bash
./test/scripts/run_fa3_ncu.sh \
  --group breakdown:qk_pv_only_no_tma \
  --group breakdown:qk_pv_only_no_tma_reg_timeline
```

The command accepts repeated `--group GROUP:MODE` selectors and uses each
profile's manifest-owned case list. See `test/scripts/README.md` for case
overrides, output layout, prerequisites, and migration notes.

The generated kernel targets `sm_90a`. It is for GPGPU-Sim/PTX bring-up and
Hopper inspection; it is not expected to run on non-Hopper hardware.

The standard wrappers compile serially and link into
`test/build/bin/sm90/fa3/standard_tests`. The split bounds NVCC memory and gives every
fatbin a unique source-derived PTX name, which GPGPU-Sim requires when loading
multiple embedded PTX images.

Smoke, packgqa, size, and sensitivity profiles are all exposed through the
`sm90/fa3` test group. A filter can select an individual GoogleTest case
without escaping the selected profile.

## Notes

This fixed workload intentionally uses `ClusterM = 1`, so GPGPU-Sim bring-up can
treat cluster-scope spellings as the local-CTA degenerate case.
