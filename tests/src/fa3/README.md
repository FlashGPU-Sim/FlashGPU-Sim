# FA3 Forward and Backward Cases

This directory contains standalone FlashAttention-3 Hopper forward and
backward tests for GPGPU-Sim bring-up.

The test wrapper is local, while FA3 kernel headers come from the shared
checkout under `tests/third_party/flash-attention/`. FA3 builds prepare the
pinned checkout and apply the local patches automatically. CUTLASS/CuTe is
provided by that checkout's nested `csrc/cutlass` submodule.

## Fixed Forward Case

- `B = 9`
- `seqlen_q = 64`
- `seqlen_k = 128`
- `nheads = 6`
- `nheads_kv = 6`
- `head_dim = 128`
- `dtype = fp16`
- `causal = false`
- `cluster_dims = (1, 1, 1)`

## CI Correctness Coverage

The smoke profile contains four shapes spanning head dimensions 64 and 128,
causal and noncausal attention, and `S=128` and `S=256`. PR CI runs both the
forward and backward variants. Deterministic inputs are checked elementwise
against a CPU attention reference: forward validates output and LSE, while
backward validates dQ, dK, and dV. The `S=256` cases exercise multi-tile
launches. CI also runs the fixed forward case above and the PackGQA smoke case.

## Files

- `fa3_fwd_hdim128_fp16_test.cu` - shared Google Test registration and workload
  wrapper
- `fa3_{fwd,bwd}_d{64,128}_{noncausal,causal}_test.cu` - thin, uniquely named
  standard-build wrappers for one kernel specialization each
- `fa3_fwd_hdim128_fp16_case.cuh` - shared CUDA workload implementation
- `fa3_fwd_packgqa_{case.cuh,test.cu}` - one-tile GQA forward case that
  validates the default and `.noinc` `cp.async.mbarrier.arrive` forms

The shared checkout, preparation script, pinned revisions, and patch bundle are
documented in `tests/third_party/flash-attention/README.md`.

## Build

From `tests/`; the first FA3 build prepares the shared dependency automatically:

```bash
./run_tests.py run --arch sm90 --group fa3 --profile smoke
./run_tests.py run --arch sm90 --group fa3 --profile packgqa
./run_tests.py run --arch sm90 --group fa3 --profile smoke \
  Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase
./run_tests.py run --arch sm90 --group fa3 --profile large \
  Fa3PrefillFp16IntegrationTest.H16D128FullB64S512
./run_tests.py run --arch sm90 --group fa3 --profile breakdown --mode baseline
./run_tests.py build --arch sm90 --group fa3 --profile concurrency --mode all
```

The generated kernel targets `sm_90a`. It is for GPGPU-Sim/PTX bring-up and
Hopper inspection; it is not expected to run on non-Hopper hardware.

The standard wrappers compile serially and link into
`tests/build/bin/sm90/fa3/standard_tests`. The split bounds NVCC memory and
gives every fatbin a unique source-derived PTX name, which GPGPU-Sim requires
when loading multiple embedded PTX images.

## Strict SASS gates

The five standard forward tests, four backward tests, two PackGQA smoke tests,
and both breakdown baselines run their unchanged assertions through
execution-driven SASS with no PTX fallback. The runner decodes the exact cubin
registered by each executable into per-run `.sassir` files; suite
JSON contains no generated SASSIR path or kernel selector. Run them using:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-13.3
source ./setup_environment
./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_forward_smoke.json
./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_backward_smoke.json
./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_packgqa.json
./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_forward_small.json
./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_backward_small.json
FA3_H1D128_PROFILE_OUT=/tmp/fa3_h1d128_profile.csv \
  ./tests/run_sass_suite.py \
  tests/src/fa3/sass_functional_suite_sm90_breakdown_baseline.json
```

For reproducible medium/large forward timing, see the
[regression workflow](../../scripts/README.md). SASS suites select the separate
`SM90_H100_SASS_FRONTEND` config.

Each declarative suite binds every specialization to one exact GTest filter.
The forward suite requires 5/5 tests: four check output and LSE against the CPU
reference, while the rectangular fixed case (`Sq=64`, `Sk=128`, `B=9`,
`H=6`, `D=128`) checks successful execution. The backward suite requires 4/4
tests and checks dQ, dK, and dV across its preprocess, mainloop, and
postprocess launches. PackGQA requires 2/2 tests.
The PackGQA suite uses an exact binary selector because its default and
`.noinc` executables intentionally register the same GTest name.

The same eight standard specializations also gate all four forward and four
backward `small` runtime cases at `B=32, S=256`. These unchanged tuning tests check
successful execution rather than a CPU reference; the adjacent smoke suites
remain the numerical output/LSE and dQ/dK/dV gates.

The separately compiled breakdown gate runs both `baseline` and
`baseline_noprofile` on their default `B=1, H=1, S=4096, D=128`,
full-attention case. Both check successful execution rather than attention
output against a CPU reference; the profiled build additionally requires
nonzero clock deltas. In functional SASS mode the global timer is a monotonic
logical value, so the emitted CSV is not a cycle-accurate timing result. The
unprofiled target kernel is byte-identical to the standard D128-full image and
therefore produces the same kernel-level SASSIR content from its own binary.

Smoke, packgqa, size, and sensitivity profiles are all exposed through the
`sm90/fa3` test group. A filter can select an individual GoogleTest case
without escaping the selected profile.

## Notes

This fixed workload intentionally uses `ClusterM = 1`, so GPGPU-Sim bring-up can
treat cluster-scope spellings as the local-CTA degenerate case.
