# FA2 Forward Cases

This directory contains standalone FlashAttention-2 forward gtests for
GPGPU-Sim bring-up. The CUDA kernel templates come from the shared checkout
under `tests/third_party/flash-attention/`. FA2 builds prepare the pinned
checkout and apply the local patches automatically.

## Cases

- 20 opt-in 32Ki-token prefill cases matching the FA3 shape table:
  `H32D64/H16D128 x full/causal x B,S`.
- 4 smoke cases matching FA3 smoke: `H32D64` full/causal at `B=2, S=128`,
  `H1D128` full at `B=1, S=256`, and `H16D128` causal at `B=2, S=128`.
- 4 small cases matching FA3 small: `B=32`, `S=256`,
  `H32D64/H16D128 x full/causal`.
- 4 medium cases matching FA3 medium: `B=16`, `S=512`,
  `H32D64/H16D128 x full/causal`.
- 9 isolated sensitivity runners for `H1D128FullB1S256`: baseline,
  `skip_cp_async`, `skip_mma`, `skip_softmax`, `fma_softmax`, `only_mma`,
  `only_cp_async`, `only_softmax`, and `nothing`.
- One fixed smoke case: `B=1`, `S=128`, `nheads=2`,
  `head_dim=64`, `dtype=fp16`, `causal=false`.

Each size profile is compiled as split binaries by `D x full/causal`, so each
translation unit only instantiates one FA2 kernel family. The SM90 and SM120
architecture manifests expose the same profiles through `sm90/fa2` and
`sm120/fa2`; each build uses its manifest's NVCC target.

The smoke profile initializes deterministic inputs and compares every output
and LSE value with a CPU attention reference. Its `H1D128FullB1S256` case also
exercises a multi-tile forward launch. This numerically checked profile is the
FA2 coverage selected by PR CI.

## Run

From `tests/`:

```bash
./run_tests.py run --arch sm90 --group fa2 --profile smoke
./run_tests.py run --arch sm90 --group fa2 --profile small
./run_tests.py run --arch sm90 --group fa2 --profile medium
./run_tests.py run --arch sm90 --group fa2 --profile large \
  Fa2PrefillFp16IntegrationTest.H32D64FullB64S512
./run_tests.py run --arch sm90 --group fa2 --profile breakdown --mode only_mma
./run_tests.py build --arch sm90 --group fa2 --profile scaling --mode all
./run_tests.py run --arch sm120 --group fa2 --profile smoke
```

## Strict SASS gate

All fixed-length FP16 suites decode their exact SM90 or SM120 executable at
runtime with the official CUDA tools. Run every original GoogleTest through
the execution-driven frontend with exact per-specialization count checks:

```bash
./tests/run_sass_suite.py tests/src/fa2/sass_functional_suite_sm90_smoke.json
./tests/run_sass_suite.py tests/src/fa2/sass_functional_suite_sm90_small.json
./tests/run_sass_suite.py tests/src/fa2/sass_functional_suite_sm90_medium.json
./tests/run_sass_suite.py tests/src/fa2/sass_functional_suite_sm90_breakdown.json
./tests/run_sass_suite.py \
  tests/src/fa2/sass_functional_suite_sm90_scaling_baseline.json
./tests/run_sass_suite.py \
  tests/src/fa2/sass_functional_suite_sm90_scaling_modes.json
FA2_RUN_32KI=1 ./tests/run_sass_suite.py \
  tests/src/fa2/sass_functional_suite_sm90_concurrency.json
./tests/run_sass_suite.py tests/src/fa2/sass_functional_suite_sm120_smoke.json
```

These paths do not parse PTX. The D64 full case covers both its 64-CTA smoke
launch and the original two-CTA fixed smoke case. Every case retains its
complete output and LSE comparison with the CPU attention reference. On SM120,
the four prefill launches cover 162 CTAs, 648 physical warps, and 1,513,216
dynamic SASS instructions. The decoded SASSIR and its provenance are
content-addressed generated cache data, not committed test inputs.
