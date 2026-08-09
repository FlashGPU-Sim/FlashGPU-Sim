# FA2 Forward Cases

This directory contains standalone FlashAttention-2 forward gtests for
GPGPU-Sim bring-up. The CUDA kernel templates come from the shared checkout
under `tests/third_party/flash-attention/`. FA2 builds prepare the pinned
checkout and apply the local patches automatically.

## Cases

- 20 opt-in 32Ki-token prefill cases matching the FA3 shape table:
  `H32D64/H16D128 x full/causal x B,S`.
- 4 smoke cases matching FA3 smoke: `B=2`, `S=128`,
  `H32D64/H16D128 x full/causal`.
- 4 small cases matching FA3 small: `B=32`, `S=256`,
  `H32D64/H16D128 x full/causal`.
- 4 medium cases matching FA3 medium: `B=16`, `S=512`,
  `H32D64/H16D128 x full/causal`.
- 9 isolated sensitivity runners for `H1D128FullB1S256`: baseline,
  `skip_cp_async`, `skip_mma`, `skip_softmax`, `fma_softmax`, `only_mma`,
  `only_cp_async`, `only_softmax`, and `nothing`.
- One legacy fixed smoke case: `B=1`, `S=128`, `nheads=2`,
  `head_dim=64`, `dtype=fp16`, `causal=false`.

Each size profile is compiled as split binaries by `D x full/causal`, so each
translation unit only instantiates one FA2 kernel family. The SM90 and SM120
architecture manifests expose the same profiles through `sm90/fa2` and
`sm120/fa2`; each build uses its manifest's NVCC target.

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

To prepare a CUDA prebuilt bundle and collect NCU results on H100 or RTX 5090,
use the manifest-backed tools from the repository root:

```bash
./tests/scripts/prepare_fa2_prebuilt.sh \
  --device h100 \
  --cuda-root /usr/local/cuda-12.8 \
  --group full --group breakdown --group scaling

./tests/scripts/run_fa2_ncu.sh \
  --device h100 \
  --prebuilt-root tests/run/fa2-prebuilt
```

Both commands accept repeated `--group` selectors. Use
`--group breakdown:only_mma` to select one mode, or `--group all` to include
all manifest-owned FA2 profiles. See `tests/scripts/README.md` for the complete
interface and migration notes.
