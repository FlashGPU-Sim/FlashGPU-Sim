# FA2 Forward Cases

This directory contains standalone FlashAttention-2 forward gtests for
GPGPU-Sim bring-up. The CUDA kernel templates come from the existing
`test/src/hopper/fa3/flash-attention/csrc/flash_attn/src` checkout prepared by
`test/src/hopper/fa3/prepare_flash_attention.sh`.

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

Each size group is compiled as split binaries by `D x full/causal`, so each
translation unit only instantiates one FA2 kernel family. All FA2 targets use
`sm_90a`. Smoke is a correctness group under `test/sm90`; larger shapes and
sensitivity modes are analysis groups under `analysis/fa2`.

## Run

From `test/`:

```bash
make prepare-fa3-flash-attention
./run_tests.sh run test --target sm90 --group fa2-smoke
./run_tests.sh run analysis --target fa2 --group small
./run_tests.sh run analysis --target fa2 --group medium
./run_tests.sh run analysis --target fa2 --group large \
  Fa2PrefillFp16IntegrationTest.H32D64FullB64S512
./run_tests.sh run analysis --target fa2 --group breakdown --mode only_mma
./run_tests.sh build analysis --target fa2 --group scaling --mode all
```

To prepare a CUDA 12.8 prebuilt bundle for H100 NCU collection:

```bash
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./scripts/prepare_fa2_sensitivity_prebuilt.sh
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./scripts/prepare_fa2_sensitivity_h1d128_prebuilt.sh
```
