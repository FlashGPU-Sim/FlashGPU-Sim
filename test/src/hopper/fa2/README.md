# FA2 Forward Cases

This directory contains standalone FlashAttention-2 forward gtests for
GPGPU-Sim bring-up. The CUDA kernel templates come from the existing
`test/src/hopper/fa3/flash-attention/csrc/flash_attn/src` submodule checkout.

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
the Hopper default `HOPPER_CUDA_ARCH=sm_90a`. Sensitivity runners are built by
the separate `hopper-fa2-sensitivity` target and do not change the normal FA2
smoke/small/medium/large binaries.

## Run

From `test/`:

```bash
./run_tests.sh build hopper-fa2
./run_tests.sh build hopper-fa2-smoke
./run_tests.sh build hopper-fa2-large-h16d128-full
./run_tests.sh build hopper-fa2-sensitivity
./run_tests.sh hopper Fa2PrefillFp16SmokeTest
./run_tests.sh hopper Fa2PrefillFp16SmallTest
./run_tests.sh hopper Fa2PrefillFp16MediumTest
./run_tests.sh hopper Fa2PrefillFp16SensitivityTest
FA2_RUN_32KI=1 ./run_tests.sh hopper Fa2PrefillFp16IntegrationTest.H32D64FullB64S512
```

To prepare a CUDA 12.8 prebuilt bundle for H100 NCU collection:

```bash
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./scripts/prepare_fa2_sensitivity_prebuilt.sh
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./scripts/prepare_fa2_sensitivity_h1d128_prebuilt.sh
```
