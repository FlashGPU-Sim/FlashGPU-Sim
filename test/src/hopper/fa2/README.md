# FA2 Forward Cases

This directory contains standalone FlashAttention-2 forward gtests for
GPGPU-Sim bring-up. The CUDA kernel templates come from the existing
`test/src/hopper/fa3/flash-attention/csrc/flash_attn/src` submodule checkout.

## Cases

- 20 opt-in 32Ki-token prefill cases matching the FA3 shape table:
  `H32D64/H16D128 x full/causal x B,S`.
- One default small smoke case: `B=1`, `seqlen_q=128`, `seqlen_k=128`,
  `nheads=2`, `head_dim=64`, `dtype=fp16`, `causal=false`.

## Run

From `test/`:

```bash
./run_tests.sh build hopper
./run_tests.sh hopper Fa2FwdFp16SmokeIntegrationTest.SmallForwardCase
FA2_RUN_32KI=1 ./run_tests.sh hopper Fa2PrefillFp16IntegrationTest.H32D64FullB64S512
```
