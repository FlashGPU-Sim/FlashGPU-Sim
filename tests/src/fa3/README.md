# FA3 Forward and Backward Cases

This directory contains standalone FlashAttention-3 Hopper forward and
backward validation workloads for FlashGPU-Sim.

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

The pinned dependency and patch bundle are documented in the
[FlashAttention dependency guide](../../third_party/flash-attention/README.md).

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

The kernels target `sm_90a`; native execution requires Hopper hardware.

Smoke, packgqa, size, and sensitivity profiles are all exposed through the
`sm90/fa3` test group. A filter can select an individual GoogleTest case
without escaping the selected profile.

## Notes

The fixed workload uses `ClusterM = 1` and exercises the local-CTA case of
cluster-scoped operations.
