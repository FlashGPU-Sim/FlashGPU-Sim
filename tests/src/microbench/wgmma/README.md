# WGMMA Microbenchmarks

The WGMMA microbenchmarks are gtest binaries built by the top-level
`tests/Makefile`.

- `wgmma_async_latency_bench.cc`: broad WGMMA shape/group sweeps, FA3-like
  timing probes, and mixed WGMMA/softmax probes.
- `wgmma_softmax_mix_bench.cc`: focused same-warpgroup FA3-shape
  `m64n176k16.ss`/`m64n128k16.rs` plus softmax-like FP/SFU mixing probes.
- `wgmma_fp16_{ss,rs}_g{1,2,4}_bench.cc`: split FP16 core sweeps for
  independent compilation by operand kind and groups-before-wait.
- `wgmma_n16_chain_bench.cc`: focused `m64n16k16` chain/count sweeps.

The old standalone mixed-softmax file was removed because the same probes now
live in `wgmma_async_latency_bench.cc`.

Build examples from `tests/`:

```bash
./run_tests.py build --arch sm90 --group microbench --profile wgmma
./run_tests.py run --arch sm90 --group microbench --profile wgmma \
  WgmmaN16Chain
```

FP16 WGMMA core sweep:

```bash
make -j6 wgmma-fp16-core

WGMMA_FP16_SWEEP_OUT_PREFIX=/tmp/wgmma_fp16_ss_g1 \
  ./build/bin/wgmma/wgmma_fp16_ss_g1_bench \
  --gtest_filter=WgmmaFp16CoreSweep.Selected
```

The unified sweep writes one CSV row per `(shape, operand, accumulator,
groups_before_wait, ops_per_group)` case. The six split binaries are:

- `wgmma_fp16_ss_g1_bench`
- `wgmma_fp16_ss_g2_bench`
- `wgmma_fp16_ss_g4_bench`
- `wgmma_fp16_rs_g1_bench`
- `wgmma_fp16_rs_g2_bench`
- `wgmma_fp16_rs_g4_bench`

Across the split binaries, the default core matrix is:

- `shape`: `m64n64k16`, `m64n128k16`, `m64n176k16`, `m64n192k16`
- `operand`: `ss`, `rs`
- `groups_before_wait`: `1`, `2`, `4`
- `ops_per_group`: `1`, `2`, `4`, `8`, `16`, `32`
- `accumulator`: `same` and `rot2` for all four shapes, plus `rot4` for
  `n64`

Use `WGMMA_FP16_SWEEP_FILTER` as a substring filter for one case or a smaller
batch, for example:

```bash
WGMMA_FP16_SWEEP_FILTER=n128_g1_o16_same \
  ./build/bin/wgmma/wgmma_fp16_ss_g1_bench \
  --gtest_filter=WgmmaFp16CoreSweep.Selected
```

Focused WGMMA/softmax mixing probe:

```bash
make -j4 wgmma-softmax-mix

WGMMA_SOFTMAX_MIX_ROUNDS=4096 \
WGMMA_SOFTMAX_MIX_OUT_PREFIX=/tmp/wgmma_softmax_mix \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.Sweep
```

For one NCU report per kernel, use `Selected`:

```bash
WGMMA_SOFTMAX_MIX_SELECTED=qk_wait_math_pv \
WGMMA_SOFTMAX_MIX_MATH_ITERS=1 \
WGMMA_SOFTMAX_MIX_ROUNDS=4096 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
ncu --set full --target-processes all --force-overwrite \
  -o ncu_wgmma_softmax_qk_wait_math_pv_m1 \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.Selected
```

Supported selected modes are `qk_only`, `pv_only`, `qk_pv_only`,
`math_only`, `qk_wait_math`, `qk_math_wait`, `qk_wait_math_pv`, and
`qk_math_wait_pv`. The FA3-like default uses `qk_ops=8` and `pv_ops=11` per
round.

For the local overlap probe, use one QK WGMMA followed by math before/after
the wait:

```bash
WGMMA_SOFTMAX_MIX_QK_OPS=1 \
WGMMA_SOFTMAX_MIX_ROUNDS=4096 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
WGMMA_SOFTMAX_MIX_OUT_PREFIX=/tmp/wgmma_local_overlap \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.LocalOverlapSweep
```

For a finer WGMMA-followed-by-small-math sweep, use:

```bash
WGMMA_SOFTMAX_MIX_QK_OPS=8 \
WGMMA_SOFTMAX_MIX_ROUNDS=8192 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
WGMMA_SOFTMAX_MIX_OUT_PREFIX=$HOME/wgmma_fine_math \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.FineMathSweep
```

`FineMathSweep` compares `math_only`, `qk_wait_math`, and `qk_math_wait` for
small per-round math blocks. It sweeps `math_ops={0,1,2,4,8,16,24,32,64}` and
the following `math_kind` values:

- `add_chain`: one dependent `add.f32` per op
- `mul_chain`: one dependent `mul.f32` per op
- `fma_chain`: one dependent `fma.f32` per op
- `max_chain`: one dependent `max.f32` per op
- `ex2`: one `ex2.approx.f32` per op
- `fma_indep8`: eight independent `fma.f32` instructions per op
- `mix4`: `max/sub/ex2/fma` per op

For the WGMMA-pending slowdown model, use the dense overlap sweep:

```bash
WGMMA_SOFTMAX_MIX_QK_OPS=8 \
WGMMA_SOFTMAX_MIX_ROUNDS=8192 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
WGMMA_SOFTMAX_MIX_OUT_PREFIX=$HOME/wgmma_overlap_model_dense \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.FineOverlapModelDenseSweep
```

`FineOverlapModelDenseSweep` uses `qk_ops=8` by default and sweeps
`math_ops={0..16}` for:

- `ex2`: dependent SFU chain
- `ex2_indep2`: two independent SFU chains per op
- `ex2_indep4`: four independent SFU chains per op
- `fma_indep8`: eight independent FP32 FMA instructions per op
- `int_add_chain`: one dependent integer add per op
- `int_add_indep8`: eight independent integer adds per op
- `lop3_indep8`: eight independent integer logic instructions per op
- `mix4`: `max/sub/ex2/fma` per op

To check whether the pending window scales with the number of QK WGMMA
instructions, use:

```bash
WGMMA_SOFTMAX_MIX_ROUNDS=8192 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
WGMMA_SOFTMAX_MIX_OUT_PREFIX=$HOME/wgmma_qk_ops_model \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.FineQkOpsModelSweep
```

`FineQkOpsModelSweep` sweeps `qk_ops={1,2,4,8,16}` and
`math_ops={0,4,8,12,16}` for dependent FP32, independent SFU/FP32, integer,
logic, and mixed softmax-like math kinds. Keep the output prefix under
`$HOME` or another persistent directory when collecting remote H100 data.

For a single NCU-friendly kernel, use `Selected` with
`WGMMA_SOFTMAX_MIX_MATH_KIND`, for example:

```bash
WGMMA_SOFTMAX_MIX_SELECTED=qk_math_wait \
WGMMA_SOFTMAX_MIX_MATH_KIND=fma_chain \
WGMMA_SOFTMAX_MIX_MATH_ITERS=16 \
WGMMA_SOFTMAX_MIX_QK_OPS=8 \
WGMMA_SOFTMAX_MIX_ROUNDS=8192 \
WGMMA_SOFTMAX_MIX_WARMUP=0 \
  ./build/bin/wgmma/wgmma_softmax_mix_bench \
  --gtest_filter=WgmmaSoftmaxMixBench.Selected
```
