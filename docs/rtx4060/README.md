# RTX 4060 (SM89) support: calibration evidence

This branch adds an experimental Ada/SM89 simulator profile and the code paths exercised by the GEMM and FlashAttention validation below. It is based on `origin/flash` at `291fbe172332eb481556632a79eb9cfc93cf46bf`.

## Results

| Workload set | Cases | Output validation | Cycle MAPE | Compared metrics |
|---|---:|---:|---:|---|
| Paper GEMM shapes, FP16+BF16, selected baseline | 80 | 80/80 pass | 14.908347% | 5-run NCU median vs `gpu_tot_sim_cycle` |
| Paper GEMM shapes, E118 shared-reserve candidate | 80 | 80/80 pass | **14.739292%** | Same fixed NCU medians vs candidate cycles |
| Triton pointer-load FlashAttention forward | 8 | 8/8 pass | 10.157663% | NCU `sm__cycles_elapsed.avg` vs simulator cycles |
| Official FA1 + FA2 forward, B=8, S=128/256 | 16 | 16/16 pass | **16.382%** combined | NCU `sm__cycles_elapsed.avg` vs simulator cycles |

The workload sets use different kernels and must not be averaged together. The 80-case GEMM set uses 40 paper shapes with FP16 and BF16, fixed 64×64×32 tiles, four warps, three stages, FP32 accumulation, and no TMA. FA1 and FA2 select different kernels. Correctness replay passed, but cycle prediction remains imperfect and these measurements do not establish full Ada calibration or match the paper's 5.24% result.

The default SM89 config includes `-gpgpu_shmem_reserved_per_block 1024`, the E118 candidate with the best measured GEMM full-set MAPE so far. It improves the selected baseline by only 0.169 percentage points; it is not a complete timing-model fix.

## PR contents

- `configs/SM89_RTX4060/`: simulator configuration, SASS hint rules and scoped GEMM validation tables.
- `src/`: generic simulator changes needed for the tested SM89 replay paths, including packed BF16 conversion support and reserved shared-memory accounting.
- `tests/arch/sm89.toml`, `tests/src/fa2/`, and `tests/src/microbench/`: SM89 workload registration, FA2 output comparison hooks, and focused BF16/`ldmatrix` probes.
- `tools/TritonTrace/` and `tests/triton_trace/calibration/`: capture, NCU, replay and summary tooling for fixed GEMM cases.
- `docs/rtx4060/`: concise calibration reports, scope, and per-case result tables.

Upstream FA1 and CUTLASS source trees and exploratory per-probe CSV collections are not duplicated in this PR. The repository's existing FA2/FA3 dependency workflow fetches pinned source at build time. The FA1/FA2 cycle result table records the experiment's pinned FA1 version and measured values; full local source trees, detailed traces, all probe CSVs, binaries, and profiler reports remain in the backup branch and original experiment worktree.

## Reproduce the GEMM full set

In the documented environment with CUDA tooling, an RTX 4060 and a built simulator:

```bash
python3 tests/triton_trace/calibration/run_paper_gemm_fullset_4060.py \
  --cases tests/triton_trace/calibration/paper_gemm_cases_4060_fullset.csv \
  --repo "$PWD" \
  --progress reference/SM89_RTX4060/gemm/full_set/progress.jsonl
python3 tests/triton_trace/calibration/summarize_paper_gemm_fullset_4060.py
```

The manifest fixes dtype, shape, strides, tile, warps, stages, seed, NCU repetition and clock/cache controls, simulator config, and tolerance. Each case is captured, profiled five times, replayed, and output-validated. Run the batch with exclusive access to the GPU.

## Evidence files

- GEMM case definitions and 80-case baseline/E118 results: [`gemm/`](gemm/).
- Paper-case full-set manifest: `tests/triton_trace/calibration/paper_gemm_cases_4060_fullset.csv`.
- Official FA1/FA2 case manifest, cross-implementation numerical checks, and cycle comparisons: [`flash_attention/`](flash_attention/).
- Triton FlashAttention pointer-path cycle results: `flash_attention/triton_pointer_cycle_results.csv`.

This PR keeps final manifests and aggregate/per-case cycle tables. The detailed session chronology, large profiler `.ncu-rep` files, generated PTX/CUBIN/SASS, complete tensor arrays, native/simulator logs for every run, and intermediate diagnostic probe collections are retained locally in `backup/feat-RTX4060-support-full-20261119` and `tmp/rtx4060_experiments/dev_kleinc`; they are not required to review the implementation diff.
