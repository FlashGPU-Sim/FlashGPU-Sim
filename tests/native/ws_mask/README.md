# B200 WS mask/collector hardware qualification

Build locally with CUDA 13.3 and a pinned CUTLASS checkout:

```
OUT=/absolute/new/package CUTLASS_ROOT=/path/to/cutlass \
GTEST_ROOT=/path/to/googletest CUDA_INSTALL_PATH=/usr/local/cuda-13.3 \
bash tests/native/ws_mask/build.sh
```

On one B200, unpack the package and run:

```
python3 collect.py --out /absolute/new/results --ncu /path/to/ncu
```

No compilation is needed on the server. The package bundles real CUDA cudart;
its collector clears simulator/preload variables. Driver/libstdc++ compatibility
is still required. The collector neither sets application clocks nor shuts down
the server. Native samples use the current GPU clock policy; NCU explicitly uses
`--clock-control base`. NCU replay timestamps never enter native timing summaries.

Numerics run first: eleven tests, FP16/BF16, N64/128/256, shared/TMEM A,
transposed B, masks changed after fill, 8/16/32-column shifts, all four collector
slots, two CTAs and two rounds. CUTLASS SW128 descriptors/layouts replace the
hand-written nonswizzled test layout. Entire shared allocations are initialized.
The oracle evaluates logical A/B independently of either simulator or CUTLASS
address arithmetic. All eleven must pass before performance collection starts.
This is a separately built native fixture, not the old CI cubin.

Timing fixes M64/N128/K16 BF16, K-major B, one owner, one accumulator. Matrix:

- shared-A and TMEM-A;
- discard, fill, use, alternating fill/lastuse;
- maximum/actual shift (0,0), (8,8), (16,16), (32,32), (32,0);
- compile-time unrolled bursts 0/1/32/128;
- mixed/all-zero masks for fill/use with the 32-column window.

176 cases have three native repetitions, each with four warmup rounds and 32
retained samples. All modes perform the same untimed warm fill before each burst.
No TMA/global traffic is inside the timed region. Separate issue and completion
markers distinguish issue backpressure from commit/wait/fence cost. Completion
is not backend-only latency. Burst zero measures control overhead and retains
the prefill result. `fill/lastuse` uses one MMA per step, alternating operations;
burst one contains only a fill. Each timing run checks all 8192 output values.
No claim is made about cold collector latency, multiple independent owners,
other N/type timing, or contention with ordinary TMEM traffic.

All 48 burst-128 cases get NCU `--set full` with eight retained rounds; the eleven
numerical kernels get another full report. Raw logs, commands, `.ncu-rep`, CSV,
GPU state, binaries and source hashes are retained. `timing.csv` and `slopes.json`
use only numerically passing, unprofiled samples. The 32-to-128 slope divides by
96 actual MMA instructions. An unsuccessful native validation is a failure,
not a timing result. Use `--dry-run` to inspect the complete matrix without GPU.
