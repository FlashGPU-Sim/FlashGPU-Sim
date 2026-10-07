Remote mbarrier arrive. Rank 0 arrives through `mapa`. Rank 1 waits locally.

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile mbarrier-remote
```

Binary: `tests/build/bin/microbench/mbarrier_remote/mbarrier_remote_bench`.

Run it under `SM90_H200_CLUSTER132`. Set `FLASHGPU_ALLOW_CC_MISMATCH=1` when
the host is not CC 9.0. `--samples 1 --warmup 0` is a smoke run.
