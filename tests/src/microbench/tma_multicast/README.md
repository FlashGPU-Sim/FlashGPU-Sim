TMA multicast size sweep (256 B–16 KiB), destination skew from `%globaltimer`,
and a 4 KiB fan-out sweep.

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile tma-multicast
```

Binary: `tests/build/bin/microbench/tma_multicast/tma_multicast_bench`.

Run it under `SM90_H200_CLUSTER132`. Set `FLASHGPU_ALLOW_CC_MISMATCH=1` when
the host is not CC 9.0. `--samples 1 --warmup 0` is a smoke run.
