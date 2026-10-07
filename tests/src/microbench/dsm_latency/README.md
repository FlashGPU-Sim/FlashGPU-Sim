DSM latency matrix, dependent remote-load round trip, store-visibility floor,
and one-pair versus all-pairs contention. The bandwidth sweep is not run.

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile dsm-latency
```

Binary: `tests/build/bin/microbench/dsm_latency/dsm_latency_bench`.

DSM measurement needs `-c SM90_H200_CLUSTER132` copied into the run directory
the same way the other standalone benches do. Set
`FLASHGPU_ALLOW_CC_MISMATCH=1` when the host is not CC 9.0. `--samples 1
--warmup 0` is a smoke run. The default is 48 samples and 8 warmup iterations.
