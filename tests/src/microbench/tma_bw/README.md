Source: https://github.com/seanzw/random commit `4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd` (`tma_bw`).
Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

Global-memory bandwidth for ordinary loads, `cp.async`, and TMA. The profile
builds the H200 L2-resident binary (40 MiB) and the cold-HBM binary (1 GiB).

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile tma-bw
```

Binaries:

- `tests/build/bin/microbench/tma_bw/bandwidth_test_h200_l2`
- `tests/build/bin/microbench/tma_bw/bandwidth_test_h200_hbm`
