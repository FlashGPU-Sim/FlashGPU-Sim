# SM90_H200_REDUCED_CLUSTER_HETERO3_2

Cheap mixed-GPC packing for Thread Block Cluster issue/barrier tests:
`-gpgpu_gpc_sms 3,2` (max m=3, n=2, total 5 SMs).

Not the functional default. Prefer `SM90_H200_REDUCED_CLUSTER16x2` for
Hopper cluster / DSM / TMA CI. Use this config when a test must see an
odd leftover SM on a GPC (cluster-of-2).

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh \
  -c SM90_H200_REDUCED_CLUSTER_HETERO3_2 run test --target sm120 --group integration \
  "*ClusterLaunch*"
```
