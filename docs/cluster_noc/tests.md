# Tests

Always:

```bash
source ./setup_environment
make FLASH=1 -j$(nproc)
```

Use `./tests/run_tests.py`. Do not invoke test binaries by hand.
Default config remains `SM120_RTX5090`. `CUDA_INSTALL_PATH` must already
point at the CUDA toolkit (`/usr/local/cuda` on the usual machine).

---

## Functional cluster / DSM / TMA

Unit tests build their own topology. The active GPU config does not matter:

```bash
./tests/run_tests.py -c SM120_RTX5090 run --arch sm120 --group unit \
  --gtest-filter 'GpuTopology*:DsmFabric*:DsmEndpoint*:Transport*:SmemService*:ClusterHang*:TbClusterAddr*'
```

Integration cases skip on `SM120_RTX5090` (one SM per GPC, DSM off), except
the launch that must fail when the cluster is larger than the GPC. The
shipped config that can run the rest is `SM90_H200_CLUSTER132`. That run is
slow and is not default CI. Set `FLASHGPU_ALLOW_CC_MISMATCH=1` when the host
GPU is not CC 9.0.

```bash
./tests/run_tests.py -c SM90_H200_CLUSTER132 run --arch sm90 --group integration \
  --gtest-filter '*ClusterLaunch*:*ClusterBasic*:*ClusterReal*:*TMACluster*:*TmaMulticast*:*DsmTest*:*MbarrierCluster*:*MultiCluster*'
```

Existing-feature regression (WGMMA / FA / MMA) is the `flash` PR CI gate, not
this list. See `tests/ci/run_ci_tests.sh` (`sm120-core`, `sm90-core`,
`sm90-fa2`, `sm90-fa3`).

Hetero leftover-SM packing is CLUSTER132-only.

---

## Topology skips

Helpers: `tests/common/gpgpusim_config_topology.h`. Each skip prints
`WARNING: skipped Suite.Test: ...` on stdout and stderr, then `GTEST_SKIP`.

| Macro | Skip when |
|-------|-----------|
| `SKIP_IF_N_CORES_PER_CLUSTER_LT(2)` | m &lt; 2 |
| `SKIP_IF_N_CLUSTERS_LT(2)` | n &lt; 2 |
| `SKIP_IF_NOT_HETERO_GPC()` | uniform `-gpgpu_gpc_sms` / none |
| `SKIP_IF_CLUSTER_NOC_OFF()` | `-gpgpu_dsm_enable` is not 1 |

---

## Suites

| Suite | Role |
|-------|------|
| `cluster_launch_api_test` | Ex launch / attrs, cluster.sync |
| `tma_cluster_multicast_test` | Peer data + mbar complete |
| `tma_multicast_mask_test` | `ctaMask` |
| `cluster_multicast_multicluster_test` | Isolation |
| `dsm_test` | SelfMapa, remote store/load/atom, drop-on-exit |
| `mbarrier_cluster_test` | Remote arrive / try_wait / expect+complete |
| `cluster_real_shaped_test` | TMA accumulate |
| unit `tb_cluster_test` | Generic shared-window decode |
| unit `dsm_fabric_test` / `dsm_endpoint_test` | Flits, VCs, ACK, shaper |

`barrier.cluster.arrive` / `barrier.cluster.wait` wait for every active warp
in every CTA of the reserved TB-cluster.

TMA multicast is functional and has no DSM-fabric callsite
(`GpuTopology.TmaMulticastDoesNotUseDsmFabric`).

---

## H200 microbenchmarks

See [`calibration.md`](calibration.md). Build them with
`./tests/run_tests.py build --arch sm90 --group microbench --profile <name>`.
Profiles: `dsm-bw`, `tma-bw`, `dsm-latency`, `tma-multicast`, `mbarrier-remote`,
and gtest `tma-latency`. Not default CI.
