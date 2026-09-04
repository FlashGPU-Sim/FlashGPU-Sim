# SM90_H200_REDUCED_CLUSTER16x2

Cluster / DSM design: `docs/cluster_noc/README.md`. Checklist: `docs/cluster_noc/todos.md`.

Reduced two-GPC **H200 (Hopper + HBM3e)** config. Each GPC has **16 enabled
SMs** on **3 CPCs × 6 slots** (2 PG'd), which is the uniform stand-in for an
H200 GPC (full die 18 SMs/GPC, product ~16.5). Two GPCs catch isolation and
cross-GPC reject bugs before the full 132-SM map.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=16** | Enabled SMs per GPC | `-gpgpu_n_cores_per_cluster` / `-gpgpu_num_sms_per_gpc` |
| **n=2** | Number of GPCs | `-gpgpu_n_clusters` / `-gpgpu_num_gpcs` |

Total SMs = m×n = **32**. **TB cluster size is not “16×2”** — it comes from the
launch API. Capacity rule: `product(clusterDim) ≤ m` (TB cluster size ≤ 16).

## Hardware baseline

Same GH100 compute as full `SM90_H200` (cc 9.0, Hopper SM, TMA, WGMMA, 50 MiB
L2 model, 80 HBM channels). Clocks and latency knobs match `SM90_H200`
(H200 NVL job **2034797**). Hop matrix is the 16×16 H200 one-way file (job
**2046238**). Only SM packing is reduced.

## Intended use

- Intra-GPC NoC (CPC / GPCMMU / two-GPC isolation)
- Co-residency of TB-cluster launches with clusterDim up to 16
- Primary **functional** H200 regression (prefer over full 132 SM)
- Hopper TMA / mbarrier / DSM; the NoC models DSM only

- `-network_mode 2` (local interconnect; BookSim icnt unused / not sized for 32 SMs)
- `-gpgpu_tma_idealized_memory 1` (functional TMA path)
- `-gpgpu_dsm_store_immediate 0` (peer DSM `st` visible only after NoC deliver; matches the code default)
- `-gpgpu_dsm_cpcs_per_gpc 3`

## Usage

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh \
  -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

## Latency

Synced with `SM90_H200` / jobs 2034797 and 2046238. Absolute performance still
differs from full 132-SM packing; use full `SM90_H200` for scale studies.

## Limitations

- Ordinary grid launches: no co-residency guarantee (validates isolation).
- Cluster launches (`cudaLaunchKernelEx` / required cluster dims): CTAs of one
  TB cluster are co-scheduled on one GPC.
- Functional correctness focus; DSM timing is still the delay-line until the
  fabric lands.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
- Not a substitute for full `SM90_H200` when measuring absolute performance.
