# SM120_RTX5090_REDUCED_CLUSTER2x2

Multi-cluster reduced config for development iteration with cluster-aware features.

- 2 clusters × 2 SMs/cluster = 4 SMs total
- Useful for testing multi-cluster isolation and cluster-aware scheduling
- Identical to `SM120_RTX5090_REDUCED_CLUSTER2` but with 2 clusters instead of 1
- `-network_mode 2` (local interconnect; BookSim icnt unused / not sized for 4 SMs)

Intended for fast iteration; not representative of full-scale performance.

## Limitations

- Plain grid launches only — validates multi-cluster **topology**, not CUDA
  cooperative Thread Block Clusters.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
