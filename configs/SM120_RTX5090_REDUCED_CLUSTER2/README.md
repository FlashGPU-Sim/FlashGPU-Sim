 # SM120_RTX5090_REDUCED_CLUSTER2 Configuration

 Lightweight GPU configuration for fast testing of 2-SM-per-cluster scheduling.

 This reduced configuration provides a minimal GPGPU-Sim setup with one cluster
 containing two SMs. It maintains fast simulation time while exercising the
 cluster round-robin CTA issuance path.

 ## Key Parameters

 - `-gpgpu_n_clusters 1`
 - `-gpgpu_n_cores_per_cluster 2`
 - `-gpgpu_n_mem 16`

 ## Usage

 ```bash
 ./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2 test ClusterBasic
 ```

 ## Notes

 - This is the preferred config for validating cluster/CTA=2 support.
 - Interconnect topology automatically adjusted to match SM count.
