 # SM120_RTX5090_CLUSTER2 Configuration

 Full-scale RTX 5090 configuration with 2 SMs per cluster.

 This variant matches the total SM count of the standard RTX 5090 config
 (170 SMs) but groups them into 85 clusters with 2 SMs each, enabling
 cluster-level features such as distributed shared memory and TMA multicast.

 ## Key Parameters

 - `-gpgpu_n_clusters 85`
 - `-gpgpu_n_cores_per_cluster 2`

 ## Usage

 ```bash
 ./test/run_tests.sh -c SM120_RTX5090_CLUSTER2 test ClusterBasic
 ```

 Use this configuration to validate cluster/CTA=2 scheduling and future
 cluster-level instructions.
