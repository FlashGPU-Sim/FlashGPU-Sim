#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("CompiledCluster.*", 3, 2, group="compiled_cluster", cluster_validation=True)
