#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("ClusterStore.*", 3, 17, group="cluster_store", cluster_validation=True)
    run_suite("PtxVectorOrder.*", 1, 1, group="vector_order")
