#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("GlobalReduce.*", 11, 11, group="global_reduce", config_overrides="-gpgpu_smem_store_visibility_latency 512")
