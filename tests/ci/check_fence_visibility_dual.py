#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("SharedFenceVisibility.*", 2, 2, group="fence_visibility",
              config_overrides="-gpgpu_smem_store_visibility_latency 512")
