#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("PtxSignedByteLoad.*", 1, 1, group="signed_load", frontends=("ptx",))
