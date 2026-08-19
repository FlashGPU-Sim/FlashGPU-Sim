#!/bin/bash

set -euo pipefail

launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTX_SIM_USE_PTX_FILE=1
export PTX_SIM_KERNELFILE="$launcher_dir/vector_ldst_v8_test.ptx"
export GPGPUSIM_SELECTED_PTX_OVERRIDE="$PTX_SIM_KERNELFILE"

exec "$launcher_dir/blackwell_tests.bin" "$@"
