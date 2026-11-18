#!/bin/bash

set -euo pipefail

launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTX_SIM_USE_PTX_FILE=1
export PTX_SIM_KERNELFILE="$launcher_dir/vector_ldst_v8_test.ptx"
export GPGPUSIM_SELECTED_PTX_OVERRIDE="$PTX_SIM_KERNELFILE"

# This is a PTX parser/execution/coalescing test, including ISA 8.8 forms
# that CI's CUDA 12.8 ptxas cannot assemble. Do not generate a SASS scheduling
# guide for these fixtures. Isolate the override so discovery and execution
# cannot change the configuration used by subsequent test groups.
config_dir="$PWD"
run_dir="$(mktemp -d "$config_dir/vector-ldst-v8.XXXXXX")"
cleanup() {
    local status=$?
    if (( status == 0 )); then
        rm -rf -- "$run_dir"
    else
        printf 'Vector-memory test artifacts retained in %s\n' "$run_dir" >&2
    fi
}
trap cleanup EXIT
cp "$config_dir/gpgpusim.config" "$run_dir/gpgpusim.config"
shopt -s nullglob
for config in "$config_dir"/*.icnt "$config_dir"/*.rules; do
    cp "$config" "$run_dir/"
done
printf '\n-gpgpu_ptx_reorder_sass_guided 0\n' >> "$run_dir/gpgpusim.config"
unset GPGPUSIM_SASS_PTXLINE_GUIDE
# Keep a caller's relative XML output path relative to its original directory.
if [[ ${GTEST_OUTPUT:-} == xml:* && ${GTEST_OUTPUT#xml:} != /* ]]; then
    export GTEST_OUTPUT="xml:$config_dir/${GTEST_OUTPUT#xml:}"
fi
cd "$run_dir"
"$launcher_dir/blackwell_tests.bin" "$@"
