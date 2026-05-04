#!/usr/bin/env bash
#
# Sync GPU config files to all test launchers directories.
#
# Usage:
#   ./sync_config.sh                    # sync to all tests
#   ./sync_config.sh test_tma_gemm      # sync to specific test
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_TRACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRITON_TRACE_DIR/../.." && pwd)"
TRACKING_DIR="$TRITON_TRACE_DIR/triton_kernel_tracking"
GPU_CONFIG_DIR="$REPO_ROOT/configs/SM120_RTX5090"

CONFIG_FILES=("gpgpusim.config" "config_ampere_islip.icnt")

# Verify source configs exist
for f in "${CONFIG_FILES[@]}"; do
    if [[ ! -f "$GPU_CONFIG_DIR/$f" ]]; then
        echo "ERROR: Source config not found: $GPU_CONFIG_DIR/$f"
        exit 1
    fi
done

# Determine which tests to sync
if [[ $# -gt 0 ]]; then
    TESTS=("$@")
else
    TESTS=()
    for d in "$TRACKING_DIR"/*/; do
        [[ -d "$d" ]] && TESTS+=("$(basename "$d")")
    done
fi

count=0
for test_name in "${TESTS[@]}"; do
    test_dir="$TRACKING_DIR/$test_name"
    if [[ ! -d "$test_dir" ]]; then
        echo "[WARN] Test directory not found: $test_name, skipping"
        continue
    fi

    while IFS= read -r -d '' launcher_dir; do
        for f in "${CONFIG_FILES[@]}"; do
            cp "$GPU_CONFIG_DIR/$f" "$launcher_dir/"
        done
        count=$((count + 1))
    done < <(find "$test_dir" -type d -name launchers -print0)
done

echo "Synced ${#CONFIG_FILES[@]} config files to $count launcher directories."
