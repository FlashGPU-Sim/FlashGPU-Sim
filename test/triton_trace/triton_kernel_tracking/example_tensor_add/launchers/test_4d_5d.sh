#!/bin/bash
# Test script for 4D and 5D TMA kernel launches
# Tests issue #31 fix for multi-dimensional TMA support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source GPGPU-Sim environment
source ~/Workspace/arch/gpgpu-sim_distribution/setup.sh
source ~/Workspace/arch/gpgpu-sim_distribution/setup_environment

echo "==================================="
echo "Testing 4D TMA Kernel Launch"
echo "==================================="
./kernel_add_4d_launch4
if [ $? -eq 0 ]; then
    echo "✓ 4D test PASSED"
else
    echo "✗ 4D test FAILED"
    exit 1
fi

echo ""
echo "==================================="
echo "Testing 5D TMA Kernel Launch"
echo "==================================="
./kernel_add_5d_launch5
if [ $? -eq 0 ]; then
    echo "✓ 5D test PASSED"
else
    echo "✗ 5D test FAILED"
    exit 1
fi

echo ""
echo "==================================="
echo "All tests PASSED!"
echo "==================================="
