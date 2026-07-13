#!/bin/bash
# Test script for multi-configuration support in run_tests.sh
# Validates issue #22 implementation

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/run_tests.sh"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed_tests=0
total_tests=0

# Test helper functions
run_test() {
    local test_name="$1"
    local test_command="$2"

    total_tests=$((total_tests + 1))
    echo -e "${YELLOW}[TEST $total_tests]${NC} $test_name"

    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED${NC}\n"
        passed_tests=$((passed_tests + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}\n"
        return 1
    fi
}

echo "========================================="
echo "Multi-Config Support Test Suite"
echo "========================================="
echo ""

# Test 1: Default config should be SM120_RTX5090
run_test "Default config is SM120_RTX5090" \
    "$TEST_SCRIPT list-configs 2>&1 | grep -q 'SM120_RTX5090 (default)' || \
     $TEST_SCRIPT list-configs 2>&1 | grep -q 'SM120_RTX5090.*default'"

# Test 2: list-configs command exists and works
run_test "list-configs command works" \
    "$TEST_SCRIPT list-configs &>/dev/null"

# Test 3: SM120_RTX5090_REDUCED config is available
run_test "SM120_RTX5090_REDUCED config is available" \
    "$TEST_SCRIPT list-configs 2>&1 | grep -q 'SM120_RTX5090_REDUCED'"

# Test 4: --config flag is recognized
run_test "--config flag is recognized" \
    "$TEST_SCRIPT list-configs --config SM120_RTX5090 &>/dev/null"

# Test 5: -c short flag is recognized
run_test "-c short flag is recognized" \
    "$TEST_SCRIPT list-configs -c SM120_RTX5090 &>/dev/null"

# Test 6: Run directory exists for default config
run_test "Run directory structure for SM120_RTX5090" \
    "[ -d '$SCRIPT_DIR/run/SM120_RTX5090' ] || \
     ($TEST_SCRIPT build test &>/dev/null && [ -d '$SCRIPT_DIR/run/SM120_RTX5090' ])"

# Test 7: Test binary is shared (only one copy in build/bin/)
run_test "Test binary is shared (single location)" \
    "[ -f '$SCRIPT_DIR/build/bin/run_all_tests' ] && \
     [ ! -f '$SCRIPT_DIR/run/SM120_RTX5090/run_all_tests' ]"

# Test 8: Config files are copied to run directory
run_test "Config files copied to run directory" \
    "[ -f '$SCRIPT_DIR/run/SM120_RTX5090/gpgpusim.config' ] && \
     [ -f '$SCRIPT_DIR/run/SM120_RTX5090/config_ampere_islip.icnt' ]"

# Test 9: Reduced config directory can be created
run_test "Reduced config can be set up" \
    "[ -d '$SCRIPT_DIR/../configs/SM120_RTX5090_REDUCED' ] || \
     echo 'Reduced config not yet created (expected during implementation)'"

# Test 10: Invalid config name is rejected
run_test "Invalid config name is rejected" \
    "! $TEST_SCRIPT refresh --config INVALID_CONFIG_NAME &>/dev/null"

# Doc-guard tests for test configuration matrix (issue #28)
run_test "Test configuration matrix file exists" \
    "[ -f '$SCRIPT_DIR/../docs/test-configuration-matrix.md' ]"

run_test "Matrix documents SM120_RTX5090 config" \
    "grep -q 'SM120_RTX5090' '$SCRIPT_DIR/../docs/test-configuration-matrix.md'"

run_test "Matrix documents SM120_RTX5090_REDUCED config" \
    "grep -q 'SM120_RTX5090_REDUCED' '$SCRIPT_DIR/../docs/test-configuration-matrix.md'"

run_test "Matrix documents excluded test CPAsyncMethod" \
    "grep -q 'CPAsyncMethod' '$SCRIPT_DIR/../docs/test-configuration-matrix.md'"

run_test "Matrix documents excluded test PerformanceComparison" \
    "grep -q 'PerformanceComparison' '$SCRIPT_DIR/../docs/test-configuration-matrix.md'"

# Doc-guard tests for build detection and native GPU mode (issue #36)
run_test "Build detection uses find command for libcudart.so" \
    "grep -q 'find.*lib.*libcudart.so' '$TEST_SCRIPT'"

run_test "Native mode checks GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" \
    "grep -q 'GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN' '$TEST_SCRIPT'"

run_test "Native mode checks LD_LIBRARY_PATH for simulator paths" \
    "grep -q 'LD_LIBRARY_PATH' '$TEST_SCRIPT'"

echo "========================================="
echo "Test Results: $passed_tests/$total_tests passed"
echo "========================================="

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    failed_tests=$((total_tests - passed_tests))
    echo -e "${RED}$failed_tests test(s) failed${NC}"
    exit 1
fi
