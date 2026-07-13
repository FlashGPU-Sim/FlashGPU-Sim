#!/bin/bash

# GPGPU-Sim Test Runner Script
# This script provides advanced test execution capabilities

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load configuration
if [ -f "test.config" ]; then
    source test.config
fi

# Default values
TEST_TIMEOUT=${TEST_TIMEOUT:-3600}
TEST_VERBOSE=${TEST_VERBOSE:-1}
DEBUG_TESTS=${DEBUG_TESTS:-0}
HOPPER_BUILD_JOBS=${HOPPER_BUILD_JOBS:-${FA2_BUILD_JOBS:-4}}
DEFAULT_GPU_CONFIG=${DEFAULT_GPU_CONFIG:-SM120_RTX5090}
GPU_CONFIG_EXPLICIT=0
if [ -n "${GPU_CONFIG:-}" ]; then
    GPU_CONFIG_EXPLICIT=1
fi
GPU_CONFIG=${GPU_CONFIG:-$DEFAULT_GPU_CONFIG}  # Default GPU configuration
REQUESTED_TARGET=""

# Resolved from the Makefile-owned suite/target registry.
ACTIVE_SUITE=""
ACTIVE_TARGET=""
ACTIVE_BUILD_GROUP=""
ACTIVE_BINARY_GROUP=""
ACTIVE_EXECUTOR=""
ACTIVE_DEFAULT_CONFIG=""
ACTIVE_REQUIRED_CC=""
ACTIVE_CUDA_ARCH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print and execute command
run_command() {
    local cmd="$*"
    print_color $YELLOW "⚡ Running: $cmd"
    "$@"
}

# Print usage
usage() {
    echo "GPGPU-Sim Test Runner"
    echo "Usage: $0 [OPTIONS] ACTION SUITE [FILTER] [OPTIONS]"
    echo ""
    echo "Actions and suites:"
    echo "  build test [--target default|sm90]"
    echo "  run test [filter] [--target default|sm90]"
    echo "  build fa --target fa2|fa3"
    echo "  run fa [filter] --target fa2|fa3"
    echo "  build microbench [--target default|sm90]"
    echo "  run microbench [filter] [--target default|sm90]"
    echo "  build dev"
    echo "  run dev [filter]"
    echo "  build trace"
    echo "  run trace [filter]"
    echo ""
    echo "Other commands:"
    echo "  clean              Clean build artifacts"
    echo "  setup              Setup test environment"
    echo "  refresh            Refresh run directory and configuration"
    echo "  list               List available tests"
    echo "  list-configs       List available GPU configurations"
    echo "  help               Show this help"
    echo ""
    echo "Options:"
    echo "  -v, --verbose      Verbose output"
    echo "  -d, --debug        Enable debug mode"
    echo "  -t, --timeout      Set test timeout (seconds)"
    echo "  -c, --config NAME  Use specific GPU configuration"
    echo "  --target NAME      Select one target within a suite"
    echo "  -h, --help         Show this help"
    echo "  Options may appear before or after the command."
    echo ""
    echo "Examples:"
    echo "  $0 run test"
    echo "  $0 run test '*MMAS8*' -c SM120_RTX5090_REDUCED"
    echo "  $0 run test --target sm90 '*WgmmaF16*'"
    echo "  $0 run fa --target fa2 'Fa2PrefillFp16SmokeTest*'"
    echo "  $0 run fa --target fa3 'Fa3FwdHdim128Fp16IntegrationTest*'"
    echo "  $0 run microbench --target sm90 '*Wgmma*'"
}

# Setup test environment
setup_environment() {
    print_color $BLUE "Setting up GPGPU-Sim test environment..."

    # Check prerequisites
    if ! command -v g++ &> /dev/null; then
        print_color $RED "Error: g++ compiler not found"
        exit 1
    fi

    if ! command -v make &> /dev/null; then
        print_color $RED "Error: make not found"
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        print_color $RED "Error: git not found"
        exit 1
    fi

    # Setup Google Test
    print_color $BLUE "Setting up Google Test..."
    run_command make setup-gtest

    # Setup run directory and configuration
    setup_run_directory

    print_color $GREEN "Environment setup complete!"
}

# Setup run directory with configuration
setup_run_directory() {
    local force_refresh="$1"
    local config_name="${GPU_CONFIG}"

    print_color $YELLOW "Setting up test run directory for config: $config_name..."

    local config_dir="run/$config_name"
    local source_config="../configs/$config_name"

    # Create run directory if it doesn't exist
    if [ ! -d "run" ]; then
        mkdir -p run
        print_color $BLUE "Created run directory"
    fi

    # Validate that source config exists
    if [ ! -d "$source_config" ]; then
        print_color $RED "Error: Configuration '$config_name' not found at $source_config"
        print_color $YELLOW "Available configs:"
        list_configs
        exit 1
    fi

    # Always sync configuration from source
    run_command cp -r "$source_config" run/
    print_color $GREEN "Synced $config_name configuration to run directory"
}

# Detect if running in native GPU mode (clean environment without simulator setup)
is_native_mode() {
    local root_dir="$(cd "$SCRIPT_DIR/.." && pwd)"

    # Check if simulator environment was sourced
    if [ -n "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
        return 1  # Simulator mode
    fi

    # Check if LD_LIBRARY_PATH contains simulator library paths from this repo
    if echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -F -q "$root_dir/lib/"; then
        print_color $YELLOW "Warning: LD_LIBRARY_PATH contains simulator paths but GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN is not set"
        print_color $YELLOW "Treating as simulator mode to avoid contamination"
        return 1  # Simulator mode (contaminated environment)
    fi

    return 0  # Native GPU mode
}

# Build GPGPU-Sim library
build_gpgpusim() {
    # Skip simulator build in native GPU mode
    if is_native_mode; then
        print_color $BLUE "Native GPU mode detected - skipping GPGPU-Sim library build"
        return 0
    fi

    print_color $BLUE "Building GPGPU-Sim library..."
    local root_dir="$(cd "$SCRIPT_DIR/.." && pwd)"

    cd "$root_dir"

    # Resolve actual libcudart.so path (fix for broken glob pattern)
    local libcudart_path=$(find lib -name libcudart.so 2>/dev/null | head -n 1)

    if [ -z "$libcudart_path" ] || [ ! -s "$libcudart_path" ] || find src -name "*.cc" -newer "$libcudart_path" 2>/dev/null | grep -q .; then
        print_color $YELLOW "GPGPU-Sim library needs rebuild..."
        run_command make FLASH=1 -j
        print_color $GREEN "GPGPU-Sim library built successfully"
    else
        print_color $GREEN "GPGPU-Sim library is up to date"
    fi
    cd "$SCRIPT_DIR"
}

build_make_group() {
    local make_target="$1"
    local cuda_arch="$2"
    local parallel="${3:-0}"
    local -a make_args=(make)

    build_gpgpusim
    print_color $BLUE "Building $ACTIVE_SUITE/$ACTIVE_TARGET via '$make_target' (CUDA arch: $cuda_arch)..."

    if [ "$parallel" -eq 1 ]; then
        make_args+=("-j$HOPPER_BUILD_JOBS")
    fi
    if [ "$DEBUG_TESTS" -eq 1 ]; then
        make_args+=("CXXFLAGS=-std=c++17 -Wall -Wextra -pthread -g -O0 -DDEBUG")
    fi
    make_args+=("CUDA_ARCH=$cuda_arch" "HOPPER_CUDA_ARCH=$cuda_arch" "$make_target")

    if run_command "${make_args[@]}"; then
        print_color $GREEN "Build successful!"
    else
        print_color $RED "Build failed!"
        return 1
    fi
}

build_active_target() {
    if [ "$ACTIVE_EXECUTOR" = "trace" ]; then
        build_gpgpusim
        run_command make -C src/trace ARCH="$ACTIVE_CUDA_ARCH" GPU_CONFIG="$GPU_CONFIG"
        return $?
    fi

    local parallel=0
    if [ "$ACTIVE_REQUIRED_CC" = "9.0" ]; then
        parallel=1
    fi
    build_make_group "$ACTIVE_BUILD_GROUP" "$ACTIVE_CUDA_ARCH" "$parallel"
}

# Query the Makefile-owned binary manifest for one test group.
test_group_binaries() {
    local group="$1"
    make -s --no-print-directory print-test-binaries TEST_GROUP="$group"
}

single_test_group_binary() {
    local group="$1"
    local binaries=""
    local count=0

    if ! binaries="$(test_group_binaries "$group")"; then
        print_color $RED "Unable to resolve test group: $group" >&2
        return 1
    fi

    count="$(printf '%s\n' "$binaries" | sed '/^[[:space:]]*$/d' | wc -l)"
    if [ "$count" -ne 1 ]; then
        print_color $RED "Expected one binary for group '$group', found $count" >&2
        return 1
    fi

    printf '%s\n' "$binaries"
}

suite_list() {
    make -s --no-print-directory list-suites
}

suite_target_list() {
    local suite="$1"
    make -s --no-print-directory list-suite-targets SUITE="$suite"
}

line_list_contains() {
    local list="$1"
    local expected="$2"
    local item=""

    while IFS= read -r item; do
        if [ "$item" = "$expected" ]; then
            return 0
        fi
    done <<< "$list"
    return 1
}

resolve_suite_target() {
    local suite="$1"
    local requested_target="${2:-}"
    local suites=""
    local targets=""
    local metadata=""

    if ! suites="$(suite_list)"; then
        print_color $RED "Unable to read suite registry"
        return 1
    fi
    if ! line_list_contains "$suites" "$suite"; then
        print_color $RED "Unknown suite: $suite"
        print_color $YELLOW "Available suites: $(printf '%s' "$suites" | tr '\n' ' ')"
        return 1
    fi

    if ! targets="$(suite_target_list "$suite")"; then
        print_color $RED "Unable to read targets for suite: $suite"
        return 1
    fi

    if [ -z "$requested_target" ]; then
        if ! requested_target="$(make -s --no-print-directory \
            print-suite-default-target SUITE="$suite")"; then
            print_color $RED "Unable to read the default target for suite: $suite"
            return 1
        fi
        if [ -z "$requested_target" ]; then
            print_color $RED "Suite '$suite' requires --target"
            print_color $YELLOW "Available targets: $(printf '%s' "$targets" | tr '\n' ' ')"
            return 1
        fi
    fi

    if ! line_list_contains "$targets" "$requested_target"; then
        print_color $RED "Unknown target '$requested_target' for suite '$suite'"
        print_color $YELLOW "Available targets: $(printf '%s' "$targets" | tr '\n' ' ')"
        return 1
    fi

    if ! metadata="$(make -s --no-print-directory print-suite-target-metadata \
        SUITE="$suite" SUITE_TARGET="$requested_target")"; then
        print_color $RED "Unable to read metadata for $suite/$requested_target"
        return 1
    fi

    IFS='|' read -r ACTIVE_BUILD_GROUP ACTIVE_BINARY_GROUP ACTIVE_EXECUTOR \
        ACTIVE_DEFAULT_CONFIG ACTIVE_REQUIRED_CC ACTIVE_CUDA_ARCH <<< "$metadata"
    ACTIVE_SUITE="$suite"
    ACTIVE_TARGET="$requested_target"

    if [ -z "$ACTIVE_BUILD_GROUP" ] || [ -z "$ACTIVE_BINARY_GROUP" ] || \
       [ -z "$ACTIVE_EXECUTOR" ] || [ -z "$ACTIVE_DEFAULT_CONFIG" ] || \
       [ -z "$ACTIVE_REQUIRED_CC" ] || [ -z "$ACTIVE_CUDA_ARCH" ]; then
        print_color $RED "Incomplete registry entry for $suite/$requested_target"
        return 1
    fi
}

validate_active_target_config() {
    local config_file="../configs/$GPU_CONFIG/gpgpusim.config"
    local major=""
    local minor=""
    local actual_cc=""

    if [ ! -f "$config_file" ]; then
        print_color $RED "Configuration '$GPU_CONFIG' is missing $config_file"
        return 1
    fi

    major="$(awk '$1 == "-gpgpu_compute_capability_major" { print $2; exit }' "$config_file")"
    minor="$(awk '$1 == "-gpgpu_compute_capability_minor" { print $2; exit }' "$config_file")"
    if ! [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]]; then
        print_color $RED "Unable to read compute capability from $config_file"
        return 1
    fi

    actual_cc="$major.$minor"
    if [ "$actual_cc" != "$ACTIVE_REQUIRED_CC" ]; then
        print_color $RED "$ACTIVE_SUITE/$ACTIVE_TARGET requires compute capability $ACTIVE_REQUIRED_CC ($ACTIVE_CUDA_ARCH), but $GPU_CONFIG declares $actual_cc"
        return 1
    fi
}

prepare_suite_target() {
    local suite="$1"
    local target="${2:-}"

    resolve_suite_target "$suite" "$target" || return $?
    if [ "$GPU_CONFIG_EXPLICIT" -eq 0 ]; then
        GPU_CONFIG="$ACTIVE_DEFAULT_CONFIG"
    fi
    validate_active_target_config || return $?
}

# List available tests
list_tests() {
    print_color $BLUE "Available test cases:"

    # Check if test binary exists
    local TEST_BINARY=""
    if ! TEST_BINARY="$(single_test_group_binary test)"; then
        return 1
    fi
    local RUN_DIR="run/$GPU_CONFIG"

    if [ ! -f "$TEST_BINARY" ]; then
        print_color $RED "Error: Test binary not found at $TEST_BINARY"
        print_color $YELLOW "Please run '$0 build test' first"
        exit 1
    fi

    # Run the test binary with --gtest_list_tests within the run directory
    # This ensures proper environment (gpgpusim.config etc.)
    (cd "$RUN_DIR" && ../../build/bin/run_all_tests --gtest_list_tests 2>/dev/null)
}

# List available GPU configurations
list_configs() {
    print_color $BLUE "Available GPU configurations:"

    local configs_dir="../configs"

    if [ ! -d "$configs_dir" ]; then
        print_color $RED "Error: configs directory not found at $configs_dir"
        exit 1
    fi

    # Find all directories in configs that contain gpgpusim.config
    local found_configs=0
    for config_path in "$configs_dir"/*; do
        if [ -d "$config_path" ] && [ -f "$config_path/gpgpusim.config" ]; then
            local config_name=$(basename "$config_path")

            # Mark default config
            if [ "$config_name" = "$GPU_CONFIG" ]; then
                print_color $GREEN "  ✓ $config_name (default)"
            else
                echo "    $config_name"
            fi

            found_configs=1
        fi
    done

    if [ $found_configs -eq 0 ]; then
        print_color $YELLOW "No GPU configurations found in $configs_dir"
    fi

    echo ""
    print_color $BLUE "To use a config:"
    echo "  $0 -c CONFIG_NAME test"
    echo "  $0 --config CONFIG_NAME test"
}

# Run a binary with a gtest filter in the config directory.
# Usage: run_binary_with_filter <abs_binary_path> <config_dir> <filter>
run_binary_with_filter() {
    local abs_bin="$1"
    local config_dir="$2"
    local filter="$3"
    local match_rc=0

    if [ ! -x "$abs_bin" ]; then
        print_color $RED "Test executable not found or not executable: $abs_bin"
        return 1
    fi

    gtest_binary_matches_filter "$abs_bin" "$config_dir" "$filter" || match_rc=$?
    if [ "$match_rc" -eq 1 ]; then
        print_color $RED "No tests in $(basename "$abs_bin") matched filter: $filter"
        return 1
    elif [ "$match_rc" -ne 0 ]; then
        print_color $RED "Failed to list tests in: $abs_bin"
        return 1
    fi

    cd "$config_dir"
    local rc=0
    if command -v timeout &> /dev/null; then
        run_command timeout $TEST_TIMEOUT "$abs_bin" --gtest_filter="$filter" || rc=$?
    else
        run_command "$abs_bin" --gtest_filter="$filter" || rc=$?
    fi
    cd - > /dev/null
    return $rc
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
}

gtest_name_matches_filter() {
    local test_name="$1"
    local filter="$2"
    local positive_patterns="$filter"
    local negative_patterns=""

    if [[ "$filter" == *-* ]]; then
        positive_patterns="${filter%%-*}"
        negative_patterns="${filter#*-}"
    fi

    if [ -z "$positive_patterns" ]; then
        positive_patterns="*"
    fi

    local pattern=""
    local -a patterns=()
    local matched_positive=1
    IFS=':' read -r -a patterns <<< "$positive_patterns"
    for pattern in "${patterns[@]}"; do
        [ -n "$pattern" ] || continue
        if [[ "$test_name" == $pattern ]]; then
            matched_positive=0
            break
        fi
    done

    if [ $matched_positive -ne 0 ]; then
        return 1
    fi

    if [ -n "$negative_patterns" ]; then
        IFS=':' read -r -a patterns <<< "$negative_patterns"
        for pattern in "${patterns[@]}"; do
            [ -n "$pattern" ] || continue
            if [[ "$test_name" == $pattern ]]; then
                return 1
            fi
        done
    fi

    return 0
}

gtest_binary_matches_filter() {
    local abs_bin="$1"
    local config_dir="$2"
    local filter="$3"

    local test_list=""
    if ! test_list="$(cd "$config_dir" && "$abs_bin" --gtest_color=no --gtest_list_tests 2>/dev/null)"; then
        return 2
    fi

    local suite=""
    local raw_line=""
    local line=""
    while IFS= read -r raw_line; do
        line="${raw_line%%#*}"
        line="$(trim_whitespace "$line")"
        [ -n "$line" ] || continue

        if [[ "$raw_line" != " "* ]]; then
            if [[ "$line" == *. ]]; then
                suite="${line%.}"
            fi
            continue
        fi

        [ -n "$suite" ] || continue
        if gtest_name_matches_filter "$suite.$line" "$filter"; then
            return 0
        fi
    done <<< "$test_list"

    return 1
}

# Run verification tests with optional pattern
run_test_targets() {
    local test_name="${1:-}"
    local build_group="$2"
    local binary_group="$3"
    local cuda_arch="$4"
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    build_make_group "$build_group" "$cuda_arch"

    local test_executable=""
    if ! test_executable="$(single_test_group_binary "$binary_group")"; then
        return 1
    fi
    if [ ! -f "$test_executable" ]; then
        print_color $RED "Test executable not found: $test_executable"
        exit 1
    fi

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    local abs_test_path="$(pwd)/$test_executable"
    local exit_code=0

    if [ -n "$test_name" ]; then
        print_color $BLUE "Running verification test: $test_name (config: $GPU_CONFIG)"
        run_binary_with_filter "$abs_test_path" "$config_dir" "*${test_name}*" || exit_code=$?
    else
        print_color $BLUE "Running all verification tests (config: $GPU_CONFIG)"
        # Excluded tests are centralized here so local runs and CI share the
        # same default skip policy.
        # - CPAsyncMethod: uses cp.async instruction
        # - PerformanceComparison: internally calls CP_ASYNC method
        # - MBarrierSanityTest: TODO: simulator try_wait can deadlock on
        #   unsatisfied barriers, so keep the sanity suite native-only for now
        local EXCLUDED_TESTS="-*CPAsyncMethod*:*PerformanceComparison*:MBarrierSanityTest.*"
        run_binary_with_filter "$abs_test_path" "$config_dir" "$EXCLUDED_TESTS" || exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Tests passed!"
    else
        print_color $RED "✗ Tests failed (exit code: $exit_code)"
    fi

    return $exit_code
}

run_single_gtest_target() {
    local label="$1"
    local test_name="${2:-}"
    local build_group="$3"
    local binary_group="$4"
    local cuda_arch="$5"
    local default_filter="${6:-*}"
    local config_dir="run/${GPU_CONFIG}"
    local parallel=0

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        return 1
    fi
    if [ "$cuda_arch" = "sm_90a" ]; then
        parallel=1
    fi
    build_make_group "$build_group" "$cuda_arch" "$parallel" || return $?

    local binary_rel=""
    if ! binary_rel="$(single_test_group_binary "$binary_group")"; then
        return 1
    fi
    local binary_path="$(pwd)/$binary_rel"
    local filter="$default_filter"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    print_color $BLUE "Running $label: ${test_name:-all} (config: $GPU_CONFIG)"
    if run_binary_with_filter "$binary_path" "$config_dir" "$filter"; then
        print_color $GREEN "✓ $label passed!"
    else
        local rc=$?
        print_color $RED "✗ $label failed (exit code: $rc)"
        return $rc
    fi
}

# Run microbenchmarks with pattern
run_microbench_tests() {
    local test_name="${1:-}"
    local build_group="$2"
    local binary_group="$3"
    local cuda_arch="$4"
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    local parallel=0
    if [ "$cuda_arch" = "sm_90a" ]; then
        parallel=1
    fi
    build_make_group "$build_group" "$cuda_arch" "$parallel"

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    local filter="*"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
        if [[ "$test_name" == *.* && "$test_name" != */* ]]; then
            local suite_name="${test_name%%.*}"
            local case_name="${test_name#*.}"
            filter="${filter}:*${suite_name}/*.${case_name}*"
        fi
    fi

    print_color $BLUE "Running microbenchmarks: ${test_name:-all} (config: $GPU_CONFIG)"

    local exit_code=0
    local bench_manifest=""
    if ! bench_manifest="$(test_group_binaries "$binary_group")"; then
        print_color $RED "Unable to resolve microbenchmark binaries"
        return 1
    fi

    local -a bench_bins=()
    while IFS= read -r bench_rel; do
        [ -n "$bench_rel" ] || continue
        local bench_bin="$(pwd)/$bench_rel"
        [ -f "$bench_bin" ] || continue
        local match_rc=0
        gtest_binary_matches_filter "$bench_bin" "$config_dir" "$filter" || match_rc=$?
        if [ "$match_rc" -eq 0 ]; then
            bench_bins+=("$bench_bin")
        elif [ "$match_rc" -ne 1 ]; then
            print_color $RED "Failed to list tests in microbenchmark: $bench_bin"
            return 1
        fi
    done <<< "$bench_manifest"

    if [ ${#bench_bins[@]} -eq 0 ]; then
        print_color $YELLOW "No microbenchmark binaries matched pattern: ${test_name:-all}"
        return 1
    fi

    for bench_bin in "${bench_bins[@]}"; do
        print_color $BLUE "Running $(basename "$bench_bin")..."
        run_binary_with_filter "$bench_bin" "$config_dir" "$filter" || exit_code=$?
    done

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Benchmarks passed!"
    else
        print_color $RED "✗ Benchmarks failed (exit code: $exit_code)"
    fi

    return $exit_code
}

# Run standalone dev tests with pattern
run_dev_tests() {
    local test_name="${1:-}"
    local build_group="$2"
    local binary_group="$3"
    local cuda_arch="$4"
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    build_make_group "$build_group" "$cuda_arch"

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    local filter="*"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    print_color $BLUE "Running dev tests: ${test_name:-all} (config: $GPU_CONFIG)"

    local dev_rel=""
    if ! dev_rel="$(single_test_group_binary "$binary_group")"; then
        return 1
    fi
    local dev_bin="$(pwd)/$dev_rel"
    if [ ! -f "$dev_bin" ]; then
        print_color $RED "Dev test binary not found: $dev_bin"
        exit 1
    fi

    local exit_code=0
    run_binary_with_filter "$dev_bin" "$config_dir" "$filter" || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Dev tests passed!"
    else
        print_color $RED "✗ Dev tests failed (exit code: $exit_code)"
    fi

    return $exit_code
}

fa2_group_for_test_name() {
    local test_name="$1"
    if [[ -z "$test_name" ]]; then
        echo "all"
    elif [[ "$test_name" == *Fa2PrefillFp16SmokeTest* || "$test_name" == *Fa2FwdFp16SmokeIntegrationTest* ]]; then
        echo "smoke"
    elif [[ "$test_name" == *Fa2PrefillFp16SmallTest* ]]; then
        echo "small"
    elif [[ "$test_name" == *Fa2PrefillFp16MediumTest* ]]; then
        echo "medium"
    elif [[ "$test_name" == *Fa2PrefillFp16SensitivityH1D128Test* ]]; then
        echo "sensitivity_h1d128"
    elif [[ "$test_name" == *Fa2PrefillFp16SensitivityLargeD128FullTest* ]]; then
        echo "sensitivity_large_d128_full"
    elif [[ "$test_name" == *Fa2PrefillFp16SensitivityTest* ]]; then
        echo "sensitivity"
    elif [[ "$test_name" == *Fa2PrefillFp16IntegrationTest* ]]; then
        echo "large"
    else
        echo "all"
    fi
}

fa2_variant_for_test_name() {
    local test_name="$1"
    if [[ "$test_name" == *H32D64Full* || "$test_name" == *SmallForwardCase* ]]; then
        echo "h32d64_full"
    elif [[ "$test_name" == *H32D64Causal* ]]; then
        echo "h32d64_causal"
    elif [[ "$test_name" == *H16D128Full* ]]; then
        echo "h16d128_full"
    elif [[ "$test_name" == *H16D128Causal* ]]; then
        echo "h16d128_causal"
    else
        echo "all"
    fi
}

fa2_make_target_for_group_variant() {
    local group="$1"
    local variant="$2"
    if [[ "$group" == "all" ]]; then
        echo "fa2"
    elif [[ "$group" == sensitivity* ]]; then
        echo "fa2-${group//_/-}"
    elif [[ "$group" == "large" && "$variant" != "all" ]]; then
        echo "fa2-large-${variant//_/-}"
    else
        echo "fa2-${group}"
    fi
}

run_fa2_split_binaries() {
    local make_target="$1"
    local filter="$2"
    local config_dir="$3"
    local exit_code=0
    local binary_manifest=""
    local matched_binaries=0

    if ! binary_manifest="$(test_group_binaries "$make_target")"; then
        print_color $RED "Unable to resolve FA2 binaries for target: $make_target"
        return 1
    fi

    while IFS= read -r binary_rel; do
        [ -n "$binary_rel" ] || continue
        local fa2_bin="$(pwd)/$binary_rel"
        if [ ! -f "$fa2_bin" ]; then
            print_color $RED "FA2 gtest executable not found: $fa2_bin"
            return 1
        fi

        local match_rc=0
        gtest_binary_matches_filter "$fa2_bin" "$config_dir" "$filter" || match_rc=$?
        if [ "$match_rc" -eq 1 ]; then
            continue
        elif [ "$match_rc" -ne 0 ]; then
            print_color $RED "Failed to list tests in FA2 binary: $fa2_bin"
            return 1
        fi

        matched_binaries=$((matched_binaries + 1))
        print_color $BLUE "Running FA2 gtests: $(basename "$fa2_bin") filter=$filter (config: $GPU_CONFIG)"
        run_binary_with_filter "$fa2_bin" "$config_dir" "$filter" || exit_code=$?
    done <<< "$binary_manifest"

    if [ "$matched_binaries" -eq 0 ]; then
        print_color $RED "No FA2 binaries matched filter: $filter"
        return 1
    fi
    return $exit_code
}

run_fa2_target() {
    local test_name="${1:-}"
    local cuda_arch="$2"
    local config_dir="run/${GPU_CONFIG}"
    local fa2_group="$(fa2_group_for_test_name "$test_name")"
    local fa2_variant="$(fa2_variant_for_test_name "$test_name")"
    local make_target="$(fa2_make_target_for_group_variant "$fa2_group" "$fa2_variant")"
    local filter="*"
    local exit_code=0

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        return 1
    fi
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    build_make_group "$make_target" "$cuda_arch" 1 || return $?
    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    run_fa2_split_binaries "$make_target" "$filter" "$config_dir" || exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        print_color $GREEN "✓ FA2 passed!"
    else
        print_color $RED "✗ FA2 failed (exit code: $exit_code)"
    fi
    return $exit_code
}

# Build and run trace tests (Triton kernel PTX smoke tests)
run_trace_tests() {
    local test_name="${1:-}"
    local cuda_arch="$2"

    build_gpgpusim

    print_color $BLUE "Building trace tests (config: $GPU_CONFIG)..."
    run_command make -C src/trace ARCH="$cuda_arch" GPU_CONFIG="$GPU_CONFIG"

    print_color $BLUE "Running trace tests..."

    local trace_bin_dir="$(pwd)/build/trace/bin"
    local exit_code=0
    local matched_tests=0

    # Data-driven test names (must match configs in gpt2_data_driven_test.cu)
    local gpt2_data_driven_tests="gelu flash_attn layernorm residual_add linear"

    # Run GPT-2 embedding test (CPU reference, separate binary)
    if [ -z "$test_name" ] || [[ "embedding" == *"$test_name"* ]]; then
        matched_tests=$((matched_tests + 1))
        print_color $BLUE "--- gpt2_embedding_test ---"
        (cd "$trace_bin_dir" && ./gpt2_embedding_test) || { exit_code=1; print_color $RED "FAILED: gpt2_embedding"; }
    fi

    # Run GPT-2 data-driven tests (single binary, test name as argument)
    for name in $gpt2_data_driven_tests; do
        if [ -n "$test_name" ] && [[ "$name" != *"$test_name"* ]]; then
            continue
        fi
        matched_tests=$((matched_tests + 1))
        print_color $BLUE "--- gpt2_data_driven_test $name ---"
        (cd "$trace_bin_dir" && ./gpt2_data_driven_test "$name") || { exit_code=1; print_color $RED "FAILED: gpt2_$name"; }
    done

    if [ "$matched_tests" -eq 0 ]; then
        print_color $RED "No trace tests matched pattern: $test_name"
        return 1
    elif [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Trace tests passed!"
    else
        print_color $RED "✗ Trace tests failed!"
    fi
    return $exit_code
}

run_active_target() {
    local filter="${1:-}"

    case "$ACTIVE_EXECUTOR" in
        test)
            run_test_targets "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH"
            ;;
        gtest-single)
            run_single_gtest_target "$ACTIVE_SUITE/$ACTIVE_TARGET" "$filter" \
                "$ACTIVE_BUILD_GROUP" "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH"
            ;;
        gtest-multi)
            run_microbench_tests "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH"
            ;;
        fa2)
            run_fa2_target "$filter" "$ACTIVE_CUDA_ARCH"
            ;;
        dev)
            run_dev_tests "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH"
            ;;
        trace)
            run_trace_tests "$filter" "$ACTIVE_CUDA_ARCH"
            ;;
        *)
            print_color $RED "Unknown executor '$ACTIVE_EXECUTOR' for $ACTIVE_SUITE/$ACTIVE_TARGET"
            return 1
            ;;
    esac
}

# Clean build artifacts
clean_tests() {
    print_color $BLUE "Cleaning test build artifacts..."
    run_command make clean
    print_color $GREEN "Clean complete!"
}

# Initialize run directory on script start
initialize_run_directory() {
    # Only setup if we're doing operations that need the config
    case "${1:-}" in
        build|run)
            setup_run_directory
            ;;
    esac
}

main() {
    local command=""
    local -a command_args=()

    # Parse global options independently from the command so options work on
    # either side of it. Positional arguments retain their original order.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                TEST_VERBOSE=2
                shift
                ;;
            -d|--debug)
                DEBUG_TESTS=1
                shift
                ;;
            -t|--timeout)
                if [ $# -lt 2 ] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    print_color $RED "--timeout requires a positive integer"
                    return 1
                fi
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            -c|--config)
                if [ $# -lt 2 ] || [ -z "$2" ]; then
                    print_color $RED "--config requires a configuration name"
                    return 1
                fi
                GPU_CONFIG="$2"
                GPU_CONFIG_EXPLICIT=1
                shift 2
                ;;
            --target)
                if [ $# -lt 2 ] || [ -z "$2" ]; then
                    print_color $RED "--target requires a target name"
                    return 1
                fi
                if [ -n "$REQUESTED_TARGET" ]; then
                    print_color $RED "--target may only be specified once"
                    return 1
                fi
                REQUESTED_TARGET="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            --)
                shift
                while [[ $# -gt 0 ]]; do
                    if [ -z "$command" ]; then
                        command="$1"
                    else
                        command_args+=("$1")
                    fi
                    shift
                done
                ;;
            -*)
                print_color $RED "Unknown option: $1"
                return 1
                ;;
            *)
                if [ -z "$command" ]; then
                    command="$1"
                else
                    command_args+=("$1")
                fi
                shift
                ;;
        esac
    done

    if [ -z "$command" ]; then
        usage
        return 0
    fi

    case "$command" in
        build)
            if [ "${#command_args[@]}" -ne 1 ]; then
                print_color $RED "build requires exactly one suite"
                return 1
            fi
            ;;
        run)
            if [ "${#command_args[@]}" -lt 1 ] || [ "${#command_args[@]}" -gt 2 ]; then
                print_color $RED "run requires a suite and accepts at most one filter"
                return 1
            fi
            ;;
        setup|refresh|clean|list|list-configs|help)
            if [ "${#command_args[@]}" -ne 0 ]; then
                print_color $RED "$command does not accept arguments"
                return 1
            fi
            ;;
        *)
            print_color $RED "Unknown command: $command"
            usage
            return 1
            ;;
    esac

    if [ -n "$REQUESTED_TARGET" ] && [ "$command" != "build" ] && [ "$command" != "run" ]; then
        print_color $RED "--target is only valid with build or run"
        return 1
    fi

    local argument="${command_args[0]:-}"
    case "$command" in
        setup)
            setup_environment
            ;;
        refresh)
            print_color $BLUE "Refreshing run directory and configuration..."
            setup_run_directory "force"
            print_color $GREEN "Run directory refreshed!"
            ;;
        build)
            prepare_suite_target "$argument" "$REQUESTED_TARGET" || return $?
            initialize_run_directory build
            build_active_target
            ;;
        run)
            local suite="$argument"
            local filter="${command_args[1]:-}"
            prepare_suite_target "$suite" "$REQUESTED_TARGET" || return $?
            initialize_run_directory run
            run_active_target "$filter"
            ;;
        clean)
            clean_tests
            ;;
        list)
            list_tests
            ;;
        list-configs)
            list_configs
            ;;
        help)
            usage
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
