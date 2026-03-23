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
GPU_CONFIG=${GPU_CONFIG:-SM120_RTX5090}  # Default GPU configuration

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
    echo "Usage: $0 [OPTIONS] COMMAND [PATTERN]"
    echo ""
    echo "Commands:"
    echo "  build              Build all (test + bench + dev)"
    echo "  build test         Build verification tests only"
    echo "  build bench        Build microbenchmarks only"
    echo "  build dev          Build standalone dev tests only"
    echo "  test               Run all verification tests (unit + integration)"
    echo "  test <pattern>     Run specific verification test"
    echo "  trace              Run Triton kernel trace smoke tests"
    echo "  trace <pattern>    Run specific trace test (e.g., 'embedding')"
    echo "  bench <pattern>    Run microbenchmarks matching pattern"
    echo "  dev <pattern>      Run standalone dev tests matching pattern"
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
    echo "  -c, --config NAME  Use specific GPU configuration (default: SM120_RTX5090)"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build verification tests"
    echo "  $0 test                               # Run all verification tests"
    echo "  $0 test \"*MMAS8*\"                     # Run specific verification test"
    echo "  $0 bench \"*MMAIssue*\"                  # Run microbenchmarks"
    echo "  $0 -c SM120_RTX5090_REDUCED test      # Run with reduced config"
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

    # Copy configuration if it doesn't exist or if forced
    if [ ! -d "$config_dir" ] || [ "$force_refresh" = "force" ]; then
        rm -rf "$config_dir" 2>/dev/null
        run_command cp -r "$source_config" run/
        print_color $GREEN "Copied $config_name configuration to run directory"
    else
        print_color $BLUE "$config_name configuration already exists in run directory"
    fi
}

# Detect if running in native GPU mode (clean environment without simulator setup)
is_native_mode() {
    # Check if simulator environment was sourced
    if [ -n "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
        return 1  # Simulator mode
    fi

    # Check if LD_LIBRARY_PATH contains simulator library paths
    if echo "$LD_LIBRARY_PATH" | grep -q "gpgpu-sim_distribution/lib"; then
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

    if [ -z "$libcudart_path" ] || find src -name "*.cc" -newer "$libcudart_path" 2>/dev/null | grep -q .; then
        print_color $YELLOW "GPGPU-Sim library needs rebuild..."
        run_command make FLASH=1 -j
        print_color $GREEN "GPGPU-Sim library built successfully"
    else
        print_color $GREEN "GPGPU-Sim library is up to date"
    fi
    cd "$SCRIPT_DIR"
}

# Build verification tests (unit + integration)
build_test_targets() {
    build_gpgpusim

    print_color $BLUE "Building verification tests..."

    if [ "$DEBUG_TESTS" -eq 1 ]; then
        run_command make CXXFLAGS="-std=c++17 -Wall -Wextra -pthread -g -O0 -DDEBUG" test
    else
        run_command make test
    fi

    if [ $? -eq 0 ]; then
        print_color $GREEN "Build successful!"
    else
        print_color $RED "Build failed!"
        exit 1
    fi
}

# Build microbenchmarks (separate binaries)
build_bench_tests() {
    build_gpgpusim

    print_color $BLUE "Building microbenchmarks..."
    run_command make bench

    if [ $? -eq 0 ]; then
        print_color $GREEN "Build successful!"
    else
        print_color $RED "Build failed!"
        exit 1
    fi
}

# Build standalone dev tests
build_dev_tests() {
    build_gpgpusim

    print_color $BLUE "Building dev tests..."
    run_command make dev

    if [ $? -eq 0 ]; then
        print_color $GREEN "Build successful!"
    else
        print_color $RED "Build failed!"
        exit 1
    fi
}

# List available tests
list_tests() {
    print_color $BLUE "Available test cases:"

    # Check if test binary exists
    local TEST_BINARY="build/bin/run_all_tests"
    local RUN_DIR="run/$GPU_CONFIG"

    if [ ! -f "$TEST_BINARY" ]; then
        print_color $RED "Error: Test binary not found at $TEST_BINARY"
        print_color $YELLOW "Please run '$0 build' first"
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

# Run verification tests with optional pattern
run_test_targets() {
    local test_name="${1:-}"
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    build_test_targets

    local test_executable="build/bin/run_all_tests"
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
        # Excluded tests (use unimplemented instructions that call abort())
        # - CPAsyncMethod: uses cp.async instruction
        # - PerformanceComparison: internally calls CP_ASYNC method
        local EXCLUDED_TESTS="-*CPAsyncMethod*:*PerformanceComparison*"
        run_binary_with_filter "$abs_test_path" "$config_dir" "$EXCLUDED_TESTS" || exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Tests passed!"
    else
        print_color $RED "✗ Tests failed (exit code: $exit_code)"
    fi

    return $exit_code
}

# Run microbenchmarks with pattern
run_bench_tests() {
    local test_name="${1:-}"
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    build_bench_tests

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    local filter="*"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    print_color $BLUE "Running microbenchmarks: ${test_name:-all} (config: $GPU_CONFIG)"

    local exit_code=0
    for bench_bin in "$(pwd)/build/bin/"*_bench; do
        [ -f "$bench_bin" ] || continue
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
    local config_dir="run/${GPU_CONFIG}"

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        exit 1
    fi

    build_dev_tests

    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi

    local filter="*"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    print_color $BLUE "Running dev tests: ${test_name:-all} (config: $GPU_CONFIG)"

    local dev_bin="$(pwd)/build/bin/run_dev_tests"
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

# Build and run trace tests (Triton kernel PTX smoke tests)
run_trace_tests() {
    local test_name="${1:-}"

    build_gpgpusim

    print_color $BLUE "Building trace tests (config: $GPU_CONFIG)..."
    run_command make -C src/trace GPU_CONFIG="$GPU_CONFIG"

    print_color $BLUE "Running trace tests..."

    local trace_bin_dir="$(pwd)/build/trace/bin"
    local exit_code=0

    # Data-driven test names (must match configs in gpt2_data_driven_test.cu)
    local gpt2_data_driven_tests="gelu flash_attn layernorm residual_add linear"

    # Run GPT-2 embedding test (CPU reference, separate binary)
    if [ -z "$test_name" ] || [[ "embedding" == *"$test_name"* ]]; then
        print_color $BLUE "--- gpt2_embedding_test ---"
        (cd "$trace_bin_dir" && ./gpt2_embedding_test) || { exit_code=1; print_color $RED "FAILED: gpt2_embedding"; }
    fi

    # Run GPT-2 data-driven tests (single binary, test name as argument)
    for name in $gpt2_data_driven_tests; do
        if [ -n "$test_name" ] && [[ "$name" != *"$test_name"* ]]; then
            continue
        fi
        print_color $BLUE "--- gpt2_data_driven_test $name ---"
        (cd "$trace_bin_dir" && ./gpt2_data_driven_test "$name") || { exit_code=1; print_color $RED "FAILED: gpt2_$name"; }
    done

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Trace tests passed!"
    else
        print_color $RED "✗ Trace tests failed!"
    fi
    return $exit_code
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
        test|bench|dev|trace|build)
            setup_run_directory
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            TEST_VERBOSE=2
            shift
            ;;
        -d|--debug)
            DEBUG_TESTS=1
            shift
            ;;
        -t|--timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        -c|--config)
            GPU_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        setup)
            setup_environment
            exit 0
            ;;
        refresh)
            print_color $BLUE "Refreshing run directory and configuration..."
            setup_run_directory "force"
            print_color $GREEN "Run directory refreshed!"
            exit 0
            ;;
        build)
            initialize_run_directory "build"
            case "${2:-}" in
                test)
                    build_test_targets
                    ;;
                bench)
                    build_bench_tests
                    ;;
                dev)
                    build_dev_tests
                    ;;
                trace)
                    build_gpgpusim
                    make -C src/trace GPU_CONFIG="$GPU_CONFIG"
                    ;;
                "")
                    build_test_targets
                    build_bench_tests
                    build_dev_tests
                    ;;
                *)
                    print_color $RED "Unknown build target: $2 (use 'test', 'bench', 'dev', or 'trace')"
                    exit 1
                    ;;
            esac
            exit 0
            ;;
        test)
            initialize_run_directory "test"
            run_test_targets "$2"
            local test_exit=$?
            # Also run trace tests (unless a specific gtest pattern was given)
            if [ -z "$2" ]; then
                run_trace_tests
                local trace_exit=$?
                [ $test_exit -ne 0 ] && exit $test_exit
                exit $trace_exit
            fi
            exit $test_exit
            ;;
        bench)
            initialize_run_directory "bench"
            run_bench_tests "$2"
            exit $?
            ;;
        dev)
            initialize_run_directory "dev"
            run_dev_tests "$2"
            exit $?
            ;;
        trace)
            initialize_run_directory "trace"
            run_trace_tests "$2"
            exit $?
            ;;
        clean)
            clean_tests
            exit 0
            ;;
        list)
            list_tests
            exit 0
            ;;
        list-configs)
            list_configs
            exit 0
            ;;
        help)
            usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# If no command specified, show usage
usage
