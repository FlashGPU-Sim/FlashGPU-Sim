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
GPGPUSIM_BUILD_JOBS=${GPGPUSIM_BUILD_JOBS:-4}
DEFAULT_GPU_CONFIG=${DEFAULT_GPU_CONFIG:-SM120_RTX5090}
GPU_CONFIG_EXPLICIT=0
if [ -n "${GPU_CONFIG:-}" ]; then
    GPU_CONFIG_EXPLICIT=1
fi
GPU_CONFIG=${GPU_CONFIG:-$DEFAULT_GPU_CONFIG}  # Default GPU configuration
REQUESTED_TARGET=""
REQUESTED_GROUP=""
REQUESTED_MODE=""

# Resolved from the Makefile-owned suite/target registry.
ACTIVE_SUITE=""
ACTIVE_TARGET=""
ACTIVE_BUILD_GROUP=""
ACTIVE_BINARY_GROUP=""
ACTIVE_EXECUTOR=""
ACTIVE_DEFAULT_CONFIG=""
ACTIVE_REQUIRED_CC=""
ACTIVE_CUDA_ARCH=""
ACTIVE_GROUP=""
ACTIVE_MODE=""
ACTIVE_DEFAULT_FILTER="*"
ACTIVE_CASE_LIST=""

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
    echo "  build test [--target sm120|sm90] --group NAME"
    echo "  run test [--target sm120|sm90] --group NAME [filter]"
    echo "  build analysis --target fa2|fa3 --group NAME [--mode NAME|all]"
    echo "  run analysis --target fa2|fa3 --group NAME [--mode NAME] [filter]"
    echo "  build microbench [--target sm120|sm90] --group NAME"
    echo "  run microbench [--target sm120|sm90] --group NAME [filter]"
    echo "  build trace [--target sm120] --group gpt2"
    echo "  run trace [--target sm120] --group gpt2 [filter]"
    echo ""
    echo "Groups:"
    echo "  test/sm120:       unit, integration"
    echo "  test/sm90:        instructions, fa2-smoke, fa3-smoke, fa3-packgqa"
    echo "  analysis/fa2|fa3: small, medium, large, breakdown, scaling, concurrency"
    echo "  microbench/sm120: mbarrier, mma, memory"
    echo "  microbench/sm90:  cp-async, mma, tma, wgmma"
    echo "  trace/sm120:      gpt2"
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
    echo "  --group NAME       Select one group within a target"
    echo "  --mode NAME        Select an analysis compile-time mode"
    echo "  -h, --help         Show this help"
    echo "  Options may appear before or after the command."
    echo ""
    echo "Examples:"
    echo "  $0 run test --target sm120 --group integration CudaVectorAdd"
    echo "  $0 run test --target sm90 --group instructions WgmmaF16"
    echo "  $0 run test --target sm90 --group fa2-smoke"
    echo "  $0 run analysis --target fa2 --group breakdown --mode only_mma"
    echo "  $0 run analysis --target fa3 --group scaling --mode baseline"
    echo "  $0 build microbench --target sm120 --group memory"
    echo "  $0 run microbench --target sm90 --group wgmma"
    echo "  $0 run trace --target sm120 --group gpt2 flash_attn"
    echo ""
    echo "Standalone calibration groups are build-only; use their local Makefiles"
    echo "for benchmark-specific runtime arguments. Mode 'all' is also build-only."
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
        run_command make FLASH=1 "-j$GPGPUSIM_BUILD_JOBS"
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

# Query the Makefile-owned binary manifest for one resolved group.
binary_group_binaries() {
    local group="$1"
    make -s --no-print-directory print-binary-group BINARY_GROUP="$group"
}

single_binary_group_binary() {
    local group="$1"
    local binaries=""
    local count=0

    if ! binaries="$(binary_group_binaries "$group")"; then
        print_color $RED "Unable to resolve binary group: $group" >&2
        return 1
    fi

    count="$(printf '%s\n' "$binaries" | sed '/^[[:space:]]*$/d' | wc -l)"
    if [ "$count" -ne 1 ]; then
        print_color $RED "Expected one binary in group '$group', found $count" >&2
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

target_group_list() {
    local suite="$1"
    local target="$2"
    make -s --no-print-directory list-target-groups \
        SUITE="$suite" SUITE_TARGET="$target"
}

target_group_mode_list() {
    local suite="$1"
    local target="$2"
    local group="$3"
    make -s --no-print-directory list-target-group-modes \
        SUITE="$suite" SUITE_TARGET="$target" TARGET_GROUP="$group"
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

resolve_target_group() {
    local requested_group="${1:-}"
    local requested_mode="${2:-}"
    local groups=""
    local modes=""
    local metadata=""

    if [ "$ACTIVE_EXECUTOR" != "group-required" ]; then
        if [ -n "$requested_group" ] || [ -n "$requested_mode" ]; then
            print_color $RED "$ACTIVE_SUITE/$ACTIVE_TARGET does not accept --group or --mode"
            return 1
        fi
        ACTIVE_DEFAULT_FILTER="*"
        return 0
    fi

    if ! groups="$(target_group_list "$ACTIVE_SUITE" "$ACTIVE_TARGET")"; then
        print_color $RED "Unable to read groups for $ACTIVE_SUITE/$ACTIVE_TARGET"
        return 1
    fi
    if [ -z "$requested_group" ]; then
        print_color $RED "$ACTIVE_SUITE/$ACTIVE_TARGET requires --group"
        print_color $YELLOW "Available groups: $(printf '%s' "$groups" | tr '\n' ' ')"
        return 1
    fi
    if ! line_list_contains "$groups" "$requested_group"; then
        print_color $RED "Unknown group '$requested_group' for $ACTIVE_SUITE/$ACTIVE_TARGET"
        print_color $YELLOW "Available groups: $(printf '%s' "$groups" | tr '\n' ' ')"
        return 1
    fi

    if ! modes="$(target_group_mode_list "$ACTIVE_SUITE" "$ACTIVE_TARGET" "$requested_group")"; then
        print_color $RED "Unable to read modes for $ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group"
        return 1
    fi
    modes="$(printf '%s\n' "$modes" | sed '/^[[:space:]]*$/d')"
    if [ -n "$modes" ] && [ -z "$requested_mode" ]; then
        print_color $RED "$ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group requires --mode"
        print_color $YELLOW "Available modes: $(printf '%s' "$modes" | tr '\n' ' ')"
        return 1
    fi
    if [ -z "$modes" ] && [ -n "$requested_mode" ]; then
        print_color $RED "$ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group does not accept --mode"
        return 1
    fi
    if [ -n "$requested_mode" ] && ! line_list_contains "$modes" "$requested_mode"; then
        print_color $RED "Unknown mode '$requested_mode' for $ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group"
        print_color $YELLOW "Available modes: $(printf '%s' "$modes" | tr '\n' ' ')"
        return 1
    fi

    if ! metadata="$(make -s --no-print-directory print-target-group-metadata \
        SUITE="$ACTIVE_SUITE" SUITE_TARGET="$ACTIVE_TARGET" \
        TARGET_GROUP="$requested_group" TARGET_MODE="$requested_mode")"; then
        print_color $RED "Unable to read metadata for $ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group"
        return 1
    fi
    IFS='|' read -r ACTIVE_BUILD_GROUP ACTIVE_BINARY_GROUP ACTIVE_EXECUTOR \
        ACTIVE_DEFAULT_FILTER ACTIVE_CASE_LIST <<< "$metadata"
    ACTIVE_GROUP="$requested_group"
    ACTIVE_MODE="$requested_mode"

    if [ -z "$ACTIVE_BUILD_GROUP" ] || [ -z "$ACTIVE_BINARY_GROUP" ] || \
       [ -z "$ACTIVE_EXECUTOR" ] || [ -z "$ACTIVE_DEFAULT_FILTER" ]; then
        print_color $RED "Incomplete group registry entry for $ACTIVE_SUITE/$ACTIVE_TARGET/$requested_group"
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
    resolve_target_group "$REQUESTED_GROUP" "$REQUESTED_MODE" || return $?
    if [ "$GPU_CONFIG_EXPLICIT" -eq 0 ]; then
        GPU_CONFIG="$ACTIVE_DEFAULT_CONFIG"
    fi
    validate_active_target_config || return $?
}

# List available tests
list_tests() {
    print_color $BLUE "Supported suite / target / group hierarchy:"

    local suites=""
    if ! suites="$(suite_list)"; then
        return 1
    fi

    local suite=""
    local target=""
    local group=""
    local targets=""
    local groups=""
    local modes=""
    while IFS= read -r suite; do
        [ -n "$suite" ] || continue
        echo "$suite"
        targets="$(suite_target_list "$suite")" || return $?
        while IFS= read -r target; do
            [ -n "$target" ] || continue
            echo "  $target"
            groups="$(target_group_list "$suite" "$target")" || return $?
            while IFS= read -r group; do
                [ -n "$group" ] || continue
                modes="$(target_group_mode_list "$suite" "$target" "$group")" || return $?
                modes="$(printf '%s\n' "$modes" | sed '/^[[:space:]]*$/d' | tr '\n' ' ')"
                if [ -n "$modes" ]; then
                    echo "    $group [modes: ${modes% }]"
                else
                    echo "    $group"
                fi
            done <<< "$groups"
        done <<< "$targets"
    done <<< "$suites"
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
    echo "  $0 -c CONFIG_NAME run test --target sm120 --group integration"
    echo "  $0 --config CONFIG_NAME run test --target sm120 --group integration"
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

# GTest writes a separate discovery report for --gtest_list_tests when
# GTEST_OUTPUT is set. Discovery helpers must not pollute CI result directories.
gtest_binary_matches_filter() {
    local abs_bin="$1"
    local config_dir="$2"
    local filter="$3"

    local test_list=""
    if ! test_list="$(cd "$config_dir" && env -u GTEST_OUTPUT "$abs_bin" \
        --gtest_color=no --gtest_list_tests 2>/dev/null)"; then
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

# Resolve a user substring to exact tests while preserving a registry-owned
# group filter. This prevents a filter from escaping its selected group when
# several groups share one binary (FA3 standard workloads do this).
gtest_binary_intersection_filter() {
    local abs_bin="$1"
    local config_dir="$2"
    local group_filter="$3"
    local test_name="$4"
    local user_filter="*${test_name}*"
    local test_list=""
    local suite=""
    local raw_line=""
    local line=""
    local resolved_filter=""
    local full_name=""

    if ! test_list="$(cd "$config_dir" && env -u GTEST_OUTPUT "$abs_bin" \
        --gtest_color=no --gtest_list_tests 2>/dev/null)"; then
        return 2
    fi

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
        full_name="$suite.$line"
        if gtest_name_matches_filter "$full_name" "$group_filter" && \
           gtest_name_matches_filter "$full_name" "$user_filter"; then
            if [ -n "$resolved_filter" ]; then
                resolved_filter+=":"
            fi
            resolved_filter+="$full_name"
        fi
    done <<< "$test_list"

    [ -n "$resolved_filter" ] || return 1
    printf '%s\n' "$resolved_filter"
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
    if ! test_executable="$(single_binary_group_binary "$binary_group")"; then
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
        # - CPAsyncMethod: source test is currently disabled; retain the
        #   pattern so it remains opt-in if re-enabled
        # - PerformanceComparison: legacy multi-iteration performance test;
        #   its CP_ASYNC entry is currently disabled in the source
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
    if ! binary_rel="$(single_binary_group_binary "$binary_group")"; then
        return 1
    fi
    local binary_path="$(pwd)/$binary_rel"
    local filter="$default_filter"
    if [ -n "$test_name" ]; then
        if [ "$default_filter" = "*" ]; then
            filter="*${test_name}*"
        else
            local filter_rc=0
            filter="$(gtest_binary_intersection_filter "$binary_path" \
                "$config_dir" "$default_filter" "$test_name")" || filter_rc=$?
            if [ "$filter_rc" -eq 1 ]; then
                print_color $RED "No tests in $label matched: $test_name"
                return 1
            elif [ "$filter_rc" -ne 0 ]; then
                print_color $RED "Failed to resolve tests in: $binary_path"
                return 1
            fi
        fi
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

# Run one logical target backed by one or more gtest binaries.
run_multi_gtest_target() {
    local label="$1"
    local test_name="${2:-}"
    local build_group="$3"
    local binary_group="$4"
    local cuda_arch="$5"
    local default_filter="${6:-*}"
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

    local filter="$default_filter"
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
        if [[ "$test_name" == *.* && "$test_name" != */* ]]; then
            local suite_name="${test_name%%.*}"
            local case_name="${test_name#*.}"
            filter="${filter}:*${suite_name}/*.${case_name}*"
        fi
    fi

    print_color $BLUE "Running $label: ${test_name:-all} (config: $GPU_CONFIG)"

    local exit_code=0
    local bench_manifest=""
    if ! bench_manifest="$(binary_group_binaries "$binary_group")"; then
        print_color $RED "Unable to resolve binaries for $label"
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
            print_color $RED "Failed to list tests in $label binary: $bench_bin"
            return 1
        fi
    done <<< "$bench_manifest"

    if [ ${#bench_bins[@]} -eq 0 ]; then
        print_color $YELLOW "No $label binaries matched pattern: ${test_name:-all}"
        return 1
    fi

    for bench_bin in "${bench_bins[@]}"; do
        print_color $BLUE "Running $(basename "$bench_bin")..."
        run_binary_with_filter "$bench_bin" "$config_dir" "$filter" || exit_code=$?
    done

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ $label passed!"
    else
        print_color $RED "✗ $label failed (exit code: $exit_code)"
    fi

    return $exit_code
}

run_fa3_profile_target() {
    local label="$1"
    local test_name="${2:-}"
    local build_group="$3"
    local binary_group="$4"
    local cuda_arch="$5"
    local default_filter="$6"
    local case_list="$7"
    local config_dir="run/${GPU_CONFIG}"
    local filter="$default_filter"
    local binary_manifest=""
    local matched_binaries=0
    local exit_code=0

    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        return 1
    fi
    if [ -n "$test_name" ]; then
        filter="*${test_name}*"
    fi

    build_make_group "$build_group" "$cuda_arch" 1 || return $?
    if ! binary_manifest="$(binary_group_binaries "$binary_group")"; then
        print_color $RED "Unable to resolve binaries for $label"
        return 1
    fi

    export GTEST_COLOR=yes
    while IFS= read -r binary_rel; do
        [ -n "$binary_rel" ] || continue
        local binary_path="$(pwd)/$binary_rel"
        local match_rc=0
        if [ ! -f "$binary_path" ]; then
            print_color $RED "FA3 gtest executable not found: $binary_path"
            return 1
        fi
        gtest_binary_matches_filter "$binary_path" "$config_dir" "$filter" || match_rc=$?
        if [ "$match_rc" -eq 1 ]; then
            continue
        elif [ "$match_rc" -ne 0 ]; then
            print_color $RED "Failed to list tests in FA3 binary: $binary_path"
            return 1
        fi

        matched_binaries=$((matched_binaries + 1))
        print_color $BLUE "Running $label: $(basename "$binary_path") (config: $GPU_CONFIG)"
        (
            export FA3_H1D128_PROFILE_CASE_LIST="$case_list"
            run_binary_with_filter "$binary_path" "$config_dir" "$filter"
        ) || exit_code=$?
    done <<< "$binary_manifest"

    if [ "$matched_binaries" -eq 0 ]; then
        print_color $RED "No FA3 binaries matched filter: $filter"
        return 1
    elif [ "$exit_code" -eq 0 ]; then
        print_color $GREEN "✓ $label passed!"
    else
        print_color $RED "✗ $label failed (exit code: $exit_code)"
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
    local label="$ACTIVE_SUITE/$ACTIVE_TARGET"
    if [ -n "$ACTIVE_GROUP" ]; then
        label="$label/$ACTIVE_GROUP"
    fi
    if [ -n "$ACTIVE_MODE" ]; then
        label="$label/$ACTIVE_MODE"
    fi

    case "$ACTIVE_EXECUTOR" in
        test)
            run_test_targets "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH"
            ;;
        gtest-single)
            run_single_gtest_target "$label" "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH" "$ACTIVE_DEFAULT_FILTER"
            ;;
        gtest-multi)
            run_multi_gtest_target "$label" "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH" "$ACTIVE_DEFAULT_FILTER"
            ;;
        fa3-profile)
            run_fa3_profile_target "$label" "$filter" "$ACTIVE_BUILD_GROUP" \
                "$ACTIVE_BINARY_GROUP" "$ACTIVE_CUDA_ARCH" \
                "$ACTIVE_DEFAULT_FILTER" "$ACTIVE_CASE_LIST"
            ;;
        trace)
            run_trace_tests "$filter" "$ACTIVE_CUDA_ARCH"
            ;;
        build-only)
            print_color $RED "$label is a standalone calibration group and cannot be run with the generic runner"
            print_color $YELLOW "Build it here, then use its local Makefile with explicit benchmark arguments"
            return 1
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
            --group)
                if [ $# -lt 2 ] || [ -z "$2" ]; then
                    print_color $RED "--group requires a group name"
                    return 1
                fi
                if [ -n "$REQUESTED_GROUP" ]; then
                    print_color $RED "--group may only be specified once"
                    return 1
                fi
                REQUESTED_GROUP="$2"
                shift 2
                ;;
            --mode)
                if [ $# -lt 2 ] || [ -z "$2" ]; then
                    print_color $RED "--mode requires a mode name"
                    return 1
                fi
                if [ -n "$REQUESTED_MODE" ]; then
                    print_color $RED "--mode may only be specified once"
                    return 1
                fi
                REQUESTED_MODE="$2"
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

    if { [ -n "$REQUESTED_TARGET" ] || [ -n "$REQUESTED_GROUP" ] || \
         [ -n "$REQUESTED_MODE" ]; } && \
       [ "$command" != "build" ] && [ "$command" != "run" ]; then
        print_color $RED "--target, --group, and --mode are only valid with build or run"
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
            if [ "$ACTIVE_MODE" = "all" ]; then
                print_color $RED "--mode all is build-only; run requires one concrete mode"
                return 1
            fi
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
