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
    echo "Usage: $0 [OPTIONS] [COMMAND] [TEST_NAME]"
    echo ""
    echo "Commands:"
    echo "  build           Build all tests"
    echo "  run             Run all tests"
    echo "  run <test>      Run specific test"
    echo "  clean           Clean build artifacts"
    echo "  setup           Setup test environment"
    echo "  refresh         Refresh run directory and configuration"
    echo "  list            List available tests"
    echo "  help            Show this help"
    echo ""
    echo "Options:"
    echo "  -v, --verbose   Verbose output"
    echo "  -d, --debug     Enable debug mode"
    echo "  -t, --timeout   Set test timeout (seconds)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Setup test environment"
    echo "  $0 build                    # Build all tests"
    echo "  $0 run                      # Run all tests"
    echo "  $0 run CudaVectorAdd        # Run specific test suite"
    echo "  $0 run BasicVectorAddition  # Run specific test case"
    echo "  $0 -v run                   # Run all tests with verbose output"
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
    print_color $YELLOW "Setting up test run directory..."
    
    local config_dir="run/SM120_RTX5090"
    
    # Create run directory if it doesn't exist
    if [ ! -d "run" ]; then
        mkdir -p run
        print_color $BLUE "Created run directory"
    fi
    
    # Copy SM120_RTX5090 configuration if it doesn't exist or if forced
    if [ ! -d "$config_dir" ] || [ "$1" = "force" ]; then
        if [ -d "../configs/SM120_RTX5090" ]; then
            rm -rf "$config_dir" 2>/dev/null
            run_command cp -r "../configs/SM120_RTX5090" run/
            print_color $GREEN "Copied SM120_RTX5090 configuration to run directory"
        else
            print_color $RED "Error: SM120_RTX5090 config not found in ../configs/"
            print_color $YELLOW "Available configs:"
            run_command ls -1 ../configs/ 2>/dev/null || echo "  No configs directory found"
            exit 1
        fi
    else
        print_color $BLUE "SM120_RTX5090 configuration already exists in run directory"
    fi
}

# Build tests
build_tests() {
    print_color $BLUE "Building GPGPU-Sim tests..."
    
    if [ "$DEBUG_TESTS" -eq 1 ]; then
        run_command make CXXFLAGS="-std=c++17 -Wall -Wextra -pthread -g -O0 -DDEBUG" all
    else
        run_command make all
    fi
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Build successful!"
    else
        print_color $RED "Build failed!"
        exit 1
    fi
}

# List available tests
list_tests() {
    print_color $BLUE "Available tests:"
    
    if [ -d "src/unit" ]; then
        print_color $YELLOW "Unit Tests:"
        for test_file in src/unit/*_test.cc; do
            if [ -f "$test_file" ]; then
                test_name=$(basename "$test_file" .cc)
                echo "  - $test_name"
            fi
        done
    fi
    
    if [ -d "src/integration" ]; then
        print_color $YELLOW "Integration Tests:"
        for test_file in src/integration/*_test.cc; do
            if [ -f "$test_file" ]; then
                test_name=$(basename "$test_file" .cc)
                echo "  - $test_name"
            fi
        done
    fi
    
    if [ ! -d "src/unit" ] && [ ! -d "src/integration" ] && [ -d "src" ]; then
        print_color $YELLOW "All Tests:"
        for test_file in src/*_test.cc; do
            if [ -f "$test_file" ]; then
                test_name=$(basename "$test_file" .cc)
                echo "  - $test_name"
            fi
        done
    fi
}

# Run specific test
run_individual_test() {
    local test_name=$1
    local test_executable="build/bin/run_all_tests"
    local config_dir="run/SM120_RTX5090"
    
    if [ ! -f "$test_executable" ]; then
        print_color $RED "Test executable not found: $test_executable"
        print_color $YELLOW "Build tests first with: $0 build"
        exit 1
    fi
    
    # Verify config directory exists (should be created by initialize_run_directory)
    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        print_color $YELLOW "This should not happen - run directory setup failed"
        exit 1
    fi
    
    print_color $BLUE "Running specific test: $test_name from $config_dir"
    
    # Set up test environment variables
    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi
    
    # Get absolute path to test executable
    local abs_test_path="$(pwd)/$test_executable"
    
    # Change to config directory and run specific test
    cd "$config_dir"
    
    # Run specific test using gtest filter
    if command -v timeout &> /dev/null; then
        run_command timeout $TEST_TIMEOUT "$abs_test_path" --gtest_filter="*${test_name}*"
    else
        run_command "$abs_test_path" --gtest_filter="*${test_name}*"
    fi
    
    local exit_code=$?
    
    # Return to original directory
    cd - > /dev/null
    
    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Test $test_name passed"
    else
        print_color $RED "✗ Test $test_name failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Run all tests
run_all_tests() {
    print_color $BLUE "Running all GPGPU-Sim tests..."
    
    local test_executable="build/bin/run_all_tests"
    local config_dir="run/SM120_RTX5090"
    
    # Check if test executable exists
    if [ ! -f "$test_executable" ]; then
        print_color $RED "Test executable not found: $test_executable"
        print_color $YELLOW "Build tests first with: $0 build"
        exit 1
    fi
    
    # Verify config directory exists (should be created by initialize_run_directory)
    if [ ! -d "$config_dir" ]; then
        print_color $RED "Configuration directory not found: $config_dir"
        print_color $YELLOW "This should not happen - run directory setup failed"
        exit 1
    fi
    
    # Set up test environment variables
    export GTEST_COLOR=yes
    if [ "$TEST_VERBOSE" -eq 2 ]; then
        export GTEST_VERBOSITY=1
    fi
    
    print_color $BLUE "Running tests from configuration directory: $config_dir"
    
    # Get absolute path to test executable
    local abs_test_path="$(pwd)/$test_executable"
    
    # Change to config directory and run tests
    cd "$config_dir"
    
    # Run the test with timeout
    if command -v timeout &> /dev/null; then
        run_command timeout $TEST_TIMEOUT "$abs_test_path"
    else
        run_command "$abs_test_path"
    fi
    
    local exit_code=$?
    
    # Return to original directory
    cd - > /dev/null
    
    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ All tests passed!"
    else
        print_color $RED "✗ Some tests failed (exit code: $exit_code)"
        exit $exit_code
    fi
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
        run|build)
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
            build_tests
            exit 0
            ;;
        run)
            initialize_run_directory "run"
            if [ -n "$2" ]; then
                # Run specific test
                build_tests
                run_individual_test "$2"
                exit $?
            else
                # Run all tests
                build_tests
                run_all_tests
                exit $?
            fi
            ;;
        clean)
            clean_tests
            exit 0
            ;;
        list)
            list_tests
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