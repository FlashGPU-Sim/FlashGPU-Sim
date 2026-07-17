#!/bin/bash
# GPGPU-Sim Docker Entrypoint Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           GPGPU-Sim Development Environment                ║${NC}"
echo -e "${BLUE}║                   CUDA 12.8 Ready                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check CUDA installation
if [ -d "$CUDA_INSTALL_PATH" ]; then
    echo -e "${GREEN}✓ CUDA installation found at: ${CUDA_INSTALL_PATH}${NC}"
    nvcc --version | head -4
else
    echo -e "${RED}✗ CUDA installation not found at: ${CUDA_INSTALL_PATH}${NC}"
fi

# Setup GPGPU-Sim environment
cd /gpgpu-sim

BUILD_TYPE="${GPGPUSIM_BUILD_TYPE:-release}"

if [ -f "setup_environment" ]; then
    echo -e "${YELLOW}→ Configuring GPGPU-Sim environment...${NC}"
    
    # Source the environment for current script
    source setup_environment $BUILD_TYPE
    
    # Add to .bashrc so it persists in interactive shells
    if ! grep -q "source /gpgpu-sim/setup_environment" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# GPGPU-Sim environment (auto-configured by entrypoint)" >> ~/.bashrc
        echo "cd /gpgpu-sim && source setup_environment $BUILD_TYPE" >> ~/.bashrc
    fi
    
    echo -e "${GREEN}✓ GPGPU-Sim environment configured${NC}"
    echo -e "  Build type: ${BUILD_TYPE}"
    echo -e "  GPGPUSIM_ROOT: ${GPGPUSIM_ROOT}"
    echo -e "  GPGPUSIM_CONFIG: ${GPGPUSIM_CONFIG}"
else
    echo -e "${YELLOW}⚠ setup_environment not found, using default paths${NC}"
fi

# Check if we should build automatically
if [ "$AUTO_BUILD" = "1" ] || [ "$AUTO_BUILD" = "true" ]; then
    echo -e "${YELLOW}→ Auto-build enabled, building GPGPU-Sim...${NC}"
    
    if [ "$FLASH_MODE" = "1" ] || [ "$FLASH_MODE" = "true" ]; then
        echo -e "${BLUE}  Building with FLASH mode (multi-threaded)${NC}"
        make FLASH=1 -j$(nproc)
    else
        echo -e "${BLUE}  Building standard version${NC}"
        make -j$(nproc)
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
    else
        echo -e "${RED}✗ Build failed!${NC}"
    fi
fi

# Check if we should run tests automatically
if [ "$AUTO_TEST" = "1" ] || [ "$AUTO_TEST" = "true" ]; then
    echo -e "${YELLOW}→ Auto-test enabled, running tests...${NC}"
    
    if [ -d "test" ]; then
        cd test
        
        # Setup test environment if needed
        if [ ! -d "gtest" ]; then
            ./run_tests.sh setup
        fi
        
        # Build and run tests
        ./run_tests.sh build test
        ./run_tests.sh run test
        
        cd /gpgpu-sim
    else
        echo -e "${RED}✗ Test directory not found${NC}"
    fi
fi

# Print helpful information
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Environment auto-configured. Quick Start Commands:${NC}"
echo -e "  ${YELLOW}make -j\$(nproc)${NC}           - Build GPGPU-Sim (standard)"
echo -e "  ${YELLOW}make FLASH=1 -j\$(nproc)${NC}   - Build GPGPU-Sim (Flash/multi-threaded)"
echo -e "  ${YELLOW}cd test && ./run_tests.sh setup${NC}  - Setup test environment"
echo -e "  ${YELLOW}cd test && ./run_tests.sh run test${NC} - Run the default test suite"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Execute the provided command or start bash
exec "$@"
