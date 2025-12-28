#!/bin/bash
# GPGPU-Sim Docker Helper Script
# Simplifies common Docker operations for GPGPU-Sim development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
SERVICE="${SERVICE:-gpgpusim}"
COMPOSE_FILE="docker/docker-compose.yml"

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           GPGPU-Sim Docker Development Environment            ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    print_banner
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  start              Start container"
    echo "  stop               Stop all containers"
    echo "  shell              Open shell in container (starts if not running)"
    echo "  logs               Show container logs"
    echo "  clean              Remove containers and images"
    echo ""
    echo "Examples:"
    echo "  $0 build           # Build Docker image"
    echo "  $0 shell           # Enter container shell"
    echo "  $0 stop            # Stop container"
    echo "  $0 clean           # Remove all containers and images"
    echo ""
    echo "Inside container:"
    echo "  source setup_environment        # Setup environment"
    echo "  make -j\$(nproc)                 # Build standard version"
    echo "  make FLASH=1 -j\$(nproc)         # Build Flash mode"
    echo ""
}

cmd_build() {
    echo -e "${BLUE}Building Docker image...${NC}"
    docker compose -f "$COMPOSE_FILE" build
    echo -e "${GREEN}✓ Build complete${NC}"
}

cmd_start() {
    echo -e "${BLUE}Starting container...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d "$SERVICE"
    echo -e "${GREEN}✓ Container started${NC}"
    echo -e "${YELLOW}Run '$0 shell' to enter the container${NC}"
}

cmd_stop() {
    echo -e "${BLUE}Stopping containers...${NC}"
    docker compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✓ Containers stopped${NC}"
}

cmd_shell() {
    echo -e "${BLUE}Opening shell...${NC}"
    
    # Check if container is running
    if ! docker compose -f "$COMPOSE_FILE" ps --status=running "$SERVICE" | grep -q "$SERVICE"; then
        echo -e "${YELLOW}Container not running, starting it first...${NC}"
        docker compose -f "$COMPOSE_FILE" up -d "$SERVICE"
        sleep 2
    fi
    
    docker compose -f "$COMPOSE_FILE" exec "$SERVICE" bash
}

cmd_logs() {
    echo -e "${BLUE}Showing logs...${NC}"
    docker compose -f "$COMPOSE_FILE" logs -f "$SERVICE"
}

cmd_clean() {
    echo -e "${YELLOW}This will remove all GPGPU-Sim containers and images.${NC}"
    read -p "Are you sure? (y/N) " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Cleaning up...${NC}"
        docker compose -f "$COMPOSE_FILE" down -v --rmi all 2>/dev/null || true
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo -e "${YELLOW}Cancelled${NC}"
    fi
}

# Main
case "${1:-help}" in
    build)
        cmd_build
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    shell)
        cmd_shell
        ;;
    logs)
        cmd_logs
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_help
        exit 1
        ;;
esac
