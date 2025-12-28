# GPGPU-Sim Docker Environment

This directory contains Docker configuration files for the GPGPU-Sim development environment.

## Files

- `Dockerfile` - Docker image definition based on NVIDIA CUDA 12.8
- `docker-compose.yml` - Docker Compose service configurations
- `entrypoint.sh` - Container entrypoint script

## Usage

All commands should be run from the **project root directory** using `docker.sh`:

```bash
./docker.sh <command>
```

## Commands

| Command | Description |
|---------|-------------|
| `build` | Build Docker image |
| `start` | Start container |
| `stop` | Stop all containers |
| `shell` | Open shell in container (starts if not running) |
| `logs` | Show container logs |
| `clean` | Remove containers and images |

## Quick Start

```bash
# Build Docker image
./docker.sh build

# Enter container shell (environment auto-configured)
./docker.sh shell

# Inside container: build directly
make -j$(nproc)                 # Standard build
# or
make FLASH=1 -j$(nproc)         # Flash mode build

# Run tests (inside container)
cd test
./run_tests.sh setup
./run_tests.sh run
```

## Environment Variables

You can customize the container by editing `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_INSTALL_PATH` | `/usr/local/cuda` | CUDA installation path |
| `PTXAS_CUDA_INSTALL_PATH` | `/usr/local/cuda` | PTXAS CUDA path |
| `GPGPUSIM_BUILD_TYPE` | `release` | Build type (release/debug) |
| `OMP_NUM_THREADS` | `8` | OpenMP threads for Flash mode |
