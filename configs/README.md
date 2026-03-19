# GPGPU-Sim Configuration Directory

This directory contains GPU hardware configurations for GPGPU-Sim simulations.

## Purpose

Configuration files define the simulated GPU architecture, including:
- Number and organization of streaming multiprocessors (SMs)
- Memory hierarchy (caches, shared memory, DRAM controllers)
- Pipeline parameters and instruction latencies
- Interconnect network topology and routing
- Clock domains and timing characteristics

## Available Configurations

### SM120_RTX5090
Full RTX 5090 configuration with 170 streaming multiprocessors.

**Use for:**
- Performance analysis and benchmarking
- Multi-SM workload validation
- Scalability testing
- Production-equivalent simulations

**Characteristics:**
- 170 SM clusters (1 core per cluster = 170 total SMs)
- 16 memory controllers
- High memory and time requirements
- Accurate performance modeling

### SM120_RTX5090_REDUCED
Lightweight configuration with 1 streaming multiprocessor.

**Use for:**
- Quick functional tests
- Development and debugging
- Continuous integration (CI/CD)
- Rapid iteration workflows

**Characteristics:**
- 1 SM cluster (1 core per cluster = 1 total SM)
- 16 memory controllers (GPGPU-Sim minimum)
- Low memory and time requirements
- Functionally equivalent for single-SM workloads

### deprecated-cfgs/
Legacy configurations maintained for reference.

### tested-cfgs/
Validated configurations from previous GPGPU-Sim versions.

## Configuration File Format

Each configuration directory must contain at minimum:

### Required Files

1. **gpgpusim.config** - Main GPU architecture configuration
   - GPU architecture parameters (SM count, memory, caches)
   - Pipeline widths and functional unit counts
   - Instruction latencies and initiation intervals
   - Memory hierarchy configuration
   - Clock domains

2. **config_*.icnt** - Interconnect network configuration
   - Network topology (butterfly fly, mesh, etc.)
   - Routing algorithm
   - Virtual channels and buffer sizes
   - Router architecture and arbitration

### Detection Requirements

For automatic detection by `test/run_tests.sh list-configs`:
- Directory must be in `configs/`
- Must contain `gpgpusim.config` file
- Directory name becomes the configuration name

## Adding Custom Configurations

### Step 1: Create Configuration Directory

```bash
mkdir configs/MY_CUSTOM_CONFIG
```

### Step 2: Copy Template Files

Start from an existing configuration:

```bash
# For full-featured config
cp configs/SM120_RTX5090/* configs/MY_CUSTOM_CONFIG/

# For lightweight config
cp configs/SM120_RTX5090_REDUCED/* configs/MY_CUSTOM_CONFIG/
```

### Step 3: Modify Configuration Parameters

Edit `configs/MY_CUSTOM_CONFIG/gpgpusim.config`:

**Key parameters to adjust:**

```bash
# Number of SM clusters (e.g., 1, 10, 85, 170)
-gpgpu_n_clusters 10

# Cores per cluster (typically 1)
-gpgpu_n_cores_per_cluster 1

# Memory controllers (must be 16 - GPGPU-Sim minimum)
-gpgpu_n_mem 16

# Memory partitions per controller
-gpgpu_n_sub_partition_per_mchannel 8

# Cache sizes (L1, L2)
-gpgpu_unified_l1d_size 128
-gpgpu_cache:dl2 S:256:128:24,L:B:m:L:P,A:192:4,32:0,32

# Shared memory size per SM
-gpgpu_shmem_size 102400
```

### Step 4: Update Interconnect Topology

Edit `configs/MY_CUSTOM_CONFIG/config_*.icnt` to match SM count:

For N SMs with 16 memory controllers, use topology:
```
topology = fly;
k = N + 16;  # Total endpoints (SMs + memory)
n = 1;       # Dimensions
```

### Step 5: Verify Configuration

```bash
# Check configuration is detected
./test/run_tests.sh list-configs

# Test with new configuration
./test/run_tests.sh test -c MY_CUSTOM_CONFIG
```

## Configuration Guidelines

### SM Count Selection

- **1 SM**: Functional testing, fast iteration
- **10-20 SMs**: Development testing with some parallelism
- **85 SMs**: Half-scale testing
- **170 SMs**: Full RTX 5090 simulation

### Memory Controller Constraints

GPGPU-Sim requires minimum 16 memory controllers:
```bash
-gpgpu_n_mem 16  # Fixed minimum
```

### Cache Configuration

L1 cache and shared memory share unified space:
```bash
# Adaptive: dynamically allocate between L1 and shared memory
-gpgpu_adaptive_cache_config 1
-gpgpu_unified_l1d_size 128  # KB per SM
-gpgpu_shmem_size 102400     # Bytes per SM
```

### Clock Domains

Format: `Core:Interconnect:L2:DRAM` (in MHz)
```bash
-gpgpu_clock_domains 2010:2010:2010:14000
```

## Testing Configurations

After creating a new configuration:

1. **Verify detection**: `./test/run_tests.sh list-configs`
2. **Run basic test**: `./test/run_tests.sh test -c MY_CONFIG CudaVectorAdd`
3. **Validate output**: Check simulation completes without errors
4. **Document**: Add README.md in config directory describing purpose and specs

## Troubleshooting

### Configuration not detected
- Verify `gpgpusim.config` exists in directory
- Check file permissions (must be readable)
- Ensure directory is directly under `configs/`

### Simulation fails to start
- Verify memory controller count is 16
- Check interconnect topology matches SM count
- Validate cache configuration syntax

### Performance issues
- Reduce SM count for faster simulation
- Use SM120_RTX5090_REDUCED for development
- Check memory subsystem configuration

## References

- [GPGPU-Sim Manual](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
- RTX 5090 Architecture: Ampere (compute capability 8.6)
- Interconnect: Booksim-based network simulator
