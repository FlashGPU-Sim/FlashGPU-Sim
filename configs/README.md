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
- FA2 SASS-guided PTX reorder runs

**Characteristics:**
- 170 SM clusters (1 core per cluster = 170 total SMs)
- 16 memory controllers
- High memory and time requirements
- Accurate performance modeling
- PTX register allocation and SASS-guided PTX reorder enabled by default
- Includes `sass_primary_hints.rules` for auto-extracted full-SASS guides

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

### SM90_H100
Full Hopper H100 configuration with 132 streaming multiprocessors.

**Use for:**
- Hopper/FA3 final validation
- WGMMA and TMA behavior with full H100 SM count
- Performance-oriented H100 experiments
- FA2/FA3 SASS-guided PTX reorder runs

**Characteristics:**
- 132 SM clusters (1 core per cluster = 132 total SMs)
- 80 HBM memory controllers, modeled as 160 L2/HBM subpartitions
- Full H100 shared memory sizing with occupancy-aware default carveout
- PTX register allocation and SASS-guided PTX reorder enabled by default
- Includes `sass_primary_hints.rules` for auto-extracted full-SASS guides

### SM90_H100_1500MHZ
Full Hopper H100 configuration with core, interconnect, and L2 clocks fixed at
1.5 GHz.

**Use for:**
- H100/NCU comparisons when the measured SM clock is near 1.5 GHz
- FA3 performance tuning runs that should not mutate the default H100 config

**Characteristics:**
- Same SM count, memory hierarchy, and TMA/WGMMA settings as `SM90_H100`
- `-gpgpu_clock_domains 1500:1500:1500:14000`

### FlashAttention H100 result configs
The current FlashAttention H100 result set uses these primary configs:

- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700`
  - FA3 baseline and generic H100 baseline.
- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA3_WGMMA_RF`
  - Current FA3 large-run config. Starts from the shared FA register-allocation/PTX-scheduling config, uses TMA 128-byte request granularity with OOB read traffic, and models pending WGMMA accumulator RF pressure as shared operand-collector read traffic.
- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA3_TMA128_OOB_READ`
  - Previous FA3 large-run config without the WGMMA RF pressure model. Starts from the FA3 baseline, uses TMA 128-byte request granularity, and models OOB TMA reads as L2 traffic while OOB writes are skipped.
- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA2_REGALLOC`
  - FA2 baseline. Enables PTX register allocation and PTX scheduling.
- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA2_IDEAL`
  - FA2 sensitivity config. Enables PTX register allocation/scheduling, idealized tensor-core issue queue, tensor-core init 4, and ideal cp.async frontend latency. Backend memory latency is still modeled.
- `SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA3_TMA128_NO_OOB`
  - FA3 sensitivity config from the final rebuttal archive. Uses TMA 128-byte request granularity and disables synthetic OOB L2 traffic.

The FA3 WGMMA RF config uses an intentionally compact pressure model. Each
WGMMA MMA injects one accumulator-sized RF-read-equivalent token:

```
4 warps * 32 lanes * accumulator registers per thread * accumulator bytes
```

For the common FA3 fp16->fp32 WGMMA shapes, the accumulator footprint is derived
from the destination vector width, falling back to `N/2` registers per thread
when the vector width is unavailable. With fp32 accumulators this is `128 *
(N/2) * 4` bytes per WGMMA, e.g. 32 KiB for `n128` and 44 KiB for `n176`.
The model drains those tokens at 512 B/cycle/SM from the same 1024 B/cycle/SM
effective RF read budget used by the normal operand collector. This is not a
full RF read/write bandwidth model for WGMMA itself; calibrated WGMMA issue,
compute, and completion timing still determine tensor throughput. The token
drain models the observed extra RF/collector pressure that pending WGMMA
accumulator traffic places on independent fp/int instructions. Accumulate and
overwrite forms are charged the same because the simulator does not yet model a
separate RF write path, and the focused H100 microbenchmarks did not show a
large enough accumulate/overwrite split to justify a more invasive model.

Intermediate parameter sweeps for cp.async, TMA response width, WGMMA/MMA queue
experiments, and most temporary `_TMP` configs are intentionally not kept here.
`TMP_FA2_CPASYNC_FORMAL_IPOLY1024_RANGE_MAP9` is retained only as a baseline
reference. The final raw logs and reports are archived outside this repository
in `flashgpu_sim_micro26_rebuttal`.

### SM90_H100_REDUCED
Lightweight Hopper H100 configuration with 1 streaming multiprocessor.

**Use for:**
- FA3 deadlock/debug iterations
- Reduced log volume
- Hopper instruction semantics debugging

**Characteristics:**
- 1 SM cluster (1 core per cluster = 1 total SM)
- 16 memory controllers (GPGPU-Sim minimum)
- Full H100 per-SM shared memory and TMA settings

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
