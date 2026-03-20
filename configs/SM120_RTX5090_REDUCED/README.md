# SM120_RTX5090_REDUCED Configuration

Lightweight GPU configuration for fast testing and development iterations.

## Purpose

This reduced configuration provides a minimal GPGPU-Sim setup that maintains functional correctness while dramatically reducing simulation time and memory usage. It is designed for:

- Quick smoke tests during development
- Continuous integration (CI/CD) pipelines
- Rapid iteration on test cases
- Debugging and development workflows
- Functional validation (non-performance testing)

## Technical Specifications

### Architecture Comparison

| Parameter | SM120_RTX5090 (Full) | SM120_RTX5090_REDUCED |
|-----------|----------------------|----------------------|
| SM Clusters | 170 | 1 |
| Cores per Cluster | 1 | 1 |
| Total SMs | 170 | 1 |
| Memory Controllers | 16 | 16 (minimum supported) |
| L2 Partitions | 16 | 16 |
| Memory Channels | 16 | 16 |

### Key Differences

**Reduced from Full Configuration:**
- **SM count**: 170 → 1 (99.4% reduction)
- **Memory system**: Unchanged (16 controllers - GPGPU-Sim minimum)
- **Cache hierarchy**: Same per-SM configuration
- **Compute capability**: Same (8.6 - Ampere architecture)

### Performance Characteristics

- **Simulation speed**: ~170x faster for SM-bound workloads
- **Memory usage**: Significantly lower (single SM state vs 170 SMs)
- **Accuracy**: Functionally equivalent for single-SM workloads
- **Limitations**: Does not capture multi-SM interactions or scaling behavior

## When to Use

### ✅ Use SM120_RTX5090_REDUCED for:
- Unit tests and functional validation
- Quick sanity checks after code changes
- Development and debugging
- CI/CD automated testing
- Testing instruction semantics and correctness

### ❌ Use SM120_RTX5090 (full) for:
- Performance analysis and benchmarking
- Multi-SM workload validation
- Scalability testing
- Final validation before release
- Reproducing production GPU behavior

## Configuration Files

### gpgpusim.config
Main simulator configuration defining:
- GPU architecture (SM count, memory controllers, cache sizes)
- Pipeline parameters (instruction latencies, functional unit counts)
- Memory hierarchy (L1/L2 cache configuration, shared memory)
- Clock domains and timing parameters

### config_ampere_islip.icnt
Interconnect network configuration defining:
- Network topology: 2x1 butterfly fly network (1 SM + 1 memory partition endpoint)
- Routing: Destination tag routing
- Flow control: Virtual channels, buffer sizes
- Router architecture: iSLIP arbitration

## Usage

```bash
# Run tests with reduced configuration
./test/run_tests.sh test -c SM120_RTX5090_REDUCED

# Build and run specific test
./test/run_tests.sh -c SM120_RTX5090_REDUCED test CudaVectorAdd

# Set as default via environment variable
export GPU_CONFIG=SM120_RTX5090_REDUCED
./test/run_tests.sh test
```

## Creating Custom Reduced Configurations

To create your own reduced configuration:

1. Copy this directory as a template
2. Modify `gpgpusim.config`:
   - Adjust `-gpgpu_n_clusters` (must be ≥ 1)
   - Keep `-gpgpu_n_mem` at 16 (GPGPU-Sim minimum)
3. Update `config_ampere_islip.icnt` if changing topology
4. Test with `./test/run_tests.sh list-configs` to verify detection

## Notes

- Memory controller count fixed at 16 (GPGPU-Sim architectural minimum)
- Single SM configuration sufficient for most functional tests
- For multi-threaded kernel testing, increase `-gpgpu_n_clusters` as needed
- Interconnect topology automatically adjusted to match SM count
