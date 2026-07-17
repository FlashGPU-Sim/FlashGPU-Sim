# SM120_RTX5090_REDUCED Configuration

Lightweight GPU configuration for fast testing and development iterations.

## Purpose

This reduced configuration keeps the full RTX 5090 simulator configuration,
including performance-simulator pipelines and backpressure mechanisms, while
reducing the simulated GPU to one SM. It is designed for:

- Quick smoke tests during development
- Continuous integration (CI/CD) pipelines
- Rapid iteration on test cases
- Debugging and development workflows
- Functional-simulator and performance-simulator correctness testing

## Technical Specifications

### Architecture Comparison

| Parameter | SM120_RTX5090 (Full) | SM120_RTX5090_REDUCED |
|-----------|----------------------|----------------------|
| SM Clusters | 170 | 1 |
| Cores per Cluster | 1 | 1 |
| Total SMs | 170 | 1 |
| Memory Controllers | 16 | 16 |
| L2 Sub-partitions | 128 | 128 |
| Memory Channels | 16 | 16 |

### Key Differences

**Reduced from Full Configuration:**
- **SM count**: 170 → 1 (99.4% reduction)
- **Memory system**: Unchanged (16 controllers and 128 L2 sub-partitions)
- **Simulator options**: Identical except for `-gpgpu_n_clusters`
- **Cache hierarchy and timing**: Identical
- **Compute capability**: Same (12.0 - Blackwell architecture)
- **Interconnect file**: Reduced topology paired with the one-SM configuration

### Performance Characteristics

- **Simulation speed**: Workload-dependent; small single-block and smoke tests
  avoid maintaining 170 SMs, while large grids can take more simulated cycles
- **Memory usage**: Significantly lower (single SM state vs 170 SMs)
- **Coverage**: Exercises the same configured functional and performance-model
  mechanisms as the full configuration
- **Limitations**: Does not capture multi-SM interactions or scaling behavior

## When to Use

### ✅ Use SM120_RTX5090_REDUCED for:
- Unit tests and functional validation
- Performance-simulator mechanism and liveness validation
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
- Network topology paired with the reduced SM count
- Routing: Destination tag routing
- Flow control: Virtual channels, buffer sizes
- Router architecture: iSLIP arbitration

## Usage

```bash
# Run tests with reduced configuration
./test/run_tests.sh -c SM120_RTX5090_REDUCED run test --target sm120 --group unit

# Build and run a specific integration test
./test/run_tests.sh -c SM120_RTX5090_REDUCED run test --target sm120 --group integration CudaVectorAdd

# Set as default via environment variable
export GPU_CONFIG=SM120_RTX5090_REDUCED
./test/run_tests.sh run test --target sm120 --group unit
```

## Creating Custom Reduced Configurations

To create your own reduced configuration:

1. Copy this directory as a template
2. Modify `gpgpusim.config`:
   - Adjust `-gpgpu_n_clusters` (must be ≥ 1)
   - Keep `-gpgpu_n_mem` at 16 to preserve RTX 5090 parity
   - Keep every other simulator option synchronized with the full config
3. Update `config_ampere_islip.icnt` if changing topology
4. Test with `./test/run_tests.sh list-configs` to verify detection

## Notes

- Memory controller count remains 16 to match the full RTX 5090 config
- Single SM configuration sufficient for most functional tests
- For multi-threaded kernel testing, increase `-gpgpu_n_clusters` as needed
- Interconnect topology is explicitly configured for the reduced SM count
- `test/scripts/check_reduced_config_parity.py` enforces full/reduced field
  parity and permits only `-gpgpu_n_clusters` to differ for RTX 5090
- Reduced results are not valid for performance or scalability conclusions
