# FlashGPU-Sim

FlashGPU-Sim is an execution-driven, cycle-accurate simulator for modern GPU
architectures and AI workloads, built upon GPGPU-Sim. The project includes
updated GPU execution models and configurations, Triton capture-and-replay
tools, validation infrastructure, and multi-threaded simulation support.

[Simulator Setup](#simulator-setup) · [Tutorials](#tutorials) ·
[Documentation](#documentation) · [Citation](#citation)

<!-- Summarize the major capabilities in a compact status table. Distinguish
Hopper/SM90 from Blackwell/SM120 and use conservative status labels such as
Supported, Partial, and Experimental. -->

| Area | Capability | Status |
| --- | --- | --- |
| Architectures | Hopper/SM90 and Blackwell/SM120 configurations | To be documented |
| Modern GPU features | TMA, memory barriers, MMA/WGMMA, and matrix operations | To be documented |
| AI workloads | Triton capture/replay and FlashAttention validation flows | To be documented |
| Simulation | Cycle-level functional and timing simulation | To be documented |
| Scalability | Multi-threaded simulation | To be documented |
| Validation | Integration tests, microbenchmarks, and hardware comparison tools | To be documented |

## Roadmap

<!-- List a small number of high-level development directions. Keep detailed
tasks and rapidly changing feature status outside the root README. -->

## Quick Start

### Dependencies

FlashGPU-Sim is developed and tested on Linux. A host build requires:

- CUDA Toolkit
- GCC/G++ with C++17 and OpenMP support
- GNU Make, Flex, Bison, zlib development headers, and X11/OpenGL development
  headers

The current build and CI environments use CUDA 12.8. A physical GPU is not
required to build or run the simulator, but is required to capture Triton
workloads or collect Nsight Compute measurements.

### Build

Configure the environment and build the simulator from the repository root:

```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
make -j $(nproc)
```

> Remember to export `CUDA_INSTALL_PATH` and source `setup_environment`
> whenever you open a new shell.

### Run a Simulation

Using the environment configured above, run the bundled CUDA vector addition example:

```bash
cd tutorial/vectorAdd
./run.sh
```

The script builds the workload, prepares the `SM120_RTX5090` configuration in
`tutorial/vectorAdd/run/`, verifies that the executable uses FlashGPU-Sim's
CUDA runtime, and starts the simulation. Output is displayed in the terminal
and saved to `tutorial/vectorAdd/run/simulation.log`. A successful run prints
`Test PASSED` for functional correctness. Inspect the simulated cycle count
with:

```bash
grep gpu_tot_sim_cycle run/simulation.log
```

## Tutorials

### CUDA Vector Addition

1. Compile the application with the shared CUDA runtime and retain PTX for the
   target architecture. For example, to target the SM120 configuration:

   ```bash
   nvcc -std=c++17 -cudart shared \
     -gencode arch=compute_120a,code=sm_120a \
     -gencode arch=compute_120a,code=compute_120a \
     application.cu -o application
   ```

   The executable must link against the shared CUDA runtime:

   - When `nvcc` performs the final link, use `-cudart shared`.
   - When `g++` performs the final link, use
     `-L"$CUDA_INSTALL_PATH/lib64" -lcudart`.

   Do not link `libcudart_static`. A statically linked CUDA runtime bypasses
   FlashGPU-Sim's replacement runtime and invokes the real NVIDIA driver, so
   the application is not intercepted by the simulator. The PTX target must
   match the selected simulator configuration.

2. Create a run directory containing the executable and the complete GPU
   configuration, then run the application from that directory:

   ```bash
   mkdir -p run/application
   cp -a configs/SM120_RTX5090/. run/application/
   cp application run/application/
   cd run/application
   ./application
   ```

FlashGPU-Sim reads `gpgpusim.config` from the current working directory, along
with any files referenced by that configuration. A successful simulation
prints `gpu_tot_sim_cycle` in its output.

### Triton Kernel Capture and Replay

<!-- Introduce and link to the standalone Triton tutorial under tutorial/.
Clearly separate native-GPU capture from simulator replay. -->

## Example Results

<!-- Show one or two reproducible correctness and cycle-comparison results.
Record the workload shape, hardware, simulator configuration, software
versions, measured metric, and diff formula alongside the numbers. -->

## Configurations and Limitations

<!-- Introduce the primary Hopper/SM90 and Blackwell/SM120 configurations, then
state the important support boundaries and experimental features. -->

## Documentation

<!-- Link to the documentation index, build and testing guides, configuration
reference, and developer documentation. -->

## Citation

We hope FlashGPU-Sim benefits your research! If you use it in your work, please cite our paper:

```text
Siying Yu, Yixun Hong, Guozhi Qiu, Feng Gu, Chenbo Geng, Zhengrong Wang, Chen Zhang, Bei Yu,
FlashGPU-sim: Enabling GPU Modeling for Modern Architectures and AI Workloads,
in 2026 IEEE/ACM 59th International Symposium on Microarchitecture (MICRO)
```

BibTeX:
```bibtex
@inproceedings{flashgpusim,
  author    = {Yu, Siying and Hong, Yixun and Qiu, Guozhi and Gu, Feng
               and Geng, Chenbo and Wang, Zhengrong and Zhang, Chen and Yu, Bei},
  title     = {{FlashGPU-sim}: Enabling GPU Modeling for Modern Architectures and AI Workloads},
  booktitle = {IEEE/ACM 59th International Symposium on Microarchitecture (MICRO)},
  year      = {2026}
}
```

## License and Acknowledgements

<!-- Link to the repository copyright and licensing notices. Keep the archived
upstream GPGPU-Sim documentation link here as historical background. -->
