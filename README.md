# FlashGPU-Sim <!-- omit from toc -->

- [Roadmap](#roadmap)
- [Quick Start](#quick-start)
  - [Dependencies](#dependencies)
  - [Build the Simulator](#build-the-simulator)
  - [Run a Simulation](#run-a-simulation)
  - [Cycle Validation](#cycle-validation)
- [Tutorials](#tutorials)
  - [Run with CUDA](#run-with-cuda)
    - [Step 1: Compile the CUDA Workload](#step-1-compile-the-cuda-workload)
    - [Step 2: Prepare the Configuration and Execute](#step-2-prepare-the-configuration-and-execute)
  - [Run with Triton](#run-with-triton)
    - [Step 1: Capture the Triton Workload](#step-1-capture-the-triton-workload)
    - [Step 2: Build the Standalone Harness](#step-2-build-the-standalone-harness)
    - [Step 3: Simulate the Captured Kernel](#step-3-simulate-the-captured-kernel)
  - [Update Configuration](#update-configuration)
  - [Set CPU Threads](#set-cpu-threads)
  - [Inspect Statistics](#inspect-statistics)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

FlashGPU-Sim is an execution-driven, cycle-accurate simulator for modern GPU
architectures and AI workloads, built upon GPGPU-Sim.

| Area | Supported capabilities |
| --- | --- |
| Architectures | Hopper/SM90 ([`SM90_H100`](configs/SM90_H100/gpgpusim.config)) and Blackwell/SM120 ([`SM120_RTX5090`](configs/SM120_RTX5090/gpgpusim.config)) configurations |
| GPU features | TMA, `mbarrier`, `mma`, `wgmma`, `ldmatrix`/`stmatrix`, etc. |
| Workload tooling | [TritonTrace](tools/README.md) kernel capture and standalone replay ([examples and validation](test/triton_trace/README.md)) |
| Simulation | Execution-driven functional simulation and cycle-level timing simulation |
| Parallelism | OpenMP-based multi-threaded SM simulation |

## Roadmap

| Direction | Status | Planned work |
| --- | --- | --- |
| Blackwell features | In progress | Extend the SM120/RTX 5090 model with features used by FlashAttention-4, including `tcgen05` |
| Distributed shared memory | In progress | Model thread-block clusters, remote shared-memory addressing, and remote accesses |
| gem5 integration | Experimental | Stabilize gem5 as an alternative memory-system backend |
| Scale-up multi-GPU simulation | Planned | Model native remote-memory loads/stores, fabric transport, unified addressing, and memory consistency |

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

### Build the Simulator

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
cd tutorials/vectorAdd
./run.sh
```

To capture and simulate the bundled Triton GEMM, open a clean shell with access
to a physical GPU and a Python environment providing PyTorch, Triton, and
NumPy:

```bash
cd tutorials/triton-gemm
python -m pip install -e ../../tools
./capture.sh
./run.sh
```

Both workflows write simulator output to `run/simulation.log` in their respective tutorial directories.  
A successful CUDA run reports `Test PASSED`, while a successful Triton replay reports `Validation PASSED`.  
In both cases, `gpu_tot_sim_cycle` confirms that the workload ran with FlashGPU-Sim.
Triton capture output is saved separately to `run/capture.log`.

### Cycle Validation

The table below compares `gpu_tot_sim_cycle` with
`sm__cycles_elapsed.avg` reported by Nsight Compute on an RTX 5090.
Difference is calculated as `(Sim - NCU) / NCU`.

| Workload | Shape | NCU cycles | Sim cycles | Difference |
| --- | --- | ---: | ---: | ---: |
| CUDA vector addition | 2,000,000 elements | 29,642.67 | 30,133 | +1.65% |
| Triton GEMM | `M=2560, N=64, K=2560` | 77,190.74 | 78,989 | +2.33% |
| Triton FlashAttention | `B=32, H=32, S=512, D=64, causal` | 797,247.12 | 794,076 | -0.40% |

> [!TIP]
> We provide RTX 5090 Nsight Compute reports and CSVs for
> [CUDA vector addition](tutorials/vectorAdd/reference/),
> [Triton GEMM](tutorials/triton-gemm/reference/), and
> [Triton FlashAttention](tutorials/triton-flash-attention/reference/).
> This enables direct validation of simulator results 
> without requiring physical access to an RTX 5090.
>
> To regenerate the provided data on compatible hardware, lock the GPU clocks
> and run
> `./tutorials/profile_ncu.sh --gpu 0 <workload>`
> (where `<workload>` can be `all`, `vectorAdd`, `triton-gemm`, or
> `triton-flash-attention`).

## Tutorials

The bundled scripts automate each example end to end; the sections below
break down the same workflows for adaptation to custom CUDA and Triton
workloads.

### Run with CUDA

This workflow manually steps through the bundled `vectorAdd` example.  
**Prerequisites**: Ensure FlashGPU-Sim is built, and `setup_environment` is sourced.  
From the repository root, enter the example directory:  

```bash
cd tutorials/vectorAdd
```

#### Step 1: Compile the CUDA Workload

Create a `run/` directory and compile `vectorAdd.cu` with the shared CUDA runtime, 
generating a code image and retaining PTX for the target
architecture. The following example targets SM120:

```bash
mkdir -p run
nvcc -std=c++17 -cudart shared \
  -gencode arch=compute_120a,code=sm_120a \
  -gencode arch=compute_120a,code=compute_120a \
  vectorAdd.cu -o run/vectorAdd
```

The `compute_120a` and `sm_120a` targets match the bundled
`configs/SM120_RTX5090/` configuration. When selecting another simulator
configuration, adjust both targets to its compute capability.

> [!IMPORTANT]
> FlashGPU-Sim intercepts CUDA runtime calls by loading its replacement
> `libcudart` through the library path set by `setup_environment`. The
> executable must therefore depend on the shared CUDA runtime.
>
> - When `nvcc` performs the final link, use `-cudart shared`.
> - When `g++` performs the final link, use
>   `-L"$CUDA_INSTALL_PATH/lib64" -lcudart`.
> - Do not link `libcudart_static.a`; embedding the CUDA runtime in the
>   executable prevents FlashGPU-Sim's replacement library from being loaded.

#### Step 2: Prepare the Configuration and Execute

Copy the selected GPU configuration into the `run/` directory. See the
[configuration guide](configs/README.md) for available GPU models, common
parameters, and FlashGPU-Sim-specific options.

```bash
cp -a ../../configs/SM120_RTX5090/. run/
cd run && ./vectorAdd 2>&1 | tee simulation.log
```

FlashGPU-Sim reads `gpgpusim.config` from the current working directory, along
with any files referenced by that configuration. A successful run reports
`Test PASSED` for functional correctness and `gpu_tot_sim_cycle` for the
simulated cycle count; the complete output is saved to `simulation.log`.

### Run with Triton

This workflow manually captures the `triton-gemm` example on a physical GPU
and replays it with FlashGPU-Sim.  
**Prerequisites**: A physical GPU, and a Python environment with PyTorch, Triton,
and NumPy.  
From the repository root, enter the example directory:

```bash
cd tutorials/triton-gemm
```

#### Step 1: Capture the Triton Workload

Install TritonTrace in the active Python environment:

```bash
python -m pip install -e ../../tools
```

Capturing another Triton workload requires minor instrumentation to import
TritonTrace, create a tracker, and enable it around the target kernel launch.
The bundled `gemm.py` already includes this instrumentation, so no code changes
are needed for this example. See the
[TritonTrace documentation](tools/README.md) for integration details, capture
internals, generated artifacts, replay workflow, and limitations.

> [!IMPORTANT]
> **Do NOT source `setup_environment` before capture.** Open a new shell and
> run the capture with the native CUDA stack on the physical GPU.

Run the capture:

```bash
python gemm.py
```

The default workload is a `2560 x 64 x 2560` GEMM. It runs Triton autotuning,
checks the result against PyTorch, and captures one kernel launch under
`run/tracking/`. The capture generates the following artifacts:
  - A standalone CUDA C++ harness
  - The compiled kernel in PTX and CUBIN formats
  - Launch metadata and serialized arguments
  - Reference outputs for validation

#### Step 2: Build the Standalone Harness

```bash
make -C run/tracking/launchers \
  -f kernel_tma_gemm_launch1_Makefile
```

The generated harness reconstructs the captured arguments and validates its
output against the values recorded during capture.

#### Step 3: Simulate the Captured Kernel

The harness replaces the original Python/Triton program during replay. It
restores the captured arguments and reissues the recorded kernel launch;
FlashGPU-Sim uses the captured PTX file to simulate the kernel.

Copy the matching GPU configuration described in the
[configuration guide](configs/README.md), configure FlashGPU-Sim, and run the
harness:

```bash
cp -a ../../configs/SM120_RTX5090/. run/tracking/launchers/
source ../../setup_environment
cd run/tracking/launchers
./kernel_tma_gemm_launch1 2>&1 | tee ../../simulation.log
```

A successful run compares every element of the simulated GEMM output with the
output recorded during capture and finishes with:

```text
Kernel execution completed successfully
Validation PASSED for arg[2]: all 163840 elements match
Done!
```

> [!NOTE]
> A Triton [FlashAttention example](tutorials/triton-flash-attention/) is also
> provided with automated capture and simulation scripts. In our testing, its
> default workload took approximately 50 minutes to simulate with
> `OMP_NUM_THREADS=4` on an Intel Core i9-14900K.

### Update Configuration

FlashGPU-Sim reads `gpgpusim.config` from the workload's current directory at
startup. Configuration changes take effect on the next simulation without
rebuilding FlashGPU-Sim.

See the [Configuration Guide](configs/README.md) for common parameters,
FlashGPU-Sim-specific options, and experimental controls.

### Set CPU Threads

Set the number of CPU threads used by FlashGPU-Sim with `OMP_NUM_THREADS`:

```bash
export OMP_NUM_THREADS=8
```

We recommend using 4 or 8 threads, because higher thread counts provided limited
additional speedup in our testing. Use no more than 32 threads. See
[Parallel Simulation](docs/development-notes.md#parallel-simulation) for
implementation details.

### Inspect Statistics

FlashGPU-Sim prints detailed simulation statistics to the run log. Use the
provided helper from the repository root to summarize the final statistics
report:

```bash
python3 test/scripts/extract_sim_stats.py tutorials/vectorAdd/run/simulation.log
```

The summary includes `simulated cycles`, `instructions`, `IPC`, `occupancy`,
`cache statistics`, and `DRAM command counts`. Pass `--all` to inspect every
statistics report or `--csv output.csv` to export the results.

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

FlashGPU-Sim is built upon GPGPU-Sim and includes AccelWattch components. We
retain their upstream copyright and license notices in
[COPYRIGHT](COPYRIGHT), and thank their authors and contributors for their
foundational work. The [archived upstream GPGPU-Sim
documentation](docs/legacy/gpgpu-sim.md) is retained for historical reference.
