# FlashGPU-Sim <!-- omit from toc -->

- [Roadmap](#roadmap)
- [Quick Start](#quick-start)
  - [Dependencies](#dependencies)
  - [Build the Simulator](#build-the-simulator)
  - [Run a Simulation](#run-a-simulation)
- [Tutorials](#tutorials)
  - [Run with CUDA](#run-with-cuda)
    - [Step 1: Compile the CUDA Workload](#step-1-compile-the-cuda-workload)
    - [Step 2: Prepare the Configuration and Execute](#step-2-prepare-the-configuration-and-execute)
  - [Run with Triton](#run-with-triton)
    - [Step 1: Add Triton Kernel Tracking](#step-1-add-triton-kernel-tracking)
    - [Step 2: Capture the Triton Workload](#step-2-capture-the-triton-workload)
    - [Step 3: Build the Standalone Harness](#step-3-build-the-standalone-harness)
    - [Step 4: Simulate the Captured Kernel](#step-4-simulate-the-captured-kernel)
  - [Update Configuration](#update-configuration)
  - [Set CPU Threads](#set-cpu-threads)
  - [Inspect Statistics](#inspect-statistics)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

FlashGPU-Sim is an execution-driven, cycle-accurate simulator for modern GPU
architectures and AI workloads, built upon GPGPU-Sim. The project includes
updated GPU execution models and configurations, Triton capture-and-replay
tools, validation infrastructure, and multi-threaded simulation support.

| Area | Supported capabilities |
| --- | --- |
| Architectures | Hopper/SM90 ([`SM90_H100`](configs/SM90_H100/gpgpusim.config)) and Blackwell/SM120 ([`SM120_RTX5090`](configs/SM120_RTX5090/gpgpusim.config)) configurations |
| GPU features | TMA, `mbarrier`, `mma`, `wgmma`, `ldmatrix`/`stmatrix`, etc. |
| Workload tooling | Triton kernel capture and standalone replay |
| Simulation | Execution-driven functional simulation and cycle-level timing simulation |
| Parallelism | OpenMP-based multi-threaded SM simulation |

## Roadmap

Near-term priorities:

- **Blackwell architecture features:** Extend beyond the current
  SM120/RTX 5090 configuration toward Blackwell features used by
  FlashAttention-4, including `tcgen05`.
- **Distributed shared memory:** Model thread-block cluster features,
  such as remote shared-memory addressing and access.

Longer-term directions:

- **gem5 Integration:** Integrate gem5 as an alternative memory-system backend.
- **Scale-up multi-GPU simulation:** Model native load/store access to remote
  GPU memory, including transaction processing, fabric transport, unified
  addressing, and memory-model behavior.

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

A successful run prints `Test PASSED` for functional correctness.
Inspect the simulated cycle count with:

```bash
grep gpu_tot_sim_cycle run/simulation.log
```

For Triton capture and replay, see the [Triton GEMM tutorial](#run-with-triton).

## Tutorials

The bundled scripts automate each example end to end; the sections below
break down the same workflows for adaptation to custom CUDA and Triton
workloads.

### Run with CUDA

This workflow manually steps through the bundled `vectorAdd` example.  
Prerequisites: Ensure FlashGPU-Sim is built, and `setup_environment` is sourced.  
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

Copy the GPU configuration into the `run/` directory and start the simulation:

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
Prerequisites: A physical GPU, and a Python environment with PyTorch, Triton,
and NumPy.
From the repository root, enter the example directory:

```bash
cd tutorials/triton-gemm
```

#### Step 1: Add Triton Kernel Tracking

Install TritonTrace in the active Python environment:

```bash
python -m pip install -e ../../tools
```

Instrument a Triton workload with the following tracking sequence:

```python
import tritontrace

tracker = tritontrace.Tracker(
    "run/tracking", save_binaries=True, capture_args=True
)
tracker.disable()
run_warmup_and_autotuning()

tracker.enable()
run_kernel_once()
tracker.save_summary()
```

Keep tracking disabled during compilation, warmup, and autotuning, then enable
it for the kernel launch to simulate. The bundled `gemm.py` already implements
this sequence, so no code changes are needed for this example.

#### Step 2: Capture the Triton Workload

Run the capture:

```bash
python gemm.py
```

The default workload is a `256 x 256 x 256` GEMM. It runs Triton autotuning,
checks the result against PyTorch, and captures one kernel launch under
`run/tracking/`. The captured launch is materialized as a standalone CUDA C++
harness, the compiled kernel in PTX and CUBIN form, launch metadata, serialized
arguments, and reference outputs.

> [!IMPORTANT]
> **Do NOT source `setup_environment` before capture.** Open a new shell and
> run the capture with the native CUDA stack on the physical GPU.

#### Step 3: Build the Standalone Harness

```bash
make -C run/tracking/launchers \
  -f kernel_tma_gemm_launch1_Makefile
```

The generated harness reconstructs the captured arguments and validates its
output against the values recorded during capture.

#### Step 4: Simulate the Captured Kernel

The harness replaces the original Python/Triton program during replay. It
restores the captured arguments and reissues the recorded kernel launch;
FlashGPU-Sim uses the captured PTX sidecar to simulate the kernel.

Copy the matching configuration, configure FlashGPU-Sim, and run the harness:

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
Validation PASSED for arg[2]: all 65536 elements match
Done!
```

### Update Configuration

### Set CPU Threads

### Inspect Statistics

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
