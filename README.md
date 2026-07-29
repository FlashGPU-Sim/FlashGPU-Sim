# FlashGPU-Sim

FlashGPU-Sim is a cycle-accurate simulator for modern GPU architectures
and AI workloads.

[Getting Started](#getting-started) · [Tutorials](#tutorials) ·
[Documentation](#documentation) · [Citation](#citation)

## Overview

<!-- Briefly introduce the problems FlashGPU-Sim addresses, its target
architectures and workloads, and its main differences from conventional
GPGPU-Sim workflows. -->

## Key Capabilities

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

## Getting Started

### Prerequisites

<!-- Summarize the supported host platform, compiler, CUDA toolkit, and other
required dependencies. -->

### Docker Setup

<!-- Provide the shortest supported Docker workflow and link to the detailed
Docker documentation. -->

### Host Setup

<!-- Provide the shortest supported host build workflow. Flash mode is enabled
by default. -->

### Verify the Installation

<!-- Provide one lightweight command that confirms the simulator was built and
used successfully. -->

## Tutorials

### CUDA Vector Addition

<!-- Introduce and link to the standalone CUDA vector-add tutorial under
tutorial/. -->

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

If FlashGPU-sim helps you in your research, you are encouraged to cite our paper. Here is an example:

```bibtex
@inproceedings{flashgpusim-micro,
  author    = {Yu, Siying and Hong, Yixun and Qiu, Guozhi and Gu, Feng
               and Geng, Chenbo and Wang, Zhengrong and Zhang, Chen and Yu, Bei},
  title     = {{FlashGPU-sim}: Enabling GPU Modeling for Modern
               Architectures and AI Workloads},
  booktitle = {IEEE/ACM 59th International Symposium on Microarchitecture (MICRO)},
  year      = {2026}
}
```

## License and Acknowledgements

<!-- Link to the repository copyright and licensing notices. Keep the archived
upstream GPGPU-Sim documentation link here as historical background. -->
