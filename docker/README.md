# FlashGPU-Sim Development Container

The development image provides CUDA 12.8, the simulator build dependencies,
and the Python stack used to develop TritonTrace:

- Python 3.12.3
- PyTorch 2.9.0
- Triton 3.5.0
- NumPy 2.4.0

The repository is mounted at `/workspace/flashgpu-sim`. The container user
uses the host user's UID and GID, so files created in the mounted repository
remain owned by the host user.

## Prerequisites

Install Docker with the Compose plugin. GPU-enabled containers additionally
require an NVIDIA driver and the NVIDIA Container Toolkit on the host.

## Build the Image

Run all commands from the repository root:

```bash
./docker.sh build
```

## Open a Development Shell

Start a disposable shell without GPU access:

```bash
./docker.sh shell
```

The shell starts in the native CUDA environment. Configure FlashGPU-Sim only
when building or running the simulator:

```bash
source setup_environment
make -j "$(nproc)"
```

For example, run the CUDA VectorAdd tutorial:

```bash
./tutorials/vectorAdd/run.sh
```

## Use a GPU

Expose all available GPUs:

```bash
./docker.sh shell --gpu
```

Select one host GPU for CUDA applications:

```bash
./docker.sh shell --gpu 1
```

The latter still attaches the GPU resources requested by Compose and sets
`CUDA_VISIBLE_DEVICES=1` inside the container.

For example, capture the Triton GEMM tutorial from a GPU-enabled shell:

```bash
cd tutorials/triton-gemm
./capture.sh
```

## Persistent Data

The source tree is a host bind mount. Python, Triton, and PyTorch caches use
the `flashgpusim-cache` Docker volume and persist across disposable shells.
Containers themselves are removed when their shells exit.

## Nsight Compute

Nsight Compute is not installed in the image. Install a version compatible
with the host GPU and driver on the host, then use
[`tutorials/profile_ncu.sh`](../tutorials/profile_ncu.sh) outside the
container.

## CI Image

[`Dockerfile.ci`](Dockerfile.ci) is a separate minimal image used by the
GitHub Actions workflow. It contains the CUDA toolchain, simulator build
dependencies, and GoogleTest 1.12.1 under `/opt/googletest`. Repository source
and test binaries are not included; CI mounts and builds the selected commit at
runtime.

The maintained image is published as:

```text
ghcr.io/flashgpu-sim/flashgpu-sim-ci:cuda12.8-gtest1.12.1-v1
```

The initial `v1` publication has this immutable reference:

```text
ghcr.io/flashgpu-sim/flashgpu-sim-ci@sha256:502e8c96182dbfab43d861b956957ddaa51bdb6b975845da44905f1ccc24a404
```

`.github/workflows/ci-image.yml` publishes both that versioned tag and a
commit-scoped `sha-<commit>` tag after a smoke test. The package is private;
GitHub Actions pulls it with the job-scoped `GITHUB_TOKEN` and `packages: read`
permission. Consumers should pin its immutable digest.

Build and inspect the image locally:

```bash
docker build -f docker/Dockerfile.ci \
  -t ghcr.io/flashgpu-sim/flashgpu-sim-ci:cuda12.8-gtest1.12.1-v1 .
docker run --rm \
  ghcr.io/flashgpu-sim/flashgpu-sim-ci:cuda12.8-gtest1.12.1-v1 \
  bash -c 'test -f "$GTEST_DIR/include/gtest/gtest.h" && nvcc --version'
```

The development image and helper do not replace or modify the CI environment.
