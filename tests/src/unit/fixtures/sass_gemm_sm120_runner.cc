#include <cuda.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  const char *detail = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &detail);
  std::cerr << operation << " failed: " << (name == nullptr ? "?" : name)
            << ": " << (detail == nullptr ? "?" : detail) << '\n';
  std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " sass_gemm_sm120.cubin\n";
    return 2;
  }

  const std::array<float, 8> a = {1.0f,  2.0f, 3.0f, 4.0f,
                                  -1.0f, 0.5f, 2.0f, -3.0f};
  const std::array<float, 8> b = {2.0f,  -1.0f, 0.5f, 3.0f,
                                  -2.0f, 4.0f,  1.0f, 2.0f};
  std::array<float, 4> c{};
  std::array<float, 4> reference{};
  for (unsigned m = 0; m < 2; ++m) {
    for (unsigned n = 0; n < 2; ++n) {
      for (unsigned k = 0; k < 4; ++k)
        reference[m * 2 + n] =
            std::fma(a[m * 4 + k], b[k * 2 + n], reference[m * 2 + n]);
    }
  }

  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");
  CUmodule module;
  check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
  CUfunction kernel;
  check(cuModuleGetFunction(&kernel, module, "sass_gemm_2x2x4"),
        "cuModuleGetFunction");

  CUdeviceptr device_a;
  CUdeviceptr device_b;
  CUdeviceptr device_c;
  check(cuMemAlloc(&device_a, sizeof(a)), "cuMemAlloc(a)");
  check(cuMemAlloc(&device_b, sizeof(b)), "cuMemAlloc(b)");
  check(cuMemAlloc(&device_c, sizeof(c)), "cuMemAlloc(c)");
  check(cuMemcpyHtoD(device_a, a.data(), sizeof(a)), "cuMemcpyHtoD(a)");
  check(cuMemcpyHtoD(device_b, b.data(), sizeof(b)), "cuMemcpyHtoD(b)");
  void *arguments[] = {&device_a, &device_b, &device_c};
  check(
      cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr),
      "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  check(cuMemcpyDtoH(c.data(), device_c, sizeof(c)), "cuMemcpyDtoH(c)");

  bool correct = true;
  std::cout << "C =";
  for (size_t i = 0; i < c.size(); ++i) {
    std::cout << ' ' << c[i];
    correct &= c[i] == reference[i];
  }
  std::cout << "\nreference =";
  for (float value : reference)
    std::cout << ' ' << value;
  std::cout << '\n';

  check(cuMemFree(device_a), "cuMemFree(a)");
  check(cuMemFree(device_b), "cuMemFree(b)");
  check(cuMemFree(device_c), "cuMemFree(c)");
  check(cuModuleUnload(module), "cuModuleUnload");
  check(cuCtxDestroy(context), "cuCtxDestroy");
  return correct ? 0 : 1;
}
