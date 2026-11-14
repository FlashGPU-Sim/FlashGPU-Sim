#include <cuda.h>

#include <array>
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

uint16_t exact_small_integer_half(int value) {
  switch (value) {
  case -3:
    return 0xc200;
  case -2:
    return 0xc000;
  case -1:
    return 0xbc00;
  case 0:
    return 0x0000;
  case 1:
    return 0x3c00;
  case 2:
    return 0x4000;
  case 3:
    return 0x4200;
  default:
    std::cerr << "unsupported fixture integer " << value << '\n';
    std::exit(2);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " sass_mma_gemm_sm120.cubin\n";
    return 2;
  }

  std::array<uint16_t, 16 * 8> a;
  std::array<uint16_t, 8 * 8> b;
  std::array<float, 16 * 8> d{};
  std::array<float, 16 * 8> reference{};
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned k = 0; k < 8; ++k) {
      const int value = static_cast<int>((row * 3 + k * 2) % 7) - 3;
      a[row * 8 + k] = exact_small_integer_half(value);
    }
  }
  for (unsigned column = 0; column < 8; ++column) {
    for (unsigned k = 0; k < 8; ++k) {
      const int value = static_cast<int>((k * 2 + column * 3) % 5) - 2;
      b[column * 8 + k] = exact_small_integer_half(value);
    }
  }
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned column = 0; column < 8; ++column) {
      for (unsigned k = 0; k < 8; ++k) {
        const int av = static_cast<int>((row * 3 + k * 2) % 7) - 3;
        const int bv = static_cast<int>((k * 2 + column * 3) % 5) - 2;
        reference[row * 8 + column] += static_cast<float>(av * bv);
      }
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
  check(cuModuleGetFunction(&kernel, module, "sass_mma_gemm_m16n8k8"),
        "cuModuleGetFunction");

  CUdeviceptr device_a;
  CUdeviceptr device_b;
  CUdeviceptr device_d;
  check(cuMemAlloc(&device_a, sizeof(a)), "cuMemAlloc(a)");
  check(cuMemAlloc(&device_b, sizeof(b)), "cuMemAlloc(b)");
  check(cuMemAlloc(&device_d, sizeof(d)), "cuMemAlloc(d)");
  check(cuMemcpyHtoD(device_a, a.data(), sizeof(a)), "cuMemcpyHtoD(a)");
  check(cuMemcpyHtoD(device_b, b.data(), sizeof(b)), "cuMemcpyHtoD(b)");
  void *arguments[] = {&device_a, &device_b, &device_d};
  check(
      cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments, nullptr),
      "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  check(cuMemcpyDtoH(d.data(), device_d, sizeof(d)), "cuMemcpyDtoH(d)");

  bool correct = true;
  for (size_t i = 0; i < d.size(); ++i)
    correct &= d[i] == reference[i];
  std::cout << "128 outputs: first=" << d.front()
            << " expected=" << reference.front() << " last=" << d.back()
            << " expected=" << reference.back()
            << " status=" << (correct ? "PASS" : "FAIL") << '\n';

  check(cuMemFree(device_a), "cuMemFree(a)");
  check(cuMemFree(device_b), "cuMemFree(b)");
  check(cuMemFree(device_d), "cuMemFree(d)");
  check(cuModuleUnload(module), "cuModuleUnload");
  check(cuCtxDestroy(context), "cuCtxDestroy");
  return correct ? 0 : 1;
}
