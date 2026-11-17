#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

@DECLARATIONS@
void cuda_check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void ffi_check(int code) {
  if (!code) return;
  TVMFFIObjectHandle error = nullptr;
  TVMFFIErrorMoveFromRaised(&error);
  std::string message = "TVM FFI call failed";
  if (error) {
    const auto cell = TVMFFIErrorGetCellPtr(error);
    message.assign(cell->message.data, cell->message.size);
    TVMFFIObjectDecRef(error);
  }
  throw std::runtime_error(message);
}
std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot read " + path.string());
  const auto size = file.tellg();
  if (size < 0) throw std::runtime_error("Cannot size " + path.string());
  std::vector<char> data(static_cast<size_t>(size));
  file.seekg(0);
  if (!file.read(data.data(), data.size())) throw std::runtime_error("Short read " + path.string());
  return data;
}
void load_buffer(const std::filesystem::path& path, size_t bytes, void*& pointer) {
  auto data = read_file(path);
  if (data.size() != bytes) throw std::runtime_error("Input size mismatch " + path.string());
  cuda_check(cudaMalloc(&pointer, std::max(bytes, size_t{1})));
  if (bytes) cuda_check(cudaMemcpy(pointer, data.data(), bytes, cudaMemcpyHostToDevice));
}
void save_buffer(const std::filesystem::path& path, void* pointer, size_t bytes) {
  std::vector<char> data(bytes);
  if (bytes) cuda_check(cudaMemcpy(data.data(), pointer, bytes, cudaMemcpyDeviceToHost));
  std::ofstream file(path, std::ios::binary);
  if (!file.write(data.data(), data.size())) throw std::runtime_error("Cannot write " + path.string());
}
void check_tensor(const DLTensor& tensor, const std::filesystem::path& path) {
  const size_t width = tensor.dtype.bits / 8;
  size_t count = 1;
  for (int i = 0; i < tensor.ndim; ++i) count *= tensor.shape[i];
  const auto expected = read_file(path);
  if (expected.size() != count * width) throw std::runtime_error("Expected output size mismatch");
  size_t span = count ? 1 : 0;
  if (count) for (int i = 0; i < tensor.ndim; ++i)
    span += (tensor.shape[i] - 1) * tensor.strides[i];
  std::vector<char> actual(span * width);
  if (span) cuda_check(cudaMemcpy(actual.data(), tensor.data, actual.size(), cudaMemcpyDeviceToHost));
  for (size_t linear = 0; linear < count; ++linear) {
    size_t remaining = linear, offset = 0;
    for (int i = tensor.ndim - 1; i >= 0; --i) {
      offset += (remaining % tensor.shape[i]) * tensor.strides[i];
      remaining /= tensor.shape[i];
    }
    if (std::memcmp(actual.data() + offset * width, expected.data() + linear * width, width))
      throw std::runtime_error("Output mismatch at element " + std::to_string(linear));
  }
  std::puts("Exact output check: PASS");
}
int main() {
  std::vector<void*> buffers(@BUFFER_COUNT@, nullptr);
  try {
    const auto root = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    cuda_check(cudaSetDevice(0));
    @BODY@
    for (auto& pointer : buffers) {
      const auto status = cudaFree(pointer);
      pointer = nullptr;
      cuda_check(status);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Replay failed: %s\n", error.what());
    for (auto pointer : buffers) if (pointer) cudaFree(pointer);
    return 1;
  }
}
