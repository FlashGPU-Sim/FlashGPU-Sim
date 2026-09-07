// Native-object regression; no GPU launch or simulator-library build.
// Compile the actual predicate from shader.cc into this executable, linking an
// existing simulator libcudart only for native constructors. From repo root:
// sed -n '/^bool tensor_core::issue_queue_enabled_for(/,/^}/p' \
//   src/gpgpu-sim/shader.cc | g++ -std=c++17 -O1 -g -fno-access-control \
//   -fsanitize=address -fno-omit-frame-pointer -Isrc -I/usr/local/cuda/include \
//   -include test/check_tensor_queue.cc -x c++ - -x none \
//   /ABS/EXISTING/SIM/LIB/libcudart.so -Wl,-rpath,/ABS/EXISTING/SIM/LIB \
//   -o /tmp/check_tensor_queue
// LD_LIBRARY_PATH=/ABS/EXISTING/SIM/LIB ASAN_OPTIONS=detect_leaks=0 \
//   /tmp/check_tensor_queue
// -fno-access-control is test-only access to the private predicate, not a
// production visibility change. Leak detection is disabled because the native
// tensor pipeline has process-lifetime allocations unrelated to this check.
#include "gpgpu-sim/shader.h"

#include <cstdio>
#include <memory>

int main() {
  shader_core_config config(nullptr);
  config.warp_size = 32;
  config.max_tensor_core_latency = 1;
  register_set output(1, "queue regression");
  tensor_core tensor(&output, &config, nullptr, 0);

  // Heap allocations have the exact base-object size, like pipeline registers;
  // ASan catches derived-field reads beyond these objects in the old predicate.
  auto classic = std::make_unique<warp_inst_t>();
  auto wgmma = std::make_unique<warp_inst_t>();
  auto non_tensor = std::make_unique<warp_inst_t>();
  classic->op = TENSOR_CORE_OP;
  wgmma->op = TENSOR_CORE_OP;
  const unsigned warps[] = {0, 1, 2, 3};
  wgmma->set_wgmma_warpgroup_info(warps, 4);
  non_tensor->op = ALU_OP;

  bool pass = true;
  for (unsigned depth : {0u, 4u}) {
    config.gpgpu_tensor_core_issue_queue_depth = depth;
    const bool classic_ok = tensor.issue_queue_enabled_for(*classic) == (depth > 0);
    const bool wgmma_ok = !tensor.issue_queue_enabled_for(*wgmma);
    const bool other_ok = !tensor.issue_queue_enabled_for(*non_tensor);
    printf("depth=%u classic=%s wgmma=%s non_tensor=%s\n", depth,
           classic_ok ? "PASS" : "FAIL", wgmma_ok ? "PASS" : "FAIL",
           other_ok ? "PASS" : "FAIL");
    pass &= classic_ok && wgmma_ok && other_ok;
  }
  return pass ? 0 : 1;
}
