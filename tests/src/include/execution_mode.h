#ifndef FLASHGPU_TEST_EXECUTION_MODE_H_
#define FLASHGPU_TEST_EXECUTION_MODE_H_

#include <cstdlib>
#include <string>

namespace flashgpu::test {

inline bool running_on_native_gpu() {
  const char* sim_env = std::getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN");
  if (sim_env != nullptr && sim_env[0] != '\0') {
    return false;
  }

  const char* ld_library_path = std::getenv("LD_LIBRARY_PATH");
  if (ld_library_path == nullptr || ld_library_path[0] == '\0') {
    return true;
  }

  const std::string ld_path(ld_library_path);
  const char* gpgpusim_root = std::getenv("GPGPUSIM_ROOT");
  if (gpgpusim_root != nullptr && gpgpusim_root[0] != '\0') {
    const std::string sim_lib_prefix = std::string(gpgpusim_root) + "/lib/";
    if (ld_path.find(sim_lib_prefix) != std::string::npos) {
      return false;
    }
  }

  return ld_path.find("gpgpu-sim") == std::string::npos;
}

}  // namespace flashgpu::test

#endif  // FLASHGPU_TEST_EXECUTION_MODE_H_
