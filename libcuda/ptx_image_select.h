#ifndef FLASH_PTX_IMAGE_SELECT_H
#define FLASH_PTX_IMAGE_SELECT_H

#include <cctype>
#include <string>
#include <vector>

// cuobjdump writes "<module>.sm_<arch>.ptx" or "<module>.<index>.sm_<arch>.ptx".
inline std::string ptx_image_module_name(const std::string &filename) {
  const size_t sm_pos = filename.rfind("sm_");
  if (sm_pos == std::string::npos || sm_pos == 0 || filename[sm_pos - 1] != '.')
    return "";
  const size_t dot_before_sm = sm_pos - 1;
  if (dot_before_sm == 0) return "";
  const size_t prev = filename.rfind('.', dot_before_sm - 1);
  if (prev != std::string::npos) {
    bool index = prev + 1 < dot_before_sm;
    for (size_t k = prev + 1; index && k < dot_before_sm; ++k) {
      if (!isdigit(static_cast<unsigned char>(filename[k]))) index = false;
    }
    if (index) return filename.substr(0, prev);
  }
  return filename.substr(0, dot_before_sm);
}

inline bool ptx_arch_has_suffix(const std::string &arch_part) {
  if (arch_part.empty()) return false;
  for (size_t j = 0; j < arch_part.length(); j++) {
    if (!isdigit(static_cast<unsigned char>(arch_part[j]))) return true;
  }
  return false;
}

inline std::string ptx_image_arch_part(const std::string &filename) {
  const size_t sm_pos = filename.rfind("sm_");
  if (sm_pos == std::string::npos) return "";
  const size_t dot_pos = filename.find('.', sm_pos);
  if (dot_pos == std::string::npos) return "";
  return filename.substr(sm_pos + 3, dot_pos - sm_pos - 3);
}

// Keep a suffix image (sm_90a) and drop the base image (sm_90) of the same
// module. Names that are not in `files` cannot appear in the result.
template <typename Range>
inline std::vector<std::string> drop_base_ptx_when_suffix_present(
    const Range &files) {
  std::vector<std::string> suffix_modules;
  for (const std::string &name : files) {
    if (!ptx_arch_has_suffix(ptx_image_arch_part(name))) continue;
    const std::string module = ptx_image_module_name(name);
    if (!module.empty()) suffix_modules.push_back(module);
  }
  std::vector<std::string> kept;
  for (const std::string &name : files) {
    const std::string module = ptx_image_module_name(name);
    bool drop = false;
    if (!module.empty() && !ptx_arch_has_suffix(ptx_image_arch_part(name))) {
      for (const std::string &suffix_module : suffix_modules) {
        if (suffix_module == module) {
          drop = true;
          break;
        }
      }
    }
    if (!drop) kept.push_back(name);
  }
  return kept;
}

#endif
