#pragma once
#include <string>

namespace flash_gpgpu_sim {
namespace sass {
// Native spellings emitted by ptxas for .pack::16b / .unpack::16b.
// A half-split PTX operation emits one SASS instruction per half-warp.
inline int tmem_transfer_half(const std::string &opcode) {
  if (opcode.find(".16dp32bit_t0_t15") != std::string::npos)
    return 0;
  if (opcode.find(".16dp32bit_t16_t31") != std::string::npos)
    return 1;
  return -1;
}
inline unsigned tmem_packed_transfer_words(const std::string &opcode) {
  const bool store = opcode.rfind("STTM", 0) == 0;
  const std::string prefix = store ? "STTM" : "LDTM";
  const std::string suffix = store ? ".EXPAND16BIT" : ".PACK16BIT";
  for (const char *shape : {"", ".16dp32bit_t0_t15", ".16dp32bit_t16_t31"})
    for (unsigned words = 1; words <= 64; words *= 2)
      if (opcode == prefix + shape +
                        (words == 1 ? "" : ".x" + std::to_string(words)) +
                        suffix)
        return words;
  return 0;
}
// Qualified native transfer payload, in 32-bit registers per lane.
inline unsigned tmem_transfer_words(const std::string &opcode) {
  if (const unsigned words = tmem_packed_transfer_words(opcode))
    return words;
  for (const char *prefix : {"STTM", "LDTM"})
    for (const char *shape : {".16dp32bit_t0_t15", ".16dp32bit_t16_t31"})
      for (unsigned words = 1; words <= 64; words *= 2)
        if (opcode == std::string(prefix) + shape +
                          (words == 1 ? "" : ".x" + std::to_string(words)))
          return words;
  if (opcode == "LDTM.x128" || opcode == "STTM.x128")
    return 128;
  if (opcode == "LDTM.x64" || opcode == "STTM.x64" ||
      opcode == "LDTM.16dp256bit.x16" || opcode == "STTM.16dp256bit.x16")
    return 64;
  if (opcode == "LDTM.x32" || opcode == "STTM.x32" ||
      opcode == "LDTM.16dp256bit.x8" || opcode == "STTM.16dp256bit.x8")
    return 32;
  if (opcode == "LDTM.x16" || opcode == "STTM.x16" ||
      opcode == "LDTM.16dp256bit.x4" || opcode == "STTM.16dp256bit.x4")
    return 16;
  if (opcode == "LDTM.16dp128bit.x4")
    return 8;
  if (opcode == "LDTM.16dp128bit.x2")
    return 4;
  if (opcode == "LDTM.16dp64bit.x4")
    return 4;
  if (opcode == "LDTM.x8" || opcode == "STTM.x8" ||
      opcode == "LDTM.16dp256bit.x2" || opcode == "STTM.16dp256bit.x2")
    return 8;
  if (opcode == "LDTM.x4" || opcode == "STTM.x4" ||
      opcode == "LDTM.16dp256bit" || opcode == "STTM.16dp256bit")
    return 4;
  if (opcode == "STTM.x2" || opcode == "LDTM.x2")
    return 2;
  if (opcode == "STTM" || opcode == "LDTM")
    return 1;
  return 0;
}
} // namespace sass
} // namespace flash_gpgpu_sim
