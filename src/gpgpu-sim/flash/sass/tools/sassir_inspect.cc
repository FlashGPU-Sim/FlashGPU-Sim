#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../decode/sassir_decoder.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: sassir_inspect SASSIR KERNEL\n";
    return 2;
  }

  flash_gpgpu_sim::sass::sassir_decoder decoder;
  const flash_gpgpu_sim::sass::kernel kernel =
      decoder.decode_kernel(argv[1], argv[2]);
  std::unordered_map<std::string, size_t> opcode_counts;
  std::unordered_map<std::string, size_t> opaque_opcode_counts;
  std::unordered_map<std::string, size_t> attribute_counts;
  size_t unknown = 0;
  size_t structured = 0;
  size_t yielding = 0;
  size_t waiting = 0;
  for (const auto &inst : kernel.instructions) {
    if (!inst.decoded) {
      ++unknown;
    } else {
      ++opcode_counts[inst.opcode];
    }
    yielding += inst.control.yield_flag;
    waiting += inst.control.wait_mask != 0;
    structured += inst.operands_structured;
    if (inst.decoded && !inst.operands_structured)
      ++opaque_opcode_counts[inst.opcode];
    for (const auto &attribute : inst.attributes)
      ++attribute_counts[attribute.name];
  }

  std::vector<std::pair<std::string, size_t>> sorted(opcode_counts.begin(),
                                                     opcode_counts.end());
  std::vector<std::pair<std::string, size_t>> opaque_sorted(
      opaque_opcode_counts.begin(), opaque_opcode_counts.end());
  std::vector<std::pair<std::string, size_t>> attributes_sorted(
      attribute_counts.begin(), attribute_counts.end());
  const auto count_descending = [](const auto &left, const auto &right) {
    if (left.second != right.second)
      return left.second > right.second;
    return left.first < right.first;
  };
  std::sort(sorted.begin(), sorted.end(), count_descending);
  std::sort(opaque_sorted.begin(), opaque_sorted.end(), count_descending);
  std::sort(attributes_sorted.begin(), attributes_sorted.end(),
            count_descending);

  std::cout << "kernel: " << kernel.name << '\n'
            << "decoder: " << kernel.decoder_name << '\n'
            << "decoder schema: " << kernel.decoder_schema << '\n'
            << "instructions: " << kernel.instructions.size() << '\n'
            << "decoded: " << kernel.instructions.size() - unknown << '\n'
            << "unknown: " << unknown << '\n'
            << "typed operands: " << structured << '\n'
            << "yield flag set: " << yielding << '\n'
            << "nonzero wait mask: " << waiting << '\n'
            << "top opcodes:\n";
  const size_t limit = std::min<size_t>(sorted.size(), 20);
  for (size_t i = 0; i < limit; ++i)
    std::cout << "  " << sorted[i].first << ": " << sorted[i].second << '\n';
  if (!opaque_sorted.empty()) {
    std::cout << "opaque operand opcodes:\n";
    for (const auto &entry : opaque_sorted)
      std::cout << "  " << entry.first << ": " << entry.second << '\n';
  }
  if (!attributes_sorted.empty()) {
    std::cout << "official instruction attributes:\n";
    for (const auto &entry : attributes_sorted)
      std::cout << "  " << entry.first << ": " << entry.second << '\n';
  }
  return unknown == 0 ? 0 : 2;
}
