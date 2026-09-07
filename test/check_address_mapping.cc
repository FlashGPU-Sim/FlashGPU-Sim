// Standalone regression using the actual decoder, without simulator startup.
// From the repository root (choose a fresh output path):
// g++ -std=c++17 -O2 -Isrc -I/usr/local/cuda/include \
//   test/check_address_mapping.cc src/gpgpu-sim/addrdec.cc \
//   src/gpgpu-sim/hashing.cc src/option_parser.cc -o /tmp/check_address_mapping
// /tmp/check_address_mapping
#include "gpgpu-sim/addrdec.h"
#include "gpgpu-sim/hashing.h"

#include <cstdio>
#include <set>
#include <tuple>

// Unused RANDOM mode still references the simulator-owned table at link time.
tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

static auto dram_tuple(const addrdec_t &t) {
  return std::make_tuple(t.chip, t.bk, t.row, t.col, t.burst);
}

int main() {
  bool all_pass = true;
  for (unsigned channels : {94u, 80u}) {
    for (const char *mode : {"0", "1", "2"}) {
      for (const char *stable : {"0", "1"}) {
        linear_to_raw_address_translation mapping;
        auto options = option_parser_create();
        mapping.addrdec_setoption(options);
        const char *args[] = {
            "check", "-gpgpu_mem_address_mask", "1",
            "-gpgpu_memory_partition_indexing", "2",
            "-gpgpu_ipoly_non_power2_balanced", mode,
            "-gpgpu_ipoly_channel_stable_l2slice", stable,
            "-gpgpu_mem_addr_mapping",
            "dramid@9;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS"};
        option_parser_cmdline(options, sizeof(args) / sizeof(*args), args);
        mapping.init(channels, 2);

        addrdec_t first, second;
        mapping.addrdec_tlx(0, &first);
        mapping.addrdec_tlx(0x200, &second);
        bool pass = dram_tuple(first) != dram_tuple(second);
        std::set<decltype(dram_tuple(first))> dram;
        std::set<std::pair<unsigned, new_addr_type>> partitions;
        for (new_addr_type base : {0ull, GLOBAL_HEAP_START}) {
          for (new_addr_type offset = 0; offset < 1024 * 1024; offset += 32) {
            const new_addr_type address = base + offset;
            addrdec_t decoded;
            mapping.addrdec_tlx(address, &decoded);
            dram.insert(dram_tuple(decoded));
            partitions.emplace(decoded.sub_partition,
                               mapping.partition_address(address));
          }
        }
        pass &= dram.size() == 65536 && partitions.size() == 65536;

        // Enumerate every decoded channel/slice seed at fixed quotient.
        // Bit 7 is the slice bit in this dramid@9 mapping.
        for (unsigned quotient = 0; quotient < 4096; ++quotient) {
          std::set<unsigned> destinations;
          for (unsigned channel = 0; channel < channels; ++channel) {
            for (unsigned slice = 0; slice < 2; ++slice) {
              const new_addr_type address =
                  ((new_addr_type(quotient) * channels + channel) << 9) |
                  (slice << 7);
              addrdec_t decoded;
              mapping.addrdec_tlx(address, &decoded);
              destinations.insert(decoded.sub_partition);
            }
          }
          pass &= destinations.size() == 2 * channels;
        }
        printf("channels=%u slices=2 legacy_mode=%s stable=%s "
               "dram_unique=%zu partition_unique=%zu %s\n",
               channels, mode, stable, dram.size(), partitions.size(),
               pass ? "PASS" : "FAIL");
        all_pass &= pass;
        option_parser_destroy(options);
      }
    }
  }
  return all_pass ? 0 : 1;
}
