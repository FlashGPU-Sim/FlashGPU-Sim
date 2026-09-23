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
#include <string>
#include <tuple>

// Unused RANDOM mode still references the simulator-owned table at link time.
tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

static auto dram_tuple(const addrdec_t &t) {
  return std::make_tuple(t.chip, t.bk, t.row, t.col, t.burst);
}

static linear_to_raw_address_translation make_mapping(unsigned channels,
                                                      const char *mode,
                                                      const char *stable) {
  linear_to_raw_address_translation mapping;
  auto options = option_parser_create();
  mapping.addrdec_setoption(options);
  const char *args[] = {
      "check",
      "-gpgpu_mem_address_mask",
      "1",
      "-gpgpu_memory_partition_indexing",
      "2",
      "-gpgpu_ipoly_non_power2_balanced",
      mode,
      "-gpgpu_ipoly_channel_stable_l2slice",
      stable,
      "-gpgpu_mem_addr_mapping",
      "dramid@9;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR."
      "RBBBCCCC.BCCSSSSS"};
  option_parser_cmdline(options, sizeof(args) / sizeof(*args), args);
  mapping.init(channels, 2);
  option_parser_destroy(options);
  return mapping;
}

static unsigned destinations_at_quotient(
    const linear_to_raw_address_translation &mapping, unsigned channels,
    unsigned quotient) {
  std::set<unsigned> destinations;
  for (unsigned channel = 0; channel < channels; ++channel) {
    for (unsigned slice = 0; slice < 2; ++slice) {
      const new_addr_type address =
          ((new_addr_type(quotient) * channels + channel) << 9) | (slice << 7);
      addrdec_t decoded;
      mapping.addrdec_tlx(address, &decoded);
      destinations.insert(decoded.sub_partition);
    }
  }
  return static_cast<unsigned>(destinations.size());
}

int main() {
  bool all_pass = true;
  const unsigned expected_subparts_80 = 160;
  const unsigned expected_subparts_94 = 188;

  for (unsigned channels : {94u, 80u}) {
    for (const char *mode : {"0", "1", "2", "3"}) {
      for (const char *stable : {"0", "1"}) {
        auto mapping = make_mapping(channels, mode, stable);

        addrdec_t first, second;
        mapping.addrdec_tlx(0, &first);
        mapping.addrdec_tlx(0x200, &second);
        bool pass = true;
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
        unsigned min_dest = ~0u;
        unsigned max_dest = 0;
        for (unsigned quotient = 0; quotient < 4096; ++quotient) {
          unsigned n = destinations_at_quotient(mapping, channels, quotient);
          if (n < min_dest)
            min_dest = n;
          if (n > max_dest)
            max_dest = n;
        }
        const bool bijective = (std::string(mode) == "3") &&
                               (std::string(stable) == "0");
        if (bijective) {
          pass &= dram_tuple(first) != dram_tuple(second);
          pass &= dram.size() == 65536 && partitions.size() == 65536;
          pass &= min_dest == 2 * channels && max_dest == 2 * channels;
        } else if (std::string(stable) == "1") {
          pass &= dram_tuple(first) != dram_tuple(second);
          pass &= dram.size() == 65536 && partitions.size() == 65536;
        } else {
          // Flash 0/1/2 range-reduce on a non-power-of-two channel count.
          // Aliasing is the historical algorithm; do not require uniqueness.
          pass &= dram.size() > 1 && partitions.size() > 1 && min_dest > 0;
        }

        printf("channels=%u slices=2 mode=%s stable=%s dram_unique=%zu "
               "partition_unique=%zu dest_min=%u dest_max=%u %s\n",
               channels, mode, stable, dram.size(), partitions.size(), min_dest,
               max_dest, pass ? "PASS" : "FAIL");
        all_pass &= pass;
      }
    }
  }

  // Mode 2 must keep the flash range-reduce algorithm (H100). Mode 3 is the
  // H200 bijective rotation and must not be an alias of 0/1/2.
  {
    auto m2 = make_mapping(80, "2", "0");
    auto m3 = make_mapping(80, "3", "0");
    auto m2_94 = make_mapping(94, "2", "0");
    auto m3_94 = make_mapping(94, "3", "0");
    bool differ_80 = false;
    bool differ_94 = false;
    bool mode3_94_full = true;
    for (unsigned q = 0; q < 64; ++q) {
      addrdec_t a, b, c, d;
      const new_addr_type addr = (new_addr_type(q) << 9);
      m2.addrdec_tlx(addr, &a);
      m3.addrdec_tlx(addr, &b);
      m2_94.addrdec_tlx(addr, &c);
      m3_94.addrdec_tlx(addr, &d);
      differ_80 |= a.sub_partition != b.sub_partition;
      differ_94 |= c.sub_partition != d.sub_partition;
    }
    mode3_94_full &=
        destinations_at_quotient(m3_94, 94, 0) == expected_subparts_94;
    bool flash2_not_full =
        destinations_at_quotient(m2_94, 94, 0) < expected_subparts_94;
    bool pass = differ_80 && differ_94 && mode3_94_full && flash2_not_full;
    printf("mode2_vs_mode3 differ80=%d differ94=%d mode3_94_full=%d "
           "mode2_94_aliases=%d %s\n",
           differ_80, differ_94, mode3_94_full, flash2_not_full,
           pass ? "PASS" : "FAIL");
    all_pass &= pass;
    (void)expected_subparts_80;
  }

  return all_pass ? 0 : 1;
}
