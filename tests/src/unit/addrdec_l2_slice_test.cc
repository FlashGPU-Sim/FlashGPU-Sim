#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <unordered_set>
#include <utility>

#include "../../../src/gpgpu-sim/addrdec.h"
#include "../../../src/gpgpu-sim/hashing.h"
#include "../../../src/option_parser.h"
#include "../../../src/tr1_hash_map.h"

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

namespace {

constexpr unsigned kChannels = 16;
constexpr unsigned kSlicesPerChannel = 12;
constexpr unsigned kTotalSlices = kChannels * kSlicesPerChannel;
// Exercise the address map used by both SM100_B200 configurations. Its 16
// detailed DRAM banks remain independent from the 12 logical L2 slices per
// channel. A possible 32-bank map belongs to a later DRAM/integration change,
// not this L2 work.
constexpr const char *kCheckedInB200AddressMap =
    "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR."
    "RBBBCCCC.BCCSSSSS";

class ConfiguredAddressMapping {
 public:
  explicit ConfiguredAddressMapping(
      unsigned indexing, unsigned non_power2_l2_slice_mapping,
      unsigned channels = kChannels,
      unsigned slices_per_channel = kSlicesPerChannel,
      unsigned ipoly_non_power2_balanced = 0,
      unsigned ipoly_channel_stable_l2slice = 0) {
    parser_ = option_parser_create();
    mapping_.addrdec_setoption(parser_);
    const std::string indexing_arg = std::to_string(indexing);
    const std::string slice_mapping_arg =
        std::to_string(non_power2_l2_slice_mapping);
    const std::string ipoly_balanced_arg =
        std::to_string(ipoly_non_power2_balanced);
    const std::string ipoly_stable_arg =
        std::to_string(ipoly_channel_stable_l2slice);
    const char *args[] = {
        "addrdec_l2_slice_test",
        "-gpgpu_mem_address_mask",
        "1",
        "-gpgpu_mem_addr_mapping",
        kCheckedInB200AddressMap,
        "-gpgpu_memory_partition_indexing",
        indexing_arg.c_str(),
        "-gpgpu_non_power2_l2_slice_mapping",
        slice_mapping_arg.c_str(),
        "-gpgpu_ipoly_non_power2_balanced",
        ipoly_balanced_arg.c_str(),
        "-gpgpu_ipoly_channel_stable_l2slice",
        ipoly_stable_arg.c_str(),
    };
    option_parser_cmdline(parser_, sizeof(args) / sizeof(args[0]), args);
    mapping_.init(channels, slices_per_channel);
  }

  ~ConfiguredAddressMapping() { option_parser_destroy(parser_); }

  linear_to_raw_address_translation *operator->() { return &mapping_; }

 private:
  option_parser_t parser_;
  linear_to_raw_address_translation mapping_;
};

TEST(AddrdecL2SliceTest, CheckedInMapReachesAll192Slices) {
  ConfiguredAddressMapping mapping(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_STABLE_ROTATION);
  std::array<unsigned long long, kTotalSlices> counts{};

  for (new_addr_type addr = 0; addr < 64ull * 1024 * 1024; addr += 128) {
    addrdec_t decoded{};
    mapping->addrdec_tlx(addr, &decoded);
    ASSERT_LT(decoded.chip, kChannels);
    ASSERT_LT(decoded.bk, 16u);
    ASSERT_LT(decoded.sub_partition, kTotalSlices);
    EXPECT_EQ(decoded.sub_partition / kSlicesPerChannel, decoded.chip);
    ++counts[decoded.sub_partition];
  }

  const auto limits = std::minmax_element(counts.begin(), counts.end());
  EXPECT_GT(*limits.first, 0u);
  const unsigned long long mean = (64ull * 1024 * 1024 / 128) / kTotalSlices;
  EXPECT_LE(*limits.second - *limits.first, mean / 10);
}

TEST(AddrdecL2SliceTest, EveryTwelveLineWindowIsABalancedPermutation) {
  for (new_addr_type group = 0; group < 4096; ++group) {
    std::array<bool, kSlicesPerChannel> seen{};
    for (unsigned remainder = 0; remainder < kSlicesPerChannel; ++remainder) {
      const new_addr_type channel_address =
          (group * kSlicesPerChannel + remainder) * 128;
      const unsigned slice = l2_slice_mapping::slice_index(
          channel_address, kSlicesPerChannel, /*balanced=*/true);
      ASSERT_LT(slice, kSlicesPerChannel);
      EXPECT_FALSE(seen[slice]);
      seen[slice] = true;
      EXPECT_EQ(l2_slice_mapping::partition_address(channel_address,
                                                    kSlicesPerChannel),
                group * 128);
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(),
                            [](bool reached) { return reached; }));
  }
}

TEST(AddrdecL2SliceTest, CheckedInMapIsStableAndKeepsEachLineOnOneSlice) {
  ConfiguredAddressMapping mapping(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_STABLE_ROTATION);

  for (new_addr_type line = 0; line < 2ull * 1024 * 1024; line += 128) {
    addrdec_t first{};
    addrdec_t repeated{};
    const new_addr_type first_partition_address =
        mapping->partition_address(line);
    mapping->addrdec_tlx(line, &first);
    mapping->addrdec_tlx(line, &repeated);
    EXPECT_EQ(first.sub_partition, repeated.sub_partition);
    EXPECT_EQ(mapping->partition_address(line), first_partition_address);

    for (unsigned offset = 32; offset < 128; offset += 32) {
      addrdec_t sector{};
      mapping->addrdec_tlx(line + offset, &sector);
      EXPECT_EQ(sector.sub_partition, first.sub_partition);
      EXPECT_EQ(mapping->partition_address(line + offset),
                first_partition_address + offset);
    }
  }
}

TEST(AddrdecL2SliceTest,
     CheckedInMapStableRotationDoesNotChangeDetailedDramAddress) {
  ConfiguredAddressMapping consecutive(/*CONSECUTIVE=*/0,
                                       NON_POWER2_L2_SLICE_PLAIN);
  ConfiguredAddressMapping rotated(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_STABLE_ROTATION);

  for (new_addr_type addr = 0; addr < 16ull * 1024 * 1024; addr += 128) {
    addrdec_t dram_decoded{};
    addrdec_t l2_decoded{};
    consecutive->addrdec_tlx(addr, &dram_decoded);
    rotated->addrdec_tlx(addr, &l2_decoded);

    EXPECT_EQ(l2_decoded.chip, dram_decoded.chip);
    EXPECT_LT(l2_decoded.bk, 16u);
    EXPECT_EQ(l2_decoded.bk, dram_decoded.bk);
    EXPECT_EQ(l2_decoded.row, dram_decoded.row);
    EXPECT_EQ(l2_decoded.col, dram_decoded.col);
    EXPECT_EQ(l2_decoded.burst, dram_decoded.burst);
    EXPECT_EQ(l2_decoded.sub_partition / kSlicesPerChannel, l2_decoded.chip);
    EXPECT_EQ(rotated->partition_address(addr),
              consecutive->partition_address(addr));
  }
}

TEST(AddrdecL2SliceTest, CheckedInMapUsesTheExplicitStableRotationFormula) {
  ConfiguredAddressMapping mapping(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_STABLE_ROTATION);

  for (new_addr_type addr = 0; addr < 16ull * 1024 * 1024; addr += 32) {
    addrdec_t decoded{};
    mapping->addrdec_tlx(addr, &decoded);

    // The checked-in dramid@8 map inserts the four channel bits at [11:8].
    // Removing those bits yields the exact channel-local address consumed by
    // the pre-split candidate's quotient/remainder + stable rotation helper.
    const new_addr_type channel_address =
        ((addr >> 12) << 8) | (addr & ((1ull << 8) - 1));
    const unsigned expected_local_slice = l2_slice_mapping::slice_index(
        channel_address, kSlicesPerChannel, /*balanced=*/true);
    EXPECT_EQ(decoded.sub_partition,
              decoded.chip * kSlicesPerChannel + expected_local_slice);
    EXPECT_EQ(mapping->partition_address(addr),
              l2_slice_mapping::partition_address(channel_address,
                                                  kSlicesPerChannel));
  }
}

TEST(AddrdecL2SliceTest, CheckedInMapSliceAndPartitionAddressDoNotAlias) {
  ConfiguredAddressMapping mapping(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_STABLE_ROTATION);
  std::unordered_set<unsigned long long> destinations;
  constexpr new_addr_type kSweepBytes = 16ull * 1024 * 1024;
  destinations.reserve(kSweepBytes / 32);

  for (new_addr_type addr = 0; addr < kSweepBytes; addr += 32) {
    addrdec_t decoded{};
    mapping->addrdec_tlx(addr, &decoded);
    const new_addr_type partition_addr = mapping->partition_address(addr);
    // The sampled partition addresses are below 2^56, leaving eight bits for
    // the 0..191 global slice ID.
    const unsigned long long destination =
        (partition_addr << 8) | decoded.sub_partition;
    EXPECT_TRUE(destinations.insert(destination).second)
        << "alias at raw address 0x" << std::hex << addr
        << ", partition address 0x" << partition_addr << std::dec << ", slice "
        << decoded.sub_partition;
  }
}

TEST(AddrdecL2SliceTest, NonPowerOfTwoSlicesRejectLegacyIndexing) {
  EXPECT_DEATH(
      {
        ConfiguredAddressMapping mapping(
            /*IPOLY=*/2, NON_POWER2_L2_SLICE_STABLE_ROTATION);
      },
      "Non-power-of-two per-channel L2-slice counts require");
}

TEST(AddrdecL2SliceTest, PowerOfTwoIPolyAndItsLegacyControlRemainUnchanged) {
  constexpr unsigned kPower2Slices = 8;
  ConfiguredAddressMapping decoded_fields(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_PLAIN, kChannels, kPower2Slices);
  ConfiguredAddressMapping legacy_ipoly(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_PLAIN, kChannels, kPower2Slices);
  ConfiguredAddressMapping legacy_stable_option(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_STABLE_ROTATION, kChannels,
      kPower2Slices, /*ipoly_non_power2_balanced=*/0,
      /*ipoly_channel_stable_l2slice=*/1);

  for (new_addr_type addr = 0; addr < 16ull * 1024 * 1024; addr += 128) {
    addrdec_t raw{};
    addrdec_t actual{};
    addrdec_t with_ignored_controls{};
    decoded_fields->addrdec_tlx(addr, &raw);
    legacy_ipoly->addrdec_tlx(addr, &actual);
    legacy_stable_option->addrdec_tlx(addr, &with_ignored_controls);

    const unsigned seed =
        raw.chip * kPower2Slices + (raw.bk & (kPower2Slices - 1));
    const unsigned expected =
        ipoly_hash_function(addr >> 8, seed, kChannels * kPower2Slices);
    EXPECT_EQ(actual.sub_partition, expected);
    EXPECT_EQ(actual.chip, expected / kPower2Slices);
    // Neither the legacy gap-only stable control nor the new non-power-of-two
    // slice policy changes a power-of-two IPOLY topology.
    EXPECT_EQ(with_ignored_controls.sub_partition, expected);
    EXPECT_EQ(with_ignored_controls.chip, expected / kPower2Slices);
  }
}

unsigned LegacyGapIPolyReference(new_addr_type addr, unsigned decoded_channel,
                                 unsigned decoded_bank,
                                 unsigned ipoly_non_power2_balanced,
                                 bool channel_stable_l2slice) {
  constexpr unsigned kGapChannels = 5;
  constexpr unsigned kGapSlices = 4;
  constexpr unsigned kTotalSubpartitions = kGapChannels * kGapSlices;
  const unsigned decoded_local_slice = decoded_bank & (kGapSlices - 1);
  unsigned subpartition = decoded_channel * kGapSlices + decoded_local_slice;

  if (channel_stable_l2slice) {
    const unsigned seed = (decoded_channel & 0xf) ^ decoded_local_slice;
    const unsigned slice_hash = ipoly_hash_function(addr >> 8, seed, 16);
    return decoded_channel * kGapSlices + (slice_hash % kGapSlices);
  }

  constexpr unsigned kDefaultVirtualSubpartitions = 8 * kGapSlices;
  const unsigned virtual_subpartitions =
      ipoly_non_power2_balanced == 2 ? 1024 : kDefaultVirtualSubpartitions;
  subpartition = ipoly_hash_function((addr >> 8) / kGapChannels, subpartition,
                                     virtual_subpartitions);
  if (ipoly_non_power2_balanced == 1) {
    const unsigned channel = subpartition % kGapChannels;
    const unsigned local_slice = (subpartition / kGapChannels) % kGapSlices;
    return channel * kGapSlices + local_slice;
  }
  if (ipoly_non_power2_balanced == 2) {
    return (static_cast<unsigned long long>(subpartition) *
            kTotalSubpartitions) /
           virtual_subpartitions;
  }
  return subpartition % kTotalSubpartitions;
}

TEST(AddrdecL2SliceTest,
     LegacyGapIPolyBalancingAndStableSliceControlsRemainUnchanged) {
  constexpr unsigned kGapChannels = 5;
  constexpr unsigned kGapSlices = 4;
  ConfiguredAddressMapping decoded_fields(
      /*CONSECUTIVE=*/0, NON_POWER2_L2_SLICE_PLAIN, kGapChannels, kGapSlices);
  ConfiguredAddressMapping legacy_modulo(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_PLAIN, kGapChannels, kGapSlices,
      /*ipoly_non_power2_balanced=*/0);
  ConfiguredAddressMapping channel_first(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_PLAIN, kGapChannels, kGapSlices,
      /*ipoly_non_power2_balanced=*/1);
  ConfiguredAddressMapping range_reduce(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_PLAIN, kGapChannels, kGapSlices,
      /*ipoly_non_power2_balanced=*/2);
  ConfiguredAddressMapping channel_stable(
      /*IPOLY=*/2, NON_POWER2_L2_SLICE_STABLE_ROTATION, kGapChannels,
      kGapSlices, /*ipoly_non_power2_balanced=*/2,
      /*ipoly_channel_stable_l2slice=*/1);

  for (new_addr_type addr = 0; addr < 16ull * 1024 * 1024; addr += 128) {
    addrdec_t raw{};
    decoded_fields->addrdec_tlx(addr, &raw);

    const std::array<std::pair<linear_to_raw_address_translation *, unsigned>,
                     3>
        balanced_modes{{{legacy_modulo.operator->(), 0},
                        {channel_first.operator->(), 1},
                        {range_reduce.operator->(), 2}}};
    for (const auto &mode : balanced_modes) {
      addrdec_t actual{};
      mode.first->addrdec_tlx(addr, &actual);
      const unsigned expected =
          LegacyGapIPolyReference(addr, raw.chip, raw.bk, mode.second,
                                  /*channel_stable_l2slice=*/false);
      EXPECT_EQ(actual.sub_partition, expected);
      EXPECT_EQ(actual.chip, expected / kGapSlices);
    }

    addrdec_t stable{};
    channel_stable->addrdec_tlx(addr, &stable);
    const unsigned expected_stable = LegacyGapIPolyReference(
        addr, raw.chip, raw.bk, /*ipoly_non_power2_balanced=*/2,
        /*channel_stable_l2slice=*/true);
    EXPECT_EQ(stable.sub_partition, expected_stable);
    EXPECT_EQ(stable.chip, raw.chip);
  }
}

}  // namespace
