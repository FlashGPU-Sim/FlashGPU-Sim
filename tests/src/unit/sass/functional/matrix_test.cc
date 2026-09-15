#include "../test_support.h"

namespace {

TEST(SassFunctionalTest, LoadsAndStoresFourM88MatrixFragments) {
  kernel image;
  image.name = "matrix_fragments";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "LDSM.16.M88.4", "R20[R10]"),
      official_instruction(0x10, "STSM.16.M88.4", "[R11],R20"),
      official_instruction(0x20, "LDSM.16.MT88.4", "R24[R10]"),
      official_instruction(0x30, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, 0x400, true);
  memory.map(memory_space::kShared, 0x1000, 0x400, true);
  for (unsigned matrix = 0; matrix < 4; ++matrix) {
    for (unsigned row = 0; row < 8; ++row) {
      for (unsigned column = 0; column < 8; ++column) {
        const uint16_t value = matrix * 1000 + row * 10 + column;
        memory.initialize(
            memory_space::kShared,
            0x100 + matrix * 0x100 + row * 0x10 + column * sizeof(uint16_t),
            value);
      }
    }
  }

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned matrix = lane / 8;
    const unsigned row = lane % 8;
    state.write_register(lane, 10, 0x100 + matrix * 0x100 + row * 0x10);
    state.write_register(lane, 11, 0x1000 + matrix * 0x100 + row * 0x10);
  }
  execution_context context{&memory};

  step_result result = sass.step("matrix_fragments", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    ASSERT_EQ(context.memory_accesses[lane].size(), 4u);
    const unsigned row = lane / 4;
    const unsigned column_bytes = (lane % 4) * sizeof(uint32_t);
    for (unsigned matrix = 0; matrix < 4; ++matrix) {
      const auto &access = context.memory_accesses[lane][matrix];
      EXPECT_EQ(access.space, memory_space::kShared);
      EXPECT_EQ(access.address,
                0x100u + matrix * 0x100u + row * 0x10u + column_bytes);
      EXPECT_EQ(access.bytes, sizeof(uint32_t));
      EXPECT_FALSE(access.write);
    }
  }
  do {
    result = sass.step("matrix_fragments", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned matrix = 0; matrix < 4; ++matrix) {
    for (unsigned row = 0; row < 8; ++row) {
      for (unsigned column = 0; column < 8; ++column) {
        EXPECT_EQ(
            memory.inspect<uint16_t>(memory_space::kShared,
                                     0x1000 + matrix * 0x100 + row * 0x10 +
                                         column * sizeof(uint16_t)),
            matrix * 1000 + row * 10 + column);
      }
    }
  }
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned column = lane / 4;
    const unsigned row = (lane % 4) * 2;
    for (unsigned matrix = 0; matrix < 4; ++matrix) {
      const uint32_t low = matrix * 1000 + row * 10 + column;
      const uint32_t high = matrix * 1000 + (row + 1) * 10 + column;
      EXPECT_EQ(state.read_register(lane, 24 + matrix), low | (high << 16));
    }
  }
}

TEST(SassFunctionalTest, ExecutesM16N8K16F16TensorFragments) {
  kernel image;
  image.name = "hmma_16816";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HMMA.16816.F32", "R40,R20,R30,R40"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  std::array<float, 16 * 16> matrix_a{};
  std::array<float, 16 * 8> matrix_b{};
  std::array<float, 16 * 8> matrix_c{};
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned k = 0; k < 16; ++k)
      matrix_a[row * 16 + k] = static_cast<int>((row * 3 + k * 2) % 7) - 3;
  }
  for (unsigned k = 0; k < 16; ++k) {
    for (unsigned column = 0; column < 8; ++column)
      matrix_b[k * 8 + column] = static_cast<int>((k * 2 + column * 3) % 5) - 2;
  }
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned column = 0; column < 8; ++column)
      matrix_c[row * 8 + column] =
          static_cast<int>(row) - static_cast<int>(column);
  }

  const auto packed_half = [](float low, float high) {
    const uint32_t low_bits = exact_integer_half(static_cast<int>(low));
    const uint32_t high_bits = exact_integer_half(static_cast<int>(high));
    return low_bits | (high_bits << 16);
  };
  const auto float_to_bits = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto bits_to_float = [](uint32_t bits) {
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned group = lane / 4;
    const unsigned thread = lane % 4;
    const unsigned k0 = thread * 2;
    state.write_register(
        lane, 20,
        packed_half(matrix_a[group * 16 + k0], matrix_a[group * 16 + k0 + 1]));
    state.write_register(lane, 21,
                         packed_half(matrix_a[(group + 8) * 16 + k0],
                                     matrix_a[(group + 8) * 16 + k0 + 1]));
    state.write_register(lane, 22,
                         packed_half(matrix_a[group * 16 + k0 + 8],
                                     matrix_a[group * 16 + k0 + 9]));
    state.write_register(lane, 23,
                         packed_half(matrix_a[(group + 8) * 16 + k0 + 8],
                                     matrix_a[(group + 8) * 16 + k0 + 9]));
    state.write_register(
        lane, 30,
        packed_half(matrix_b[k0 * 8 + group], matrix_b[(k0 + 1) * 8 + group]));
    state.write_register(lane, 31,
                         packed_half(matrix_b[(k0 + 8) * 8 + group],
                                     matrix_b[(k0 + 9) * 8 + group]));

    const unsigned column0 = thread * 2;
    state.write_register(lane, 40,
                         float_to_bits(matrix_c[group * 8 + column0]));
    state.write_register(lane, 41,
                         float_to_bits(matrix_c[group * 8 + column0 + 1]));
    state.write_register(lane, 42,
                         float_to_bits(matrix_c[(group + 8) * 8 + column0]));
    state.write_register(
        lane, 43, float_to_bits(matrix_c[(group + 8) * 8 + column0 + 1]));
  }

  execution_context context;
  step_result result;
  do {
    result = sass.step("hmma_16816", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned group = lane / 4;
    const unsigned column0 = (lane % 4) * 2;
    const unsigned rows[4] = {group, group, group + 8, group + 8};
    const unsigned columns[4] = {column0, column0 + 1, column0, column0 + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float reference = matrix_c[rows[output] * 8 + columns[output]];
      for (unsigned k = 0; k < 16; ++k)
        reference = std::fma(matrix_a[rows[output] * 16 + k],
                             matrix_b[k * 8 + columns[output]], reference);
      EXPECT_FLOAT_EQ(bits_to_float(state.read_register(lane, 40 + output)),
                      reference);
    }
  }
}

TEST(SassFunctionalTest, ExecutesHopperRegisterSharedF16WarpgroupMma) {
  kernel image;
  image.name = "hgmma_64x8x16";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "WARPGROUP.ARRIVE"),
      official_instruction(0x10, "HGMMA.64x8x16.F32",
                           "R40,R20,gdesc[UR12],R40,UP0,gsb0"),
      official_instruction(0x20, "WARPGROUP.DEPBAR.LE", "gsb0,0x0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  std::array<float, 16 * 16> matrix_a{};
  std::array<float, 16 * 8> matrix_b{};
  std::array<float, 16 * 8> matrix_c{};
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned k = 0; k < 16; ++k)
      matrix_a[row * 16 + k] = static_cast<int>((row * 3 + k * 2) % 7) - 3;
  }
  for (unsigned k = 0; k < 16; ++k) {
    for (unsigned column = 0; column < 8; ++column)
      matrix_b[k * 8 + column] = static_cast<int>((k * 2 + column * 3) % 5) - 2;
  }
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned column = 0; column < 8; ++column)
      matrix_c[row * 8 + column] =
          static_cast<int>(row) - static_cast<int>(column);
  }

  constexpr uint64_t leading_bytes = 256;
  constexpr uint64_t stride_bytes = 16;
  const uint64_t descriptor =
      ((leading_bytes >> 4) << 16) | ((stride_bytes >> 4) << 32);
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, 1024, true);
  for (unsigned k = 0; k < 16; ++k) {
    for (unsigned column = 0; column < 8; ++column) {
      const uint64_t address =
          (k / 8) * leading_bytes + column * stride_bytes + (k % 8) * 2;
      memory.initialize(
          memory_space::kShared, address,
          exact_integer_half(static_cast<int>(matrix_b[k * 8 + column])));
    }
  }

  const auto packed_half = [](float low, float high) {
    const uint32_t low_bits = exact_integer_half(static_cast<int>(low));
    const uint32_t high_bits = exact_integer_half(static_cast<int>(high));
    return low_bits | (high_bits << 16);
  };
  const auto float_to_bits = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto bits_to_float = [](uint32_t bits) {
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  warp_state state;
  state.write_uniform_register(14, static_cast<uint32_t>(descriptor));
  state.write_uniform_register(15, static_cast<uint32_t>(descriptor >> 32));
  state.write_uniform_predicate(0, true);
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned row = lane / 4;
    const unsigned k0 = (lane % 4) * 2;
    state.write_register(
        lane, 20,
        packed_half(matrix_a[row * 16 + k0], matrix_a[row * 16 + k0 + 1]));
    state.write_register(lane, 21,
                         packed_half(matrix_a[(row + 8) * 16 + k0],
                                     matrix_a[(row + 8) * 16 + k0 + 1]));
    state.write_register(
        lane, 22,
        packed_half(matrix_a[row * 16 + k0 + 8], matrix_a[row * 16 + k0 + 9]));
    state.write_register(lane, 23,
                         packed_half(matrix_a[(row + 8) * 16 + k0 + 8],
                                     matrix_a[(row + 8) * 16 + k0 + 9]));
    const unsigned column = (lane % 4) * 2;
    state.write_register(lane, 40, float_to_bits(matrix_c[row * 8 + column]));
    state.write_register(lane, 41,
                         float_to_bits(matrix_c[row * 8 + column + 1]));
    state.write_register(lane, 42,
                         float_to_bits(matrix_c[(row + 8) * 8 + column]));
    state.write_register(lane, 43,
                         float_to_bits(matrix_c[(row + 8) * 8 + column + 1]));
  }
  execution_context context{&memory};
  for (unsigned lane = 0; lane < 32; ++lane)
    context.thread_linear_id[lane] = lane;

  step_result result;
  do {
    result = sass.step("hgmma_64x8x16", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned row = lane / 4;
    const unsigned column = (lane % 4) * 2;
    const unsigned rows[4] = {row, row, row + 8, row + 8};
    const unsigned columns[4] = {column, column + 1, column, column + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float reference = matrix_c[rows[output] * 8 + columns[output]];
      for (unsigned k = 0; k < 16; ++k)
        reference = std::fma(matrix_a[rows[output] * 16 + k],
                             matrix_b[k * 8 + columns[output]], reference);
      EXPECT_FLOAT_EQ(bits_to_float(state.read_register(lane, 40 + output)),
                      reference);
    }
  }
}

TEST(SassFunctionalTest, ExecutesHopperWgmmaCommitGroupSentinel) {
  kernel image;
  image.name = "hgmma_commit_group";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HGMMA.64x8x16.F16",
                           "RZ,gdesc[URZ],RZ,!UPT,gsb0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  state.write_register(0, 0, 0x12345678);
  execution_context context;
  const step_result result = sass.step("hgmma_commit_group", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.pc, 0x10u);
  EXPECT_EQ(state.read_register(0, 0), 0x12345678u);
}

TEST(SassFunctionalTest, ExecutesHopperSharedSharedF16WarpgroupMma) {
  kernel image;
  image.name = "hgmma_64x64x16";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HGMMA.64x64x16.F32",
                           "R24,gdesc[UR4],R24,UP0,gsb0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  std::array<float, 64 * 16> matrix_a{};
  std::array<float, 16 * 64> matrix_b{};
  std::array<float, 64 * 64> matrix_c{};
  for (unsigned row = 0; row < 64; ++row) {
    for (unsigned k = 0; k < 16; ++k)
      matrix_a[row * 16 + k] = static_cast<int>((row + 2 * k) % 5) - 2;
  }
  for (unsigned k = 0; k < 16; ++k) {
    for (unsigned column = 0; column < 64; ++column)
      matrix_b[k * 64 + column] = static_cast<int>((3 * k + column) % 5) - 2;
  }
  for (unsigned row = 0; row < 64; ++row) {
    for (unsigned column = 0; column < 64; ++column)
      matrix_c[row * 64 + column] =
          static_cast<int>(row % 7) - static_cast<int>(column % 5);
  }

  constexpr uint64_t a_base = 32;
  constexpr uint64_t b_base = 8192 + 48;
  constexpr uint64_t leading_bytes = 16;
  constexpr uint64_t stride_bytes = 1024;
  const auto descriptor = [](uint64_t base) {
    return ((base >> 4) & 0x3fffull) | ((leading_bytes >> 4) << 16) |
           ((stride_bytes >> 4) << 32) | (uint64_t{1} << 62);
  };
  const auto shared_address = [](uint64_t base, unsigned row, unsigned k) {
    const uint64_t logical =
        uint64_t{row / 8} * stride_bytes + uint64_t{row % 8} * 128 +
        uint64_t{k / 8} * leading_bytes + uint64_t{k % 8} * 2;
    constexpr uint64_t y_mask = uint64_t{0x7} << 7;
    constexpr uint64_t period = 128 * 8;
    const uint64_t period_base = base & ~(period - 1);
    const uint64_t period_offset = (base - period_base) + logical;
    return period_base + (period_offset ^ ((period_offset & y_mask) >> 3));
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, 16384, true);
  for (unsigned row = 0; row < 64; ++row) {
    for (unsigned k = 0; k < 16; ++k) {
      memory.initialize(
          memory_space::kShared, shared_address(a_base, row, k),
          exact_integer_half(static_cast<int>(matrix_a[row * 16 + k])));
      memory.initialize(
          memory_space::kShared, shared_address(b_base, row, k),
          exact_integer_half(static_cast<int>(matrix_b[k * 64 + row])));
    }
  }

  const auto float_to_bits = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto bits_to_float = [](uint32_t bits) {
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  warp_state state;
  const uint64_t descriptor_a = descriptor(a_base);
  const uint64_t descriptor_b = descriptor(b_base);
  state.write_uniform_register(4, static_cast<uint32_t>(descriptor_a));
  state.write_uniform_register(5, static_cast<uint32_t>(descriptor_a >> 32));
  state.write_uniform_register(6, static_cast<uint32_t>(descriptor_b));
  state.write_uniform_register(7, static_cast<uint32_t>(descriptor_b >> 32));
  state.write_uniform_predicate(0, true);
  execution_context context{&memory};
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned thread_id = 64 + lane;
    context.thread_idx_x[lane] = thread_id;
    const unsigned row_in_octet = (thread_id / 4) % 8;
    const unsigned quarter = thread_id / 32;
    const unsigned column_pair = (thread_id % 4) * 2;
    for (unsigned output = 0; output < 32; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row =
          row_in_octet + 16 * quarter + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      state.write_register(lane, 24 + output,
                           float_to_bits(matrix_c[row * 64 + column]));
    }
  }

  const step_result result = sass.step("hgmma_64x64x16", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned thread_id = 64 + lane;
    const unsigned row_in_octet = (thread_id / 4) % 8;
    const unsigned quarter = thread_id / 32;
    const unsigned column_pair = (thread_id % 4) * 2;
    for (unsigned output = 0; output < 32; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row =
          row_in_octet + 16 * quarter + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      float reference = matrix_c[row * 64 + column];
      for (unsigned k = 0; k < 16; ++k)
        reference = std::fma(matrix_a[row * 16 + k], matrix_b[k * 64 + column],
                             reference);
      EXPECT_FLOAT_EQ(bits_to_float(state.read_register(lane, 24 + output)),
                      reference);
    }
  }
}

TEST(SassFunctionalTest, ExecutesWideHopperRegisterSharedF16WarpgroupMma) {
  kernel image;
  image.name = "hgmma_64x64x16_register_shared";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "WARPGROUP.DEPBAR.LE", "gsb0,0x1"),
      official_instruction(0x10, "HGMMA.64x64x16.F32",
                           "R24,R120,gdesc[UR4].tnspB,R24,gsb0"),
      official_instruction(0x20, "HGMMA.64x64x16.F32",
                           "R24,R120,gdesc[UR4].tnspB,RZ,!UPT,gsb0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  std::array<float, 16 * 16> matrix_a{};
  std::array<float, 16 * 64> matrix_b{};
  std::array<float, 16 * 64> matrix_c{};
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned k = 0; k < 16; ++k)
      matrix_a[row * 16 + k] = static_cast<int>((row + 2 * k) % 5) - 2;
  }
  for (unsigned k = 0; k < 16; ++k) {
    for (unsigned column = 0; column < 64; ++column)
      matrix_b[k * 64 + column] = static_cast<int>((3 * k + column) % 5) - 2;
  }
  for (unsigned row = 0; row < 16; ++row) {
    for (unsigned column = 0; column < 64; ++column)
      matrix_c[row * 64 + column] =
          static_cast<int>(row % 7) - static_cast<int>(column % 5);
  }

  constexpr uint64_t stride_bytes = 1024;
  const uint64_t descriptor_b =
      ((stride_bytes >> 4) << 32) | (uint64_t{1} << 62);
  const auto shared_address = [](unsigned row, unsigned k) {
    const uint64_t logical =
        uint64_t{row} * 2 + uint64_t{k / 8} * 1024 + uint64_t{k % 8} * 128;
    constexpr uint64_t y_mask = uint64_t{0x7} << 7;
    return logical ^ ((logical & y_mask) >> 3);
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, 4096, true);
  for (unsigned column = 0; column < 64; ++column) {
    for (unsigned k = 0; k < 16; ++k) {
      memory.initialize(
          memory_space::kShared, shared_address(column, k),
          exact_integer_half(static_cast<int>(matrix_b[k * 64 + column])));
    }
  }

  const auto float_to_bits = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto bits_to_float = [](uint32_t bits) {
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  warp_state state;
  state.write_uniform_register(6, static_cast<uint32_t>(descriptor_b));
  state.write_uniform_register(7, static_cast<uint32_t>(descriptor_b >> 32));
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned row_in_octet = lane / 4;
    const unsigned column_pair = (lane % 4) * 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const unsigned row = row_in_octet + ((reg & 1) != 0 ? 8 : 0);
      const unsigned k = (lane % 4) * 2 + ((reg & 2) != 0 ? 8 : 0);
      const uint32_t packed =
          exact_integer_half(static_cast<int>(matrix_a[row * 16 + k])) |
          (uint32_t{
               exact_integer_half(static_cast<int>(matrix_a[row * 16 + k + 1]))}
           << 16);
      state.write_register(lane, 120 + reg, packed);
    }
    for (unsigned output = 0; output < 32; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row = row_in_octet + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      state.write_register(lane, 24 + output,
                           float_to_bits(matrix_c[row * 64 + column]));
    }
  }
  execution_context context{&memory};

  ASSERT_EQ(sass.step("hgmma_64x64x16_register_shared", state, context).status,
            step_status::kAdvanced);
  const step_result result =
      sass.step("hgmma_64x64x16_register_shared", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    const unsigned row_in_octet = lane / 4;
    const unsigned column_pair = (lane % 4) * 2;
    for (unsigned output = 0; output < 32; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row = row_in_octet + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      float reference = matrix_c[row * 64 + column];
      for (unsigned k = 0; k < 16; ++k)
        reference = std::fma(matrix_a[row * 16 + k], matrix_b[k * 64 + column],
                             reference);
      EXPECT_FLOAT_EQ(bits_to_float(state.read_register(lane, 24 + output)),
                      reference);
    }
  }

  const step_result clearing_result =
      sass.step("hgmma_64x64x16_register_shared", state, context);
  ASSERT_EQ(clearing_result.status, step_status::kAdvanced)
      << clearing_result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 32; ++output) {
      float reference = 0.0f;
      const unsigned row_in_octet = lane / 4;
      const unsigned column_pair = (lane % 4) * 2;
      const unsigned fragment = output % 4;
      const unsigned row = row_in_octet + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      for (unsigned k = 0; k < 16; ++k)
        reference = std::fma(matrix_a[row * 16 + k], matrix_b[k * 64 + column],
                             reference);
      EXPECT_FLOAT_EQ(bits_to_float(state.read_register(lane, 24 + output)),
                      reference);
    }
  }
}

TEST(SassFunctionalTest, ExecutesWideHopperWarpgroupMmaCompactOperands) {
  kernel image;
  image.name = "hgmma_wide_compact";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HGMMA.64x192x16.F32", "R24,gdesc[UR4],R24"),
      official_instruction(0x10, "HGMMA.64x192x16.F32",
                           "R24,gdesc[UR4],R24,gsb0"),
      official_instruction(0x20, "HGMMA.64x192x16.F32",
                           "R24,gdesc[UR4],RZ,!UPT"),
      official_instruction(0x30, "HGMMA.64x256x16.F32",
                           "R24,gdesc[UR4],RZ,!UPT"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  constexpr uint64_t leading_bytes = 16;
  constexpr uint64_t stride_bytes = 1024;
  const auto descriptor = [](uint64_t base) {
    return ((base >> 4) & 0x3fffull) | ((leading_bytes >> 4) << 16) |
           ((stride_bytes >> 4) << 32) | (uint64_t{1} << 62);
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, 64 * 1024, true);
  warp_state state;
  const uint64_t descriptor_a = descriptor(0);
  const uint64_t descriptor_b = descriptor(8192);
  state.write_uniform_register(4, static_cast<uint32_t>(descriptor_a));
  state.write_uniform_register(5, static_cast<uint32_t>(descriptor_a >> 32));
  state.write_uniform_register(6, static_cast<uint32_t>(descriptor_b));
  state.write_uniform_register(7, static_cast<uint32_t>(descriptor_b >> 32));
  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 128; ++output)
      state.write_register(lane, 24 + output, 0x3f800000);
  }
  execution_context context{&memory};

  for (unsigned instruction = 0; instruction < 2; ++instruction) {
    const step_result result = sass.step("hgmma_wide_compact", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned output = 0; output < 96; ++output)
        EXPECT_EQ(state.read_register(lane, 24 + output), 0x3f800000u);
    }
  }
  const step_result clearing_result =
      sass.step("hgmma_wide_compact", state, context);
  ASSERT_EQ(clearing_result.status, step_status::kAdvanced)
      << clearing_result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 96; ++output)
      EXPECT_EQ(state.read_register(lane, 24 + output), 0u);
    for (unsigned output = 96; output < 128; ++output)
      EXPECT_EQ(state.read_register(lane, 24 + output), 0x3f800000u);
  }

  const step_result wide_clearing_result =
      sass.step("hgmma_wide_compact", state, context);
  ASSERT_EQ(wide_clearing_result.status, step_status::kAdvanced)
      << wide_clearing_result.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 128; ++output)
      EXPECT_EQ(state.read_register(lane, 24 + output), 0u);
  }
}

TEST(SassFunctionalTest, ExecutesM16N8K8BF16TensorFragments) {
  kernel image;
  image.name = "hmma_1688_bf16";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HMMA.1688.F32.BF16", "R40,R20,R30,RZ"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  constexpr uint32_t packed_bfloat_ones = 0x3f803f80u;
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 20, packed_bfloat_ones);
    state.write_register(lane, 21, packed_bfloat_ones);
    state.write_register(lane, 30, packed_bfloat_ones);
  }

  execution_context context;
  step_result result;
  do {
    result = sass.step("hmma_1688_bf16", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 4; ++output) {
      float value = 0.0f;
      const uint32_t bits = state.read_register(lane, 40 + output);
      std::memcpy(&value, &bits, sizeof(value));
      EXPECT_FLOAT_EQ(value, 8.0f);
    }
  }
}

TEST(SassFunctionalTest, ExecutesM16N8K4TF32TensorFragments) {
  kernel image;
  image.name = "hmma_1684_tf32";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "HMMA.1684.F32.TF32", "R40,R20,R30,RZ"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  float unrounded = 1.0001f;
  uint32_t unrounded_bits = 0;
  std::memcpy(&unrounded_bits, &unrounded, sizeof(unrounded_bits));
  ASSERT_NE(unrounded_bits & 0x1fffu, 0u);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 20, unrounded_bits);
    state.write_register(lane, 21, unrounded_bits);
    state.write_register(lane, 30, unrounded_bits);
  }

  execution_context context;
  step_result result;
  do {
    result = sass.step("hmma_1684_tf32", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned output = 0; output < 4; ++output) {
      float value = 0.0f;
      const uint32_t bits = state.read_register(lane, 40 + output);
      std::memcpy(&value, &bits, sizeof(value));
      EXPECT_FLOAT_EQ(value, 4.0f);
    }
  }
}

TEST(SassFunctionalTest, ExecutesSignedM16N8K32IntegerFragments) {
  kernel image;
  image.name = "imma_16832_s8";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "IMMA.16832.S8.S8", "R40,R20.ROW,R30.COL,RZ"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 20, 0x01010101u);
    state.write_register(lane, 21, 0x02020202u);
    state.write_register(lane, 22, 0x03030303u);
    state.write_register(lane, 23, 0x04040404u);
    state.write_register(lane, 30, 0x01010101u);
    state.write_register(lane, 31, 0xfefefefeu);
  }

  execution_context context;
  step_result result;
  do {
    result = sass.step("imma_16832_s8", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(static_cast<int32_t>(state.read_register(lane, 40)), -48);
    EXPECT_EQ(static_cast<int32_t>(state.read_register(lane, 41)), -48);
    EXPECT_EQ(static_cast<int32_t>(state.read_register(lane, 42)), -80);
    EXPECT_EQ(static_cast<int32_t>(state.read_register(lane, 43)), -80);
  }
}

}  // namespace
