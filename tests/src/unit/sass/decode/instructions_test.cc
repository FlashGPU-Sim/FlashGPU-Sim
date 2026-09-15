#include "../test_support.h"

namespace {

TEST(Sm120ControlTest, DecodesAllStaticSchedulingFields) {
  const auto control =
      decode_sm120_control(make_control(11, true, 2, 5, 0b100101, 0b101));
  EXPECT_EQ(control.stall, 11);
  EXPECT_TRUE(control.yield_flag);
  EXPECT_EQ(control.write_barrier, 2);
  EXPECT_EQ(control.read_barrier, 5);
  EXPECT_EQ(control.wait_mask, 0b100101);
  EXPECT_EQ(control.reuse_mask, 0b101);
  EXPECT_TRUE(control.waits_on(0));
  EXPECT_TRUE(control.waits_on(2));
  EXPECT_TRUE(control.waits_on(5));
  EXPECT_FALSE(control.waits_on(1));
}

TEST(Sm90ControlTest, RetainsTheFourthReuseFlag) {
  const auto control =
      decode_sm90_control(make_control(3, false, 4, 2, 0b010101, 0b1001));
  EXPECT_EQ(control.stall, 3);
  EXPECT_FALSE(control.yield_flag);
  EXPECT_EQ(control.write_barrier, 4);
  EXPECT_EQ(control.read_barrier, 2);
  EXPECT_EQ(control.wait_mask, 0b010101);
  EXPECT_EQ(control.reuse_mask, 0b1001);
}

TEST(SassOperandParserTest, ParsesOfficialNvdisasmOperandForms) {
  const auto pieces =
      split_operand_text("R0,desc[UR4][R2.64+0x80],c[0x0][0x390],!PT,SR_TID.X");
  ASSERT_EQ(pieces.size(), 5u);

  const auto reg = parse_operand("R11.reuse");
  EXPECT_EQ(reg.kind, operand_kind::kRegister);
  EXPECT_EQ(reg.index, 11u);
  EXPECT_TRUE(reg.reuse);

  const auto row_register = parse_operand("R10.ROW");
  EXPECT_EQ(row_register.kind, operand_kind::kRegister);
  EXPECT_EQ(row_register.matrix_layout, operand_matrix_layout::kRow);
  const auto column_register = parse_operand("R13.COL");
  EXPECT_EQ(column_register.kind, operand_kind::kRegister);
  EXPECT_EQ(column_register.matrix_layout, operand_matrix_layout::kColumn);

  const auto negative_immediate = parse_operand("-0x6000");
  EXPECT_EQ(negative_immediate.kind, operand_kind::kImmediate);
  EXPECT_EQ(negative_immediate.immediate, -0x6000);
  EXPECT_FALSE(negative_immediate.negated);

  const auto address = parse_operand("desc[UR4][R2.64+0x80]");
  EXPECT_EQ(address.kind, operand_kind::kDescriptorAddress);
  EXPECT_EQ(address.descriptor_register, 4u);
  EXPECT_EQ(address.address_register, 2u);
  EXPECT_EQ(address.address_offset, 0x80);

  const auto negative_address = parse_operand("desc[UR12][R2.64+-0x40]");
  EXPECT_EQ(negative_address.kind, operand_kind::kDescriptorAddress);
  EXPECT_EQ(negative_address.descriptor_register, 12u);
  EXPECT_EQ(negative_address.address_register, 2u);
  EXPECT_EQ(negative_address.address_offset, -0x40);

  const auto tensor_map_descriptor = parse_operand("desc[UR20]");
  EXPECT_EQ(tensor_map_descriptor.kind, operand_kind::kTensorMapDescriptor);
  EXPECT_EQ(tensor_map_descriptor.descriptor_register, 20u);

  const auto gmmas_descriptor = parse_operand("gdesc[UR12]");
  EXPECT_EQ(gmmas_descriptor.kind, operand_kind::kGmmaDescriptor);
  EXPECT_EQ(gmmas_descriptor.descriptor_register, 12u);
  EXPECT_FALSE(gmmas_descriptor.descriptor_transpose_a);
  EXPECT_FALSE(gmmas_descriptor.descriptor_transpose_b);
  const auto transposed_gmma_descriptor = parse_operand("gdesc[UR4].tnspB");
  EXPECT_EQ(transposed_gmma_descriptor.kind, operand_kind::kGmmaDescriptor);
  EXPECT_EQ(transposed_gmma_descriptor.descriptor_register, 4u);
  EXPECT_TRUE(transposed_gmma_descriptor.descriptor_transpose_b);
  const auto transposed_a_gmma_descriptor = parse_operand("gdesc[UR8].tnspA");
  EXPECT_EQ(transposed_a_gmma_descriptor.kind, operand_kind::kGmmaDescriptor);
  EXPECT_EQ(transposed_a_gmma_descriptor.descriptor_register, 8u);
  EXPECT_TRUE(transposed_a_gmma_descriptor.descriptor_transpose_a);

  const auto group_scoreboard = parse_operand("gsb0");
  EXPECT_EQ(group_scoreboard.kind, operand_kind::kScoreboardRegister);
  EXPECT_EQ(group_scoreboard.index, 0u);

  const auto constant = parse_operand("c[0x0][0x390]");
  EXPECT_EQ(constant.kind, operand_kind::kConstantMemory);
  EXPECT_EQ(constant.constant_bank, 0u);
  EXPECT_TRUE(constant.constant_offset_static);
  EXPECT_EQ(constant.constant_offset, 0x390u);

  const auto dynamic_constant = parse_operand("c[0x4][R10]");
  ASSERT_EQ(dynamic_constant.kind, operand_kind::kConstantMemory);
  EXPECT_EQ(dynamic_constant.constant_bank, 4u);
  EXPECT_FALSE(dynamic_constant.constant_offset_static);
  ASSERT_EQ(dynamic_constant.constant_address.terms.size(), 1u);
  EXPECT_EQ(dynamic_constant.constant_address.terms[0].kind,
            address_term_kind::kRegister);
  EXPECT_EQ(dynamic_constant.constant_address.terms[0].index, 10u);

  const auto predicate = parse_operand("!UPT");
  EXPECT_EQ(predicate.kind, operand_kind::kUniformPredicate);
  EXPECT_EQ(predicate.index, 7u);
  EXPECT_TRUE(predicate.negated);

  const auto prefixed_address = parse_operand("R0[UR48+0x8]");
  ASSERT_EQ(prefixed_address.kind, operand_kind::kMemoryAddress);
  EXPECT_TRUE(prefixed_address.has_prefix);
  EXPECT_EQ(prefixed_address.prefix_kind, operand_kind::kRegister);
  EXPECT_EQ(prefixed_address.prefix_index, 0u);
  ASSERT_EQ(prefixed_address.address_groups.size(), 1u);
  ASSERT_EQ(prefixed_address.address_groups[0].terms.size(), 2u);
  EXPECT_EQ(prefixed_address.address_groups[0].terms[0].kind,
            address_term_kind::kUniformRegister);
  EXPECT_EQ(prefixed_address.address_groups[0].terms[0].index, 48u);
  EXPECT_EQ(prefixed_address.address_groups[0].terms[1].kind,
            address_term_kind::kImmediate);
  EXPECT_EQ(prefixed_address.address_groups[0].terms[1].immediate, 8);

  const auto grouped_address = parse_operand("[UR48][UR40]");
  ASSERT_EQ(grouped_address.kind, operand_kind::kMemoryAddress);
  EXPECT_FALSE(grouped_address.has_prefix);
  ASSERT_EQ(grouped_address.address_groups.size(), 2u);
  EXPECT_EQ(grouped_address.address_groups[0].terms[0].index, 48u);
  EXPECT_EQ(grouped_address.address_groups[1].terms[0].index, 40u);

  const auto mixed_address = parse_operand("RZ[R10.64+UR40]");
  ASSERT_EQ(mixed_address.kind, operand_kind::kMemoryAddress);
  EXPECT_EQ(mixed_address.prefix_index, 255u);
  ASSERT_EQ(mixed_address.address_groups[0].terms.size(), 2u);
  EXPECT_EQ(mixed_address.address_groups[0].terms[0].kind,
            address_term_kind::kRegister);
  EXPECT_TRUE(mixed_address.address_groups[0].terms[0].wide);
  EXPECT_EQ(mixed_address.address_groups[0].terms[1].kind,
            address_term_kind::kUniformRegister);

  const auto register_set = parse_operand("{5,4,3,2,1,0}");
  ASSERT_EQ(register_set.kind, operand_kind::kRegisterSet);
  EXPECT_EQ(register_set.register_set,
            (std::vector<uint32_t>{5, 4, 3, 2, 1, 0}));

  instruction p2r;
  p2r.predicate_text = "PR";
  p2r.operand_text = "R3,RZ,0x20";
  parse_instruction_operands(p2r);
  EXPECT_TRUE(p2r.operands_structured);
  EXPECT_FALSE(p2r.has_guard);
  EXPECT_TRUE(p2r.has_auxiliary_predicate);
  EXPECT_EQ(p2r.auxiliary_predicate.kind, operand_kind::kPredicateRegisterFile);

  EXPECT_EQ(parse_operand("new_future_operand{x}").kind, operand_kind::kOpaque);
}

TEST(SassSassirTest, LoadsAValidatedStaticKernel) {
  const std::string path =
      "/tmp/flashgpu_sassir_valid_test_" + std::to_string(getpid()) + ".sassir";
  const uint64_t raw_hi = make_control(5, true, 1, 3, 0b100010, 0b011);
  {
    std::ofstream output(path);
    output << "SASSIR\t1\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "INST\t0x0\t0x1234\t0x" << std::hex << raw_hi << std::dec
           << "\t1\tIADD3\tIADD3_R_R_R\tX\t5\t1\t1\t3\t34\t3\n"
           << "FIELD\tRd\t16\t8\t4\t1\treg\n"
           << "ENDINST\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, "kernel");
  ASSERT_EQ(image.instructions.size(), 1u);
  EXPECT_EQ(image.instructions[0].raw.lo, 0x1234u);
  EXPECT_EQ(image.instructions[0].opcode, "IADD3");
  ASSERT_EQ(image.instructions[0].fields.size(), 1u);
  EXPECT_EQ(image.instructions[0].fields[0].value, 4u);
  std::remove(path.c_str());
}

TEST(SassSassirTest, LoadsAnSm90KernelWithHopperControl) {
  const std::string path =
      "/tmp/flashgpu_sassir_sm90_test_" + std::to_string(getpid()) + ".sassir";
  const uint64_t raw_hi = make_control(3, false, 4, 2, 0b010101, 0b1001);
  {
    std::ofstream output(path);
    output << "SASSIR\t3\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tkernel\tsm90\t16\n"
           << "RESOURCES\t8\t0\t0\t0\n"
           << "PARAMBANK\t0x210\t0x0\n"
           << "INST\t0x0\t0x0\t0x" << std::hex << raw_hi << std::dec
           << "\t1\tnvdisasm\t\tNOP\t\t3\t0\t4\t2\t21\t9\n"
           << "ENDINST\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  frontend sass;
  sass.load(decoder, path, "kernel");
  const kernel *image = sass.find_kernel("kernel");
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->arch, architecture::kSm90);
  ASSERT_EQ(image->instructions.size(), 1u);
  EXPECT_EQ(image->instructions[0].control.reuse_mask, 0b1001);
  std::remove(path.c_str());
}

TEST(SassSassirTest, LoadsOfficialKernelParameterAbi) {
  const std::string path =
      "/tmp/flashgpu_sassir_abi_test_" + std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t2\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "PARAMBANK\t0x380\t0xc\n"
           << "PARAM\t0\t0x0\t0x8\n"
           << "PARAM\t1\t0x8\t0x4\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, "kernel");
  EXPECT_TRUE(image.has_parameter_bank);
  EXPECT_EQ(image.parameter_base, 0x380u);
  EXPECT_EQ(image.parameter_size, 0xcu);
  ASSERT_EQ(image.parameters.size(), 2u);
  EXPECT_EQ(image.parameters[1].ordinal, 1u);
  EXPECT_EQ(image.parameters[1].offset, 8u);
  EXPECT_EQ(image.parameters[1].size, 4u);
  std::remove(path.c_str());
}

TEST(SassSassirTest, DistinguishesParameterlessAbiFromMissingAbi) {
  const std::string path = "/tmp/flashgpu_sassir_no_args_test_" +
                           std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t2\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, "kernel");
  EXPECT_TRUE(image.has_parameter_bank);
  EXPECT_EQ(image.parameter_base, 0x380u);
  EXPECT_EQ(image.parameter_size, 0u);
  EXPECT_TRUE(image.parameters.empty());
  std::remove(path.c_str());
}

TEST(SassSassirTest, LoadsV3ResourcesFromAMultiKernelSassir) {
  const std::string path = "/tmp/flashgpu_sassir_resources_test_" +
                           std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t3\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tfirst\tsm120\t16\n"
           << "RESOURCES\t4\t0\t0\t0\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n"
           << "KERNEL\tsecond\tsm120\t16\n"
           << "RESOURCES\t9\t1024\t16\t32\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, "second");
  EXPECT_TRUE(image.has_resources);
  EXPECT_EQ(image.resources.registers, 9u);
  EXPECT_EQ(image.resources.static_shared, 1024u);
  EXPECT_EQ(image.resources.local_memory, 16u);
  EXPECT_EQ(image.resources.stack_size, 32u);
  std::remove(path.c_str());
}

TEST(SassSassirTest, LoadsV4StaticConstantBanks) {
  const std::string path = "/tmp/flashgpu_sassir_constants_test_" +
                           std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t4\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "RESOURCES\t8\t0\t0\t0\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "CONSTBANK\t2\tb0090000e009000090090000\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, "kernel");
  ASSERT_EQ(image.constant_banks.size(), 1u);
  EXPECT_EQ(image.constant_banks[0].index, 2u);
  ASSERT_EQ(image.constant_banks[0].data.size(), 12u);
  EXPECT_EQ(image.constant_banks[0].data[0], 0xb0u);
  EXPECT_EQ(image.constant_banks[0].data[11], 0x00u);
  std::remove(path.c_str());
}

TEST(SassSassirTest, MatchesRebuiltAnonymousNamespaceKernelByDemangledName) {
  const std::string path =
      "/tmp/flashgpu_sassir_anon_test_" + std::to_string(getpid()) + ".sassir";
  const std::string sassir_name =
      "_ZN64_GLOBAL__N__a850efb3_23_address_operand_test_cu_e655aaf2_"
      "257156930load_with_large_address_offsetEPjm";
  const std::string rebuilt_name =
      "_ZN64_GLOBAL__N__a850efb3_23_address_operand_test_cu_e655aaf2_"
      "122450330load_with_large_address_offsetEPjm";
  {
    std::ofstream output(path);
    output << "SASSIR\t3\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\t" << sassir_name << "\tsm120\t16\n"
           << "RESOURCES\t8\t0\t0\t0\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(path, rebuilt_name);
  EXPECT_EQ(image.name, sassir_name);
  std::remove(path.c_str());
}

TEST(SassSassirTest, RejectsAmbiguousDemangledKernelMatch) {
  const std::string path = "/tmp/flashgpu_sassir_anon_ambiguous_test_" +
                           std::to_string(getpid()) + ".sassir";
  const std::string prefix =
      "_ZN64_GLOBAL__N__a850efb3_23_address_operand_test_cu_e655aaf2_";
  const std::string suffix = "30load_with_large_address_offsetEPjm";
  {
    std::ofstream output(path);
    output << "SASSIR\t3\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\t" << prefix << "2571569" << suffix << "\tsm120\t16\n"
           << "RESOURCES\t8\t0\t0\t0\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n"
           << "KERNEL\t" << prefix << "1224503" << suffix << "\tsm120\t16\n"
           << "RESOURCES\t8\t0\t0\t0\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  EXPECT_DEATH(decoder.decode_kernel(path, prefix + "9999999" + suffix),
               "FlashGPU-Sim panic:");
  std::remove(path.c_str());
}

TEST(SassSassirTest, RejectsV3KernelWithoutResources) {
  const std::string path = "/tmp/flashgpu_sassir_no_resources_test_" +
                           std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t3\n"
           << "DECODER\tnvdisasm\tversion\t1.0.0\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "PARAMBANK\t0x380\t0x0\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  EXPECT_DEATH(decoder.decode_kernel(path, "kernel"), "FlashGPU-Sim panic:");
  std::remove(path.c_str());
}

TEST(SassSassirTest, RejectsControlThatDoesNotMatchRawInstruction) {
  const std::string path =
      "/tmp/flashgpu_sassir_test_" + std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t1\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "INST\t0x0\t0x0\t0x0\t1\tNOP\tNOP\t\t1\t0\t7\t7\t0\t0\n"
           << "ENDINST\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  EXPECT_DEATH(decoder.decode_kernel(path, "kernel"), "FlashGPU-Sim panic:");
  std::remove(path.c_str());
}

TEST(SassSassirTest, RejectsV2WithoutDecoderProvenance) {
  const std::string path =
      "/tmp/flashgpu_sassir_v2_test_" + std::to_string(getpid()) + ".sassir";
  {
    std::ofstream output(path);
    output << "SASSIR\t2\n"
           << "KERNEL\tkernel\tsm120\t16\n"
           << "ENDKERNEL\n";
  }

  sassir_decoder decoder;
  EXPECT_DEATH(decoder.decode_kernel(path, "kernel"), "FlashGPU-Sim panic:");
  std::remove(path.c_str());
}

}  // namespace
