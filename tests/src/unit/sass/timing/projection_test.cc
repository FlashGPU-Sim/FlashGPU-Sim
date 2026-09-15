#include "../test_support.h"

namespace {

TEST(SassTimingProjectionTest, ProjectsVectorAddDependencyControl) {
  instruction source = official_instruction(0x110, "FADD", "R9,R2.reuse,R5");
  source.control.stall = 5;
  source.control.yield_flag = true;
  source.control.write_barrier = 2;
  source.control.read_barrier = 4;
  source.control.wait_mask = 0b010101;
  source.control.reuse_mask = 0b0011;

  timing_profile profile;
  profile.fp32_latency[0] = 4;
  profile.fp32_initiation[0] = 1;
  const timing_instruction projected(source, profile);

  EXPECT_TRUE(projected.valid());
  EXPECT_EQ(projected.pc, 0x110u);
  EXPECT_EQ(projected.isize, 16u);
  EXPECT_EQ(projected.op, SP_OP);
  EXPECT_EQ(projected.op_pipe, SP__OP);
  EXPECT_EQ(projected.oprnd_type, FP_OP);
  EXPECT_EQ(projected.latency, 4u);
  EXPECT_EQ(projected.initiation_interval, 1u);
  EXPECT_EQ(projected.outcount, 1u);
  EXPECT_EQ(projected.incount, 2u);
  EXPECT_EQ(projected.arch_reg.dst[0], 10);
  EXPECT_EQ(projected.arch_reg.src[0], 3);
  EXPECT_EQ(projected.arch_reg.src[1], 6);
  EXPECT_EQ(projected.get_num_operands(), 2u);
  EXPECT_EQ(projected.get_num_regs(), 2u);
  ASSERT_EQ(projected.get_register_file_source_count(), 2u);
  EXPECT_EQ(projected.get_register_file_source(0).reg, 2u);
  EXPECT_EQ(projected.get_register_file_source(0).slot, 0u);
  EXPECT_TRUE(projected.get_register_file_source(0).retain);
  EXPECT_EQ(projected.get_register_file_source(1).reg, 5u);
  EXPECT_EQ(projected.get_register_file_source(1).slot, 1u);
  EXPECT_FALSE(projected.get_register_file_source(1).retain);
  ASSERT_EQ(projected.get_register_file_destination_count(), 1u);
  EXPECT_EQ(projected.get_register_file_destination(0).reg, 9u);

  const auto &control = projected.get_dependency_control();
  EXPECT_TRUE(projected.has_explicit_dependency_control());
  EXPECT_EQ(control.stall_cycles, 5u);
  EXPECT_TRUE(control.yield);
  EXPECT_EQ(control.write_barrier, 2u);
  EXPECT_EQ(control.read_barrier, 4u);
  EXPECT_EQ(control.wait_mask, 0b010101u);
  EXPECT_EQ(control.reuse_mask, 0b0011u);
}

TEST(SassTimingProjectionTest, LeavesDynamicAsyncProxyTimingToRuntime) {
  instruction fence = official_instruction(0x110, "FENCE.VIEW.ASYNC.S");
  fence.control.stall = 4;
  instruction membar = official_instruction(0x120, "MEMBAR.ALL.CTA");
  membar.control.stall = 12;
  instruction acquire = official_instruction(0x130, "ACQBULK");

  timing_profile profile;
  profile.async_proxy_fence_extra_stall = 28;
  const timing_instruction projected_fence(fence, profile);
  const timing_instruction projected_membar(membar, profile);
  const timing_instruction projected_acquire(acquire, profile);

  EXPECT_EQ(projected_fence.get_dependency_control().stall_cycles, 4u);
  EXPECT_TRUE(projected_fence.is_async_proxy_fence());
  EXPECT_EQ(projected_membar.get_dependency_control().stall_cycles, 12u);
  EXPECT_FALSE(projected_membar.is_async_proxy_fence());
  EXPECT_EQ(projected_acquire.op, MEMORY_BARRIER_OP);
}

TEST(SassTimingProjectionTest, ProjectsDeferredCtaBarrierAsArrival) {
  const timing_instruction deferred(
      official_instruction(0x140, "BAR.SYNC.DEFER_BLOCKING", "0x0,0x80"),
      timing_profile{});
  EXPECT_EQ(deferred.op, BARRIER_OP);
  EXPECT_EQ(deferred.bar_type, ARRIVE);
  EXPECT_EQ(deferred.bar_id, 0u);
  EXPECT_EQ(deferred.bar_count, 0x80u);
}

TEST(SassTimingProjectionTest, NormalizesReusePositionsAcrossOpcodeLayouts) {
  const timing_instruction iabs(official_instruction(0, "IABS", "R7,R6.reuse"));
  ASSERT_EQ(iabs.get_register_file_source_count(), 1u);
  EXPECT_EQ(iabs.get_register_file_source(0).reg, 6u);
  EXPECT_EQ(iabs.get_register_file_source(0).slot, 0u);
  EXPECT_TRUE(iabs.get_register_file_source(0).retain);

  const timing_instruction shift(
      official_instruction(0x10, "SHF.R.U32.HI", "R13,RZ,0x4,R7.reuse"));
  ASSERT_EQ(shift.get_register_file_source_count(), 1u);
  EXPECT_EQ(shift.get_register_file_source(0).reg, 7u);
  EXPECT_EQ(shift.get_register_file_source(0).slot, 1u);
  EXPECT_TRUE(shift.get_register_file_source(0).retain);
}

TEST(SassTimingProjectionTest, KeepsWideTensorSourcesInTheirLogicalSlots) {
  const timing_instruction projected(
      official_instruction(0, "HMMA.16816.F32", "R8,R12.reuse,R16,R8"));
  ASSERT_EQ(projected.get_register_file_destination_count(), 4u);
  for (unsigned index = 0; index < 4; ++index)
    EXPECT_EQ(projected.get_register_file_destination(index).reg, 8u + index);

  ASSERT_EQ(projected.get_register_file_source_count(), 10u);
  for (unsigned index = 0; index < 4; ++index) {
    EXPECT_EQ(projected.get_register_file_source(index).reg, 12u + index);
    EXPECT_EQ(projected.get_register_file_source(index).slot, 0u);
    EXPECT_TRUE(projected.get_register_file_source(index).retain);
  }
  EXPECT_EQ(projected.get_register_file_source(4).slot, 1u);
  EXPECT_EQ(projected.get_register_file_source(5).slot, 1u);
  for (unsigned index = 6; index < 10; ++index)
    EXPECT_EQ(projected.get_register_file_source(index).slot, 2u);

  const timing_instruction tf32(
      official_instruction(0, "HMMA.1688.F32.TF32", "R8,R12,R16,RZ"));
  EXPECT_EQ(tf32.get_register_file_source_count(), 6u);
  EXPECT_EQ(tf32.get_register_file_destination_count(), 4u);
  EXPECT_EQ(tf32.incount, 6u);  // RZ contributes no physical registers.
}

TEST(SassTimingProjectionTest, ParsesBackendExecutionPipelineProfile) {
  timing_profile_options options;
  options.integer_latency = "4,5,6,7,21,14";
  options.integer_initiation = "1,2,3,4,5,6";
  options.fp32_latency = "4,5,6,7,39";
  options.fp32_initiation = "1,2,3,4,5";
  options.sfu_latency = "28";
  options.sfu_initiation = "8";
  options.tensor_latency = "34,32,16,31,30,29,15";
  options.tensor_initiation = "34,16,8,15,15,14,7";
  options.wgmma_ss_latency = "4,5,6,7";
  options.wgmma_rs_latency = "3,4,5,6";
  options.wgmma_ss_initiation = "2,3,4,5";
  options.wgmma_rs_initiation = "1,2,3,4";
  options.wgmma_ss_completion = "66,67,68,69";
  options.wgmma_rs_completion = "62,63,64,65";
  options.wgmma_int_ss_completion = "56,57,58,59";
  options.wgmma_int_rs_completion = "52,53,54,55";
  options.wgmma_compute_throughput = "4096,2048,8192,8192,65536";
  options.tma_latency = "37";
  options.tma_initiation = "19";
  options.cp_async_latency = "7";
  options.cp_async_initiation = "3";
  options.cp_async_commit_latency = "5";
  options.cp_async_commit_initiation = "2";
  options.cp_async_wait_latency = "6";
  options.cp_async_wait_initiation = "1";

  const timing_profile profile = parse_timing_profile(options);
  EXPECT_EQ(profile.integer_latency[3], 7u);
  EXPECT_EQ(profile.integer_initiation[5], 6u);
  EXPECT_EQ(profile.fp32_latency[2], 6u);
  EXPECT_EQ(profile.fp32_initiation[3], 4u);
  EXPECT_EQ(profile.sfu_latency, 28u);
  EXPECT_EQ(profile.sfu_initiation, 8u);
  EXPECT_EQ(profile.tensor_latency[0], 34u);
  EXPECT_EQ(profile.tensor_initiation[6], 7u);
  EXPECT_EQ(profile.wgmma_ss_latency[3], 7u);
  EXPECT_EQ(profile.wgmma_rs_initiation[2], 3u);
  EXPECT_EQ(profile.wgmma_ss_completion[0], 66u);
  EXPECT_EQ(profile.wgmma_int_rs_completion[3], 55u);
  EXPECT_EQ(profile.wgmma_compute_throughput[4], 65536u);
  EXPECT_EQ(profile.tma_latency, 37u);
  EXPECT_EQ(profile.tma_initiation, 19u);
  EXPECT_EQ(profile.cp_async_latency, 7u);
  EXPECT_EQ(profile.cp_async_initiation, 3u);
  EXPECT_EQ(profile.cp_async_commit_latency, 5u);
  EXPECT_EQ(profile.cp_async_commit_initiation, 2u);
  EXPECT_EQ(profile.cp_async_wait_latency, 6u);
  EXPECT_EQ(profile.cp_async_wait_initiation, 1u);
}

TEST(SassTimingProjectionTest, BroadcastsLegacyTensorPipelineProfile) {
  timing_profile_options options;
  options.integer_latency = "4,4,4,4,21,14";
  options.integer_initiation = "1,1,1,1,2,4";
  options.fp32_latency = "4,4,4,4,39";
  options.fp32_initiation = "1,1,1,1,2";
  options.sfu_latency = "28";
  options.sfu_initiation = "8";
  options.tensor_latency = "64";
  options.tensor_initiation = "32";
  options.tma_latency = "32";
  options.tma_initiation = "32";
  options.cp_async_latency = "7";
  options.cp_async_initiation = "7";
  options.cp_async_commit_latency = "7";
  options.cp_async_commit_initiation = "7";
  options.cp_async_wait_latency = "5";
  options.cp_async_wait_initiation = "5";

  const timing_profile profile = parse_timing_profile(options);
  for (unsigned value : profile.tensor_latency) EXPECT_EQ(value, 64u);
  for (unsigned value : profile.tensor_initiation) EXPECT_EQ(value, 32u);

  options.tensor_latency = "64,";
  EXPECT_DEATH(parse_timing_profile(options), "FlashGPU-Sim panic:");
}

TEST(SassTimingProjectionTest, RejectsInvalidExecutionPipelineProfile) {
  timing_profile_options options;
  options.integer_latency = "4,4,4,4,21,14";
  options.integer_initiation = "1,1,1,1,2,15";
  options.fp32_latency = "4,4,4,4,39";
  options.fp32_initiation = "1,1,1,1,2";
  options.sfu_latency = "28";
  options.sfu_initiation = "8";
  options.tensor_latency = "34,32,16,32,32,32,16";
  options.tensor_initiation = "34,32,16,32,32,32,16";
  options.tma_latency = "32";
  options.tma_initiation = "32";
  options.cp_async_latency = "7";
  options.cp_async_initiation = "7";
  options.cp_async_commit_latency = "7";
  options.cp_async_commit_initiation = "7";
  options.cp_async_wait_latency = "5";
  options.cp_async_wait_initiation = "5";

  EXPECT_DEATH(parse_timing_profile(options), "FlashGPU-Sim panic:");

  options.integer_initiation = "1,1,1,1,2,4";
  options.tma_initiation = "33";
  EXPECT_DEATH(parse_timing_profile(options), "FlashGPU-Sim panic:");
}

TEST(SassTimingProjectionTest, SelectsNativeExecutionPipelineClass) {
  timing_profile profile;
  profile.integer_latency = {{10, 11, 12, 13, 14, 15}};
  profile.integer_initiation = {{1, 2, 3, 4, 5, 6}};
  profile.fp32_latency = {{20, 21, 22, 23, 24}};
  profile.fp32_initiation = {{1, 2, 3, 4, 5}};
  profile.tensor_latency = {{30, 31, 32, 33, 34, 35, 36}};
  profile.tensor_initiation = {{10, 11, 12, 13, 14, 15, 16}};
  profile.wgmma_ss_latency = {{4, 4, 4, 4}};
  profile.wgmma_rs_latency = {{3, 3, 3, 3}};
  profile.wgmma_ss_initiation = {{4, 4, 4, 4}};
  profile.wgmma_rs_initiation = {{3, 3, 3, 3}};
  profile.wgmma_ss_completion = {{66, 66, 66, 66}};
  profile.wgmma_rs_completion = {{64, 65, 64, 64}};
  profile.wgmma_compute_throughput = {{4096, 2048, 8192, 8192, 65536}};

  const timing_instruction imad(
      official_instruction(0, "IMAD.WIDE", "R2,R9,0x4,R2"), profile);
  const timing_instruction viadd(
      official_instruction(0x08, "VIADD", "R3,R2,UR8"), profile);
  const timing_instruction viadd_minmax(
      official_instruction(0x0c, "VIADDMNMX", "R18,R18,R9,0xffffffff,!PT"),
      profile);
  const timing_instruction setmaxreg(
      official_instruction(0x10, "USETMAXREG.TRY_ALLOC.CTAPOOL", "UP0,0xa0"),
      profile);
  const timing_instruction warpgroup_arrive(
      official_instruction(0x18, "WARPGROUP.ARRIVE"), profile);
  const timing_instruction nanosleep(
      official_instruction(0x1c, "NANOSLEEP.SYNCS", "0x989680"), profile);
  const timing_instruction hgmma_ss(
      official_instruction(0x20, "HGMMA.64x176x16.F32",
                           "R24,gdesc[UR40],R24,gsb0"),
      profile);
  const timing_instruction hgmma_rs(
      official_instruction(0x30, "HGMMA.64x128x16.F32",
                           "R112,R176,gdesc[UR8].tnspB,R112"),
      profile);
  const timing_instruction hgmma_commit(
      official_instruction(0x34, "HGMMA.64x8x16.F16",
                           "RZ,gdesc[URZ],RZ,!UPT,gsb0"),
      profile);
  const timing_instruction warpgroup_wait(
      official_instruction(0x38, "WARPGROUP.DEPBAR.LE", "gsb0,0x1"), profile);
  const timing_instruction shuffle(
      official_instruction(0x10, "SHFL.IDX", "R2,P0,R3,R4,R5"), profile);
  const timing_instruction multiply(
      official_instruction(0x20, "FMUL", "R2,R3,R4"), profile);
  const timing_instruction fma(
      official_instruction(0x30, "FFMA", "R2,R3,R4,R5"), profile);
  const timing_instruction half2_fma(
      official_instruction(0x40, "HFMA2", "R2,R3,R4,R5"), profile);
  const timing_instruction uniform_multiply(
      official_instruction(0x50, "UFMUL.FTZ", "UR2,UR3,UR4"), profile);
  const timing_instruction mufu(official_instruction(0x58, "MUFU.EX2", "R2,R3"),
                                profile);
  const timing_instruction ldsm(
      official_instruction(0x5a, "LDSM.16.M88.4", "R8[R2]"), profile);
  const timing_instruction stsm(
      official_instruction(0x5c, "STSM.16.M88.4", "[R2],R8"), profile);
  const timing_instruction lds(official_instruction(0x5e, "LDS.128", "R8[R2]"),
                               profile);
  ASSERT_EQ(lds.outcount, 4u);
  ASSERT_EQ(lds.get_register_file_destination_count(), 4u);
  for (unsigned word = 0; word < 4; ++word) {
    EXPECT_EQ(lds.arch_reg.dst[word], 9 + static_cast<int>(word));
    EXPECT_EQ(lds.get_register_file_destination(word).reg, 8u + word);
  }
  const timing_instruction sts(official_instruction(0x5f, "STS.128", "[R2],R8"),
                               profile);
  const timing_instruction hm16(
      official_instruction(0x60, "HMMA.16816.F32", "R8,R12,R16,R8"), profile);
  const timing_instruction hm8(
      official_instruction(0x70, "HMMA.1688.F32", "R8,R12,R16,R8"), profile);

  EXPECT_EQ(imad.latency, 13u);
  EXPECT_EQ(imad.initiation_interval, 4u);
  EXPECT_EQ(viadd.op_pipe, INTP__OP);
  EXPECT_EQ(viadd.latency, 10u);
  EXPECT_EQ(viadd.initiation_interval, 1u);
  EXPECT_EQ(viadd_minmax.op_pipe, INTP__OP);
  EXPECT_EQ(viadd_minmax.latency, 10u);
  EXPECT_EQ(viadd_minmax.initiation_interval, 1u);
  EXPECT_EQ(setmaxreg.op, ALU_OP);
  EXPECT_EQ(setmaxreg.op_pipe, INTP__OP);
  EXPECT_EQ(setmaxreg.outcount, 1u);
  EXPECT_EQ(setmaxreg.arch_reg.dst[0], 521);
  EXPECT_FALSE(
      warpgroup_arrive.get_wgmma_static_info().is_warpgroup_instruction());
  EXPECT_EQ(nanosleep.op, ALU_OP);
  EXPECT_EQ(nanosleep.op_pipe, INTP__OP);
  EXPECT_EQ(hgmma_ss.op, TENSOR_CORE_OP);
  EXPECT_EQ(hgmma_ss.op_pipe, TENSOR_CORE__OP);
  EXPECT_EQ(hgmma_ss.latency, 12u);
  EXPECT_EQ(hgmma_ss.initiation_interval, 12u);
  EXPECT_EQ(hgmma_ss.wgmma_compute_latency, 88u);
  EXPECT_EQ(hgmma_ss.wgmma_completion_tail_latency, 66u);
  EXPECT_EQ(hgmma_ss.get_wgmma_static_info().accumulator_bytes_per_thread,
            352u);
  EXPECT_FALSE(hgmma_ss.get_wgmma_static_info().uses_register_a());
  EXPECT_TRUE(hgmma_ss.get_wgmma_static_info().commit_group_after_issue);
  EXPECT_EQ(hgmma_rs.latency, 6u);
  EXPECT_EQ(hgmma_rs.initiation_interval, 6u);
  EXPECT_EQ(hgmma_rs.wgmma_compute_latency, 64u);
  EXPECT_EQ(hgmma_rs.wgmma_completion_tail_latency, 64u);
  EXPECT_TRUE(hgmma_rs.get_wgmma_static_info().uses_register_a());
  EXPECT_FALSE(hgmma_rs.get_wgmma_static_info().commit_group_after_issue);
  EXPECT_EQ(hgmma_commit.op, ALU_OP);
  EXPECT_EQ(hgmma_commit.op_pipe, INTP__OP);
  EXPECT_TRUE(hgmma_commit.get_wgmma_static_info().is_group_control());
  EXPECT_EQ(hgmma_commit.get_wgmma_static_info().operation,
            inst_t::wgmma_static_info_t::WGMMA_COMMIT_GROUP);
  EXPECT_EQ(hgmma_commit.wgmma_compute_latency, 0u);
  EXPECT_EQ(hgmma_commit.outcount, 0u);
  EXPECT_EQ(hgmma_commit.incount, 0u);
  EXPECT_TRUE(warpgroup_wait.get_wgmma_static_info().is_group_control());
  EXPECT_EQ(warpgroup_wait.get_wgmma_static_info().wait_group_num, 1u);
  EXPECT_EQ(shuffle.latency, 15u);
  EXPECT_EQ(shuffle.initiation_interval, 6u);
  EXPECT_EQ(multiply.latency, 22u);
  EXPECT_EQ(multiply.initiation_interval, 3u);
  EXPECT_EQ(fma.latency, 23u);
  EXPECT_EQ(fma.initiation_interval, 4u);
  EXPECT_EQ(half2_fma.op_pipe, SP__OP);
  EXPECT_EQ(half2_fma.latency, 23u);
  EXPECT_EQ(half2_fma.initiation_interval, 4u);
  EXPECT_EQ(uniform_multiply.op_pipe, SP__OP);
  EXPECT_EQ(uniform_multiply.latency, 22u);
  EXPECT_EQ(uniform_multiply.initiation_interval, 3u);
  EXPECT_EQ(uniform_multiply.native_text(), "UFMUL.FTZ UR2,UR3,UR4");
  EXPECT_TRUE(mufu.is_mio_client());
  EXPECT_TRUE(ldsm.is_mio_client());
  EXPECT_TRUE(stsm.is_mio_client());
  EXPECT_TRUE(lds.is_mio_client());
  EXPECT_TRUE(sts.is_mio_client());
  EXPECT_FALSE(multiply.is_mio_client());
  EXPECT_EQ(hm16.latency, 30u);
  EXPECT_EQ(hm16.initiation_interval, 10u);
  EXPECT_EQ(hm8.latency, 33u);
  EXPECT_EQ(hm8.initiation_interval, 13u);
}

TEST(SassTimingProjectionTest, SelectsConfiguredUniformExecutionClass) {
  timing_profile profile;
  profile.uniform_unit_index = 0;
  profile.uniform_latency = 4;
  profile.uniform_initiation = 2;

  const timing_instruction multiply(
      official_instruction(0, "UFMUL.FTZ", "UR2,UR3,UR4"), profile);
  const timing_instruction add(
      official_instruction(0x10, "UIADD3", "UR2,UR3,UR4,UR5"), profile);
  const timing_instruction minmax(
      official_instruction(0x20, "UVIMNMX.S32", "UR2,UR3,UR4,UPT"), profile);
  const timing_instruction predicate_lut(
      official_instruction(0x30, "UPLOP3.LUT", "UP0,UP1,UPT,UPT,UPT,0x80,0x8"),
      profile);
  const timing_instruction permute(
      official_instruction(0x40, "UPRMT", "UR15,URZ,0x654,UR15"), profile);
  EXPECT_EQ(multiply.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(multiply.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(multiply.latency, 4u);
  EXPECT_EQ(multiply.initiation_interval, 2u);
  EXPECT_EQ(add.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(add.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(add.initiation_interval, 2u);
  EXPECT_EQ(minmax.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(minmax.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(minmax.initiation_interval, 2u);
  EXPECT_EQ(predicate_lut.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(permute.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(permute.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(predicate_lut.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(predicate_lut.outcount, 2u);
}

TEST(SassTimingProjectionTest, ProjectsUniformPredicateRegisterPacking) {
  timing_profile profile;
  profile.uniform_unit_index = 0;
  profile.uniform_latency = 4;
  profile.uniform_initiation = 2;

  const timing_instruction projected(
      official_instruction(0, "UP2UR", "UR45,UPR,URZ,0x1"), profile);
  EXPECT_EQ(projected.op, SPECIALIZED_UNIT_1_OP);
  EXPECT_EQ(projected.op_pipe, SPECIALIZED__OP);
  EXPECT_EQ(projected.outcount, 1u);
  EXPECT_EQ(projected.incount, 7u);
  EXPECT_EQ(projected.arch_reg.dst[0], 302);
  for (unsigned predicate = 0; predicate < 7; ++predicate)
    EXPECT_EQ(projected.arch_reg.src[predicate],
              521 + static_cast<int>(predicate));
}

TEST(SassTimingProjectionTest, ProjectsDescriptorGlobalLoadRegisters) {
  const instruction source =
      official_instruction(0xd0, "LDG.E", "R2,desc[UR4][R2.64]");
  const timing_instruction projected(source);

  EXPECT_EQ(projected.op, LOAD_OP);
  EXPECT_EQ(projected.op_pipe, MEM__OP);
  EXPECT_EQ(projected.memory_op, memory_load);
  EXPECT_EQ(projected.space.get_type(), global_space);
  EXPECT_EQ(projected.data_size, 4u);
  EXPECT_EQ(projected.outcount, 1u);
  EXPECT_EQ(projected.incount, 3u);
  EXPECT_EQ(projected.arch_reg.dst[0], 3);
  EXPECT_EQ(projected.arch_reg.src[0], 261);
  EXPECT_EQ(projected.arch_reg.src[1], 3);
  EXPECT_EQ(projected.arch_reg.src[2], 4);
  EXPECT_EQ(projected.get_num_operands(), 1u);
  EXPECT_EQ(projected.get_num_regs(), 1u);
}

TEST(SassTimingProjectionTest, ProjectsGlobalAtomicAsResponseDrivenAtomic) {
  const timing_instruction projected(official_instruction(
      0xe0, "ATOMG.E.ADD.STRONG.GPU", "PT,R2,desc[UR8][R2.64],R11"));

  EXPECT_EQ(projected.op, LOAD_OP);
  EXPECT_EQ(projected.op_pipe, MEM__OP);
  EXPECT_EQ(projected.memory_op, memory_load);
  EXPECT_EQ(projected.space.get_type(), global_space);
  EXPECT_EQ(projected.cache_op, CACHE_GLOBAL);
  EXPECT_EQ(projected.data_size, 4u);
  EXPECT_TRUE(projected.isatomic());
  EXPECT_FALSE(projected.executes_atomic_callback());

  ASSERT_EQ(projected.outcount, 1u);
  EXPECT_EQ(projected.arch_reg.dst[0], 3);  // R2, not the leading PT.
  ASSERT_EQ(projected.incount, 4u);
  EXPECT_EQ(projected.arch_reg.src[0], 265);  // UR8.
  EXPECT_EQ(projected.arch_reg.src[1], 3);    // R2.64 address low.
  EXPECT_EQ(projected.arch_reg.src[2], 4);    // R2.64 address high.
  EXPECT_EQ(projected.arch_reg.src[3], 12);   // R11 addend.
  EXPECT_EQ(projected.get_num_operands(), 2u);
  EXPECT_EQ(projected.get_num_regs(), 2u);

  ASSERT_EQ(projected.get_register_file_source_count(), 1u);
  EXPECT_EQ(projected.get_register_file_source(0).reg, 11u);
  EXPECT_EQ(projected.get_register_file_source(0).slot, 0u);
  ASSERT_EQ(projected.get_register_file_destination_count(), 1u);
  EXPECT_EQ(projected.get_register_file_destination(0).reg, 2u);
}

TEST(SassTimingProjectionTest, ProjectsSharedAtomicAsTimingOnlyL2Read) {
  const auto source = official_instruction(0, "ATOMS.ADD", "R9[UR4],R0");
  const timing_instruction projected(source);
  EXPECT_EQ(projected.op, LOAD_OP);
  EXPECT_EQ(projected.space.get_type(), global_space);
  EXPECT_EQ(projected.cache_op, CACHE_GLOBAL);
  EXPECT_EQ(projected.data_size, 4u);
  EXPECT_TRUE(projected.isatomic());
  EXPECT_FALSE(projected.executes_atomic_callback());
  ASSERT_EQ(projected.outcount, 1u);
  EXPECT_EQ(projected.arch_reg.dst[0], 10);  // R9.
  ASSERT_EQ(projected.incount, 2u);
  EXPECT_EQ(projected.arch_reg.src[0], 261);  // UR4 address.
  EXPECT_EQ(projected.arch_reg.src[1], 1);    // R0 addend.
  ASSERT_EQ(projected.get_register_file_destination_count(), 1u);
  EXPECT_EQ(projected.get_register_file_destination(0).reg, 9u);
  ASSERT_EQ(projected.get_register_file_source_count(), 1u);
  EXPECT_EQ(projected.get_register_file_source(0).reg, 0u);
  const timing_instruction discarded(
      official_instruction(0, "ATOMS.ADD", "RZ[UR4],R0"));
  EXPECT_EQ(discarded.outcount, 0u);
  EXPECT_EQ(discarded.get_register_file_destination_count(), 0u);
  EXPECT_TRUE(orders_deferred_cta_barrier(source, architecture::kSm90));
  EXPECT_TRUE(orders_deferred_cta_barrier(source, architecture::kSm120));
}

TEST(SassTimingProjectionTest, PreservesGlobalLoadCacheScope) {
  const timing_instruction all(
      official_instruction(0x00, "LDG.E.STRONG.SM", "R4,desc[UR8][R2.64]"));
  const timing_instruction global(
      official_instruction(0x10, "LDG.E.STRONG.GPU", "R4,desc[UR8][R2.64]"));
  const timing_instruction system(
      official_instruction(0x20, "LDG.E.STRONG.SYS", "R4,desc[UR8][R2.64]"));

  EXPECT_EQ(all.cache_op, CACHE_ALL);
  EXPECT_EQ(global.cache_op, CACHE_GLOBAL);
  EXPECT_EQ(system.cache_op, CACHE_VOLATILE);
}

TEST(SassTimingProjectionTest, ProjectsDescriptorGenericMemoryRegisters) {
  const timing_instruction load(
      official_instruction(0xd0, "LD.E.128", "R8,desc[UR4][R2.64]"));
  EXPECT_EQ(load.op, LOAD_OP);
  EXPECT_EQ(load.memory_op, memory_load);
  EXPECT_EQ(load.space.get_type(), generic_space);
  EXPECT_EQ(load.data_size, 16u);
  EXPECT_EQ(load.outcount, 4u);
  EXPECT_EQ(load.incount, 3u);

  const timing_instruction store(
      official_instruction(0xe0, "ST.E", "desc[UR14][R6.64],RZ"));
  EXPECT_EQ(store.op, STORE_OP);
  EXPECT_EQ(store.memory_op, memory_store);
  EXPECT_EQ(store.space.get_type(), generic_space);
  EXPECT_EQ(store.data_size, 4u);
  EXPECT_EQ(store.outcount, 0u);
  EXPECT_EQ(store.incount, 3u);
  EXPECT_EQ(store.arch_reg.src[0], 271);
  EXPECT_EQ(store.arch_reg.src[1], 7);
  EXPECT_EQ(store.arch_reg.src[2], 8);
}

TEST(SassTimingProjectionTest, ProjectsNativeTmaPipeline) {
  timing_profile profile;
  profile.tma_latency = 37;
  profile.tma_initiation = 19;
  profile.cp_async_commit_latency = 13;
  profile.cp_async_commit_initiation = 7;

  const timing_instruction load(
      official_instruction(0x80, "UTMALDG.2D", "[UR8][UR4],desc[UR28]"),
      profile);
  EXPECT_EQ(load.op, TENSOR_MEMORY_ACCELERATOR_OP);
  EXPECT_EQ(load.latency, 37u);
  EXPECT_EQ(load.initiation_interval, 19u);
  EXPECT_EQ(load.get_tma_static_info().tensor_dim, 2u);
  EXPECT_EQ(load.get_tma_static_info().src_space,
            inst_t::tma_static_info_t::TMA_GLOBAL);
  EXPECT_EQ(load.get_tma_static_info().dst_space,
            inst_t::tma_static_info_t::TMA_SHARED_CTA);

  const timing_instruction store(
      official_instruction(0x90, "UTMASTG.2D", "[UR8][UR4]"), profile);
  EXPECT_EQ(store.op, TENSOR_MEMORY_ACCELERATOR_OP);
  EXPECT_EQ(store.latency, 37u);
  EXPECT_EQ(store.initiation_interval, 19u);
  EXPECT_EQ(store.get_tma_static_info().tensor_dim, 2u);
  EXPECT_EQ(store.get_tma_static_info().src_space,
            inst_t::tma_static_info_t::TMA_SHARED_CTA);
  EXPECT_EQ(store.get_tma_static_info().dst_space,
            inst_t::tma_static_info_t::TMA_GLOBAL);

  const timing_instruction prefetch(
      official_instruction(0xa0, "UTMAPF.L2.4D", "[UR8][UR12]"), profile);
  EXPECT_EQ(prefetch.op, TENSOR_MEMORY_ACCELERATOR_OP);
  EXPECT_EQ(prefetch.latency, 37u);
  EXPECT_EQ(prefetch.initiation_interval, 19u);
  EXPECT_EQ(prefetch.outcount, 0u);
  EXPECT_EQ(prefetch.incount, 2u);

  const timing_instruction cp_arrive(
      official_instruction(0xb0, "ARRIVES.LDGSTSBAR.64.TRANSCNT", "[UR4]"),
      profile);
  const timing_instruction cp_arrive_noinc(
      official_instruction(0xc0, "ARRIVES.LDGSTSBAR.64.ARVCNT", "[UR4]"),
      profile);
  EXPECT_EQ(cp_arrive.op, ASYNC_COPY_OP);
  EXPECT_EQ(cp_arrive.op_pipe, CP_ASYNC__OP);
  EXPECT_TRUE(cp_arrive.m_is_cp_async_mbarrier_arrive);
  EXPECT_EQ(cp_arrive.latency, 13u);
  EXPECT_EQ(cp_arrive.initiation_interval, 7u);
  EXPECT_TRUE(
      cp_arrive.get_async_copy_static_info().mbarrier_increment_pending);
  EXPECT_FALSE(
      cp_arrive_noinc.get_async_copy_static_info().mbarrier_increment_pending);

  const timing_instruction bulk_load(
      official_instruction(0xb0, "UBLKCP.S.G", "[UR10][UR14],UR13"), profile);
  const timing_instruction bulk_store(
      official_instruction(0xc0, "UBLKCP.G.S", "[UR10][UR14],UR13"), profile);
  const timing_instruction bulk_reduce(
      official_instruction(0xd0, "UBLKRED.G.S.ADD.F32.RN",
                           "[UR26][UR25],UR9,desc[UR30]"),
      profile);
  EXPECT_EQ(bulk_load.op, TENSOR_MEMORY_ACCELERATOR_OP);
  EXPECT_EQ(bulk_load.get_tma_static_info().dst_space,
            inst_t::tma_static_info_t::TMA_SHARED_CTA);
  EXPECT_EQ(bulk_load.get_tma_static_info().src_space,
            inst_t::tma_static_info_t::TMA_GLOBAL);
  EXPECT_EQ(bulk_store.get_tma_static_info().dst_space,
            inst_t::tma_static_info_t::TMA_GLOBAL);
  EXPECT_EQ(bulk_store.get_tma_static_info().src_space,
            inst_t::tma_static_info_t::TMA_SHARED_CTA);
  EXPECT_EQ(bulk_reduce.get_tma_static_info().tma_type,
            inst_t::tma_static_info_t::TMA_NORMAL);
  EXPECT_EQ(bulk_reduce.latency, 37u);
  EXPECT_EQ(bulk_reduce.initiation_interval, 19u);
}

TEST(SassTimingProjectionTest, ProjectsLdGstsAsyncCopyPipeline) {
  timing_profile profile;
  profile.cp_async_latency = 7;
  profile.cp_async_initiation = 3;
  profile.cp_async_commit_latency = 5;
  profile.cp_async_commit_initiation = 2;
  profile.cp_async_wait_latency = 6;
  profile.cp_async_wait_initiation = 1;

  const timing_instruction copy(
      official_instruction(0xad0, "LDGSTS.E.BYPASS.128",
                           "[R51],desc[UR10][R20.64]"),
      profile);
  EXPECT_EQ(copy.op, ASYNC_COPY_OP);
  EXPECT_EQ(copy.op_pipe, CP_ASYNC__OP);
  EXPECT_TRUE(copy.m_is_ldgsts);
  EXPECT_EQ(copy.memory_op, memory_load);
  EXPECT_EQ(copy.space.get_type(), global_space);
  EXPECT_EQ(copy.data_size, 16u);
  EXPECT_EQ(copy.latency, 7u);
  EXPECT_EQ(copy.initiation_interval, 3u);
  EXPECT_EQ(copy.outcount, 0u);
  EXPECT_EQ(copy.incount, 4u);

  const timing_instruction commit(official_instruction(0xc70, "LDGDEPBAR", ""),
                                  profile);
  EXPECT_EQ(commit.op, ASYNC_COPY_OP);
  EXPECT_TRUE(commit.m_is_ldgdepbar);
  EXPECT_EQ(commit.latency, 5u);
  EXPECT_EQ(commit.initiation_interval, 2u);

  const timing_instruction wait(
      official_instruction(0x11a0, "DEPBAR.LE", "SB0,0x2"), profile);
  EXPECT_EQ(wait.op, ASYNC_COPY_OP);
  EXPECT_TRUE(wait.m_is_depbar);
  EXPECT_EQ(wait.m_depbar_group_no, 2u);
  EXPECT_EQ(wait.latency, 6u);
  EXPECT_EQ(wait.initiation_interval, 1u);
  EXPECT_EQ(wait.get_tma_static_info().tma_type,
            inst_t::tma_static_info_t::TMA_BULK_WAIT);
  EXPECT_EQ(wait.get_tma_static_info().bulk_wait_num, 2u);

  const timing_instruction tma_commit(
      official_instruction(0x11b0, "UTMACMDFLUSH", ""), profile);
  EXPECT_EQ(tma_commit.op, TENSOR_MEMORY_ACCELERATOR_OP);
  EXPECT_EQ(tma_commit.get_tma_static_info().tma_type,
            inst_t::tma_static_info_t::TMA_BULK_COMMIT);
  EXPECT_EQ(tma_commit.latency, 5u);
  EXPECT_EQ(tma_commit.initiation_interval, 2u);
}

TEST(SassTimingProjectionTest, CountsGuardAsARegisterOperand) {
  const timing_instruction projected(
      official_instruction(0x70, "EXIT", "", "@P0"));

  EXPECT_EQ(projected.op, EXIT_OPS);
  EXPECT_EQ(projected.outcount, 0u);
  EXPECT_EQ(projected.incount, 1u);
  EXPECT_EQ(projected.get_num_operands(), 1u);
  EXPECT_EQ(projected.get_num_regs(), 1u);
  EXPECT_GE(projected.get_num_operands(), projected.get_num_regs());
}

TEST(SassTimingProjectionTest, ProjectsCollectiveControlMarkers) {
  const timing_instruction begin(
      official_instruction(0x30, "WARPSYNC.COLLECTIVE", "R0,0x50"));
  const timing_instruction end(official_instruction(0x40, "ENDCOLLECTIVE"));

  EXPECT_EQ(begin.op, BRANCH_OP);
  EXPECT_EQ(end.op, BRANCH_OP);
}

TEST(SassTimingProjectionTest, ProjectsUniformIndirectBranch) {
  const timing_instruction branch(
      official_instruction(0x980, "BRXU", "UR6,-0x990"));
  EXPECT_EQ(branch.op, BRANCH_OP);
  EXPECT_EQ(branch.outcount, 0u);
  EXPECT_EQ(branch.incount, 2u);
  EXPECT_EQ(branch.arch_reg.src[0], 263);
  EXPECT_EQ(branch.arch_reg.src[1], 264);
}

TEST(SassTimingProjectionTest, AcceptsOfficiallyResolvedSymbolCall) {
  instruction call =
      official_instruction(0xff0, "CALL.REL.NOINC", "$__internal_helper");
  ASSERT_FALSE(call.operands_structured);
  call.attributes.push_back({"target-pc", "0x5160"});
  const timing_instruction projected(call);
  EXPECT_EQ(projected.op, CALL_OPS);
  EXPECT_EQ(projected.outcount, 0u);

  const timing_instruction returned(
      official_instruction(0x5520, "RET.REL.NODEC", "R2,relative_call"));
  EXPECT_EQ(returned.op, CALL_OPS);
  EXPECT_EQ(returned.outcount, 0u);
  EXPECT_EQ(returned.incount, 1u);
}

TEST(SassTimingProjectionTest, ProjectsWideConstantLoadAndStore) {
  const timing_instruction load(
      official_instruction(0x80, "LDC.64", "R2,c[0x0][0x380]"));
  EXPECT_EQ(load.op, LOAD_OP);
  EXPECT_EQ(load.space.get_type(), const_space);
  EXPECT_EQ(load.data_size, 8u);
  EXPECT_EQ(load.outcount, 2u);
  EXPECT_EQ(load.arch_reg.dst[0], 3);
  EXPECT_EQ(load.arch_reg.dst[1], 4);
  EXPECT_EQ(load.get_num_operands(), 1u);
  EXPECT_EQ(load.get_num_regs(), 0u);

  const timing_instruction uniform_load(
      official_instruction(0xa0, "ULDC.64", "UR4,c[0x0][0x198]"));
  EXPECT_EQ(uniform_load.op, LOAD_OP);
  EXPECT_EQ(uniform_load.op_pipe, MEM__OP);
  EXPECT_EQ(uniform_load.space.get_type(), const_space);
  EXPECT_EQ(uniform_load.data_size, 8u);
  EXPECT_EQ(uniform_load.outcount, 2u);
  EXPECT_EQ(uniform_load.arch_reg.dst[0], 261);
  EXPECT_EQ(uniform_load.arch_reg.dst[1], 262);

  const timing_instruction imad_wide(
      official_instruction(0xc0, "IMAD.WIDE", "R2,R9,0x4,R2"));
  EXPECT_EQ(imad_wide.outcount, 2u);
  EXPECT_EQ(imad_wide.arch_reg.dst[0], 3);
  EXPECT_EQ(imad_wide.arch_reg.dst[1], 4);
  EXPECT_EQ(imad_wide.incount, 2u);
  EXPECT_EQ(imad_wide.arch_reg.src[0], 10);
  EXPECT_EQ(imad_wide.arch_reg.src[1], 3);
  EXPECT_EQ(imad_wide.get_num_operands(), 3u);
  EXPECT_EQ(imad_wide.get_num_regs(), 2u);

  const timing_instruction store(
      official_instruction(0x120, "STG.E", "desc[UR4][R6.64],R9"));
  EXPECT_EQ(store.op, STORE_OP);
  EXPECT_EQ(store.memory_op, memory_store);
  EXPECT_EQ(store.space.get_type(), global_space);
  EXPECT_EQ(store.outcount, 0u);
  EXPECT_EQ(store.incount, 4u);
  EXPECT_EQ(store.arch_reg.src[0], 261);
  EXPECT_EQ(store.arch_reg.src[1], 7);
  EXPECT_EQ(store.arch_reg.src[2], 8);
  EXPECT_EQ(store.arch_reg.src[3], 10);
  EXPECT_EQ(store.get_num_operands(), 2u);
  EXPECT_EQ(store.get_num_regs(), 2u);
}

TEST(SassTimingProjectionTest, ProjectsCombinedMbarrierArriveExpectTx) {
  const timing_instruction projected(official_instruction(
      0x1980, "SYNCS.ARRIVE.TRANS64", "RZ[R7+URZ],R0", "@P1"));

  ASSERT_EQ(projected.op, MBARRIER_OP);
  const auto &info = projected.get_mbarrier_static_info();
  EXPECT_EQ(info.operation,
            inst_t::mbarrier_static_info_t::MBARRIER_ARRIVE_EXPECT_TX);
  EXPECT_TRUE(info.arrive);
  EXPECT_TRUE(info.expect_tx);
}

TEST(SassTimingProjectionTest, ProjectsMbarrierTransactionCompletion) {
  const timing_instruction projected(official_instruction(
      0x1990, "SYNCS.ARRIVE.TRANS64.RED.A0TX", "RZ[R7+URZ],R0"));

  ASSERT_EQ(projected.op, MBARRIER_OP);
  const auto &info = projected.get_mbarrier_static_info();
  EXPECT_EQ(info.operation,
            inst_t::mbarrier_static_info_t::MBARRIER_COMPLETE_TX);
  EXPECT_FALSE(info.arrive);
  EXPECT_FALSE(info.expect_tx);
}

TEST(SassTimingProjectionTest, ProjectsUniformIntegerPredicateComparison) {
  const timing_instruction projected(
      official_instruction(0x2500, "UISETP.GT.AND", "UP0,UPT,UR55,0x1,UPT"));

  EXPECT_EQ(projected.op, ALU_OP);
  EXPECT_EQ(projected.op_pipe, INTP__OP);
}

TEST(SassTimingProjectionTest, ProjectsUniformFloatPredicateComparison) {
  const timing_instruction projected(official_instruction(
      0x2510, "UFSETP.GEU.AND", "UP1,UPT,|UR4|,1.17549435e-38,UPT"));

  EXPECT_EQ(projected.op, SP_OP);
  EXPECT_EQ(projected.op_pipe, SP__OP);
}

TEST(SassTimingProjectionTest, ProjectsLocalStackCachePolicies) {
  const timing_instruction load(
      official_instruction(0x6540, "LDL.LU", "R3,[R1+0x1c]"));
  EXPECT_EQ(load.space.get_type(), local_space);
  EXPECT_EQ(load.cache_op, CACHE_LAST_USE);

  const timing_instruction store(
      official_instruction(0x53f0, "STL", "[R1+0x8],R6"));
  EXPECT_EQ(store.space.get_type(), local_space);
  EXPECT_EQ(store.cache_op, CACHE_WRITE_BACK);
}

TEST(SassTimingProjectionTest, FailsClosedForUnknownTimingOpcode) {
  EXPECT_DEATH(
      timing_instruction(official_instruction(0, "FUTURE.OP", "R1,R2")),
      "FlashGPU-Sim panic:");
}

TEST(SassTimingProjectionTest, ProjectsEveryRealTritonGemmInstruction) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/triton_gemm_sm120.sassir";
  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(sassir, "kernel_tma_gemm");

  ASSERT_EQ(image.instructions.size(), 936u);
  for (const instruction &source : image.instructions) {
    (void)timing_instruction{source};
  }
}

TEST(SassTimingProjectionTest, ProjectsEveryRealTritonAttentionInstruction) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/triton_attention_sm120.sassir";
  sassir_decoder decoder;
  const kernel image =
      decoder.decode_kernel(sassir, "_llama3_gqa_attn_fwd_qkv_tiled");

  // Keep the gate tied to the complete checked-in attention cubin.
  ASSERT_EQ(image.instructions.size(), 3640u);
  for (const instruction &source : image.instructions) {
    (void)timing_instruction{source};
  }
}

TEST(SassTimingProjectionTest, ProjectsEveryRealLlamaQkvInstruction) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/llama_qkv_sm120.sassir";
  sassir_decoder decoder;
  const kernel image =
      decoder.decode_kernel(sassir, "_llama3_layer_matmul_tile");

  ASSERT_EQ(image.instructions.size(), 1040u);
  for (const instruction &source : image.instructions) {
    (void)timing_instruction{source};
  }
}

TEST(SassTimingProjectionTest, ProjectsEveryRealLlamaResidualInstruction) {
  const std::string sassir =
      std::string(SASS_GENERATED_SASSIR_DIR) + "/llama_residual_sm120.sassir";
  sassir_decoder decoder;
  const kernel image =
      decoder.decode_kernel(sassir, "_llama3_layer_matmul_residual_tile");

  ASSERT_EQ(image.instructions.size(), 1224u);
  for (const instruction &source : image.instructions) {
    (void)timing_instruction{source};
  }
}

}  // namespace
