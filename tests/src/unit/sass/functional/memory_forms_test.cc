#include "../test_support.h"
namespace {
TEST(SassSharedBulk, UniformStoreRecordsOneByteExtentAndPreservesGuards) {
  for (const auto active : {0xffffffffu, 1u << 7}) {
    for (const unsigned words : {0u, 2u}) {
      kernel image;
      image.name="bulk"; image.arch=architecture::kSm100; image.instruction_bytes=16;
      image.instructions={official_instruction(0,"UMEMSETS.64","URZ[UR4],UR5")};
      frontend sass; sass.add_kernel(std::move(image));
      register_sm100_functional_semantics(sass);
      mapped_functional_memory memory;
      memory.map(memory_space::kShared,0,32,true);
      for(unsigned i=0;i<8;++i) memory.initialize<uint32_t>(memory_space::kShared,i*4,123u);
      execution_context context{&memory};
      warp_state state; state.active_mask=active;
      state.write_uniform_register(4,8); state.write_uniform_register(5,words);
      const auto result=sass.step("bulk",state,context);
      ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
      unsigned accesses=0;
      for(unsigned lane=0;lane<32;++lane)
        for(const auto &access:context.memory_accesses[lane]) {
          ++accesses;
          EXPECT_EQ(access.address,8u);
          EXPECT_EQ(access.bytes,words*8u);
        }
      EXPECT_EQ(accesses,words?1u:0u);
      for(unsigned i=0;i<8;++i)
        EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared,i*4),
                  words && i>=2 && i<6 ? 0u : 123u);
    }
  }
}
TEST(SassSharedBulk, RejectsNonzeroFillAndMisalignment) {
  for(bool bad_fill:{false,true}) {
    kernel image; image.name="invalid_bulk";image.arch=architecture::kSm100;image.instruction_bytes=16;
    image.instructions={official_instruction(0,"UMEMSETS.64",bad_fill?"UR6[UR4],UR5":"URZ[UR4],UR5")};
    frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
    mapped_functional_memory memory;memory.map(memory_space::kShared,0,32,true);
    execution_context context{&memory};warp_state state;
    state.write_uniform_register(4,bad_fill?8:4);state.write_uniform_register(5,1);
    state.write_uniform_register(6,123);
    EXPECT_EQ(sass.step("invalid_bulk",state,context).status,step_status::kUnsupported);
    EXPECT_EQ(state.pc,0u);
  }
}
} // namespace

namespace {
TEST(SassGlobal256, NoncontiguousGroupsPredicatesAndFullDependencyWidth) {
  const auto inst=official_instruction(0,"LDG.E.NA.EFL2.LTC256B.256.CONSTANT",
                                      "R20,R4,desc[UR8][R2.64]","@P0");
  kernel image;image.name="wide_load";image.arch=architecture::kSm100;image.instruction_bytes=16;
  image.instructions={inst};
  frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
  mapped_functional_memory memory;memory.map(memory_space::kGlobal,0x100,32,true);
  for(unsigned i=0;i<8;++i) memory.initialize<uint32_t>(memory_space::kGlobal,0x100+i*4,100+i);
  execution_context context{&memory};warp_state state;state.active_mask=3;
  for(unsigned lane=0;lane<2;++lane) {
    state.write_register(lane,2,0x100);state.write_register(lane,3,0);
    state.write_predicate(lane,0,lane==0);
    for(unsigned i=0;i<4;++i) {
      state.write_register(lane,4+i,999);state.write_register(lane,20+i,999);
    }
  }
  const auto result=sass.step("wide_load",state,context);
  ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
  for(unsigned i=0;i<4;++i) {
    EXPECT_EQ(state.read_register(0,4+i),100+i);
    EXPECT_EQ(state.read_register(0,20+i),104+i);
    EXPECT_EQ(state.read_register(1,4+i),999u);
    EXPECT_EQ(state.read_register(1,20+i),999u);
  }
  ASSERT_EQ(context.memory_accesses[0].size(),1u);
  EXPECT_EQ(context.memory_accesses[0][0].bytes,32u);
  EXPECT_TRUE(context.memory_accesses[1].empty());
  const timing_instruction projected(inst);
  EXPECT_EQ(projected.data_size,32u);
  ASSERT_EQ(projected.get_register_file_destination_count(),8u);
  for(unsigned i=0;i<8;++i)
    EXPECT_EQ(projected.get_register_file_destination(i).reg,i<4?20+i:4+i-4);
  std::vector<unsigned> dependencies;
  for(unsigned i=0;i<projected.outcount;++i) dependencies.push_back(projected.out[i]);
  dependencies.insert(dependencies.end(),projected.extra_out.begin(),projected.extra_out.end());
  EXPECT_EQ(dependencies,(std::vector<unsigned>{21,22,23,24,5,6,7,8}));
}
TEST(SassGlobal256, RejectsUnqualifiedMaskedForm) {
  kernel image;image.name="masked_load";image.arch=architecture::kSm100;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"LDG.E.NA.EFL2.LTC256B.256.CONSTANT",
                                          "R20,R4,desc[UR8][R2.64],0xf3")};
  frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
  mapped_functional_memory memory;execution_context context{&memory};warp_state state;
  EXPECT_EQ(sass.step("masked_load",state,context).status,step_status::kUnsupported);
}
} // namespace
