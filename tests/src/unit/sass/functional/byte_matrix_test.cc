#include "../test_support.h"

namespace {

TEST(ByteMatrixNative, MultiMatrixAddressesAndBytes) {
  for (unsigned count : {1,2,4}) {
    const std::string opcode = "STSM.8.MT168" + (count == 1 ? std::string{} : "." + std::to_string(count));
    kernel image;
    image.name="byte_matrix"; image.arch=architecture::kSm100; image.instruction_bytes=16;
    image.instructions={official_instruction(0,opcode,"[R10],R20")};
    frontend sass; sass.add_kernel(std::move(image)); register_sm100_functional_semantics(sass);
    mapped_functional_memory memory; memory.map(memory_space::kShared,0x100,0x400,true);
    warp_state state;
    for(unsigned lane=0;lane<32;++lane) {
      state.write_register(lane,10,0x100+(lane/8)*256+(lane%8)*32);
      for(unsigned matrix=0;matrix<count;++matrix)
        state.write_register(lane,20+matrix,0x03020100u+0x04040404u*lane+0x11111111u*matrix);
    }
    execution_context context{&memory};
    auto result=sass.step("byte_matrix",state,context);
    ASSERT_EQ(result.status,step_status::kAdvanced) << result.detail;
    for(unsigned lane=0;lane<32;++lane) {
      ASSERT_EQ(context.memory_accesses[lane].size(),4*count);
      for(const auto &access:context.memory_accesses[lane]) {
        EXPECT_EQ(access.bytes,1u); EXPECT_TRUE(access.write);
      }
    }
    for(unsigned m=0;m<count;++m)
      for(unsigned col=0;col<8;++col)
        for(unsigned row=0;row<16;++row) {
          const unsigned lane=(row%8)*4+col/2, byte=(row/8)*2+col%2;
          const uint32_t packed=0x03020100u+0x04040404u*lane+0x11111111u*m;
          EXPECT_EQ(memory.inspect<uint8_t>(memory_space::kShared,0x100+m*256+col*32+row),uint8_t(packed>>(byte*8)));
        }
  }
}
TEST(ByteMatrixNative, RejectsPartialWarpAndUnalignedAddressBeforeWriting) {
  for(unsigned invalid: {0,1}) {
    kernel image;
    image.name="bad_matrix"; image.arch=architecture::kSm100; image.instruction_bytes=16;
    image.instructions={official_instruction(0,"STSM.8.MT168.4","[R10],R20")};
    frontend sass; sass.add_kernel(std::move(image)); register_sm100_functional_semantics(sass);
    mapped_functional_memory memory; memory.map(memory_space::kShared,0x100,0x400,true);
    warp_state state;
    if(invalid==0) state.active_mask=0x7fffffffu;
    for(unsigned lane=0;lane<32;++lane) state.write_register(lane,10,0x100+lane*16+(invalid==1 && lane==31));
    execution_context context{&memory};
    auto result=sass.step("bad_matrix",state,context);
    EXPECT_EQ(result.status,step_status::kUnsupported) << result.detail;
    for(const auto &accesses:context.memory_accesses) EXPECT_TRUE(accesses.empty());
  }
}

} // namespace
