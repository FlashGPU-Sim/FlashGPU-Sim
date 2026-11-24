#include "../test_support.h"
TEST(WarpShift,WrappedRightCountsAndSourceAlias) {
  for(const char *opcode:{"SHF.R.W.U32","SHF.R.W.U32.HI"})
    for(unsigned n:{0u,1u,31u,32u,33u,63u,255u,0xffffffffu}) {
      const bool hi=std::string(opcode).find(".HI")!=std::string::npos;
      kernel image;image.name="shift";image.arch=architecture::kSm100;image.instruction_bytes=16;
      image.instructions={official_instruction(0,opcode,hi ? "R2,RZ,R3,R2":"R1,R1,R3,R2")};
      frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
      warp_state state;state.active_mask=1;state.write_register(0,1,0x12345678);state.write_register(0,2,0xfedcba98);state.write_register(0,3,n);
      execution_context ctx;auto result=sass.step("shift",state,ctx);
      ASSERT_EQ(result.status,step_status::kAdvanced) << result.detail;
      const unsigned k=n&31;
      const unsigned expected=hi ? 0xfedcba98u>>k : k ? (0x12345678u>>k)|(0xfedcba98u<<(32-k)):0x12345678u;
      EXPECT_EQ(state.read_register(0,hi?2:1),expected) << n;
    }
}
