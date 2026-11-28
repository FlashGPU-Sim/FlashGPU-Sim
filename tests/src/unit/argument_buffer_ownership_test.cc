#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <list>
#include "../../../libcuda/gpgpu_context.h"
#include "cuda-sim/ptx_ir.h"
#include "ptx.tab.h"

TEST(ArgumentBufferOwnership, CopiesSurviveSourceMutationAndDestruction) {
  auto *ctx = GPGPU_Context();
  type_info type(nullptr, type_info_key(param_space_local, B8_TYPE, 0, 1, 0, 32));
  symbol destination("payload", &type, "ownership-test", 32, ctx);
  operand_info source(ctx);
  const std::array<unsigned char, 4> expected{1, 2, 3, 4};
  std::list<arg_buffer_t> saved;
  arg_buffer_t assigned(ctx);
  {
    auto bytes = expected;
    arg_buffer_t original(&destination, source, bytes.data(), bytes.size());
    saved.push_back(original);
    assigned = original;
    assigned = assigned;  // Must not free its own source storage.
    bytes.fill(99);
    std::array<unsigned char, 32> replacement{};
    original = arg_buffer_t(&destination, source, replacement.data(), replacement.size());
  }
  for (const auto *copy : {&saved.front(), &assigned}) {
    ASSERT_EQ(copy->get_param_buffer_size(), expected.size());
    EXPECT_EQ(std::memcmp(copy->get_param_buffer(), expected.data(), expected.size()), 0);
    EXPECT_EQ(copy->get_dst(), &destination);
  }
  EXPECT_NE(saved.front().get_param_buffer(), assigned.get_param_buffer());
}
