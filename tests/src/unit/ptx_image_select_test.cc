#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../../libcuda/ptx_image_select.h"

TEST(PtxImageSelect, SuffixDropsSameModuleBaseOnly) {
  const std::vector<std::string> files = {
      "new_module.sm_90a.ptx",
      "new_module.sm_90.ptx",
  };
  const std::vector<std::string> kept =
      drop_base_ptx_when_suffix_present(files);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_EQ(kept[0], "new_module.sm_90a.ptx");
  for (const auto &name : kept)
    EXPECT_NE(name, "old_module.sm_90a.ptx");
}
