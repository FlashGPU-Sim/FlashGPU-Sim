#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <limits>
#include "gpgpu-sim/flash/tensormap.h"
namespace {
// Put the descriptor immediately before a guard page. This reproduces an
// invalid captured Driver map being sent to the public error printer without
// relying on ASAN, stack layout, or a particular out-of-bounds value.
void print_at_guard(unsigned rank) {
  const auto page = size_t(sysconf(_SC_PAGESIZE));
  auto *base = static_cast<unsigned char *>(mmap(nullptr, page * 2,
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (base == MAP_FAILED || mprotect(base + page, page, PROT_NONE)) _exit(3);
  auto *d = reinterpret_cast<tensormap_descriptor_t *>(base + page - sizeof(tensormap_descriptor_t));
  std::memset(d, 0, sizeof(*d));
  d->fields.tensorRank = rank;
  d->fields.tensorDataType = TMA_DTYPE_U8;
  for (unsigned i = 0; i < 5; ++i) {
    d->fields.boxDim[i] = 2;
    d->fields.globalDim[i] = 2;
    d->fields.elementStrides[i] = 1;
  }
  d->print();
  munmap(base, page * 2);
  _exit(0);
}
TEST(TensorMapDiagnostics, InvalidRanksNeverReadBeyondDescriptor) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  for (unsigned rank : {5u, 2581408u, std::numeric_limits<unsigned>::max()})
    EXPECT_EXIT(print_at_guard(rank), ::testing::ExitedWithCode(0), "") << rank;
}
TEST(TensorMapDiagnostics, ValidRanksRemainPrintable) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  for (unsigned rank=0; rank<5; ++rank)
    EXPECT_EXIT(print_at_guard(rank), ::testing::ExitedWithCode(0), "") << rank;
}
} // namespace
