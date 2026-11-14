#include <gtest/gtest.h>
#include "../../../libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"

#include <cstdio>
#include <vector>

namespace {

struct TestPort : mem_fetch_interface {
  std::vector<mem_fetch *> sent;
  bool full(unsigned, bool) const override { return false; }
  void push(mem_fetch *mf) override { sent.push_back(mf); }
};

class InspectableCache : public data_cache {
 public:
  using data_cache::data_cache;
  tag_array *tags() const { return m_tag_array; }
};

class CacheInvalidationTest
    : public testing::TestWithParam<std::tuple<char, bool>> {};

TEST_P(CacheInvalidationTest, OutstandingFillDoesNotOwnReusedTag) {
  const auto [type, replace_with_store] = GetParam();
  gpgpu_context context;
  memory_config memory(&context);
  memory.SST_mode = true;
  memory.icnt_flit_size = 32;

  cache_config config;
  char options[80];
  std::snprintf(options, sizeof(options), "%c:1:128:2,L:T:m:L:L,A:8:8,8:8,32",
                type);
  config.init(options, FuncCachePreferNone);
  TestPort port;
  InspectableCache cache("invalidation", config, 0, 0, &port, nullptr,
                         IN_L1D_MISS_QUEUE, L1_WR_ALLOC_R, L1_WRBK_ACC, nullptr,
                         L1_GPU_CACHE);
  std::vector<mem_fetch *> requests;
  mem_access_sector_mask_t sector;
  sector.set(0);
  auto request = [&](unsigned addr, unsigned time, bool write = false) {
    const unsigned size = write ? config.get_atom_sz() : 32;
    mem_access_byte_mask_t bytes;
    for (unsigned i = 0; i < size; ++i) bytes.set(i);
    mem_access_t access(write ? GLOBAL_ACC_W : GLOBAL_ACC_R, addr, size, write,
                        active_mask_t().set(), bytes, sector, &context);
    auto *mf = new mem_fetch(access, nullptr, 0, 8, 0, 0, 0, &memory, time);
    requests.push_back(mf);
    std::list<cache_event> events;
    EXPECT_EQ(cache.access(addr, mf, time, events), MISS);
    cache.cycle();
    return mf;
  };
  auto probe = [&](unsigned addr) {
    unsigned index = 0;
    return cache.tags()->probe(addr, index, sector, false);
  };
  auto *a = request(0x1000, 1);
  // Another warp finishes a GPU-scope fence while A is still outstanding.
  cache.invalidate();
  auto *b = request(0x2000, 2, replace_with_store);  // Reuses A's tag.
  request(0x1000, 3);  // Merges into A's existing MSHR.
  a->set_reply();
  cache.fill(a, 4);
  // A's stale index must not complete B's reservation or resurrect A.
  const bool old_fill_safe =
      probe(0x2000) == (replace_with_store ? HIT : HIT_RESERVED) &&
      probe(0x1000) == MISS;
  if (!replace_with_store) {
    b->set_reply();
    cache.fill(b, 5);
  }
  unsigned returned = 0;
  while (cache.access_ready()) {
    cache.next_access();
    ++returned;
  }
  unsigned reserved = 0, dirty = 0;
  auto *tags = cache.tags();
  for (unsigned i = 0; i < tags->size(); ++i) {
    reserved += tags->get_block(i)->is_reserved_line();
    dirty += tags->get_block(i)->is_modified_line();
  }
  EXPECT_TRUE(old_fill_safe);
  EXPECT_EQ(returned, replace_with_store ? 2u : 3u);
  EXPECT_EQ(port.sent.size(), 2u);
  EXPECT_EQ(reserved, 0u);
  EXPECT_EQ(probe(0x2000), HIT);
  EXPECT_EQ(dirty, replace_with_store ? 1u : 0u);
  for (auto *mf : requests) delete mf;
}

INSTANTIATE_TEST_SUITE_P(NormalAndSector, CacheInvalidationTest,
                         testing::Combine(testing::Values('N', 'S'),
                                          testing::Bool()));
}  // namespace
