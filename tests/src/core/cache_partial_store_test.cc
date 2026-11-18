#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <vector>

#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "libcuda/gpgpu_context.h"
#include "option_parser.h"

namespace {

struct StoreTestPort : mem_fetch_interface {
  std::vector<mem_fetch *> sent;
  unsigned writes = 0;
  bool full(unsigned, bool) const override { return false; }
  void push(mem_fetch *mf) override {
    sent.push_back(mf);
    if (mf->get_is_write()) ++writes;
  }
};

class StoreTestCache : public data_cache {
 public:
  using data_cache::data_cache;
  tag_array &tags() const { return *m_tag_array; }
};

void CheckPartialStore(gpgpu_context &context, gpgpu_sim &gpu, char type,
                       bool invalidate) {
  memory_config memory(&context);
  memory.SST_mode = true;
  memory.icnt_flit_size = 32;
  cache_config config;
  char options[80];
  std::snprintf(options, sizeof(options), "%c:1:128:1,L:B:m:F:L,A:8:8,8:8,32",
                type);
  config.init(options, FuncCachePreferNone);
  StoreTestPort port;
  StoreTestCache cache("partial-store", config, 0, 0, &port, nullptr,
                       IN_L1D_MISS_QUEUE, L1_WR_ALLOC_R, L1_WRBK_ACC, &gpu,
                       L1_GPU_CACHE);
  mem_access_sector_mask_t sector;
  sector.set(0);
  auto make_request = [&](unsigned addr, unsigned time, bool write) {
    const unsigned size = write ? 4 : 32;
    mem_access_byte_mask_t bytes;
    for (unsigned i = 0; i < size; ++i) bytes.set(i);
    mem_access_t access(write ? GLOBAL_ACC_W : GLOBAL_ACC_R, addr, size, write,
                        active_mask_t().set(), bytes, sector, &context);
    return std::make_unique<mem_fetch>(access, nullptr, 0, 8, 0, 0, 0, &memory,
                                       time);
  };
  auto issue = [&](mem_fetch *mf, unsigned time) {
    std::list<cache_event> events;
    const auto status = cache.access(mf->get_addr(), mf, time, events);
    cache.cycle();
    return status;
  };
  constexpr unsigned a_addr = 0x1000;
  constexpr unsigned b_addr = 0x2000;
  auto a = make_request(a_addr, 1, false);
  ASSERT_EQ(issue(a.get(), 1), MISS);
  ASSERT_EQ(port.sent.size(), 1u);

  // B completes in A's former slot while A's response is still outstanding.
  if (invalidate) {
    cache.invalidate();
    auto b = make_request(b_addr, 2, false);
    ASSERT_EQ(issue(b.get(), 2), MISS);
    ASSERT_EQ(port.sent.size(), 2u);
    b->set_reply();
    cache.fill(b.get(), 3);
    ASSERT_TRUE(cache.access_ready());
    EXPECT_EQ(cache.next_access(), b.get());
    EXPECT_FALSE(cache.access_ready());
    cache.cycle();
  }

  auto store = make_request(a_addr, 4, true);
  auto store_status = issue(store.get(), 4);
  ASSERT_TRUE(store_status == MISS || store_status == RESERVATION_FAIL);
  if (store_status == RESERVATION_FAIL) {
    ASSERT_TRUE(invalidate);
    EXPECT_EQ(port.sent.size(), 2u);
    unsigned index = 0;
    ASSERT_EQ(cache.tags().probe(b_addr, index, sector, false), HIT);
    EXPECT_FALSE(cache.tags().get_block(index)->is_modified_line());
  }
  a->set_reply();
  cache.fill(a.get(), 5);
  ASSERT_TRUE(cache.access_ready());
  if (store_status == RESERVATION_FAIL) {
    // A ready response still owns the MSHR until its final consumer retires.
    EXPECT_EQ(issue(store.get(), 5), RESERVATION_FAIL);
    EXPECT_EQ(port.sent.size(), 2u);
  }
  EXPECT_EQ(cache.next_access(), a.get());

  // A safe implementation may stall the store until A's old MSHR drains.
  if (store_status == RESERVATION_FAIL) {
    ASSERT_FALSE(cache.access_ready());
    const auto sent_before_retry = port.sent.size();
    store_status = issue(store.get(), 6);
    ASSERT_EQ(store_status, MISS);
    ASSERT_EQ(port.sent.size(), sent_before_retry + 1);
    auto *fetch = port.sent.back();
    ASSERT_EQ(fetch->get_access_type(), L1_WR_ALLOC_R);
    fetch->set_reply();
    cache.fill(fetch, 7);
  }

  ASSERT_TRUE(cache.access_ready());
  std::unique_ptr<mem_fetch> write_allocate(cache.next_access());
  EXPECT_EQ(write_allocate->get_access_type(), L1_WR_ALLOC_R);
  EXPECT_EQ(write_allocate->get_original_wr_mf(), store.get());
  EXPECT_FALSE(cache.access_ready());
  EXPECT_EQ(port.writes, 0u);

  // An accepted write-back store must retain its bytes in A for writeback.
  // The invalidated old fill must not silently retire the store without them.
  unsigned index = 0;
  const auto a_status = cache.tags().probe(a_addr, index, sector, true);
  std::fprintf(stderr,
               "partial store type=%c invalidate=%d status=%s A=%s "
               "downstream_writes=%u\n",
               type, invalidate, cache_request_status_str(store_status),
               cache_request_status_str(a_status), port.writes);
  EXPECT_EQ(a_status, HIT) << "completed partial store has no cache tag";
  if (a_status == HIT) {
    auto *block = cache.tags().get_block(index);
    EXPECT_TRUE(block->is_modified_line());
    EXPECT_EQ(block->get_dirty_byte_mask() & store->get_access_byte_mask(),
              store->get_access_byte_mask());
  }
}

void RunPartialStore(char type, bool invalidate) {
  gpgpu_context context;
  gpgpu_sim_config config(&context);
  auto *options = option_parser_create();
  context.ptx_reg_options(options);
  context.func_sim->ptx_opcocde_latency_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  option_parser_delimited_string(
      options,
      "-gpgpu_n_clusters 1 -gpgpu_n_cores_per_cluster 1 -network_mode 2 "
      "-gpgpu_memlatency_stat 0",
      " ");
  config.init();
  exec_gpgpu_sim gpu(config, &context);
  gpu.init();
  CheckPartialStore(context, gpu, type, invalidate);
  // The subprocess owns GPU, parser and interconnect globals.
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}

class CachePartialStoreTest
    : public testing::TestWithParam<std::tuple<char, bool>> {};

TEST_P(CachePartialStoreTest, CompletedFetchOnWriteRetainsDirtyBytes) {
  const auto [type, invalidate] = GetParam();
  ASSERT_EXIT(RunPartialStore(type, invalidate), testing::ExitedWithCode(0),
              "");
}

INSTANTIATE_TEST_SUITE_P(NormalAndSector, CachePartialStoreTest,
                         testing::Combine(testing::Values('N', 'S'),
                                          testing::Bool()));
}  // namespace
