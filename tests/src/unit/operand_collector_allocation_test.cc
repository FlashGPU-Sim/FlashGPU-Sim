#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "gpgpu-sim/shader.h"

namespace {
class CollectorTestConfig : public core_config {
 public:
  CollectorTestConfig() : core_config(nullptr) { warp_size = 32; }
  void init() override {}
};

class CollectorTestWarp : public warp_inst_t {
 public:
  CollectorTestWarp(const core_config *config, unsigned uid, unsigned scheduler)
      : warp_inst_t(config) {
    m_empty = false;
    m_uid = uid;
    m_scheduler_id = scheduler;
    m_warp_id = uid;
    m_dynamic_warp_id = uid;
    set_num_operands(0);
  }
};
}  // namespace

// Exercise production allocation and register-set movement, including busy
// collectors, without launching a simulated kernel.
class OperandCollectorAllocationTest : public ::testing::Test {
 protected:
  void Init(bool partitioned = true, bool second_set = false) {
    rfu_.sub_core_model = partitioned;
    rfu_.m_num_warp_scheds = 4;
    rfu_.m_num_banks = 16;
    rfu_.m_num_banks_per_sched = 4;
    rfu_.add_cu_set(0, 16, 0);
    if (second_set) rfu_.add_cu_set(1, 16, 0);
    for (auto &entry : rfu_.m_cus) {
      for (unsigned k = 0; k < entry.second.size(); ++k) {
        entry.second[k].init(k, 16, &config_, &rfu_, partitioned, k / 4, 4);
      }
    }
    rfu_.m_arbiter.init(rfu_.m_cu.size(), 16);
    opndcoll_rfu_t::port_vector_t inputs{&input_}, outputs{&output_};
    opndcoll_rfu_t::uint_vector_t sets{0};
    if (second_set) sets.push_back(1);
    rfu_.add_port(inputs, outputs, sets);
  }

  void Put(register_set &regs, unsigned slot, unsigned uid, unsigned scheduler,
           bool partitioned = true) {
    warp_inst_t **target = regs.get_free(partitioned, slot);
    ASSERT_NE(target, nullptr);
    **target = CollectorTestWarp(&config_, uid, scheduler);
  }

  void Occupy(unsigned set, unsigned scheduler) {
    for (unsigned k = scheduler * 4; k < (scheduler + 1) * 4; ++k) {
      busy_inputs_.emplace_back(new register_set(4, "busy"));
      Put(*busy_inputs_.back(), scheduler, 100 + k, scheduler);
      ASSERT_TRUE(
          rfu_.m_cus[set][k].allocate(busy_inputs_.back().get(), &output_));
    }
  }

  void Allocate() { rfu_.allocate_cu(0); }
  bool HasHead(unsigned scheduler) { return input_.has_ready(true, scheduler); }
  unsigned HeadUid(unsigned scheduler) {
    return (*input_.get_ready(true, scheduler))->get_uid();
  }
  bool Free(unsigned set, unsigned index) {
    return rfu_.m_cus[set][index].is_free();
  }
  unsigned CollectorWarp(unsigned set, unsigned index) {
    return rfu_.m_cus[set][index].get_warp_id();
  }

  std::vector<unsigned> SelectReady(unsigned count, bool partitioned = true) {
    opndcoll_rfu_t::dispatch_unit_t dispatch(&rfu_.m_cus[0]);
    dispatch.init(partitioned, 4);
    std::vector<unsigned> selected;
    for (unsigned i = 0; i < count; ++i) {
      auto *cu = dispatch.find_ready();
      if (!cu) break;
      selected.push_back(cu->get_id());
    }
    return selected;
  }

  CollectorTestConfig config_;
  opndcoll_rfu_t rfu_;
  register_set input_{4, "input"}, output_{4, "output"};
  std::vector<std::unique_ptr<register_set>> busy_inputs_;
};

TEST_F(OperandCollectorAllocationTest, BusyPartitionDoesNotBlockFreePartition) {
  Init();
  Occupy(0, 0);
  Put(input_, 0, 10, 0);
  Put(input_, 1, 11, 1);
  Allocate();
  ASSERT_TRUE(HasHead(0));
  EXPECT_EQ(HeadUid(0), 10u);
  EXPECT_FALSE(HasHead(1));
  ASSERT_FALSE(Free(0, 4));
  EXPECT_EQ(CollectorWarp(0, 4), 11u);
}

TEST_F(OperandCollectorAllocationTest, SelectsOldestEligiblePartition) {
  Init();
  Put(input_, 0, 20, 0);
  Put(input_, 1, 10, 1);
  Allocate();
  EXPECT_TRUE(HasHead(0));
  EXPECT_FALSE(HasHead(1));
  EXPECT_EQ(CollectorWarp(0, 4), 10u);
}

TEST_F(OperandCollectorAllocationTest, AllBusyPreservesBothHeads) {
  Init();
  for (unsigned scheduler = 0; scheduler < 4; ++scheduler) Occupy(0, scheduler);
  Put(input_, 0, 10, 0);
  Put(input_, 1, 11, 1);
  Allocate();
  ASSERT_TRUE(HasHead(0));
  ASSERT_TRUE(HasHead(1));
  EXPECT_EQ(HeadUid(0), 10u);
  EXPECT_EQ(HeadUid(1), 11u);
}

TEST_F(OperandCollectorAllocationTest, NonPartitionedKeepsGlobalOldest) {
  Init(false);
  Put(input_, 0, 20, 0, false);
  Put(input_, 0, 10, 1, false);
  Allocate();
  ASSERT_TRUE(input_.has_ready());
  EXPECT_EQ((*input_.get_ready())->get_uid(), 20u);
  EXPECT_EQ(CollectorWarp(0, 0), 10u);
}

TEST_F(OperandCollectorAllocationTest, KeepsFirstEligibleCollectorSet) {
  Init(true, true);
  Occupy(0, 0);
  Put(input_, 0, 10, 0);
  Put(input_, 1, 11, 1);
  Allocate();
  EXPECT_TRUE(HasHead(0));
  EXPECT_FALSE(HasHead(1));
  EXPECT_TRUE(Free(1, 0));
  ASSERT_FALSE(Free(0, 4));
  EXPECT_EQ(CollectorWarp(0, 4), 11u);
}

// Keep production collectors continuously eligible, as if each selected slot
// were drained and refilled before the next arbitration. Exercise find_ready
// itself, not a duplicate of its selection algorithm.
TEST_F(OperandCollectorAllocationTest, DispatchDoesNotStarveWithinSubcore) {
  Init();
  for (unsigned scheduler = 0; scheduler < 4; ++scheduler) Occupy(0, scheduler);
  auto selected = SelectReady(64);
  ASSERT_EQ(selected.size(), 64u);
  std::vector<unsigned> counts(16, 0);
  for (unsigned i = 0; i < selected.size(); ++i) {
    ++counts[selected[i]];
    EXPECT_EQ(selected[i] / 4, (i + 1) % 4);
  }
  for (unsigned slot = 0; slot < counts.size(); ++slot) {
    SCOPED_TRACE(slot);
    EXPECT_EQ(counts[slot], 4u);
  }
}

TEST_F(OperandCollectorAllocationTest, DispatchSkipsBlockedSubcoreOutput) {
  Init();
  for (unsigned scheduler = 0; scheduler < 4; ++scheduler) Occupy(0, scheduler);
  Put(output_, 1, 1000, 1);
  auto selected = SelectReady(48);
  ASSERT_EQ(selected.size(), 48u);
  std::vector<unsigned> counts(16, 0);
  for (unsigned slot : selected) ++counts[slot];
  for (unsigned slot = 0; slot < counts.size(); ++slot) {
    SCOPED_TRACE(slot);
    EXPECT_EQ(counts[slot], slot / 4 == 1 ? 0u : 4u);
  }
}

TEST_F(OperandCollectorAllocationTest, DispatchEmptyCollectorsHaveNoWinner) {
  Init();
  EXPECT_TRUE(SelectReady(16).empty());
}

TEST_F(OperandCollectorAllocationTest, DispatchNonPartitionedKeepsLegacyOrder) {
  Init();
  for (unsigned scheduler = 0; scheduler < 4; ++scheduler) Occupy(0, scheduler);
  auto selected = SelectReady(32, false);
  ASSERT_EQ(selected.size(), 32u);
  for (unsigned i = 0; i < selected.size(); ++i)
    EXPECT_EQ(selected[i], (i + 1) % 16);
}
