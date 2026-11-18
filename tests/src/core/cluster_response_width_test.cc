#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"

namespace {
class ResponseTestGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  simt_core_cluster *cluster() { return m_cluster[0]; }
};

class ResponseTestCore : public shader_core_ctx {
 public:
  static void cycle_lsu(shader_core_ctx &core) {
    auto *lsu = core.*(&ResponseTestCore::m_ldst_unit);
    lsu->cycle();
    lsu->end_memory_transport_cycle();
  }
};

struct ResponseCase {
  const char *name;
  bool ingress;
  unsigned width, capacity, ticks;
  std::vector<mem_access_type> types;
  std::vector<unsigned> sectors;
  unsigned accepted;
};

template <typename Check>
void RunWithResponseTestGpu(bool ingress, unsigned width, unsigned capacity,
                            Check check) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.func_sim->ptx_opcocde_latency_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const std::string overrides =
      "-gpgpu_n_clusters 1 -gpgpu_n_cores_per_cluster 2 -network_mode 2 "
      "-gpgpu_memlatency_stat 0 -gpgpu_tma_response_width 1 "
      "-gpgpu_cp_async_response_width 1 "
      "-gpgpu_cluster_response_ingress_sectors_per_cycle " +
      std::to_string(ingress ? width : 0) +
      " -gpgpu_cluster_response_dispatch_sectors_per_cycle " +
      std::to_string(ingress ? 0 : width) +
      " -gpgpu_n_cluster_ejection_buffer_size " +
      std::to_string(ingress ? capacity : 32) +
      " -gpgpu_n_ldst_response_buffer_size " +
      std::to_string(ingress ? 32 : capacity);
  option_parser_delimited_string(options, overrides.c_str(), " ");
  config.init();
  ResponseTestGpu gpu(config, &ctx);
  gpu.init();
  check(gpu, ctx);
  // Each subprocess owns its GPU, interconnect globals and queued replies.
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}

mem_fetch *MakeReply(gpgpu_context &ctx, const memory_config *memory,
                     unsigned sid, mem_access_type type, new_addr_type addr,
                     unsigned sectors) {
  const bool write = type == GLOBAL_ACC_W;
  mem_access_t access(type, addr, sectors * SECTOR_SIZE, write, &ctx);
  auto *mf = new mem_fetch(access, nullptr, 0, 8, 0, sid, 0, memory, 0);
  mf->set_reply();
  mf->set_return_timestamp(0);
  return mf;
}

// Exercise the production cluster loop and per-SM queues using a deterministic
// interconnect producer. Consumers do not retire replies in these cases.
void CheckResponse(ResponseTestGpu &gpu, gpgpu_context &ctx,
                   const ResponseCase &test) {
  const auto *shader = gpu.getShaderCoreConfig();
  const auto *memory = gpu.getMemoryConfig();
  std::vector<std::deque<mem_fetch *>> incoming(2);
  std::vector<std::vector<mem_fetch *>> packets(2);
  for (unsigned sid = 0; sid < 2; ++sid) {
    for (unsigned i = 0; i < test.types.size(); ++i) {
      auto *mf = MakeReply(ctx, memory, sid, test.types[i], 0x1000 + i * 128,
                           test.sectors[i]);
      packets[sid].push_back(mf);
      if (test.ingress) incoming[sid].push_back(mf);
      else {
        mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE, 0);
        gpu.cluster()->push_response_fifo(mf);
      }
    }
  }
  icnt_top = [&](unsigned node) -> void * {
    for (unsigned sid = 0; sid < 2; ++sid)
      if (node == shader->topology().global_sm_node_id(sid))
        return incoming[sid].empty() ? nullptr : incoming[sid].front();
    return nullptr;
  };
  icnt_pop = [&](unsigned node) -> void * {
    auto *mf = static_cast<mem_fetch *>(icnt_top(node));
    if (mf) incoming[mf->get_sid()].pop_front();
    return mf;
  };
  for (unsigned tick = 0; tick < test.ticks; ++tick) {
    gpu.gpu_sim_cycle = tick + 1;
    gpu.cluster()->icnt_cycle();
  }
  for (unsigned sid = 0; sid < 2; ++sid) {
    unsigned accepted = 0;
    for (unsigned i = 0; i < packets[sid].size(); ++i) {
      const auto state = packets[sid][i]->get_status();
      const bool moved = test.ingress
          ? std::find(incoming[sid].begin(), incoming[sid].end(), packets[sid][i]) == incoming[sid].end()
          : state == IN_SHADER_LDST_RESPONSE_FIFO || state == IN_TMA_RESPONSE_FIFO;
      EXPECT_EQ(moved, i < test.accepted) << test.name << " SM=" << sid << " packet=" << i;
      accepted += moved;
    }
    EXPECT_EQ(accepted, test.accepted) << test.name << " SM=" << sid;
    if (test.ingress) {
      EXPECT_EQ(incoming[sid].size(), packets[sid].size() - test.accepted);
    }
  }
}

void PrintTo(const ResponseCase &test, std::ostream *out) {
  *out << test.name << " width=" << test.width << " capacity=" << test.capacity
       << " ticks=" << test.ticks << " expected=" << test.accepted;
}

class ClusterResponseWidthTest : public testing::TestWithParam<ResponseCase> {};
TEST_P(ClusterResponseWidthTest, HonorsBudgetAndBackpressure) {
  const auto &test = GetParam();
  ASSERT_EXIT(
      RunWithResponseTestGpu(test.ingress, test.width, test.capacity,
                             [&](ResponseTestGpu &gpu, gpgpu_context &ctx) {
                               CheckResponse(gpu, ctx, test);
                             }),
      testing::ExitedWithCode(0), "");
}

void CheckOversizedDispatch(unsigned capacity, unsigned expected_tick) {
  RunWithResponseTestGpu(
      false, 3, capacity, [&](ResponseTestGpu &gpu, gpgpu_context &ctx) {
        auto *cluster = gpu.cluster();
        const auto *memory = gpu.getMemoryConfig();
        std::vector<mem_fetch *> oversized;
        for (unsigned sid = 0; sid < cluster->num_cores(); ++sid) {
          // An ACK uses one dispatch slot and can retire through the real LSU
          // without creating a load instruction or a pending cache miss.
          cluster->get_core(sid)->inc_store_req(0);
          auto *ack = MakeReply(ctx, memory, sid, GLOBAL_ACC_W, 0x1000, 1);
          auto *load = MakeReply(ctx, memory, sid, GLOBAL_ACC_R, 0x1080, 4);
          for (auto *reply : {ack, load}) {
            reply->set_status(IN_CLUSTER_TO_SHADER_QUEUE, 0);
            cluster->push_response_fifo(reply);
          }
          oversized.push_back(load);
        }

        for (unsigned tick = 1; tick <= 3; ++tick) {
          gpu.gpu_sim_cycle = tick;
          cluster->icnt_cycle();
          for (unsigned sid = 0; sid < oversized.size(); ++sid) {
            const auto expected = tick < expected_tick
                                      ? IN_CLUSTER_TO_SHADER_QUEUE
                                      : IN_SHADER_LDST_RESPONSE_FIFO;
            const auto actual = oversized[sid]->get_status();
            if (actual != expected) {
              std::fprintf(stderr,
                           "SM %u tick %u: oversized reply in LSU FIFO=%d, "
                           "expected dispatch on tick %u\n",
                           sid, tick, actual == IN_SHADER_LDST_RESPONSE_FIFO,
                           expected_tick);
              ADD_FAILURE();
            }
            if (tick == 1) {
              auto &core = *cluster->get_core(sid);
              ASSERT_EQ(core.ldst_unit_response_buffer_full(), capacity == 1);
              ResponseTestCore::cycle_lsu(core);
              ASSERT_FALSE(core.ldst_unit_response_buffer_full());
            }
          }
        }
      });
}

TEST(ClusterResponseDispatchCreditTest, FullLsuFifoPreventsCarry) {
  ASSERT_EXIT(CheckOversizedDispatch(1, 3), testing::ExitedWithCode(0), "");
}

TEST(ClusterResponseDispatchCreditTest, AvailableLsuFifoAllowsCarry) {
  ASSERT_EXIT(CheckOversizedDispatch(2, 2), testing::ExitedWithCode(0), "");
}

std::vector<ResponseCase> Cases() {
  std::vector<ResponseCase> cases;
  for (bool ingress : {false, true}) {
    const char *prefix = ingress ? "Ingress" : "Dispatch";
    // Names are generated by the parameter suffix below.
    for (unsigned width : {0u, 1u, 2u, 4u})
      cases.push_back({prefix, ingress, width, 16, 1,
                       std::vector<mem_access_type>(6, GLOBAL_ACC_R),
                       std::vector<unsigned>(6, 1), width ? width : 1});
    cases.push_back({prefix, ingress, 4, 1, 1,
                     {GLOBAL_ACC_R, GLOBAL_ACC_R}, {1, 1}, 1});
    cases.push_back({prefix, ingress, 4, 0, 1,
                     {GLOBAL_ACC_R, GLOBAL_ACC_R}, {1, 1}, 0});
    cases.push_back({prefix, ingress, 4, 16, 1,
                     {GLOBAL_ACC_R, TMA_ACC_R, CP_ASYNC_ACC_R, GLOBAL_ACC_R, TMA_ACC_R},
                     {1, 1, 1, 1, 1}, 4});
    cases.push_back({prefix, ingress, 1, 16, 4, {GLOBAL_ACC_R}, {4}, 1});
    cases.push_back({prefix, ingress, 1, 16, 3, {GLOBAL_ACC_R}, {4}, 0});
    cases.push_back({prefix, ingress, 4, 16, 1,
                     std::vector<mem_access_type>(6, GLOBAL_ACC_W),
                     std::vector<unsigned>(6, 1), 4});
  }
  // Instruction replies are transport packets too; this only tests ingress,
  // not the independent one-fill-per-tick instruction-cache consumer.
  cases.push_back({"InstructionIngress", true, 4, 16, 1,
                   {INST_ACC_R, GLOBAL_ACC_R, INST_ACC_R, GLOBAL_ACC_R},
                   {1, 1, 1, 1}, 4});
  // Legacy async packet widths remain separate consumer limits. Explicit
  // ingress budgets govern transport; dispatch still obeys the TMA port cap.
  cases.push_back({"AsyncLegacy", true, 0, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R}, {1, 1}, 1});
  cases.push_back({"AsyncIngress", true, 4, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R}, {1, 1}, 2});
  cases.push_back({"AsyncDispatch", false, 4, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R, GLOBAL_ACC_R}, {1, 1, 1}, 1});
  return cases;
}
INSTANTIATE_TEST_SUITE_P(PerSm, ClusterResponseWidthTest,
                         testing::ValuesIn(Cases()));
}  // namespace
