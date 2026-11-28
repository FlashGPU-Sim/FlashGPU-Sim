#include "ptx_cluster_store.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "../shader.h"
#include "cluster_store.h"
#include "panic.h"
#include "tb_cluster.h"
#include <cstring>
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_cluster_store(const ptx_instruction *inst,
                               ptx_thread_info *thread) {
  const auto &options = inst->get_options();
  if (std::find(options.begin(), options.end(), ASYNC_OPTION) == options.end())
    return false;
  auto fail = [&]() {
    panic("unqualified PTX async cluster store: " + inst->to_string());
  };
  if (!inst->is_cluster_async_store() || inst->get_num_operands() != 3 ||
      inst->get_scalar_type().size() != 1)
    fail();
  for (auto option : options)
    if (option != ASYNC_OPTION && option != TMA_MBAR_COMPLETE_BYTES &&
        option != CLUSTER_OPTION && option != WEAK_DIRECTIVE &&
        option != V2_TYPE && option != V4_TYPE)
      fail();
  const unsigned type = inst->get_type(), vec = inst->get_vector();
  const unsigned count = vec == 0         ? 1
                         : vec == V2_TYPE ? 2
                         : vec == V4_TYPE ? 4
                                          : 0;
  const bool bits32 = type == B32_TYPE || type == U32_TYPE ||
                      type == S32_TYPE || type == F32_TYPE;
  const bool bits64 = type == B64_TYPE || type == U64_TYPE ||
                      type == S64_TYPE || type == F64_TYPE;
  const bool bits128 = type == BB128_TYPE;
  const unsigned element_bytes = bits32 ? 4 : bits64 ? 8 : bits128 ? 16 : 0;
  const unsigned bytes = count * element_bytes;
  if (!count || !element_bytes || bytes > 16 || (bits128 && vec))
    fail();
  const auto space = inst->get_space();
  // The lexer represents .shared::cluster as CLUSTER_OPTION; it does not
  // necessarily populate the ordinary load/store space field.
  const bool explicit_shared =
      space == shared_space || inst->membar_level() == CLUSTER_OPTION;
  if ((space != shared_space && space != generic_space &&
       space != undefined_space) ||
      (explicit_shared && inst->membar_level() != CLUSTER_OPTION))
    fail();
  auto *core = dynamic_cast<shader_core_ctx *>(thread->get_core());
  tb_cluster_target_t target;
  auto address = [&](const operand_info &operand, bool is_destination) {
    const uint64_t raw =
        thread->get_operand_value(operand, operand, U64_TYPE, thread, 0).u64;
    tb_cluster_target_t resolved;
    uint32_t state_address = 0;
    if (explicit_shared) {
      if (raw > UINT32_MAX ||
          !decode_ptx_shared_cluster(thread, raw, &resolved))
        fail();
      state_address = raw;
    } else {
      unsigned owner;
      addr_t offset;
      if (!decode_shared_generic(raw, &owner, &offset) ||
          !resolve_tb_cluster_owner_sm(core, thread->get_hw_ctaid(), owner,
                                       &resolved))
        fail();
      state_address = shared_cluster_address(
          resolved.core->get_cta_cluster_rank(resolved.cta_slot), offset);
    }
    if (is_destination)
      target = resolved;
    return state_address;
  };
  const uint32_t destination = address(inst->dst(), true);
  const uint32_t barrier = address(inst->src2(), false);
  const auto dim = thread->get_kernel().get_cluster_dim();
  cluster_store_request request;
  if (!decode_cluster_store(destination, barrier, bytes, dim.x * dim.y * dim.z,
                            request))
    fail();
  ptx_reg_t values[4];
  if (!vec) {
    if (inst->src1().is_vector())
      fail();
    values[0] =
        thread->get_operand_value(inst->src1(), inst->dst(), type, thread, 1);
  } else {
    if (!inst->src1().is_vector() || inst->src1().get_vect_nelem() != count)
      fail();
    thread->get_vector_operand_values(inst->src1(), values, count);
  }
  uint8_t data[16]{};
  for (unsigned i = 0; i < count; ++i)
    std::memcpy(data + i * element_bytes, &values[i].u128, element_bytes);
  target.smem->write(request.destination, bytes, data, thread, inst);
  inst_t::mbarrier_info_t info;
  info.bar_id = request.barrier;
  info.bar_count = bytes;
  info.is_remote = target.sm_id != thread->get_hw_sid() ||
                   target.cta_slot != thread->get_hw_ctaid();
  info.remote_cid = target.local_sm;
  info.remote_hw_cta = target.cta_slot;
  const_cast<ptx_instruction *>(inst)->set_mbarrier_info(thread->get_laneid(),
                                                         info);
  return true;
}
} // namespace flash_gpgpu_sim
