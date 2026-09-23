#include "tensormap.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

//=============================================================================
// Helper Functions
//=============================================================================

// Get a 32-bit unsigned value from an operand
static uint32_t get_operand_u32(ptx_thread_info *thread,
                                const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
  return reg.u32;
}

// Get a 64-bit unsigned value from an operand
static uint64_t get_operand_u64(ptx_thread_info *thread,
                                const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
  return reg.u64;
}

} // namespace flash_gpgpu_sim

//=============================================================================
// tensormap_descriptor_t Member Functions
//=============================================================================

void tensormap_descriptor_t::print() const {
  char buf[1024];
  size_t pos = 0;
  uint32_t dims = num_dims();

  auto append_checked = [&](const char *fmt, ...) {
    if (pos >= sizeof(buf))
      return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + pos, sizeof(buf) - pos, fmt, ap);
    va_end(ap);
    if (n < 0) {
      pos = sizeof(buf) - 1;
      buf[pos] = '\0';
      return;
    }
    size_t rem = sizeof(buf) - pos;
    if ((size_t)n >= rem) {
      pos = sizeof(buf) - 1;
      buf[pos] = '\0';
    } else
      pos += (size_t)n;
  };

  append_checked("TensorMap Descriptor:\n");
  append_checked("  global_address: 0x%llx\n",
                 (unsigned long long)fields.globalAddress);
  append_checked("  rank: %u (num_dims=%u)\n", fields.tensorRank, dims);
  append_checked("  elemtype: %u (size=%u bytes)\n", fields.tensorDataType,
                 get_element_size());

  append_checked("  box_dim: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0)
      append_checked(", ");
    append_checked("%u", fields.boxDim[i]);
  }
  append_checked("]\n");

  append_checked("  global_dim: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0)
      append_checked(", ");
    append_checked("%u", fields.globalDim[i]);
  }
  append_checked("]\n");

  append_checked("  global_strides: [");
  if (dims > 1) {
    for (uint32_t i = 0; i < dims - 1; i++) {
      if (i > 0)
        append_checked(", ");
      append_checked("0x%llx", (unsigned long long)fields.globalStrides[i]);
    }
  }
  append_checked("]\n");

  append_checked("  element_strides: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0)
      append_checked(", ");
    append_checked("%u", fields.elementStrides[i]);
  }
  append_checked("]\n");

  // Map numeric mode values to human-readable macro names.
  auto interleave_to_string = [](uint32_t v) -> const char * {
    switch (v) {
    case TMA_INTERLEAVE_NONE:
      return "TMA_INTERLEAVE_NONE";
    case TMA_INTERLEAVE_16B:
      return "TMA_INTERLEAVE_16B";
    case TMA_INTERLEAVE_32B:
      return "TMA_INTERLEAVE_32B";
    default:
      return "TMA_INTERLEAVE_UNKNOWN";
    }
  };
  auto swizzle_to_string = [](uint32_t v) -> const char * {
    switch (v) {
    case TMA_SWIZZLE_NONE:
      return "TMA_SWIZZLE_NONE";
    case TMA_SWIZZLE_32B:
      return "TMA_SWIZZLE_32B";
    case TMA_SWIZZLE_64B:
      return "TMA_SWIZZLE_64B";
    case TMA_SWIZZLE_128B:
      return "TMA_SWIZZLE_128B";
    case TMA_SWIZZLE_96B:
      return "TMA_SWIZZLE_96B";
    default:
      return "TMA_SWIZZLE_UNKNOWN";
    }
  };
  auto oobfill_to_string = [](uint32_t v) -> const char * {
    switch (v) {
    case TMA_OOB_ZERO:
      return "TMA_OOB_ZERO";
    case TMA_OOB_NAN:
      return "TMA_OOB_NAN";
    default:
      return "TMA_OOB_UNKNOWN";
    }
  };

  append_checked("  interleave: %s (%u), swizzle: %s (%u), oobFill: %s (%u)\n",
                 interleave_to_string(fields.interleave), fields.interleave,
                 swizzle_to_string(fields.swizzle), fields.swizzle,
                 oobfill_to_string(fields.oobFill), fields.oobFill);

  append_checked("  tile_size_bytes: %u\n", get_tile_size_bytes());

  printf("%s", buf);
}

tensormap_descriptor_t
tensormap_descriptor_t::read_from_shared(memory_space *shared_mem,
                                         uint32_t addr) {
  tensormap_descriptor_t desc;
  // Read entire 128-byte descriptor as raw bytes
  shared_mem->read(addr, 128, desc.raw_bytes);
  return desc;
}

void tensormap_descriptor_t::write_to_shared(memory_space *shared_mem,
                                             uint32_t addr,
                                             ptx_thread_info *thd,
                                             const ptx_instruction *pI) const {
  // Write entire 128-byte descriptor as raw bytes
  shared_mem->write(addr, 128, raw_bytes, thd, pI);
}

//=============================================================================
// tensormap Instruction Functional Simulation
//=============================================================================

void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  const auto &options = pI->get_options();

  // Check for tensormap instruction types
  bool is_replace = false;
  bool is_cp_fenceproxy = false;
  bool is_tile = false;

  // Field type for replace operations
  bool is_global_address = false;
  bool is_rank = false;
  bool is_box_dim = false;
  bool is_global_dim = false;
  bool is_global_stride = false;
  bool is_element_stride = false;
  bool is_elemtype = false;
  bool is_interleave_layout = false;
  bool is_swizzle_mode = false;
  bool is_swizzle_atomicity = false;
  bool is_fill_mode = false;

  for (auto op : options) {
    switch (op) {
    case REPLACE_OPTION:
      is_replace = true;
      break;
    case CP_FENCEPROXY_OPTION:
      is_cp_fenceproxy = true;
      break;
    case TILE_OPTION:
      is_tile = true;
      break;
    case GLOBAL_ADDRESS_OPTION:
      is_global_address = true;
      break;
    case RANK_OPTION:
      is_rank = true;
      break;
    case BOX_DIM_OPTION:
      is_box_dim = true;
      break;
    case GLOBAL_DIM_OPTION:
      is_global_dim = true;
      break;
    case GLOBAL_STRIDE_OPTION:
      is_global_stride = true;
      break;
    case ELEMENT_STRIDE_OPTION:
      is_element_stride = true;
      break;
    case ELEMTYPE_OPTION:
      is_elemtype = true;
      break;
    case INTERLEAVE_LAYOUT_OPTION:
      is_interleave_layout = true;
      break;
    case SWIZZLE_MODE_OPTION:
      is_swizzle_mode = true;
      break;
    case SWIZZLE_ATOMICITY_OPTION:
      is_swizzle_atomicity = true;
      break;
    case FILL_MODE_OPTION:
      is_fill_mode = true;
      break;
    default:
      break;
    }
  }

  memory_space *shared_mem = thread->m_shared_mem;

  if (is_replace && is_tile) {
    // tensormap.replace.tile.<field> [dst], value
    // or tensormap.replace.tile.<field> [dst], dim_idx, value

    // Get destination address (tensormap address in shared memory)
    uint32_t tensormap_addr = get_operand_u32(thread, pI->dst());

    // Read existing descriptor, modify field, write back
    tensormap_descriptor_t desc =
        tensormap_descriptor_t::read_from_shared(shared_mem, tensormap_addr);

    if (is_global_address) {
      // tensormap.replace.tile.global_address [dst], value
      uint64_t value = get_operand_u64(thread, pI->src1());
      desc.fields.globalAddress = value;
      GPPRINTF_INST_EXEC(
          TMA, "tensormap.replace.tile.global_address [0x%x] = 0x%llx\n",
          tensormap_addr, (unsigned long long)value);

    } else if (is_rank) {
      // tensormap.replace.tile.rank [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.tensorRank = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.rank [0x%x] = %u\n",
                         tensormap_addr, value);

    } else if (is_box_dim) {
      // tensormap.replace.tile.box_dim [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5)
        desc.fields.boxDim[dim_idx] = value;
      GPPRINTF_INST_EXEC(
          TMA, "tensormap.replace.tile.box_dim [0x%x], dim=%u, value=%u\n",
          tensormap_addr, dim_idx, value);

    } else if (is_global_dim) {
      // tensormap.replace.tile.global_dim [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5)
        desc.fields.globalDim[dim_idx] = value;
      GPPRINTF_INST_EXEC(
          TMA, "tensormap.replace.tile.global_dim [0x%x], dim=%u, value=%u\n",
          tensormap_addr, dim_idx, value);

    } else if (is_global_stride) {
      // tensormap.replace.tile.global_stride [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint64_t value = get_operand_u64(thread, pI->src2());
      if (dim_idx < 5)
        desc.fields.globalStrides[dim_idx] = value;
      GPPRINTF_INST_EXEC(
          TMA,
          "tensormap.replace.tile.global_stride [0x%x], dim=%u, value=%llu\n",
          tensormap_addr, dim_idx, (unsigned long long)value);

    } else if (is_element_stride) {
      // tensormap.replace.tile.element_stride [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5)
        desc.fields.elementStrides[dim_idx] = value;
      GPPRINTF_INST_EXEC(
          TMA,
          "tensormap.replace.tile.element_stride [0x%x], dim=%u, value=%u\n",
          tensormap_addr, dim_idx, value);

    } else if (is_elemtype) {
      // tensormap.replace.tile.elemtype [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.tensorDataType = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.elemtype [0x%x] = %u\n",
                         tensormap_addr, value);

    } else if (is_interleave_layout) {
      // tensormap.replace.tile.interleave_layout [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.interleave = value;
      GPPRINTF_INST_EXEC(
          TMA, "tensormap.replace.tile.interleave_layout [0x%x] = %u\n",
          tensormap_addr, value);
      assert(value == 0 &&
             "Only TMA_INTERLEAVE_NONE (0) is currently supported");

    } else if (is_swizzle_mode) {
      // tensormap.replace.tile.swizzle_mode [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.swizzle = value;
      GPPRINTF_INST_EXEC(TMA,
                         "tensormap.replace.tile.swizzle_mode [0x%x] = %u\n",
                         tensormap_addr, value);

    } else if (is_swizzle_atomicity) {
      // tensormap.replace.tile.swizzle_atomicity [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      assert(value == 0 &&
             "Only 16B swizzle atomicity (0x0) is currently supported");
      GPPRINTF_INST_EXEC(
          TMA, "tensormap.replace.tile.swizzle_atomicity [0x%x] = %u (16B)\n",
          tensormap_addr, value);

    } else if (is_fill_mode) {
      // tensormap.replace.tile.fill_mode [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.oobFill = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.fill_mode [0x%x] = %u\n",
                         tensormap_addr, value);

    } else {
      printf("GPGPU-Sim ERROR: unrecognized tensormap.replace.tile field\n");
      pI->print_insn();
      abort();
    }

    // Write modified descriptor back to shared memory
    desc.write_to_shared(shared_mem, tensormap_addr, thread, pI);

  } else if (is_cp_fenceproxy) {
    // tensormap.cp_fenceproxy.global.shared::cta... [dst_global], [src_shared],
    // size Copy tensormap from shared to global memory with fence

    // TODO: Add option check
    uint64_t dst_addr = get_operand_u64(thread, pI->dst());
    uint32_t src_addr = get_operand_u32(thread, pI->src1());
    uint32_t size_in_bytes = get_operand_u32(thread, pI->src2());

    memory_space *global_mem = thread->get_global_memory();
    ;

    tensormap_descriptor_t desc =
        tensormap_descriptor_t::read_from_shared(shared_mem, src_addr);
    global_mem->write(dst_addr, size_in_bytes, desc.raw_bytes, thread, pI);

  } else {
    printf("GPGPU-Sim ERROR: unrecognized tensormap instruction variant\n");
    pI->print_insn();
    abort();
  }
}
