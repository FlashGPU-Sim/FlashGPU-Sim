#include "ldmatrix.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {

  // We need the thread for DPRINTF.
  int tid_lane0 = (core->get_gpu()->is_functional_sim() ? inst.warp_id_func()
                                                        : inst.warp_id()) *
                  core->get_warp_size();
  auto thread = core->get_thread_info()[tid_lane0];

  DPRINTF_INST_EXEC(WIP, "Handling %s\n", pI->to_string().c_str());
  // Process the options.
  bool is_sync = false;
  bool is_aligned = false;

  constexpr int invalid_matrix_shape = -1;
  int matrix_shape = invalid_matrix_shape;

  constexpr int invalid_num_matrixs = -1;
  constexpr int max_num_matrixs = 4;
  int num_matrixs = invalid_num_matrixs;

  bool is_transpose = false;

  constexpr int invalid_scalar_type = -1;
  int scalar_type = invalid_scalar_type;

  for (auto option : pI->get_options()) {
    switch (option) {
    case SYNC_OPTION:
      is_sync = true;
      break;
    case ALIGNED_OPTION:
      is_aligned = true;
      break;
    case M8N8_OPTION:
      matrix_shape = M8N8_OPTION;
      break;
    case X1_OPTION:
      num_matrixs = 1;
      break;
    case X2_OPTION:
      num_matrixs = 2;
      break;
    case X4_OPTION:
      num_matrixs = 4;
      break;
    default:
      printf("ldmatrix: unknown option %d\n", option);
      abort();
    }
  }

  for (auto scalar_type_option : pI->get_scalar_type()) {
    scalar_type = scalar_type_option;
  }

  /**
   * Sanity check the options. Be strict to avoid future bug.
   */
  assert(matrix_shape == M8N8_OPTION &&
         "Currently only m8n8 shape is supported in ldmatrix");
  assert(num_matrixs != invalid_num_matrixs &&
         "Number of matrixs option is required in ldmatrix");
  assert(scalar_type == B16_TYPE &&
         "Currently only b16 scalar type is supported in ldmatrix");
  assert(is_sync && "ldmatrix is always sync");
  assert(is_aligned && "ldmatrix is always aligned");
  assert(!is_transpose && "Currently transpose ldmatrix is not supported");
  assert(pI->get_num_operands() == 2 && "ldmatrix should have 2 operands");
  assert(pI->dst().get_vect_nelem() == num_matrixs &&
         "Destination operand size mismatch in ldmatrix");

  DPRINTF_INST_EXEC(WIP,
                    "ldmatrix options: matrix_shape=%d, num_matrixs=%d, "
                    "scalar_type=%d, num_operands=%d\n",
                    matrix_shape, num_matrixs, scalar_type,
                    pI->get_num_operands());

  auto get_u32_value = [](ptx_thread_info *thread, const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  ptx_reg_t result_regs[max_num_matrixs];
  for (auto lane_id = 0; lane_id < core->get_warp_size(); lane_id++) {
    auto thread = core->get_thread_info()[tid_lane0 + lane_id];

    // Every 4 lanes handle one row.
    auto matrix_row_id = lane_id / 4;
    // Each lane handles 2 columns of b16.
    auto matrix_col_id = (lane_id % 4) * 2;

    for (auto matrix_id = 0; matrix_id < num_matrixs; matrix_id++) {

      auto row_address_lane_id = matrix_id * 8 + matrix_row_id;
      auto row_address_thread =
          core->get_thread_info()[tid_lane0 + row_address_lane_id];
      auto row_address = get_u32_value(row_address_thread, pI->src1());

      auto address = row_address + matrix_col_id * sizeof(uint16_t);

      // Read 2 b16 values from shared memory.
      auto &data = result_regs[matrix_id];
      thread->m_shared_mem->read(address, sizeof(uint16_t) * 2,
                                 reinterpret_cast<char *>(data.u16_2));

      DPRINTF_INST_EXEC(
          WIP,
          "ldmatrix: lane_id=%d, matrix_id=%d, row_address_lane_id=%d, "
          "matrix_row_id=%d, matrix_col_id=%d, row_address=0x%x, "
          "address=0x%x data=%04x %04x\n",
          lane_id, matrix_id, row_address_lane_id, matrix_row_id, matrix_col_id,
          row_address, address, result_regs[matrix_id].u16_2[0],
          result_regs[matrix_id].u16_2[1]);
    }

    // Write to the destination register.
    thread->set_vector_operand_values(pI->dst(), result_regs[0], result_regs[1],
                                      result_regs[2], result_regs[3]);
  }
}