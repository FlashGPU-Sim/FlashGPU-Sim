#include "ld_st_matrix.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

// Direction enum for load vs store
enum class MatrixDirection { LOAD, STORE };

// Common implementation for ldmatrix and stmatrix
template <MatrixDirection Direction>
static void handle_ld_st_matrix_inst_impl(const ptx_instruction *pI,
                                          core_t *core, warp_inst_t &inst) {
  constexpr bool is_load = (Direction == MatrixDirection::LOAD);
  const char *inst_name = is_load ? "ldmatrix" : "stmatrix";

  // We need the thread for DPRINTF.
  int tid_lane0 = (core->get_gpu()->is_functional_sim() ? inst.warp_id_func()
                                                        : inst.warp_id()) *
                  core->get_warp_size();
  auto thread = core->get_thread_info()[tid_lane0];

  GPPRINTF_INST_EXEC(WIP, "Handling %s\n", pI->to_string().c_str());

  // Process the options.
  bool is_sync = false;
  bool is_aligned = false;

  constexpr int invalid_matrix_shape = -1;
  int matrix_shape = invalid_matrix_shape;

  constexpr int invalid_num_matrixs = -1;
  constexpr int max_num_matrixs = 4;
  int num_matrixs = invalid_num_matrixs;

  bool is_transpose = false;

  // Matrix layout constants for 8x8 matrix distributed across 32-lane warp
  constexpr int MATRIX_DIMENSION = 8;          // 8x8 matrix size
  constexpr int LANES_PER_MATRIX_ROW = 4;      // 32 lanes / 8 rows = 4 lanes per row
  constexpr int COLUMNS_PER_LANE = 2;          // Each lane handles 2 consecutive columns

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
      printf("%s: unknown option %d\n", inst_name, option);
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
         "Currently only m8n8 shape is supported");
  assert(num_matrixs != invalid_num_matrixs &&
         "Number of matrixs option is required");
  assert(scalar_type == B16_TYPE &&
         "Currently only b16 scalar type is supported");
  assert(is_sync && "ld/stmatrix is always sync");
  assert(is_aligned && "ld/stmatrix is always aligned");
  assert(!is_transpose && "Currently transpose ld/stmatrix is not supported");
  assert(pI->get_num_operands() == 2 && "ld/stmatrix should have 2 operands");

  // For ldmatrix: dst is vector register, src1 is address
  // For stmatrix: dst is address, src1 is vector register
  if constexpr (is_load) {
    assert(pI->dst().get_vect_nelem() == num_matrixs &&
           "Destination operand size mismatch in ldmatrix");
  } else {
    assert(pI->src1().get_vect_nelem() == num_matrixs &&
           "Source operand size mismatch in stmatrix");
  }

  GPPRINTF_INST_EXEC(WIP,
                    "%s options: matrix_shape=%d, num_matrixs=%d, "
                    "scalar_type=%d, num_operands=%d\n",
                    inst_name, matrix_shape, num_matrixs, scalar_type,
                    pI->get_num_operands());

  auto get_u32_value = [](ptx_thread_info *thread, const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  // Get the address operand reference based on direction
  const auto &addr_operand = is_load ? pI->src1() : pI->dst();

  ptx_reg_t result_regs[max_num_matrixs];
  for (auto lane_id = 0; lane_id < core->get_warp_size(); lane_id++) {
    auto thread = core->get_thread_info()[tid_lane0 + lane_id];

    // Map lane ID to matrix position
    auto matrix_row_id = lane_id / LANES_PER_MATRIX_ROW;
    auto matrix_col_id = (lane_id % LANES_PER_MATRIX_ROW) * COLUMNS_PER_LANE;

    // For stmatrix: read data from source registers first
    if constexpr (!is_load) {
      thread->get_vector_operand_values(pI->src1(), &result_regs[0],
                                        num_matrixs);
    }

    for (auto matrix_id = 0; matrix_id < num_matrixs; matrix_id++) {

      auto row_address_lane_id = matrix_id * MATRIX_DIMENSION + matrix_row_id;
      auto row_address_thread =
          core->get_thread_info()[tid_lane0 + row_address_lane_id];
      auto row_address = get_u32_value(row_address_thread, addr_operand);

      auto address = row_address + matrix_col_id * sizeof(uint16_t);

      auto &data = result_regs[matrix_id];

      if constexpr (is_load) {
        // Read 2 b16 values from shared memory.
        thread->m_shared_mem->read(address, sizeof(uint16_t) * 2,
                                   reinterpret_cast<char *>(data.u16_2));
      } else {
        // Write 2 b16 values to shared memory.
        thread->m_shared_mem->write(address, sizeof(uint16_t) * 2,
                                    reinterpret_cast<char *>(data.u16_2),
                                    thread, pI);
      }

      GPPRINTF_INST_EXEC(WIP,
                        "%s: lane_id=%d, matrix_id=%d, row_address_lane_id=%d, "
                        "matrix_row_id=%d, matrix_col_id=%d, row_address=0x%x, "
                        "address=0x%x data=%04x %04x\n",
                        inst_name, lane_id, matrix_id, row_address_lane_id,
                        matrix_row_id, matrix_col_id, row_address, address,
                        result_regs[matrix_id].u16_2[0],
                        result_regs[matrix_id].u16_2[1]);
    }

    // For ldmatrix: write to the destination register after loading all
    // matrices
    if constexpr (is_load) {
      thread->set_vector_operand_values(pI->dst(), result_regs[0],
                                        result_regs[1], result_regs[2],
                                        result_regs[3]);
    }
  }
}

void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  handle_ld_st_matrix_inst_impl<MatrixDirection::LOAD>(pI, core, inst);
}

void handle_stmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  handle_ld_st_matrix_inst_impl<MatrixDirection::STORE>(pI, core, inst);
}
