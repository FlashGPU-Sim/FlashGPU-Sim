#include "../../tma_driver_tensormap.h"
#include "../../tma_load_convert.h"
#include "internal.h"
#include <algorithm>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
step_result read_driver_tensor_map(const instruction &inst,
                                   execution_context &context, uint64_t address,
                                   unsigned rank,
                                   const std::array<uint64_t, 8> &raw,
                                   bool &recognized,
                                   functional_tensor_map_view &view) {
  uint32_t dtype, swizzle;
  const bool host_candidate =
      rank >= 1 && rank <= 5 && uint32_t(raw[1]) == rank - 1;
  recognized = host_candidate ||
               sm100_driver_tma_format(uint32_t(raw[1]), rank, dtype, swizzle);
  if (!recognized)
    return {step_status::kAdvanced, {}};
  tensormap_descriptor_t encoded{}, decoded{};
  std::copy(raw.begin(), raw.end(), encoded.raw_u64);
  if (!context.memory->read(memory_space::kGlobal, address + sizeof(raw),
                            encoded.raw_u64 + raw.size(), sizeof(raw)))
    return memory_fault(inst, memory_space::kGlobal, address + sizeof(raw),
                        sizeof(raw));
  if (host_candidate) {
    if (encoded.is_packed_u4() && encoded.num_dims() == rank &&
        tma_packed4_supported(encoded)) {
      decoded = encoded;
    } else if (!decode_tma_float_host(encoded, rank, decoded)) {
      recognized = false;
      return {step_status::kAdvanced, {}};
    }
    dtype = decoded.fields.tensorDataType;
    swizzle = decoded.fields.swizzle;
  } else if (!decode_sm100_driver_tensormap(encoded, rank, decoded))
    return unsupported(inst, "unqualified SM100 Driver tensor-map fields");
  functional_tensor_map_view result;
  result.rank = rank;
  result.oob_zero = decoded.fields.oobFill == TMA_OOB_ZERO;
  result.global_address = decoded.fields.globalAddress;
  result.data_type = dtype;
  result.element_bytes = decoded.get_element_size();
  result.element_type =
      dtype == TMA_DTYPE_F32   ? tensor_map_element_type::kFloat32
      : dtype == TMA_DTYPE_U32 ? tensor_map_element_type::kUnsigned32
                               : tensor_map_element_type::kUnknown;
  result.swizzle_mode = swizzle;
  result.swizzle_bytes = swizzle ? 16u << swizzle : 0;
  for (unsigned i = 0; i < rank; ++i) {
    result.global_dim[i] = decoded.fields.globalDim[i];
    result.box_dim[i] = decoded.fields.boxDim[i];
    result.element_stride[i] = 1;
    result.global_stride_bytes[i] =
        i ? decoded.fields.globalStrides[i - 1] : result.element_bytes;
  }
  view = result;
  return {step_status::kAdvanced, {}};
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
