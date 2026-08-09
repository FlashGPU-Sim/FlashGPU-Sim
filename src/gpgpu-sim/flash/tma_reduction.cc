#include "tma_reduction.h"

#include "tensormap.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "../../cuda-sim/half.h"
#include "../../cuda-sim/half.hpp"

namespace {

template <typename T> T load_element(const void *buffer, size_t index) {
  T value;
  memcpy(&value, static_cast<const uint8_t *>(buffer) + index * sizeof(T),
         sizeof(T));
  return value;
}

template <typename T>
void store_element(void *buffer, size_t index, const T &value) {
  memcpy(static_cast<uint8_t *>(buffer) + index * sizeof(T), &value, sizeof(T));
}

float f16_to_f32(uint16_t value) {
  return half_float::detail::half2float<float>(value);
}

uint16_t f32_to_f16(float value) {
  return half_float::detail::float2half<std::round_to_nearest>(value);
}

float bf16_to_f32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t f32_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) |
                                 ((bits & 0xffffu) != 0u ? 1u : 0u));
  }
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

template <typename T>
void reduce_integer(tma_reduction_op_t op, void *dst, const void *src,
                    size_t count) {
  using unsigned_t = typename std::make_unsigned<T>::type;
  for (size_t i = 0; i < count; ++i) {
    const T dst_value = load_element<T>(dst, i);
    const T src_value = load_element<T>(src, i);
    T result{};
    switch (op) {
    case tma_reduction_op_t::ADD: {
      const unsigned_t sum = static_cast<unsigned_t>(dst_value) +
                             static_cast<unsigned_t>(src_value);
      memcpy(&result, &sum, sizeof(result));
      break;
    }
    case tma_reduction_op_t::MIN:
      result = std::min(dst_value, src_value);
      break;
    case tma_reduction_op_t::MAX:
      result = std::max(dst_value, src_value);
      break;
    case tma_reduction_op_t::INC:
      result = dst_value >= src_value ? 0 : dst_value + 1;
      break;
    case tma_reduction_op_t::DEC:
      result =
          (dst_value == 0 || dst_value > src_value) ? src_value : dst_value - 1;
      break;
    case tma_reduction_op_t::BIT_AND:
      result = static_cast<T>(static_cast<unsigned_t>(dst_value) &
                              static_cast<unsigned_t>(src_value));
      break;
    case tma_reduction_op_t::BIT_OR:
      result = static_cast<T>(static_cast<unsigned_t>(dst_value) |
                              static_cast<unsigned_t>(src_value));
      break;
    case tma_reduction_op_t::BIT_XOR:
      result = static_cast<T>(static_cast<unsigned_t>(dst_value) ^
                              static_cast<unsigned_t>(src_value));
      break;
    default:
      throw std::invalid_argument("invalid integer TMA reduction operation");
    }
    store_element(dst, i, result);
  }
}

float flush_subnormal(float value) {
  return std::fpclassify(value) == FP_SUBNORMAL ? std::copysign(0.0f, value)
                                                : value;
}

void reduce_f32_add(bool ftz, void *dst, const void *src, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const float raw_dst_value = load_element<float>(dst, i);
    const float raw_src_value = load_element<float>(src, i);
    const float dst_value =
        ftz ? flush_subnormal(raw_dst_value) : raw_dst_value;
    const float src_value =
        ftz ? flush_subnormal(raw_src_value) : raw_src_value;
    const float sum = dst_value + src_value;
    const float result = ftz ? flush_subnormal(sum) : sum;
    store_element(dst, i, result);
  }
}

void reduce_f16_like(tma_reduction_op_t op, bool bf16, void *dst,
                     const void *src, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const uint16_t dst_bits = load_element<uint16_t>(dst, i);
    const uint16_t src_bits = load_element<uint16_t>(src, i);
    const float dst_value = bf16 ? bf16_to_f32(dst_bits) : f16_to_f32(dst_bits);
    const float src_value = bf16 ? bf16_to_f32(src_bits) : f16_to_f32(src_bits);
    float result;
    switch (op) {
    case tma_reduction_op_t::ADD:
      result = dst_value + src_value;
      break;
    case tma_reduction_op_t::MIN:
      result = std::fmin(dst_value, src_value);
      break;
    case tma_reduction_op_t::MAX:
      result = std::fmax(dst_value, src_value);
      break;
    default:
      throw std::invalid_argument("invalid 16-bit TMA reduction operation");
    }
    const uint16_t result_bits =
        bf16 ? f32_to_bf16(result) : f32_to_f16(result);
    store_element(dst, i, result_bits);
  }
}

bool is_bitwise(tma_reduction_op_t op) {
  return op == tma_reduction_op_t::BIT_AND ||
         op == tma_reduction_op_t::BIT_OR || op == tma_reduction_op_t::BIT_XOR;
}

} // namespace

const char *tma_reduction_op_name(tma_reduction_op_t op) {
  switch (op) {
  case tma_reduction_op_t::NONE:
    return "none";
  case tma_reduction_op_t::ADD:
    return "add";
  case tma_reduction_op_t::MIN:
    return "min";
  case tma_reduction_op_t::MAX:
    return "max";
  case tma_reduction_op_t::INC:
    return "inc";
  case tma_reduction_op_t::DEC:
    return "dec";
  case tma_reduction_op_t::BIT_AND:
    return "and";
  case tma_reduction_op_t::BIT_OR:
    return "or";
  case tma_reduction_op_t::BIT_XOR:
    return "xor";
  }
  return "unknown";
}

bool tma_tensor_reduction_supported(tma_reduction_op_t op,
                                    uint32_t tensor_data_type) {
  switch (tensor_data_type) {
  case TMA_DTYPE_U32:
    return op == tma_reduction_op_t::ADD || op == tma_reduction_op_t::MIN ||
           op == tma_reduction_op_t::MAX || op == tma_reduction_op_t::INC ||
           op == tma_reduction_op_t::DEC || is_bitwise(op);
  case TMA_DTYPE_S32:
    return op == tma_reduction_op_t::ADD || op == tma_reduction_op_t::MIN ||
           op == tma_reduction_op_t::MAX || is_bitwise(op);
  case TMA_DTYPE_U64:
    return op == tma_reduction_op_t::ADD || op == tma_reduction_op_t::MIN ||
           op == tma_reduction_op_t::MAX || is_bitwise(op);
  case TMA_DTYPE_S64:
    return op == tma_reduction_op_t::MIN || op == tma_reduction_op_t::MAX ||
           is_bitwise(op);
  case TMA_DTYPE_F16:
  case TMA_DTYPE_BF16:
    return op == tma_reduction_op_t::ADD || op == tma_reduction_op_t::MIN ||
           op == tma_reduction_op_t::MAX;
  case TMA_DTYPE_F32:
  case TMA_DTYPE_F32_FTZ:
    return op == tma_reduction_op_t::ADD;
  default:
    return false;
  }
}

void apply_tma_tensor_reduction(tma_reduction_op_t op,
                                uint32_t tensor_data_type, void *dst,
                                const void *src, size_t size_in_bytes) {
  if (!tma_tensor_reduction_supported(op, tensor_data_type))
    throw std::invalid_argument("unsupported TMA tensor reduction type");

  const size_t element_size =
      tensor_data_type == TMA_DTYPE_F16 || tensor_data_type == TMA_DTYPE_BF16
          ? sizeof(uint16_t)
          : (tensor_data_type == TMA_DTYPE_U64 ||
                     tensor_data_type == TMA_DTYPE_S64
                 ? sizeof(uint64_t)
                 : sizeof(uint32_t));
  if (size_in_bytes % element_size != 0)
    throw std::invalid_argument("partial element in TMA tensor reduction");

  switch (tensor_data_type) {
  case TMA_DTYPE_U32:
    reduce_integer<uint32_t>(op, dst, src, size_in_bytes / sizeof(uint32_t));
    return;
  case TMA_DTYPE_S32:
    reduce_integer<int32_t>(op, dst, src, size_in_bytes / sizeof(int32_t));
    return;
  case TMA_DTYPE_U64:
    reduce_integer<uint64_t>(op, dst, src, size_in_bytes / sizeof(uint64_t));
    return;
  case TMA_DTYPE_S64:
    reduce_integer<int64_t>(op, dst, src, size_in_bytes / sizeof(int64_t));
    return;
  case TMA_DTYPE_F16:
    reduce_f16_like(op, false, dst, src, size_in_bytes / sizeof(uint16_t));
    return;
  case TMA_DTYPE_BF16:
    reduce_f16_like(op, true, dst, src, size_in_bytes / sizeof(uint16_t));
    return;
  case TMA_DTYPE_F32:
  case TMA_DTYPE_F32_FTZ:
    reduce_f32_add(tensor_data_type == TMA_DTYPE_F32_FTZ, dst, src,
                   size_in_bytes / sizeof(float));
    return;
  default:
    throw std::invalid_argument("unsupported TMA tensor reduction type");
  }
}
