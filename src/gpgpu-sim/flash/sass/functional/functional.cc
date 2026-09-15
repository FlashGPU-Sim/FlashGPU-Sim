#include "functional.h"

#include "internal.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace {

void register_common_functional_semantics(frontend &target) {
  functional_detail::register_data_movement_semantics(target);
  functional_detail::register_integer_semantics(target);
  functional_detail::register_float_semantics(target);
  functional_detail::register_control_semantics(target);
  functional_detail::register_memory_semantics(target);
  functional_detail::register_mbarrier_semantics(target);
}

} // namespace

void register_sm90_functional_semantics(frontend &target) {
  register_common_functional_semantics(target);
  functional_detail::register_tma_semantics(target);
  functional_detail::register_mma_semantics(target);
  functional_detail::register_wgmma_semantics(target);
}

void register_sm120_functional_semantics(frontend &target) {
  register_common_functional_semantics(target);
  functional_detail::register_tma_semantics(target);
  functional_detail::register_mma_semantics(target);
}

} // namespace sass
} // namespace flash_gpgpu_sim
