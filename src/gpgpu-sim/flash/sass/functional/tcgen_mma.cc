#include "tcgen_mma_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
void register_tcgen_mma_semantics(frontend &target) {
  target.register_semantics("UTCHMMA", half_mma);
  target.register_semantics("UTCHMMA.WS", half_mma);
  target.register_semantics("UTCQMMA", mixed_mma);
  target.register_semantics("UTCOMMA", fp4_mma);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
