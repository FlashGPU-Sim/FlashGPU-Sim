#ifndef FLASH_GPGPU_SIM_SASS_FRONTEND_H_
#define FLASH_GPGPU_SIM_SASS_FRONTEND_H_

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "decode/decoder.h"

namespace flash_gpgpu_sim {
namespace sass {

constexpr unsigned kWarpLanes = 32;
constexpr unsigned kVectorRegisters = 256;
constexpr unsigned kUniformRegisters = 256;
constexpr unsigned kPredicateRegisters = 8;
constexpr unsigned kBarrierRegisters = 16;
constexpr unsigned kCtaBarrierSlots = 16;
constexpr unsigned kMaxCtaWarps = 32;
constexpr unsigned kTruePredicate = 7;
constexpr unsigned kZeroRegister = 255;
constexpr unsigned kZeroUniformRegister = 255;

struct reconvergence_path {
  uint64_t pc = 0;
  uint32_t active_mask = 0;
};

struct reconvergence_state {
  bool valid = false;
  bool reliable = false;
  uint64_t target_pc = 0;
  uint32_t participating_mask = 0;
  uint32_t arrived_mask = 0;
  std::vector<reconvergence_path> pending_paths;
  std::vector<reconvergence_path> escaped_paths;
};

enum class mbarrier_try_wait_state : uint8_t {
  kNone,
  kFunctionalComplete,
  kFunctionalPending,
};

// Physical architectural state for an execution-driven SASS warp. This state
// is independent of PTX virtual registers and SIMT-stack objects.
struct warp_state {
  uint64_t pc = 0;
  uint32_t active_mask = 0xffffffffu;
  std::array<std::array<uint32_t, kVectorRegisters>, kWarpLanes> registers{};
  std::array<std::bitset<kPredicateRegisters>, kWarpLanes> predicates{};
  std::array<uint32_t, kUniformRegisters> uniform_registers{};
  std::bitset<kPredicateRegisters> uniform_predicates{};
  std::array<reconvergence_state, kBarrierRegisters> reconvergence{};
  std::vector<unsigned> reconvergence_stack;
  std::vector<reconvergence_state> warp_sync_stack;
  std::vector<reconvergence_path> independent_paths;
  bool waiting_at_cta_barrier = false;
  unsigned cta_barrier_id = 0;
  uint64_t cta_barrier_generation = 0;
  // BAR.SYNC.DEFER_BLOCKING records an arrival without stopping the warp at
  // the BAR itself.  A later shared-memory or GMMA consumer is the point at
  // which an unreleased generation becomes blocking.  Keep this state in the
  // architectural SASS frontend rather than teaching the common PTX backend
  // an SM90-specific barrier rule.
  std::array<bool, kCtaBarrierSlots> deferred_cta_barrier_pending{};
  std::array<uint64_t, kCtaBarrierSlots> deferred_cta_barrier_generation{};
  // Functional TMA completes eagerly while its timing transaction remains in
  // flight, so functional and timing barrier phases may legitimately differ.
  // Retain whether TRYWAIT was already complete or still needs its destination
  // predicate written when the timing backend releases the warp.
  mbarrier_try_wait_state mbarrier_try_wait = mbarrier_try_wait_state::kNone;
  uint32_t pending_mbarrier_try_wait_mask = 0;
  unsigned pending_mbarrier_try_wait_predicate = kTruePredicate;

  uint32_t read_register(unsigned lane, unsigned reg) const;
  void write_register(unsigned lane, unsigned reg, uint32_t value);
  uint32_t read_uniform_register(unsigned reg) const;
  void write_uniform_register(unsigned reg, uint32_t value);
  bool read_predicate(unsigned lane, unsigned pred) const;
  void write_predicate(unsigned lane, unsigned pred, bool value);
  bool read_uniform_predicate(unsigned pred) const;
  void write_uniform_predicate(unsigned pred, bool value);
  void record_mbarrier_try_wait(unsigned pred, uint32_t pending_lane_mask);
  mbarrier_try_wait_state complete_mbarrier_try_wait();
};

enum class step_status {
  kAdvanced,
  kBlocked,
  kExited,
  kUnsupported,
  kMissingPc
};

struct step_result {
  step_status status = step_status::kUnsupported;
  std::string detail;
};

enum class memory_space { kConstant, kGlobal, kShared, kLocal };

enum class tensor_map_element_type {
  kUnknown,
  kUnsigned32,
  kFloat16,
  kFloat32,
  kFloat32Ftz,
};

// The functional frontend deliberately sees memory through this narrow
// architectural interface.  A standalone correctness test can provide flat
// buffers while the simulator can later provide an adapter to its own memory
// implementation; neither side has to borrow PTX thread state.
class functional_memory {
public:
  virtual ~functional_memory() = default;
  virtual bool read(memory_space space, uint64_t address, void *data,
                    size_t bytes) const = 0;
  // Static nonzero cubin constant banks are distinct from the launch ABI in
  // bank zero. Existing functional-memory implementations remain bank-zero
  // only unless they explicitly provide a static-bank view.
  virtual bool read_constant(uint32_t bank, uint64_t address, void *data,
                             size_t bytes) const {
    return bank == 0 && read(memory_space::kConstant, address, data, bytes);
  }
  virtual bool write(memory_space space, uint64_t address, const void *data,
                     size_t bytes) = 0;
};

// Frontend-neutral functional metadata for a tensor map with up to two
// dimensions. A CUDA/runtime adapter can register this directly; validated raw
// tensor-map decoders produce the same form for execution from cubin state.
// Dimension zero is contiguous, and dimension one is set to one for rank-1
// maps.
struct functional_tensor_map_2d {
  uint64_t global_address = 0;
  uint32_t element_bytes = 0;
  // Equal-width formats can still require different reduction arithmetic.
  tensor_map_element_type element_type = tensor_map_element_type::kUnknown;
  std::array<uint32_t, 2> global_dim{};
  std::array<uint32_t, 2> box_dim{};
  uint64_t row_stride_bytes = 0;
  std::array<uint32_t, 2> element_stride{{1, 1}};
  uint32_t swizzle_bytes = 0;
  bool oob_zero = true;
};

// Instruction effects needed by a timing backend after architectural SASS
// execution.  These are kept in ISA-neutral scalar form: the frontend does
// not depend on the simulator's warp_inst_t or tensor-map byte layout.
struct functional_tensor_map_view {
  unsigned rank = 0;
  uint64_t global_address = 0;
  uint32_t element_bytes = 0;
  tensor_map_element_type element_type = tensor_map_element_type::kUnknown;
  std::array<uint32_t, 5> global_dim{{1, 1, 1, 1, 1}};
  std::array<uint32_t, 5> box_dim{{1, 1, 1, 1, 1}};
  std::array<uint64_t, 5> global_stride_bytes{};
  std::array<uint32_t, 5> element_stride{{1, 1, 1, 1, 1}};
  uint32_t swizzle_bytes = 0;
  bool oob_zero = true;
};

struct functional_tma_effect {
  bool valid = false;
  uint64_t destination_address = 0;
  uint64_t source_address = 0;
  uint32_t size_bytes = 0;
  uint32_t mbarrier_address = UINT32_MAX;
  std::array<int32_t, 5> coordinates{};
  functional_tensor_map_view tensor_map;
};

struct functional_mbarrier_effect {
  bool valid = false;
  uint32_t address = UINT32_MAX;
  uint32_t count = UINT32_MAX;
  bool parity = false;
};

struct functional_named_barrier_effect {
  bool valid = false;
  uint32_t id = UINT32_MAX;
  uint32_t participant_count = UINT32_MAX;
};

// Functional state for CUDA named CTA barriers. Each slot is reusable: the
// generation advances only after every configured warp has arrived. Timing
// policy (deferred blocking, issue order, and latency) deliberately does not
// live here.
class cta_execution_state {
public:
  explicit cta_execution_state(unsigned warp_count);

  unsigned warp_count() const { return warp_count_; }
  uint64_t generation(unsigned barrier_id) const;
  bool arrive(unsigned barrier_id, unsigned warp_id,
              unsigned expected_warp_count = 0);
  void retire_warp(unsigned warp_id);
  bool released(unsigned barrier_id, uint64_t generation) const;
  bool initialize_mbarrier(uint64_t address, uint32_t expected_arrivals);
  bool mbarrier_initialized(uint64_t address) const;
  bool arrive_mbarrier(uint64_t address, uint32_t arrivals,
                       uint64_t transaction_bytes);
  bool complete_mbarrier_transaction(uint64_t address,
                                     uint64_t transaction_bytes);
  bool invalidate_mbarrier(uint64_t address);
  uint32_t mbarrier_pending_arrivals(uint64_t address) const;
  uint64_t mbarrier_pending_transaction_bytes(uint64_t address) const;
  uint64_t mbarrier_phase(uint64_t address) const;
  bool register_tensor_map_2d(uint64_t descriptor_address,
                              functional_tensor_map_2d descriptor);
  const functional_tensor_map_2d *
  find_tensor_map_2d(uint64_t descriptor_address) const;

private:
  struct barrier_state {
    unsigned arrivals = 0;
    unsigned expected_arrivals = 0;
    uint64_t generation = 0;
    // A count-less CTA barrier snapshots the live CTA size on its first
    // arrival.  Track which warps already contributed so a later EXIT only
    // removes a warp that had not yet arrived from that snapshot.
    bool tracks_live_participants = false;
    uint32_t arrived_warps = 0;
  };
  struct mbarrier_state {
    uint32_t expected_arrivals = 0;
    uint32_t pending_arrivals = 0;
    uint64_t pending_transaction_bytes = 0;
    uint64_t phase = 0;
  };

  unsigned warp_count_ = 0;
  uint32_t expected_warps_ = 0;
  std::array<barrier_state, kCtaBarrierSlots> barriers_{};
  std::unordered_map<uint64_t, mbarrier_state> mbarriers_;
  std::unordered_map<uint64_t, functional_tensor_map_2d> tensor_maps_2d_;
};

// Returns whether the instruction belongs to one of the consumer classes
// experimentally observed to trigger BAR.SYNC.DEFER_BLOCKING on SM90.
bool triggers_deferred_cta_barrier(const instruction &inst);

// Returns whether the instruction must remain ordered behind an unresolved
// deferred CTA barrier for the decoded architecture. Named-barrier operations
// are ordering points on both supported architectures. Direct SM120 hardware
// probes additionally establish mbarrier phase checks and UTMA operations as
// ordering points; equivalent SM90 probes have not yet been collected.
bool orders_deferred_cta_barrier(const instruction &inst, architecture arch);

// Clears generations that have completed and reports whether every deferred
// CTA barrier recorded by this warp is now released.
bool deferred_cta_barriers_released(warp_state &warp,
                                    const cta_execution_state &cta);

struct execution_context {
  struct functional_memory_access {
    memory_space space = memory_space::kGlobal;
    uint64_t address = 0;
    size_t bytes = 0;
    bool write = false;
  };

  execution_context() = default;
  explicit execution_context(functional_memory *backing_memory)
      : memory(backing_memory) {}

  functional_memory *memory = nullptr;
  const kernel *kernel_image = nullptr;
  std::array<uint32_t, kWarpLanes> thread_idx_x{};
  std::array<uint32_t, kWarpLanes> thread_idx_y{};
  std::array<uint32_t, kWarpLanes> thread_idx_z{};
  std::array<uint32_t, kWarpLanes> thread_linear_id{};
  uint32_t cta_id_x = 0;
  uint32_t cta_id_y = 0;
  uint32_t cta_id_z = 0;
  uint32_t cga_cta_id = 0;
  uint64_t local_memory_base = 0;
  uint64_t local_memory_thread_stride = 0;
  uint64_t instructions_executed = 0;
  uint64_t architectural_cycle = 0;
  uint64_t architectural_core_frequency_hz = 0;
  bool architectural_cycle_valid = false;
  uint32_t virtual_smid = 0;
  cta_execution_state *cta = nullptr;
  unsigned warp_id = 0;
  std::array<std::vector<functional_memory_access>, kWarpLanes> memory_accesses;
  std::array<functional_tma_effect, kWarpLanes> tma_effects{};
  std::array<functional_mbarrier_effect, kWarpLanes> mbarrier_effects{};
  functional_named_barrier_effect named_barrier_effect;

  void clear_instruction_effects();
  bool decode_local_memory_offset(unsigned lane, uint64_t address,
                                  uint64_t &offset) const;
  void record_memory_access(unsigned lane, memory_space space, uint64_t address,
                            size_t bytes, bool write);
  void record_tma_effect(unsigned lane, functional_tma_effect effect);
  void record_mbarrier_effect(unsigned lane, functional_mbarrier_effect effect);
  void record_named_barrier_effect(functional_named_barrier_effect effect);
};

using semantic_handler = std::function<step_result(
    const instruction &, warp_state &, execution_context &)>;

// Owns static decoded kernels and drives dynamic execution by fetching at the
// warp's current PC.  There is intentionally no dynamic instruction stream.
class frontend {
public:
  void load(const decoder &source, const std::string &image_path,
            const std::string &kernel_name);
  void add_kernel(kernel decoded_kernel);

  const kernel *find_kernel(const std::string &kernel_name) const;
  const instruction *fetch(const std::string &kernel_name, uint64_t pc) const;

  void register_semantics(const std::string &opcode, semantic_handler handler);
  step_result step(const std::string &kernel_name, warp_state &state,
                   execution_context &context) const;

private:
  std::unordered_map<std::string, kernel> kernels_;
  std::unordered_map<std::string, semantic_handler> semantics_;
};

// Round-robin functional CTA driver. It owns independent physical warp state,
// shares only architectural CTA state/memory, and turns a warp-local barrier
// block into continued progress by another warp. This is not a timing model.
class cta_executor {
public:
  cta_executor(const frontend &source, std::string kernel_name,
               unsigned warp_count, functional_memory *memory = nullptr);

  step_result step();
  warp_state &warp(unsigned warp_id);
  const warp_state &warp(unsigned warp_id) const;
  execution_context &context(unsigned warp_id);
  const execution_context &context(unsigned warp_id) const;

  unsigned warp_count() const { return warps_.size(); }
  unsigned last_warp_id() const { return last_warp_id_; }
  uint64_t instructions_executed() const { return instructions_executed_; }
  cta_execution_state &cta_state() { return cta_; }
  const cta_execution_state &cta_state() const { return cta_; }

private:
  const frontend &source_;
  std::string kernel_name_;
  cta_execution_state cta_;
  std::vector<warp_state> warps_;
  std::vector<execution_context> contexts_;
  std::vector<bool> exited_;
  unsigned next_warp_id_ = 0;
  unsigned last_warp_id_ = 0;
  uint64_t instructions_executed_ = 0;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_FRONTEND_H_
