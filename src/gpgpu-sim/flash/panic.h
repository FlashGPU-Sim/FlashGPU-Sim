#ifndef FLASH_GPGPU_SIM_PANIC_H
#define FLASH_GPGPU_SIM_PANIC_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace flash_gpgpu_sim {

// Diagnostic context only: no recovery or exception unwinding. References
// remain live for the scope; normal instruction execution allocates nothing.
class panic_context {
public:
  panic_context(const std::string &label, const uint64_t &position,
                bool hexadecimal = false)
      : label_(label), position_(position), hexadecimal_(hexadecimal),
        previous_(current()) {
    current() = this;
  }
  ~panic_context() { current() = previous_; }
  panic_context(const panic_context &) = delete;
  panic_context &operator=(const panic_context &) = delete;

  static const panic_context *&current() {
    static thread_local const panic_context *context = nullptr;
    return context;
  }
  void print() const {
    if (previous_)
      previous_->print();
    std::fprintf(stderr, hexadecimal_ ? "%s at pc 0x%llx: " : "%s:%llu: ",
                 label_.c_str(), static_cast<unsigned long long>(position_));
  }

private:
  const std::string &label_;
  const uint64_t &position_;
  bool hexadecimal_;
  const panic_context *previous_;
};

[[noreturn]] inline void panic(const std::string &message) {
  std::fprintf(stderr, "FlashGPU-Sim panic: ");
  if (panic_context::current())
    panic_context::current()->print();
  std::fprintf(stderr, "%s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
}

} // namespace flash_gpgpu_sim

#endif
