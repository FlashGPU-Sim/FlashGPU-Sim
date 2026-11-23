#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "src/kernel_info.h"

// Every exit waits until all other exits have decremented the live count.
// A later snapshot of running() would let every worker claim retirement;
// ownership must come from the decrement that actually reached zero.
TEST(KernelRetirementTest, SimultaneousExitsElectOneOwner) {
  gpgpu_context context;
  function_info function(1, &context);
  for (unsigned count : {2u, 16u, 32u}) {
    kernel_info_t kernel(dim3(count), dim3(32), &function, 0);
    for (unsigned i = 0; i < count; ++i) kernel.inc_running();
    std::atomic<unsigned> owners{0};
    unsigned arrived = 0;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < count; ++i) {
      workers.emplace_back([&] {
        bool owns_retirement = kernel.dec_running();
        {
          std::unique_lock<std::mutex> lock(mutex);
          ++arrived;
          ready.notify_all();
          ready.wait(lock, [&] { return arrived == count; });
        }
        if (owns_retirement) ++owners;
      });
    }
    for (auto &worker : workers) worker.join();
    EXPECT_EQ(owners.load(), 1u) << "CTA exits=" << count;
    EXPECT_FALSE(kernel.running());
    // Reusing the same live-count state elects one new owner, too.
    kernel.inc_running();
    EXPECT_TRUE(kernel.dec_running());
  }
}
