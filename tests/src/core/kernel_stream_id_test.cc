#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <new>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/ptx_ir.h"
#include "kernel_info.h"

TEST(KernelStreamId, TextureSnapshotConstructorInitializesDefaultStream) {
  function_info entry(1, GPGPU_Context());
  entry.set_name("stream_id_test");
  alignas(kernel_info_t) unsigned char storage[sizeof(kernel_info_t)];
  for (unsigned char fill : {0x00, 0xa5, 0xff}) {
    SCOPED_TRACE(static_cast<unsigned>(fill));
    std::memset(storage, fill, sizeof(storage));
    auto *kernel = new (storage) kernel_info_t(
        dim3(1), dim3(32), &entry,
        std::map<std::string, const cudaArray *>(),
        std::map<std::string, const textureInfo *>());
    EXPECT_EQ(kernel->get_streamID(), 0ull);
    kernel->~kernel_info_t();
  }
}

TEST(KernelStreamId, ExplicitStreamConstructorPreservesFullWidthId) {
  function_info entry(1, GPGPU_Context());
  entry.set_name("explicit_stream_id_test");
  for (auto id : {0ull, 7ull, 0x123456789abcdef0ull}) {
    kernel_info_t kernel(dim3(1), dim3(32), &entry, id);
    EXPECT_EQ(kernel.get_streamID(), id);
  }
}
