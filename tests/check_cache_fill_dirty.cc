// Native tag-array regression; no GPU launch or simulator-library build.
// From repo root, link an existing simulator library:
// g++ -std=c++17 -O1 -g -fno-access-control -Isrc -I/usr/local/cuda/include \
//   test/check_cache_fill_dirty.cc /ABS/SIM/LIB/libcudart.so \
//   -Wl,-rpath,/ABS/SIM/LIB -o /tmp/check_cache_fill_dirty
// LD_LIBRARY_PATH=/ABS/SIM/LIB /tmp/check_cache_fill_dirty
// To test current fill source against existing library constructors instead:
// sed -n -e '/^void tag_array::fill(new_addr_type addr, unsigned time,$/,/^}/p' \
//   -e '/^void tag_array::fill(unsigned index, unsigned time, mem_fetch \*mf)/,/^}/p' \
//   src/gpgpu-sim/gpu-cache.cc | g++ -std=c++17 -O1 -g \
//   -fno-access-control -Isrc -I/usr/local/cuda/include \
//   -include test/check_cache_fill_dirty.cc -x c++ - -x none \
//   /ABS/SIM/LIB/libcudart.so -Wl,-rpath,/ABS/SIM/LIB \
//   -o /tmp/check_cache_fill_dirty_source
// Private access is only for checking the counter against real block state.
#include "gpgpu-sim/gpu-sim.h"
#include "../libcuda/gpgpu_context.h"

#include <cstdio>

int main() {
  bool pass = true;
  gpgpu_context context;
  memory_config memory(&context);
  // This check needs only the mem_fetch masks, not an address decoder or GPU.
  memory.SST_mode = true;
  memory.icnt_flit_size = 32;
  for (bool indexed : {false, true})
  for (char type : {'N', 'S'}) {
    for (bool retain_modified : {false, true}) {
      char options[80];
      std::snprintf(options, sizeof(options),
                    "%c:1:128:1,L:B:m:W:L,A:4:4,4:4,32", type);
      cache_config config;
      config.init(options, FuncCachePreferNone);
      tag_array tags(config, 0, 0);
      mem_access_sector_mask_t first, second;
      first.set(0);
      second.set(1);
      mem_access_byte_mask_t bytes;
      bytes.set();
      tags.fill(new_addr_type(0), 1, first, bytes, false);
      if (type == 'S') tags.fill(new_addr_type(32), 2, second, bytes, false);

      // Same state/count updates as a write hit; no mocked cache blocks.
      auto *block = tags.get_block(0);
      block->set_status(MODIFIED, first);
      if (type == 'S') block->set_status(MODIFIED, second);
      tags.inc_dirty();
      block->set_modified_on_fill(retain_modified, first);
      auto check = [&](const char *stage) {
        unsigned actual = 0;
        for (unsigned i = 0; i < tags.size(); ++i)
          actual += tags.get_block(i)->is_modified_line();
        const bool ok = tags.m_dirty == actual;
        std::printf("%s %c retain=%u %s counted=%u actual=%u %s\n",
                    indexed ? "index" : "address", type, retain_modified,
                    stage, tags.m_dirty, actual,
                    ok ? "PASS" : "FAIL");
        pass &= ok;
      };
      auto refill = [&](new_addr_type address, unsigned time,
                        mem_access_sector_mask_t mask) {
        if (indexed) {
          mem_access_t access(GLOBAL_ACC_R, address, 32, false,
                              active_mask_t(), bytes, mask, &context);
          mem_fetch fetch(access, nullptr, 0, 8, 0, 0, 0, &memory, time);
          tags.fill(unsigned(0), time, &fetch);
        } else {
          tags.fill(address, time, mask, bytes, false);
        }
      };
      check("before");
      refill(0, 3, first);
      check(type == 'S' ? "other-sector-dirty" : "after-hit-fill");
      if (type == 'S') {
        refill(32, 4, second);
        check("after-last-sector-fill");
      }
    }
  }
  return pass ? 0 : 1;
}
