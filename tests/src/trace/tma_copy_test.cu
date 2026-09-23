// Triton-style 1D TMA copy trace test
//
// Loads a Triton-generated PTX kernel that copies a 1D fp16 tensor through
// shared memory using cp.async.bulk.tensor.1d.  The kernel is adapted to
// receive pointers to 128-byte tensormap descriptors in global memory rather
// than the descriptors by value in parameter space, because GPGPU-Sim reads
// the tensormap with global_mem->read().

#include "trace_test_utils.h"

#define TMA_DTYPE_F16 6u

// Host-side mirror of the 128-byte tensormap descriptor layout used by
// src/gpgpu-sim/flash/tensormap.h.  Packed so the offsets match exactly.
typedef struct __attribute__((packed, aligned(128))) {
    uint64_t globalAddress;
    uint32_t tensorRank;
    uint32_t boxDim[5];
    uint32_t globalDim[5];
    uint64_t globalStrides[5];
    uint32_t elementStrides[5];
    uint32_t tensorDataType;
    uint32_t interleave;
    uint32_t swizzle;
    uint32_t oobFill;
} tensormap_host_t;

static void init_tensormap(tensormap_host_t* tm, void* global_buf,
                           uint32_t tile_dim, uint32_t total_dim) {
    memset(tm, 0, sizeof(*tm));
    tm->globalAddress = (uint64_t)global_buf;
    tm->tensorRank = 0;          // 1D tensor (rank = num_dims - 1)
    tm->boxDim[0] = tile_dim;
    tm->globalDim[0] = total_dim;
    tm->elementStrides[0] = 1;
    tm->tensorDataType = TMA_DTYPE_F16;
    tm->interleave = 0;
    tm->swizzle = 0;
    tm->oobFill = 0;
}

int main() {
    printf("=== Triton TMA 1D Copy Trace Test ===\n");

    const uint32_t TILE = 64;
    const uint32_t N = 256;
    const size_t data_size = N * sizeof(uint16_t);

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char fatbin_path[2048];
    build_path(fatbin_path, sizeof(fatbin_path), exe_dir,
               "tma_copy_kernel.fatbin");

    CUmodule module = load_fatbin(fatbin_path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "tma_copy_kernel");
    if (!func) return 1;
    set_shared_mem(func, 512);

    // Input/output fp16 buffers
    void* d_in = alloc_gpu(data_size);
    void* d_out = alloc_gpu(data_size);

    uint16_t* h_in = (uint16_t*)malloc(data_size);
    for (uint32_t i = 0; i < N; i++) {
        h_in[i] = fp32_to_fp16((float)i * 0.5f);
    }
    cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);

    // Allocate 128-byte aligned global tensormap descriptors
    void *d_src_tmap_raw, *d_dst_tmap_raw;
    cudaMalloc(&d_src_tmap_raw, 256);
    cudaMalloc(&d_dst_tmap_raw, 256);
    cudaMemset(d_src_tmap_raw, 0, 256);
    cudaMemset(d_dst_tmap_raw, 0, 256);

    tensormap_host_t* h_src_tmap = (tensormap_host_t*)malloc(sizeof(tensormap_host_t));
    tensormap_host_t* h_dst_tmap = (tensormap_host_t*)malloc(sizeof(tensormap_host_t));
    init_tensormap(h_src_tmap, d_in, TILE, N);
    init_tensormap(h_dst_tmap, d_out, TILE, N);

    // Align descriptors to 128 bytes within the 256-byte allocations
    void* d_src_tmap = (void*)(((uintptr_t)d_src_tmap_raw + 127) & ~127);
    void* d_dst_tmap = (void*)(((uintptr_t)d_dst_tmap_raw + 127) & ~127);
    cudaMemcpy(d_src_tmap, h_src_tmap, sizeof(tensormap_host_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_tmap, h_dst_tmap, sizeof(tensormap_host_t),
               cudaMemcpyHostToDevice);

    uint32_t arg_N = N;
    void* args[] = { &d_src_tmap, &d_dst_tmap, &arg_N };
    unsigned grid = N / TILE;

    printf("Launching tma_copy_kernel: grid=(%u,1,1) block=(128,1,1) smem=512\n",
           grid);
    if (launch_kernel(func, grid, 1, 1, 128, 1, 1, 512, args)) return 1;
    printf("Kernel completed\n");

    printf("Validating output equals input...\n");
    int rc = validate_fp16_from_host(d_out, h_in, N);

    free(h_in);
    free(h_src_tmap);
    free(h_dst_tmap);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_src_tmap_raw);
    cudaFree(d_dst_tmap_raw);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
