// Shared utilities for GPT-2 trace tests
// Each test loads a PTX kernel (via fatbin), allocates GPU memory,
// loads/generates data, launches the kernel, and validates output.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

// FP16 conversion helpers
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float val = mantissa / 1024.0f / 16384.0f;
        return sign ? -val : val;
    } else if (exponent == 31) {
        return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, (float)((int)exponent - 15));
    return sign ? -val : val;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent > 15) return (sign << 15) | 0x7C00;  // inf
    if (exponent < -14) return (sign << 15);           // zero/denorm
    uint16_t h = (sign << 15) | ((exponent + 15) << 10) | (mantissa >> 13);
    return h;
}

// Initialize CUDA driver API context
static int init_cuda(CUcontext* ctx) {
    if (cuInit(0) != CUDA_SUCCESS) return 1;
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;
    return cuCtxCreate(ctx, 0, dev) == CUDA_SUCCESS ? 0 : 1;
}

// Get directory of the executable
static void get_exe_dir(char* buf, size_t bufsize) {
    ssize_t len = readlink("/proc/self/exe", buf, bufsize - 1);
    if (len == -1) { buf[0] = '.'; buf[1] = '\0'; return; }
    buf[len] = '\0';
    char* slash = strrchr(buf, '/');
    if (slash) *slash = '\0';
}

// Load fatbin module from file
static CUmodule load_fatbin(const char* fatbin_path) {
    CUmodule module;
    CUresult r = cuModuleLoad(&module, fatbin_path);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Failed to load %s: %s\n", fatbin_path, err);
        return NULL;
    }
    return module;
}

// Get kernel function from module
static CUfunction get_kernel(CUmodule module, const char* name) {
    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, module, name);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Failed to get function '%s': %s\n", name, err);
        return NULL;
    }
    return func;
}

// Set max dynamic shared memory for a kernel.
static int set_shared_mem(CUfunction func, int bytes) {
    CUresult r = cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to set shared memory size: %d\n", int(r));
        return 1;
    }
    return 0;
}

static int read_exact_file(const char* path, void* data, size_t size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", path);
        return 1;
    }
    size_t read = fread(data, 1, size, fp);
    bool failed = read != size || ferror(fp);
    fclose(fp);
    if (failed) {
        fprintf(stderr, "Short or failed read from %s: got %zu, expected %zu\n",
                path, read, size);
        return 1;
    }
    return 0;
}

static bool fp16_matches_reference(uint16_t actual, uint16_t expected) {
    float act_f = fp16_to_fp32(actual), exp_f = fp16_to_fp32(expected);
    if (!isfinite(act_f) || !isfinite(exp_f)) return false;
    float diff = fabsf(exp_f - act_f);
    float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
    return diff <= 1e-2f || diff <= 1e-3f * max_abs;
}

// Allocate GPU memory and zero it; failure propagates to the caller.
static void* alloc_gpu(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error == cudaSuccess) error = cudaMemset(ptr, 0, size);
    if (error != cudaSuccess) {
        fprintf(stderr, "GPU allocation/initialization failed: %s\n",
                cudaGetErrorString(error));
        if (ptr) cudaFree(ptr);
        return nullptr;
    }
    return ptr;
}

static void* load_bin_to_gpu(const char* path, size_t size) {
    void* h_ptr = malloc(size);
    if (!h_ptr) return nullptr;
    if (read_exact_file(path, h_ptr, size)) {
        free(h_ptr);
        return nullptr;
    }
    void* d_ptr = alloc_gpu(size);
    if (!d_ptr) {
        free(h_ptr);
        return nullptr;
    }
    cudaError_t error = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    free(h_ptr);
    if (error != cudaSuccess) {
        fprintf(stderr, "Input upload failed: %s\n", cudaGetErrorString(error));
        cudaFree(d_ptr);
        return nullptr;
    }
    return d_ptr;
}

// Launch kernel and synchronize
static int launch_kernel(CUfunction func,
                         unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned smem, void** args) {
    CUresult r = cuLaunchKernel(func, gx, gy, gz, bx, by, bz, smem, 0, args, 0);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Kernel launch failed: %s\n", err);
        return 1;
    }
    r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "Kernel synchronization failed: %d\n", int(r));
        return 1;
    }
    return 0;
}

// Compare complete finite FP16 output against a host reference.
static int validate_fp16_from_host(void* d_actual, const uint16_t* h_expected,
                                    size_t num_elements) {
    size_t size = num_elements * sizeof(uint16_t);
    uint16_t* h_actual = (uint16_t*)malloc(size);
    if (!h_actual) return 1;
    cudaError_t error = cudaMemcpy(h_actual, d_actual, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Output download failed: %s\n", cudaGetErrorString(error));
        free(h_actual);
        return 1;
    }
    int mismatches = 0;
    for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
        if (!fp16_matches_reference(h_actual[i], h_expected[i])) {
            fprintf(stderr, "Mismatch [%zu]: expected=%g actual=%g\n", i,
                    fp16_to_fp32(h_expected[i]), fp16_to_fp32(h_actual[i]));
            mismatches++;
        }
    }
    free(h_actual);
    if (mismatches) return 1;
    printf("Validation PASSED (%zu fp16 elements)\n", num_elements);
    return 0;
}

static int validate_fp16_from_file(void* d_actual, const char* expected_path,
                                    size_t size) {
    if (size % sizeof(uint16_t)) return 1;
    uint16_t* expected = (uint16_t*)malloc(size);
    if (!expected) return 1;
    if (read_exact_file(expected_path, expected, size)) {
        free(expected);
        return 1;
    }
    int rc = validate_fp16_from_host(d_actual, expected, size / sizeof(uint16_t));
    free(expected);
    return rc;
}

// Build a path relative to the executable directory
static void build_path(char* out, size_t outsize, const char* exe_dir, const char* rel) {
    snprintf(out, outsize, "%s/%s", exe_dir, rel);
}
