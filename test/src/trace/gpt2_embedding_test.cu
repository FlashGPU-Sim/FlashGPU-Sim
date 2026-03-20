// GPT-2 Embedding Kernel Trace Test (CPU Reference)
//
// Kernel semantics: out[t, c] = tok_emb[token_ids[t], c] + pos_emb[t, c]
// Uses small dimensions (T=1, V=64, C=768) with random data.
// Validates GPU output against CPU reference computation.

#include "trace_test_utils.h"

int main() {
    printf("=== GPT-2 Embedding Trace Test (CPU Reference) ===\n");

    // Small dimensions for fast CI
    const int T = 1;
    const int C = 768;
    const int V = 64;

    const size_t tokens_size = T * sizeof(int32_t);
    const size_t tok_emb_size = V * C * sizeof(uint16_t);
    const size_t pos_emb_size = T * C * sizeof(uint16_t);
    const size_t out_size = T * C * sizeof(uint16_t);

    // Initialize CUDA
    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    // Locate fatbin relative to executable
    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char fatbin_path[2048];
    build_path(fatbin_path, sizeof(fatbin_path), exe_dir, "embedding_kernel.fatbin");

    // Load module and get kernel
    CUmodule module = load_fatbin(fatbin_path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "embedding_kernel");
    if (!func) return 1;
    set_shared_mem(func, 2056);

    // Generate random test data on host
    srand(42);
    int32_t* h_tokens = (int32_t*)malloc(tokens_size);
    uint16_t* h_tok_emb = (uint16_t*)malloc(tok_emb_size);
    uint16_t* h_pos_emb = (uint16_t*)malloc(pos_emb_size);

    for (int i = 0; i < T; i++)
        h_tokens[i] = rand() % V;

    for (int i = 0; i < V * C; i++)
        h_tok_emb[i] = fp32_to_fp16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);

    for (int i = 0; i < T * C; i++)
        h_pos_emb[i] = fp32_to_fp16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);

    // CPU reference: out[t,c] = tok_emb[tokens[t], c] + pos_emb[t, c]
    uint16_t* h_expected = (uint16_t*)malloc(out_size);
    for (int t = 0; t < T; t++) {
        int tok_id = h_tokens[t];
        for (int c = 0; c < C; c++) {
            float te = fp16_to_fp32(h_tok_emb[tok_id * C + c]);
            float pe = fp16_to_fp32(h_pos_emb[t * C + c]);
            h_expected[t * C + c] = fp32_to_fp16(te + pe);
        }
    }

    // Upload to GPU
    void* d_tokens;
    cudaMalloc(&d_tokens, tokens_size);
    cudaMemcpy(d_tokens, h_tokens, tokens_size, cudaMemcpyHostToDevice);

    void* d_tok_emb;
    cudaMalloc(&d_tok_emb, tok_emb_size);
    cudaMemcpy(d_tok_emb, h_tok_emb, tok_emb_size, cudaMemcpyHostToDevice);

    void* d_pos_emb;
    cudaMalloc(&d_pos_emb, pos_emb_size);
    cudaMemcpy(d_pos_emb, h_pos_emb, pos_emb_size, cudaMemcpyHostToDevice);

    void* d_out = alloc_gpu(out_size);

    // Triton runtime scratch buffers
    void* global_scratch = alloc_gpu(2048);
    void* profile_scratch = NULL;

    // Kernel args: tokens, tok_emb, pos_emb, out, T, C, V, global_scratch, profile_scratch
    int32_t arg_T = T;
    int32_t arg_C = C;
    int32_t arg_V = V;
    void* args[] = {
        &d_tokens, &d_tok_emb, &d_pos_emb, &d_out,
        &arg_T, &arg_C, &arg_V,
        &global_scratch, &profile_scratch
    };

    printf("Launching embedding_kernel: grid=(%d,1,1) block=(128,1,1) smem=2056\n", T);
    if (launch_kernel(func, T, 1, 1, 128, 1, 1, 2056, args)) return 1;
    printf("Kernel completed\n");

    // Validate
    printf("Validating against CPU reference...\n");
    int rc = validate_fp16_from_host(d_out, h_expected, T * C);

    // Cleanup
    free(h_tokens); free(h_tok_emb); free(h_pos_emb); free(h_expected);
    cudaFree(d_tokens); cudaFree(d_tok_emb); cudaFree(d_pos_emb);
    cudaFree(d_out); cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
