// Helper to convert FP16 to float for comparison
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float val = mantissa / 1024.0f / 16384.0f;
        return sign ? -val : val;
    }
    if (exponent == 31) {
        return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, (float)((int)exponent - 15));
    return sign ? -val : val;
}

// Helper to convert BF16 to float for comparison
float bf16_to_fp32(uint16_t h) {
    uint32_t f32_bits = ((uint32_t)h) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Helper function to validate output tensor with dtype support
int validate_tensor_output(void* d_actual, const char* exe_path, const char* rel_expected_path,
                           size_t size, int arg_idx, const char* dtype) {
    char expected_path[2048];
    snprintf(expected_path, sizeof(expected_path), "%s/%s", exe_path, rel_expected_path);
    FILE* fp = fopen(expected_path, "rb");
    if (!fp) {
        printf("  No expected output file for arg[%d], skipping validation\n", arg_idx);
        return 0;
    }

    void* h_expected = malloc(size);
    void* h_actual = malloc(size);
    fread(h_expected, 1, size, fp);
    fclose(fp);
    cudaMemcpy(h_actual, d_actual, size, cudaMemcpyDeviceToHost);

    int is_fp16 = (strstr(dtype, "float16") != NULL || strstr(dtype, "fp16") != NULL);
    int is_bf16 = (strstr(dtype, "bfloat16") != NULL || strstr(dtype, "bf16") != NULL);
    int is_fp32 = (strstr(dtype, "float32") != NULL || strstr(dtype, "fp32") != NULL);
    int is_fp64 = (strstr(dtype, "float64") != NULL || strstr(dtype, "fp64") != NULL);
    int is_int8 = (strstr(dtype, "int8") != NULL);
    int is_int16 = (strstr(dtype, "int16") != NULL);
    int is_int32 = (strstr(dtype, "int32") != NULL);
    int is_int64 = (strstr(dtype, "int64") != NULL);
    int is_uint8 = (strstr(dtype, "uint8") != NULL);
    int is_uint16 = (strstr(dtype, "uint16") != NULL);
    int is_uint32 = (strstr(dtype, "uint32") != NULL);
    int is_uint64 = (strstr(dtype, "uint64") != NULL);

    size_t elem_size = 4;
    if (is_fp16 || is_bf16 || is_int16 || is_uint16) elem_size = 2;
    else if (is_fp64 || is_int64 || is_uint64) elem_size = 8;
    else if (is_int8 || is_uint8) elem_size = 1;
    else if (is_fp32 || is_int32 || is_uint32) elem_size = 4;

    size_t num_elements = size / elem_size;
    int mismatches = 0;

    printf("  Validating arg[%d]: dtype=%s, elem_size=%zu, num_elements=%zu\n",
           arg_idx, dtype, elem_size, num_elements);

    if (is_fp16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        float rel_tolerance = 1e-3f;
        float abs_tolerance = 1e-2f;

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = fp16_to_fp32(expected_data[i]);
            float act_f = fp16_to_fp32(actual_data[i]);
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_bf16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        float rel_tolerance = 1e-3f;
        float abs_tolerance = 1e-2f;

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = bf16_to_fp32(expected_data[i]);
            float act_f = bf16_to_fp32(actual_data[i]);
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_fp32) {
        float* expected_data = (float*)h_expected;
        float* actual_data = (float*)h_actual;
        float rel_tolerance = 1e-3f;
        float abs_tolerance = 1e-2f;

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = expected_data[i];
            float act_f = actual_data[i];
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_fp64) {
        double* expected_data = (double*)h_expected;
        double* actual_data = (double*)h_actual;
        double rel_tolerance = 1e-6;
        double abs_tolerance = 1e-12;

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            double exp_d = expected_data[i];
            double act_d = actual_data[i];
            double diff = fabs(exp_d - act_d);
            double max_abs = fmax(fabs(exp_d), fabs(act_d));
            int mismatch = (max_abs > abs_tolerance)
                ? (diff / max_abs > rel_tolerance)
                : (diff > abs_tolerance);

            if (mismatch) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                double rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
                printf("    Element %zu: expected=%.15f, actual=%.15f, diff=%.6e, rel_err=%.6e\n",
                       i, exp_d, act_d, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_int8) {
        int8_t* expected_data = (int8_t*)h_expected;
        int8_t* actual_data = (int8_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%d, actual=%d\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint8) {
        uint8_t* expected_data = (uint8_t*)h_expected;
        uint8_t* actual_data = (uint8_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%u, actual=%u\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int16) {
        int16_t* expected_data = (int16_t*)h_expected;
        int16_t* actual_data = (int16_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%d, actual=%d\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%u, actual=%u\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int32) {
        int32_t* expected_data = (int32_t*)h_expected;
        int32_t* actual_data = (int32_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%d, actual=%d\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint32) {
        uint32_t* expected_data = (uint32_t*)h_expected;
        uint32_t* actual_data = (uint32_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%u, actual=%u\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int64) {
        int64_t* expected_data = (int64_t*)h_expected;
        int64_t* actual_data = (int64_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%lld, actual=%lld\n",
                       i, (long long)expected_data[i], (long long)actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint64) {
        uint64_t* expected_data = (uint64_t*)h_expected;
        uint64_t* actual_data = (uint64_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) printf("\n  Validation FAILED for arg[%d]:\n", arg_idx);
                printf("    Element %zu: expected=%llu, actual=%llu\n",
                       i, (unsigned long long)expected_data[i], (unsigned long long)actual_data[i]);
                mismatches++;
            }
        }
    } else {
        printf("\n  ERROR: Unsupported dtype '%s' for validation of arg[%d]\n", dtype, arg_idx);
        printf("  Supported types: float16, bfloat16, float32, float64, int8/16/32/64, uint8/16/32/64\n");
        free(h_expected);
        free(h_actual);
        return -1;
    }

    if (mismatches == 0) {
        printf("  Validation PASSED for arg[%d]: all %zu elements match\n",
               arg_idx, num_elements);
    } else {
        printf("  Total mismatches for arg[%d]: %d (showing first 10)\n", arg_idx, mismatches);
    }

    free(h_expected);
    free(h_actual);
    return mismatches == 0 ? 0 : 1;
}
