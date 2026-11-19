#include "ggml_env/env_info.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>

namespace ggml_env {

void info() {
    printf("========== [GGML Environment Check] ==========\n");

    // 1. 验证基础 ggml 上下文与张量创建是否成功
    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[ERROR] Failed to initialize ggml_context!\n");
        return;
    }

    ggml_tensor* test_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    if (test_tensor) {
        printf("[OK] GGML initialized successfully! (Tensor overhead: %zu bytes)\n",
               ggml_tensor_overhead());
    }
    ggml_free(ctx);

    // 2. 加载所有后端并枚举已注册后端驱动
    ggml_backend_load_all();
    size_t reg_count = ggml_backend_reg_count();
    printf("\n--- Registered Backend Drivers (%zu) ---\n", reg_count);
    for (size_t i = 0; i < reg_count; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        printf("  [%zu] %s\n", i, ggml_backend_reg_name(reg));
    }

    // 3. 枚举所有硬件设备并筛选可用 GPU 后端 (含独立 GPU 与统一内存 iGPU)
    size_t dev_count = ggml_backend_dev_count();
    size_t gpu_count = 0;
    printf("\n--- Available GPU Backends ---\n");
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);

        if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
            dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_count++;
            size_t free_mem = 0, total_mem = 0;
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            const char* type_str = (dtype == GGML_BACKEND_DEVICE_TYPE_GPU) ? "Discrete GPU" : "Integrated/UMA GPU";
            printf("  [GPU #%zu] Name: %-10s | Type: %-18s | Desc: %s | Memory: %.2f / %.2f GB\n",
                   gpu_count,
                   ggml_backend_dev_name(dev),
                   type_str,
                   ggml_backend_dev_description(dev),
                   (double)free_mem / (1024.0 * 1024.0 * 1024.0),
                   (double)total_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }

    if (gpu_count == 0) {
        printf("  [WARN] No GPU backend detected. Built with CPU backend only.\n");
        printf("  [TIP]  Enable GPU in CMake via -DGGML_METAL=ON, -DGGML_CUDA=ON, or -DGGML_VULKAN=ON\n");
    }
    printf("==============================================\n");
}

} // namespace ggml_env
