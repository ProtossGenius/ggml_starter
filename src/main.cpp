#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml_env/env_info.hpp"
#include <cstdio>
#include <vector>

int main() {
    // 先打印子模块的环境信息
    // ggml_env::info();

    // =========================================================================
    // 步骤 1: 选择并初始化后端 (优先 GPU/Metal/CUDA，否则回退 CPU)
    // =========================================================================
    ggml_backend_t backend = ggml_backend_init_best();
    printf("\n[Step 1] Using Backend: %s\n", ggml_backend_name(backend));

    // =========================================================================
    // 步骤 2: 创建元数据上下文 (no_alloc = true) 并声明输入张量
    // =========================================================================
    // 我们要计算矩阵乘法: A (4行 x 2列) 与 B (3行 x 2列)
    const int rows_A = 4, cols_A = 2;
    const int rows_B = 3, cols_B = 2;

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true, // 关键：不在 ctx 中分配实际数据内存，交给
                               // backend 分配
    };
    ggml_context *ctx = ggml_init(params);

    // 注意：ne[0] 是列数，ne[1] 是行数
    ggml_tensor *tensor_a =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    ggml_tensor *tensor_b =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    // 在所选后端(如 GPU 显存)中为上下文中的张量分配真实物理内存
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // =========================================================================
    // 步骤 3: 将主机(RAM)数据拷贝到后端缓冲区(VRAM/RAM)
    // =========================================================================
    float host_A[ rows_A * cols_A ] = {2.0f, 8.0f, 5.0f, 1.0f,
                                       4.0f, 2.0f, 8.0f, 6.0f};
    float host_B[ rows_B * cols_B ] = {10.0f, 5.0f, 9.0f, 9.0f, 5.0f, 4.0f};

    // =========================================================================
    // 步骤 4: 构建前向计算图 (C = B * A^T)
    // =========================================================================
    ggml_cgraph *gf     = ggml_new_graph(ctx);
    ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b);
    ggml_build_forward_expand(gf, result);

    // =========================================================================
    // 步骤 5: 使用图内存分配器 (ggml_gallocr) 分配中间计算缓冲区并执行计算图
    // =========================================================================
    ggml_gallocr_t allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    printf("[Step 5] set tensor_a and tensor_b:\n");
    ggml_backend_tensor_set(tensor_a, host_A, 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, host_B, 0, ggml_nbytes(tensor_b));

    // 在后端上执行计算图
    ggml_backend_graph_compute(backend, gf);

    // =========================================================================
    // 步骤 6: 从后端取回结果张量数据并释放资源
    // =========================================================================
    std::vector<float> out_data(ggml_nelements(result));
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    printf("[Step 6] Mul_Mat Result (ne[0]=%lld, ne[1]=%lld):\n",
           (long long) result->ne[ 0 ], (long long) result->ne[ 1 ]);
    for (int r = 0; r < result->ne[ 1 ]; ++r) {
        printf("  [ ");
        for (int c = 0; c < result->ne[ 0 ]; ++c) {
            printf("%6.1f ", out_data[ r * result->ne[ 0 ] + c ]);
        }
        printf("]\n");
    }

    // 释放所有资源
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
