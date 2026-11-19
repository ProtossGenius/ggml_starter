add_library(ggml_env STATIC
    src/env_info.cpp
)

# 暴露给外部调用者的头文件目录 (include 与 src 分离)
target_include_directories(ggml_env
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接 ggml 核心与后端库
target_link_libraries(ggml_env
    PUBLIC
        ggml
)
