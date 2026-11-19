cmake_minimum_required(VERSION 3.20)
project(ggml_starter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 关闭 ggml 默认示例与测试，加快编译速度
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# 1. 引入外部依赖模块 (ggml)
add_subdirectory(third_party/ggml)

# 2. 引入项目内部子模块 (ggml_env)
add_subdirectory(modules/ggml_env)

# 3. 主程序目标 (src 与 include 分离)
add_executable(app src/main.cpp)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(app PRIVATE ggml_env)
