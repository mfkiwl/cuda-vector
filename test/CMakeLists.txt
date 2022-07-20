cmake_minimum_required(VERSION 3.15)

project(test LANGUAGES CXX CUDA)

# Exclude the examples from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")

# Exclude the benchmarks from the build
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

# Exclude the tests from the build
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

add_subdirectory(stdgpu)

add_library(foo stdgpu/src/stdgpu/)

target_link_libraries(foo PUBLIC stdgpu::stdgpu)

add_executable(prog main.cu)

