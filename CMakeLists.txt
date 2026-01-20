cmake_minimum_required(VERSION 3.20)
project(flash_attention_kernel1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(flash_attention
    src/main.cu
)

add_executable(flash_test
    src/main.cu
)

target_include_directories(flash_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(flash_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(flash_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable extended constexpr for CUDA headers when needed.
target_compile_options(flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_options(flash_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
