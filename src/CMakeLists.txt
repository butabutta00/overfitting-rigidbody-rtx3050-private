cmake_minimum_required(VERSION 3.24)
project(rigidbody_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RTX 3050 (Ampere): SM 86
set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(rigidbody_cuda SHARED
    rigidbody_cuda.cu
)

target_compile_definitions(rigidbody_cuda PRIVATE RB_CUDA_EXPORTS)
set_target_properties(rigidbody_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(rigidbody_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage>
)

add_executable(rigidbody_cuda_test
    rigidbody_cuda_test.cu
)

target_link_libraries(rigidbody_cuda_test PRIVATE rigidbody_cuda)

target_compile_options(rigidbody_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage>
)
