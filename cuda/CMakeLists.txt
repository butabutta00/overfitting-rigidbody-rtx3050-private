cmake_minimum_required(VERSION 3.20)
project(mass_spring_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MSS_CUDA_ARCHITECTURES "86-real;86-virtual" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES ${MSS_CUDA_ARCHITECTURES})

option(MSS_ENABLE_LINEINFO "Enable line info for Nsight" ON)

add_library(mass_spring_native SHARED
    mass_spring_native.cu
)

add_executable(main
    main.cu
)

target_compile_definitions(mass_spring_native PRIVATE MASS_SPRING_NATIVE_EXPORTS=1)

set(MSS_NVCC_FLAGS
    -O3
    --use_fast_math
    -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage
)

if (MSS_ENABLE_LINEINFO)
    list(APPEND MSS_NVCC_FLAGS -lineinfo)
endif()

target_compile_options(mass_spring_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${MSS_NVCC_FLAGS}>
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${MSS_NVCC_FLAGS}>
)

target_link_libraries(main PRIVATE
    mass_spring_native
    CUDA::cudart
)

set_target_properties(mass_spring_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    OUTPUT_NAME "mass_spring_native"
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    OUTPUT_NAME "main"
    BUILD_RPATH "$ORIGIN"
)
