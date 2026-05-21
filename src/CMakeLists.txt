cmake_minimum_required(VERSION 3.24)
project(rigidbody_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RTX 3050 (Ampere): SM 86
set(CMAKE_CUDA_ARCHITECTURES 86)

option(RB_BUILD_FP16 "Build FP16 tensor-core variant" ON)
option(RB_BUILD_BF16 "Build BF16 tensor-core variant" ON)
option(RB_CUDA_DIAGNOSTICS "Enable CUDA diagnostics exports" OFF)

if (NOT RB_BUILD_FP16 AND NOT RB_BUILD_BF16)
    message(FATAL_ERROR "Enable at least one variant: RB_BUILD_FP16 or RB_BUILD_BF16")
endif()

function(rb_add_variant variant precision_def)
    set(lib_target "rigidbody_cuda_${variant}")
    set(test_target "rigidbody_cuda_test_${variant}")

    add_library(${lib_target} SHARED
        rigidbody_cuda.cu
    )

    target_compile_definitions(${lib_target} PRIVATE
        RB_CUDA_EXPORTS
        ${precision_def}
        $<$<BOOL:${RB_CUDA_DIAGNOSTICS}>:RB_CUDA_DIAGNOSTICS=1>
    )
    set_target_properties(${lib_target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(${lib_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage>
    )

    add_executable(${test_target}
        rigidbody_cuda_test.cu
    )

    target_compile_definitions(${test_target} PRIVATE ${precision_def})
    target_link_libraries(${test_target} PRIVATE ${lib_target})

    target_compile_options(${test_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage>
    )

    if (variant STREQUAL "fp")
        set_target_properties(${lib_target} PROPERTIES OUTPUT_NAME "rigidbody_cuda")
        set_target_properties(${test_target} PROPERTIES OUTPUT_NAME "rigidbody_cuda_test")
    else()
        set_target_properties(${lib_target} PROPERTIES OUTPUT_NAME "rigidbody_cuda_${variant}")
        set_target_properties(${test_target} PROPERTIES OUTPUT_NAME "rigidbody_cuda_test_${variant}")
    endif()
endfunction()

if (RB_BUILD_FP16)
    rb_add_variant("fp" "RB_CUDA_TC_USE_FP16=1")
endif()

if (RB_BUILD_BF16)
    rb_add_variant("bf" "RB_CUDA_TC_USE_BF16=1")
endif()

# Backward-compatible target names point to FP16 build.
if (TARGET rigidbody_cuda_fp)
    add_library(rigidbody_cuda ALIAS rigidbody_cuda_fp)
endif()
