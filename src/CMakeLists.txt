cmake_minimum_required(VERSION 3.24)
project(rigidbody_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    # Ensure host compiler reads source as UTF-8 to avoid C4819 with CUDA headers on CP949 systems.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# Use shared CUDA runtime on Windows to avoid LIBCMT vs /MD conflicts (LNK4098).
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Default targets:
# - SM 61 (Pascal): GTX 1050 / GTX 1650 / GTX 950 / GTX 960
# - SM 75 (Turing): RTX 2060 / RTX 2070 / RTX 2080
# - SM 86 (Ampere): RTX 3050
set(RB_CUDA_ARCHITECTURES "61;75;86" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_ARCHITECTURES ${RB_CUDA_ARCHITECTURES})
set(RB_CUDA_FORCE_GENCODE ON CACHE BOOL "Force explicit -gencode flags for runtime image compatibility")

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(RB_CUDA_GENCODE_FLAGS
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_86,code=compute_86)

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
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    target_compile_options(${lib_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage -Xcompiler=/utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${RB_CUDA_FORCE_GENCODE}>>:${RB_CUDA_GENCODE_FLAGS}>
    )

    add_executable(${test_target}
        rigidbody_cuda_test.cu
    )

    target_compile_definitions(${test_target} PRIVATE ${precision_def})
    target_link_libraries(${test_target} PRIVATE ${lib_target})

    target_compile_options(${test_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage -Xcompiler=/utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${RB_CUDA_FORCE_GENCODE}>>:${RB_CUDA_GENCODE_FLAGS}>
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
