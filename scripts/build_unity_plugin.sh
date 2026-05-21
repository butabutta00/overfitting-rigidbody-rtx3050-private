#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
precision="${1:-fp16}"
platform="${2:-linux}"
diagnostics="${3:-off}"
cuda_arch="${4:-61;75;86}"

case "${precision}" in
    fp16)
        fp_opt="ON"
        bf_opt="OFF"
        lib_name="librigidbody_cuda.so"
        ;;
    bf16)
        fp_opt="OFF"
        bf_opt="ON"
        lib_name="librigidbody_cuda_bf.so"
        ;;
    *)
        echo "Unsupported precision: ${precision}" >&2
        echo "Usage: $0 [fp16|bf16] [linux] [on|off] [cuda-arch]" >&2
        exit 2
        ;;
esac

diag_opt="OFF"
if [[ "${diagnostics}" == "on" ]]; then
    diag_opt="ON"
elif [[ "${diagnostics}" != "off" ]]; then
    echo "Unsupported diagnostics option: ${diagnostics}" >&2
    echo "Usage: $0 [fp16|bf16] [linux] [on|off] [cuda-arch]" >&2
    exit 4
fi

if [[ "${platform}" != "linux" ]]; then
    echo "Unsupported platform: ${platform}" >&2
    echo "Currently only linux output is scripted." >&2
    exit 3
fi

build_dir="${repo_root}/build-unity-${precision}"
plugin_dir="${repo_root}/unity/Assets/Plugins/x86_64"

cmake --fresh -S "${repo_root}/src" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRB_BUILD_FP16="${fp_opt}" \
    -DRB_BUILD_BF16="${bf_opt}" \
    -DRB_CUDA_DIAGNOSTICS="${diag_opt}" \
    -DRB_CUDA_ARCHITECTURES="${cuda_arch}"

cmake --build "${build_dir}" --config Release -j

mkdir -p "${plugin_dir}"

if [[ "${precision}" == "fp16" ]]; then
    cp "${build_dir}/${lib_name}" "${plugin_dir}/librigidbody_cuda.so"
else
    cp "${build_dir}/${lib_name}" "${plugin_dir}/librigidbody_cuda_bf.so"
fi

echo "Built Unity plugin precision=${precision} diagnostics=${diagnostics} arch=${cuda_arch}"
echo "Output: ${plugin_dir}"
