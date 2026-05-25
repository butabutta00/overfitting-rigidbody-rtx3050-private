#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_dir="${project_root}/cuda"
build_dir="${cuda_dir}/build"
cuda_arch="${1:-native}"

rm -f "${build_dir}/CMakeCache.txt"
rm -rf "${build_dir}/CMakeFiles"

cmake -S "${cuda_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSS_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DMSS_ENABLE_LINEINFO=ON

cmake --build "${build_dir}" --config Release --target main -j

entrypoint_bin="${build_dir}/main"
if [[ ! -x "${entrypoint_bin}" ]]; then
    entrypoint_bin="${build_dir}/Release/main"
fi

if [[ ! -x "${entrypoint_bin}" ]]; then
    echo "main binary not found in ${build_dir}" >&2
    exit 4
fi

echo "CUDA entrypoint build complete: ${entrypoint_bin}"
echo "Run example: ${entrypoint_bin} 1.0 0.0 0.016 1.0 120.0 0.2 8"
