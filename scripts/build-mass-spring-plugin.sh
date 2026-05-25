#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_dir="${project_root}/cuda"
build_dir="${project_root}/cuda/build"
plugin_dir="${project_root}/VRTermProject/Assets/Plugins/x86_64"
cuda_arch="${1:-native}"

rm -f "${build_dir}/CMakeCache.txt"
rm -rf "${build_dir}/CMakeFiles"

cmake -S "${cuda_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSS_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DMSS_ENABLE_LINEINFO=ON

cmake --build "${build_dir}" --config Release -j

mkdir -p "${plugin_dir}"

if [[ -f "${build_dir}/libmass_spring_native.so" ]]; then
    cp "${build_dir}/libmass_spring_native.so" "${plugin_dir}/libmass_spring_native.so"
elif [[ -f "${build_dir}/Release/mass_spring_native.dll" ]]; then
    cp "${build_dir}/Release/mass_spring_native.dll" "${plugin_dir}/mass_spring_native.dll"
elif [[ -f "${build_dir}/mass_spring_native.dll" ]]; then
    cp "${build_dir}/mass_spring_native.dll" "${plugin_dir}/mass_spring_native.dll"
else
    echo "Built plugin binary not found in ${build_dir}" >&2
    exit 4
fi

echo "Mass spring plugin build complete."
echo "Copied binary to ${plugin_dir}"
