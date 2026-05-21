#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
count="${1:-4194304}"
iters="${2:-30}"

build_and_run_local() {
    local variant="$1"
    local build_dir="${repo_root}/build-${variant}"
    local bf_opt="OFF"
    local fp_opt="OFF"

    if [[ "${variant}" == "fp" ]]; then
        fp_opt="ON"
    else
        bf_opt="ON"
    fi

    cmake -S "${repo_root}/src" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRB_BUILD_FP16="${fp_opt}" \
        -DRB_BUILD_BF16="${bf_opt}"
    cmake --build "${build_dir}" --config Release -j

    if [[ "${variant}" == "fp" ]]; then
        "${build_dir}/rigidbody_cuda_test" "${count}" "${iters}"
    else
        "${build_dir}/rigidbody_cuda_test_bf" "${count}" "${iters}"
    fi
}

# Check if running inside a Docker container
is_in_docker() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
        return 0
    fi
    if grep -q docker /proc/1/cgroup 2>/dev/null; then
        return 0
    fi
    return 1
}

if is_in_docker; then
    echo "Running inside Docker container - building and testing both variants..."
    echo "[FP16]"
    build_and_run_local fp
    echo "[BF16]"
    build_and_run_local bf
else
    echo "Running outside Docker container - building Docker image..."

    image_name="rigidbody-cuda-test"
    docker build -t "${image_name}" "${repo_root}"
    docker run --rm --gpus all "${image_name}" /workspace/scripts/run_gpu_test.sh "${count}" "${iters}"
fi
