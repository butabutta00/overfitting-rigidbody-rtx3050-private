#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
count="${1:-1048576}"

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
    echo "Running inside Docker container - building and testing locally..."
    
    # Build directly inside the container
    cmake -S "${repo_root}/src" -B "${repo_root}/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${repo_root}/build" --config Release -j
    
    # Run the test
    "${repo_root}/build/rigidbody_cuda_test" "${count}"
else
    echo "Running outside Docker container - building Docker image..."
    
    image_name="rigidbody-cuda-test"
    docker build -t "${image_name}" "${repo_root}"
    docker run --rm --gpus all "${image_name}" /workspace/build/rigidbody_cuda_test "${count}"
fi
