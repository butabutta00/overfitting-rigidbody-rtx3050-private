#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_name="rigidbody-cuda-test"
count="${1:-1048576}"

docker build -t "${image_name}" "${repo_root}"
docker run --rm --gpus all "${image_name}" /workspace/build/rigidbody_cuda_test "${count}"
