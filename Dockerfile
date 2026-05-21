FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY src /workspace/src
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_gpu_test.sh

# GPU smoke test (override count as needed):
# docker run --rm --gpus all rigidbody-cuda-test /workspace/scripts/run_gpu_test.sh 4194304 30
CMD ["/workspace/scripts/run_gpu_test.sh", "4194304", "30"]
