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

RUN cmake -S /workspace/src -B /workspace/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /workspace/build --config Release -j

# GPU smoke test (override count as needed):
# docker run --rm --gpus all rigidbody-cuda-test /workspace/build/rigidbody_cuda_test 1048576
CMD ["/workspace/build/rigidbody_cuda_test", "1048576"]
