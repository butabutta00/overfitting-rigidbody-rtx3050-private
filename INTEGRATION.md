# CUDA RigidBody Integration Plan (RTX 3050 overfitting)

## 1) What was implemented

- CUDA native plugin in `src/rigidbody_cuda.cu` with RTX 3050-targeted build settings:
  - `CMAKE_CUDA_ARCHITECTURES=86` (Ampere GA10x path for RTX 3050)
  - `-O3 --use_fast_math -Xptxas=-v,-dlcm=ca,-warn-spills,-warn-lmem-usage -lineinfo`
  - `__launch_bounds__(256, 2)` for main kernels
  - SoA (`float4` arrays) integration kernel for coalesced memory access
- Two requested methods are implemented in CUDA:
  - Semi-Implicit Euler (`RB_CUDA_SEMI_IMPLICIT`)
  - Velocity Verlet (`RB_CUDA_VELOCITY_VERLET`)
- Unity-side bridge is wired in `unity/Assets/Scripts/RigidBodyDynamicsLab.cs` via `DllImport("rigidbody_cuda")`.
- Standalone CUDA test executable in `src/rigidbody_cuda_test.cu` validates GPU vs CPU reference for both methods.

## 2) RTX 3050 overfitting checklist mapping

1. Register Pressure Management
   - `__launch_bounds__` used in all critical kernels.
   - PTXAS verbosity and spill/lmem warnings enabled (`-Xptxas=-v,-warn-spills,-warn-lmem-usage`).
2. Shared Memory / L1 tuning
   - `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout, 100)` used.
   - `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 49152)` used for async staging kernel.
   - `cuda::memcpy_async` applied in AoS->SoA staging kernel.
3. WMMA tile-based compute
   - 16x16x16 WMMA path added (`TensorCoreScaleBf16Kernel`) for acceleration*dt / angularAcceleration*dt transforms.
4. BF16/TF32 mixed precision
   - BF16 used in WMMA path (`__nv_bfloat16`, `wmma::precision::bfloat16`) with FP32 accumulation.
5. Vectorized load/store
   - SoA layout uses `float4` for all state channels.
6. Branchless programming
   - Integration mode selection in main kernel converted to mask-based branchless blend.
   - `__byte_perm` used for packed method decode.
7. Warp shuffle
   - `__shfl_sync` used to broadcast scale/dt in warp-level paths.
8. Texture unit read-only cache usage
   - acceleration / angularAcceleration read through 1D texture objects (`tex1Dfetch<float4>`).

## 3) Unity integration strategy

### Native plugin deployment

1. Build `librigidbody_cuda.so` (Linux) or platform equivalent (`rigidbody_cuda.dll`, `librigidbody_cuda.dylib`).
2. Copy to Unity plugin path by platform, e.g.:
   - Linux: `unity/Assets/Plugins/x86_64/librigidbody_cuda.so`
   - Windows: `unity/Assets/Plugins/x86_64/rigidbody_cuda.dll`
3. In Unity Import Settings, enable the target platform/CPU architecture for the plugin.

### Runtime fallback behavior

- `RigidBodyDynamicsLab` attempts CUDA integration when `useCudaIntegration=true`.
- If plugin is missing or entry point mismatch occurs, it auto-falls back to CPU integrator.
- Existing spring force/torque accumulation is preserved; only integration step is offloaded.

### API contract

- Unity sends one `RbCudaState` for current object state + computed accelerations.
- Plugin returns updated position/rotation/linear velocity/angular velocity.
- API supports both single (`RbCudaStepSingle`) and batch (`RbCudaStepBatch`) stepping.

## 4) External GPU-server test strategy

### Docker build

From repository root `rtx3050/overfitting-rigidbody-rtx3050`:

```bash
docker build -t rigidbody-cuda-test .
```

To run the full build-and-test flow with one command, use:

```bash
bash ./run_gpu_test.sh
```

### Docker run (requires NVIDIA Container Toolkit)

```bash
docker run --rm --gpus all rigidbody-cuda-test
```

Optional custom particle count:

```bash
docker run --rm --gpus all rigidbody-cuda-test /workspace/build/rigidbody_cuda_test 2097152
```

Expected output includes:

- `PASS count=... methods=semi-implicit,velocity-verlet`
