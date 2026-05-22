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
- Standalone CUDA test executable in `src/rigidbody_cuda_test.cu` validates GPU vs CPU reference for both methods and prints precision-tagged benchmark timing.
- Two precision variants are build-configurable via CMake options:
  - `RB_BUILD_FP16=ON/OFF`
  - `RB_BUILD_BF16=ON/OFF`

## 2) RTX 3050 overfitting checklist mapping

1. Register Pressure Management
   - `__launch_bounds__` used in all critical kernels.
   - PTXAS verbosity and spill/lmem warnings enabled (`-Xptxas=-v,-warn-spills,-warn-lmem-usage`).
2. Shared Memory / L1 tuning
   - `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout, 100)` used.
   - `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 49152)` kept for kernel tuning compatibility.
3. WMMA tile-based compute
   - 16x16x16 WMMA path added (`TensorCoreScaleTcKernel`) for acceleration*dt / angularAcceleration*dt transforms.
4. BF16/FP16 mixed precision variants
   - Build-time macro selects precision path:
     - FP16: `RB_CUDA_TC_USE_FP16=1` (`__half` + FP32 accumulation)
     - BF16: `RB_CUDA_TC_USE_BF16=1` (`__nv_bfloat16` + FP32 accumulation)
   - BF16 conversion uses CUDA math intrinsics (`__float2bfloat16`) per CUDA math API.
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

### Unity precision selection at build time

- `RigidBodyDynamicsLab` now dispatches through a compile-time bridge:
  - Default (no define): FP16 plugin (`librigidbody_cuda.so`)
  - With define `RB_CUDA_BF16`: BF16 plugin (`librigidbody_cuda_bf.so`)
- Build/copy helper script:

```bash
bash ./scripts/build_unity_plugin.sh fp16
bash ./scripts/build_unity_plugin.sh bf16
```

- In Unity Player Settings -> Scripting Define Symbols:
  - Leave unset for FP16
  - Add `RB_CUDA_BF16` for BF16

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

From repository root:

```bash
docker build -t rigidbody-cuda-test .
```

To run the full build-and-test flow with one command (both FP16 and BF16), use:

```bash
bash ./scripts/run_gpu_test.sh
```

### Docker run (requires NVIDIA Container Toolkit)

```bash
docker run --rm --gpus all rigidbody-cuda-test
```

Optional custom particle count and iterations:

```bash
docker run --rm --gpus all rigidbody-cuda-test /workspace/scripts/run_gpu_test.sh 4194304 30
```

Expected output includes per-precision lines:

- `PASS precision=fp16 count=... methods=semi-implicit,velocity-verlet`
- `BENCH precision=fp16 iters=... total_ms=... avg_ms=...`
- `PASS precision=bf16 count=... methods=semi-implicit,velocity-verlet`
- `BENCH precision=bf16 iters=... total_ms=... avg_ms=...`
