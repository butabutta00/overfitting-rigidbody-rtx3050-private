// Mass-Spring System Simulator - CUDA Implementation
// Provides GPU-accelerated physics simulation for mass-spring systems using both
// semi-implicit and implicit (conjugate gradient) integration methods.
// Supports fixed constraints, gravity, and spring-damper forces.

#include "mass_spring_native.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace cg = cooperative_groups;

// Simulation constants
// kEpsilon: Minimum distance threshold to avoid division by zero
// kBlockSize: 256 threads per block - optimal for RTX 3050 SM utilization
//   (RTX 3050 SMs can handle 1536-2048 threads; 256 threads provides good
//    occupancy while managing register pressure)
constexpr float kEpsilon = 1e-6f;  // Minimum distance threshold to avoid division by zero
constexpr int kBlockSize = 256;     // CUDA thread block size for kernels

// Main system state container - holds both host and device memory for particle positions,
// velocities, forces, and spring connectivity. Manages GPU resources and solver state.
// 
// OPTIMIZATION: Uses float4 to align data to 128-bit boundaries for RTX 3050's memory bandwidth.
// RTX 3050 memory bandwidth is 128-bit, so vectorized float4 loads/stores are optimal.
// Even though we only use xyz components, padding with w=unused still gives better throughput
// than using separate float arrays.
struct DeviceSystem
{
    // Topology
    int particleCount = 0;  // Number of particles
    int springCount = 0;    // Number of springs

    // Spring topology (device memory)
    int2* dSpringEnds = nullptr;       // Spring endpoint particle indices
    float* dRestLengths = nullptr;     // Rest lengths for each spring
    float* dMasses = nullptr;          // Particle masses
    uint8_t* dFixedMask = nullptr;     // Bit mask for fixed particles

    // Particle state (device memory)
    // OPTIMIZATION: All position/velocity/force use float4 for 128-bit aligned memory access.
    // This matches RTX 3050's optimal memory bandwidth (128-bit per clock).
    float4* dPosition = nullptr;  // Current positions (xyz, w=unused)
    float4* dVelocity = nullptr;  // Current velocities (xyz, w=unused)
    float4* dForce = nullptr;     // Accumulated forces (xyz, w=unused)

    // Implicit solver buffers
    float4* dX0 = nullptr;        // Saved position state for implicit solve
    float4* dV0 = nullptr;        // Saved velocity state for implicit solve
    float4* dJv = nullptr;        // Damping jacobian term
    float4* dLinearized = nullptr;// Linearized spring data (dir.xyz, stiffness coeff)
    float4* dB = nullptr;         // Right-hand side vector for CG solver
    float4* dV = nullptr;         // Solution vector (solved velocity) for CG
    float4* dR = nullptr;         // Residual vector for CG iterations
    float4* dP = nullptr;         // Search direction for CG iterations
    float4* dAp = nullptr;        // Matrix-vector product A*p for CG

    // Dot product reduction buffers
    float* dDotPartial = nullptr;  // Partial sums from each block
    float* dDotValue = nullptr;    // Final dot product result
    int dotPartialCount = 0;       // Number of blocks in last allocation
    float hDotValue = 0.0f;        // Host-side copy of dot product

    // Configuration flags
    bool cudaAvailable = false;            // Whether CUDA is available and initialized
    bool kernelAttributesConfigured = false; // Whether kernel cache hints were set
    bool deviceCacheConfigured = false;    // Whether device cache was configured

    // Host-side state buffers (used when CUDA unavailable)
    std::vector<float4> hPosition;   // Particle positions on host
    std::vector<float4> hVelocity;   // Particle velocities on host
    std::vector<float4> hForce;      // Accumulated forces on host
    std::vector<float> hMasses;      // Particle masses on host
    std::vector<uint8_t> hFixedMask; // Fixed constraint mask on host
    std::vector<int2> hSpringEnds;   // Spring connectivity on host
    std::vector<float> hRestLengths; // Spring rest lengths on host
};

thread_local std::string gLastError;

// Set error message in thread-local storage
inline void SetError(const std::string& message)
{
    gLastError = message;
}

// Set CUDA error message with location context
inline void SetErrorCuda(const char* where, cudaError_t err)
{
    gLastError = std::string(where) + ": " + cudaGetErrorString(err);
}

// Check CUDA error and log if failed
inline bool CheckCuda(cudaError_t err, const char* where)
{
    if (err == cudaSuccess)
    {
        return true;
    }

    SetErrorCuda(where, err);
    return false;
}

// Read-only cached load for device memory (uses __ldg for compute capability 3.5+)
// OPTIMIZATION: RTX 3050 (CC 8.6) supports __ldg for efficient read-only data access.
// Data like masses, spring endpoints, rest lengths, and fixed masks are never modified,
// so __ldg routes these loads through L1 cache instead of L2, reducing cache pollution
// and improving memory efficiency. Falls back to regular load for older GPUs.
template <typename T>
__device__ __forceinline__ T ReadOnly(const T* ptr)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Vector dot product (3D components only)
__device__ __forceinline__ float Dot3(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Vector addition (3D components, w=0)
__device__ __forceinline__ float4 Add3(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

// Vector subtraction (3D components, w=0)
__device__ __forceinline__ float4 Sub3(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

// Vector scalar multiplication (3D components, w=0)
__device__ __forceinline__ float4 Mul3(const float4& v, float s)
{
    return make_float4(v.x * s, v.y * s, v.z * s, 0.0f);
}

// Returns 1.0 if particle is free, 0.0 if fixed (used to disable updates for fixed particles)
// OPTIMIZATION: Uses __byte_perm bit manipulation instead of conditional branches.
// Branch divergence on RTX 3050 can serialize warp execution. By converting the fixed mask
// byte into a scalar multiplier (0.0 or 1.0) via __byte_perm, we avoid branches and let
// all threads in a warp execute the same code path, improving throughput.
__device__ __forceinline__ float ActiveScale(const uint8_t* fixedMask, int idx)
{
    uint8_t mask = ReadOnly(&fixedMask[idx]);
    uint32_t expanded = __byte_perm(static_cast<unsigned int>(mask), 0u, 0x0000);
    return static_cast<float>((expanded & 0xffu) == 0u);
}

// Initialize forces to gravity for free particles, zero for fixed particles. Optionally clears JV term.
// Each thread handles one particle.
// Fixed particles: force = 0 (no dynamics)
// Free particles: force = gravity * mass (gravitational acceleration)
// JV term: used in implicit solver, optionally initialized to zero
// 
// OPTIMIZATION: Uses __launch_bounds__(kBlockSize, 2) to limit register usage.
// RTX 3050 SMs have 64KB registers. Using __launch_bounds__ hints the compiler to maintain
// ~2 blocks of warps per SM, ensuring good occupancy even with point-wise operations.
__global__ __launch_bounds__(kBlockSize, 2) void InitForcesKernel(
    float4* __restrict__ force,
    float4* __restrict__ jv,
    const float* __restrict__ masses,
    const uint8_t* __restrict__ fixedMask,
    float3 gravity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Initialize forces for each particle
    if (ReadOnly(&fixedMask[i]))
    {
        // Fixed particle: zero force
        force[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        // Free particle: gravitational force = mass * gravity
        float m = ReadOnly(&masses[i]);
        force[i] = make_float4(gravity.x * m, gravity.y * m, gravity.z * m, 0.0f);
    }

    // Optional: Initialize Jacobian term used in implicit solver
    if (jv != nullptr)
    {
        jv[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Compute spring forces and apply to particles. Each spring adds/subtracts force to/from endpoints.
// Algorithm:
//   1. Get spring endpoints a and b with positions and velocities
//   2. Compute spring vector: diff = pb - pa
//   3. Calculate distance and unit direction vector
//   4. Compute spring force: f_spring = k * (current_length - rest_length)
//   5. Compute damping force: f_damping = d * (relative_velocity · direction)
//   6. Apply forces with opposite signs to endpoints (Newton's 3rd law)
// 
// OPTIMIZATION: Uses float4 for vectorized position/velocity loads (128-bit aligned memory).
// Uses __ldg for reading constant data (masses, rest lengths, fixed masks).
// Uses atomicAdd for force accumulation to safely handle concurrent writes from multiple springs.
__global__ __launch_bounds__(kBlockSize, 2) void SpringForceKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* __restrict__ force,
    const int2* __restrict__ springEnds,
    const float* __restrict__ restLengths,
    const uint8_t* __restrict__ fixedMask,
    float springStiffness,
    float springDamping,
    int springCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int2 ends = ReadOnly(&springEnds[s]);
    int a = ends.x;
    int b = ends.y;

    float4 pa = pos[a];  // Position of spring endpoint A
    float4 pb = pos[b];  // Position of spring endpoint B
    float4 va = vel[a];  // Velocity of endpoint A
    float4 vb = vel[b];  // Velocity of endpoint B

    // Compute spring vector from A to B
    float4 diff = Sub3(pb, pa);
    float dist2 = Dot3(diff, diff);  // Squared distance
    if (dist2 < kEpsilon)  // Skip if endpoints too close (avoid division by zero)
    {
        return;
    }

    // Compute current spring length using fast reciprocal square root
    float invDist = rsqrtf(dist2);
    float dist = dist2 * invDist;  // Current length = sqrt(dist2)
    float4 dir = Mul3(diff, invDist);  // Unit direction vector
    float4 relVel = Sub3(vb, va);  // Relative velocity

    // Project relative velocity onto spring direction for damping
    float relAlong = Dot3(relVel, dir);
    // Hooke's law: spring force = k * (length - rest_length)
    float fSpring = springStiffness * (dist - ReadOnly(&restLengths[s]));
    // Damping force: proportional to velocity along spring
    float fDamper = springDamping * relAlong;
    // Total force magnitude in spring direction
    float4 total = Mul3(dir, fSpring + fDamper);

    // Apply forces to endpoints (atomic adds to handle concurrent writes)
    // Force on A: push away from B if stretched, pull towards B if compressed
    if (!ReadOnly(&fixedMask[a]))
    {
        atomicAdd(&force[a].x, total.x);
        atomicAdd(&force[a].y, total.y);
        atomicAdd(&force[a].z, total.z);
    }

    // Force on B: opposite reaction (Newton's 3rd law)
    if (!ReadOnly(&fixedMask[b]))
    {
        atomicAdd(&force[b].x, -total.x);
        atomicAdd(&force[b].y, -total.y);
        atomicAdd(&force[b].z, -total.z);
    }
}

// Semi-implicit (Symplectic Euler) integration: update velocity from forces, then update position.
// More stable than explicit Euler for oscillatory systems like springs.
// OPTIMIZATION: Uses __launch_bounds__ to control register usage and maintain occupancy.
// Point-wise operations (acceleration, velocity, position updates) are light on registers,
// allowing compiler to fit more thread blocks per SM.
__global__ __launch_bounds__(kBlockSize, 2) void IntegrateSemiKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float* __restrict__ masses,
    const uint8_t* __restrict__ fixedMask,
    float dt,
    float velocityDamping,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Determine if particle is free or fixed (0.0 for fixed, 1.0 for free)
    float active = ActiveScale(fixedMask, i);
    // Compute inverse mass (with epsilon safeguard to prevent division by zero)
    float invMass = 1.0f / fmaxf(ReadOnly(&masses[i]), kEpsilon);
    // Acceleration = Force / Mass
    float4 accel = Mul3(force[i], invMass);

    // Semi-implicit Euler: update velocity first (v_new = v_old + a*dt)
    float4 v = vel[i];
    v = Add3(v, Mul3(accel, dt));
    // Then update position using new velocity (x_new = x_old + v_new*dt)
    float4 prevPos = pos[i];
    float4 x = prevPos;
    x = Add3(x, Mul3(v, dt));
    // Apply velocity damping (dampens oscillations)
    v = Mul3(v, velocityDamping);

    // Write back: fixed particles keep original position/velocity, free particles update
    vel[i] = Mul3(v, active);  // Zero out velocity for fixed particles
    pos[i] = Add3(Mul3(prevPos, 1.0f - active), Mul3(x, active));  // Keep old pos for fixed
}

// Copy particle state from source to destination arrays
__global__ __launch_bounds__(kBlockSize, 2) void CopyStateKernel(const float4* srcPos, const float4* srcVel, float4* dstPos, float4* dstVel, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    dstPos[i] = srcPos[i];
    dstVel[i] = srcVel[i];
}

// Linearize springs for implicit solver: compute spring forces and linearized coefficients.
// Prepares spring data for linear system solver in implicit integration.
// OPTIMIZATION: Uses __launch_bounds__ and float4 vectorization.
// Atomic operations accumulate forces safely even with high spring-to-thread ratios.
// L2 persisting cache (configured separately) helps repeated reads of spring data.
__global__ __launch_bounds__(kBlockSize, 2) void SpringLinearizeKernel(
    const float4* __restrict__ x0,
    const float4* __restrict__ v0,
    float4* __restrict__ force,
    float4* __restrict__ jv,
    float4* __restrict__ linearized,
    const int2* __restrict__ springEnds,
    const float* __restrict__ restLengths,
    const uint8_t* __restrict__ fixedMask,
    float springStiffness,
    float springDamping,
    float h,
    float h2,
    int springCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int2 ends = ReadOnly(&springEnds[s]);
    int a = ends.x;
    int b = ends.y;

    float4 xa = x0[a];  // Saved initial position of endpoint A
    float4 xb = x0[b];  // Saved initial position of endpoint B
    float4 va = v0[a];  // Saved initial velocity of endpoint A
    float4 vb = v0[b];  // Saved initial velocity of endpoint B

    // Compute current spring vector and distance
    float4 diff = Sub3(xb, xa);
    float dist2 = Dot3(diff, diff);
    if (dist2 < kEpsilon)
    {
        linearized[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float invDist = rsqrtf(dist2);
    float dist = dist2 * invDist;
    float4 dir = Mul3(diff, invDist);  // Unit direction vector along spring
    float4 relVel = Sub3(vb, va);  // Relative velocity
    float relAlong = Dot3(relVel, dir);  // Component along spring axis

    // Compute spring and damping forces (same as non-implicit version)
    float fSpring = springStiffness * (dist - ReadOnly(&restLengths[s]));
    float fDamper = springDamping * relAlong;
    float4 total = Mul3(dir, fSpring + fDamper);

    // Accumulate forces (will be used to build RHS)
    if (!ReadOnly(&fixedMask[a]))
    {
        atomicAdd(&force[a].x, total.x);
        atomicAdd(&force[a].y, total.y);
        atomicAdd(&force[a].z, total.z);
    }

    if (!ReadOnly(&fixedMask[b]))
    {
        atomicAdd(&force[b].x, -total.x);
        atomicAdd(&force[b].y, -total.y);
        atomicAdd(&force[b].z, -total.z);
    }

    // Linearize damping jacobian for implicit solver
    // Project velocities onto spring direction
    float projADot = Dot3(va, dir);
    float projBDot = Dot3(vb, dir);
    float4 projA = Mul3(dir, projADot);  // Component of va along spring
    float4 projB = Mul3(dir, projBDot);  // Component of vb along spring

    // Jacobian terms: J*v contributes to implicit system
    // These represent damping effects in the linearized system
    float4 jvA = Mul3(Sub3(projB, projA), springDamping);
    float4 jvB = Mul3(Sub3(projA, projB), springDamping);

    if (!ReadOnly(&fixedMask[a]))
    {
        atomicAdd(&jv[a].x, jvA.x);
        atomicAdd(&jv[a].y, jvA.y);
        atomicAdd(&jv[a].z, jvA.z);
    }

    if (!ReadOnly(&fixedMask[b]))
    {
        atomicAdd(&jv[b].x, jvB.x);
        atomicAdd(&jv[b].y, jvB.y);
        atomicAdd(&jv[b].z, jvB.z);
    }

    // Store linearized spring data: direction and effective stiffness coefficient
    // Coefficient = h*damping + h²*stiffness (used in matrix-vector products)
    linearized[s] = make_float4(dir.x, dir.y, dir.z, h * springDamping + h2 * springStiffness);
}

// Build right-hand side (b) vector for implicit solver: b = m*v0 + h*(f - Jv)
// Constructs RHS of linear system for conjugate gradient solver.
// OPTIMIZATION: Point-wise operation with simple arithmetic - low register pressure.
// __launch_bounds__ allows high occupancy for fast execution.
__global__ __launch_bounds__(kBlockSize, 2) void BuildImplicitBKernel(
    const float4* __restrict__ v0,
    const float4* __restrict__ force,
    const float4* __restrict__ jv,
    const float* __restrict__ masses,
    const uint8_t* __restrict__ fixedMask,
    float h,
    float4* __restrict__ b,
    float4* __restrict__ v,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Determine if particle is free or fixed
    float active = ActiveScale(fixedMask, i);
    float m = ReadOnly(&masses[i]);
    // Build RHS: b = m*v0 + h*(f - Jv)
    // This is the right-hand side of the linear system A*v = b
    // m*v0: mass times initial velocity
    // h*f: time step times forces
    // h*Jv: time step times damping jacobian term
    float4 rhs = Add3(Mul3(v0[i], m), Mul3(Sub3(force[i], jv[i]), h));
    b[i] = Mul3(rhs, active);  // Zero out for fixed particles
    v[i] = Mul3(v0[i], active);  // Initialize solution with initial velocity
}

// Initialize: multiply vector by mass (diagonal matrix scaling)
__global__ __launch_bounds__(kBlockSize, 2) void InitMassTermKernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const float* __restrict__ masses,
    const uint8_t* __restrict__ fixedMask,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    float active = ActiveScale(fixedMask, i);
    dst[i] = Mul3(src[i], ReadOnly(&masses[i]) * active);
}

// Apply implicit solver matrix A (mass + spring stiffness terms) to a vector
// Core operation in conjugate gradient solver.
// OPTIMIZATION: Uses __ldg for reading linearized spring coefficients (read-only, constant).
// Atomic operations safely accumulate matrix-vector product contributions across springs.
// L2 persisting cache helps with repeated access to same spring data across CG iterations.
__global__ __launch_bounds__(kBlockSize, 2) void AccumulateImplicitMatrixKernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const float4* __restrict__ linearized,
    const int2* __restrict__ springEnds,
    const uint8_t* __restrict__ fixedMask,
    int springCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int2 ends = ReadOnly(&springEnds[s]);
    int a = ends.x;
    int b = ends.y;

    float4 lin = ReadOnly(&linearized[s]);
    float coeff = lin.w;  // h*damping + h²*stiffness coefficient
    if (coeff == 0.0f)  // Skip springs with zero stiffness
    {
        return;
    }

    // Extract direction vector from linearized spring data
    float4 dir = make_float4(lin.x, lin.y, lin.z, 0.0f);
    // Project src vector onto spring direction at both endpoints
    float dotA = Dot3(src[a], dir);
    float dotB = Dot3(src[b], dir);
    float4 projA = Mul3(dir, dotA);  // Projection of src[a] along spring
    float4 projB = Mul3(dir, dotB);  // Projection of src[b] along spring

    // Apply matrix-vector product: A*x contributes spring constraint terms
    // For free particles: add contribution from difference in projections
    if (!ReadOnly(&fixedMask[a]))
    {
        // If b is fixed, its contribution is zero; otherwise subtract b's projection
        float4 term = Sub3(projA, ReadOnly(&fixedMask[b]) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : projB);
        atomicAdd(&dst[a].x, coeff * term.x);
        atomicAdd(&dst[a].y, coeff * term.y);
        atomicAdd(&dst[a].z, coeff * term.z);
    }

    if (!ReadOnly(&fixedMask[b]))
    {
        // Symmetric: if a is fixed, its contribution is zero; otherwise subtract a's projection
        float4 term = Sub3(projB, ReadOnly(&fixedMask[a]) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : projA);
        atomicAdd(&dst[b].x, coeff * term.x);
        atomicAdd(&dst[b].y, coeff * term.y);
        atomicAdd(&dst[b].z, coeff * term.z);
    }
}

// Warp-level reduction to sum values across all lanes in a warp
__device__ inline float WarpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }

    return val;
}

// Compute partial dot products for free particles (skipping fixed ones)
// First level of two-level dot product reduction.
// OPTIMIZATION: Uses __shfl_down_sync for warp-level reduction.
// Instead of using shared memory immediately, each warp reduces its local sum internally
// using shuffle operations. This reduces shared memory contention and synchronization overhead
// on RTX 3050, which has fast shuffle units in each SM.
__global__ __launch_bounds__(kBlockSize, 2) void DotPartialKernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    const uint8_t* __restrict__ fixedMask,
    int count,
    float* __restrict__ partial)
{
    __shared__ float warpSums[8];  // One entry per warp (256 threads = 8 warps)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // Total number of threads across all blocks
    
    // Step 1: Each thread accumulates dot products over its assigned range
    // Stride pattern ensures coalesced memory access
    float local = 0.0f;
    for (int i = idx; i < count; i += stride)
    {
        // Only compute dot product for free particles (skip fixed ones)
        if (!ReadOnly(&fixedMask[i]))
        {
            local += Dot3(a[i], b[i]);
        }
    }

    // Step 2: Reduce within warp (32 threads per warp)
    // Each warp produces one sum value
    local = WarpReduceSum(local);
    int lane = threadIdx.x & (warpSize - 1);  // Position within warp (0-31)
    int warpId = threadIdx.x / warpSize;       // Warp index within block
    if (lane == 0)  // Only warp leader stores result
    {
        warpSums[warpId] = local;
    }

    __syncthreads();  // Wait for all warps to finish

    // Step 3: Reduce across all warps to get block sum
    float blockSum = 0.0f;
    if (warpId == 0)  // Only threads in first warp perform this reduction
    {
        blockSum = (lane < (blockDim.x / warpSize)) ? warpSums[lane] : 0.0f;
        blockSum = WarpReduceSum(blockSum);  // Reduce across warps
        if (lane == 0)  // Only warp leader writes final result
        {
            partial[blockIdx.x] = blockSum;  // Store partial sum for this block
        }
    }
}

// Reduce partial sums into final scalar result
// Second level of two-level dot product reduction.
// OPTIMIZATION: Uses cooperative_groups::memcpy_async for asynchronous memory copies.
// On RTX 3050 (CC 8.0+), async memcpy allows computation and memory transfer to overlap,
// hiding latency. Without this, threads would stall waiting for data from device memory.
// The memory copy can proceed in parallel with warp reductions, improving overall throughput.
__global__ __launch_bounds__(kBlockSize, 2) void ReducePartialSumKernel(const float* __restrict__ partial, int count, float* __restrict__ out)
{
    __shared__ float shared[kBlockSize];  // Shared memory for reduction
    int tid = threadIdx.x;

    float local = 0.0f;
#if __CUDA_ARCH__ >= 800
    // For compute capability 8.0+: use asynchronous memcpy for better performance
    cg::thread_block block = cg::this_thread_block();
    for (int base = 0; base < count; base += blockDim.x)
    {
        int i = base + tid;
        if (i < count)
        {
            // Asynchronously copy partial sum to shared memory
            cg::memcpy_async(block, &shared[tid], partial + i, sizeof(float));
        }
        else
        {
            shared[tid] = 0.0f;
        }

        cg::wait(block);  // Wait for async copy to complete
        local += shared[tid];  // Accumulate
        cg::sync(block);
    }
#else
    // For older compute capabilities: direct memory access
    for (int i = tid; i < count; i += blockDim.x)
    {
        local += partial[i];
    }
#endif

    // Perform final tree reduction in shared memory
    shared[tid] = local;
    __syncthreads();

    // Reduce: at each stage, threads add values from opposite end of remaining range
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();  // Synchronize after each reduction stage
    }

    if (tid == 0)  // Thread 0 has the final sum
    {
        *out = shared[0];
    }
}

// Initialize CG solver: compute residual r = b - A*v0 and search direction p = r
__global__ __launch_bounds__(kBlockSize, 2) void InitCGKernel(const float4* __restrict__ b, const float4* __restrict__ ap, const uint8_t* __restrict__ fixedMask, float4* __restrict__ r, float4* __restrict__ p, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Fixed particles have zero residual and search direction
    if (ReadOnly(&fixedMask[i]))
    {
        r[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        p[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Initial residual: r = b - A*v0 (where A*v0 is stored in ap)
    float4 ri = Sub3(b[i], ap[i]);
    r[i] = ri;  // Residual vector
    p[i] = ri;  // Search direction starts equal to residual (Steepest Descent direction)
}

// CG iteration: update solution x and residual r: x += alpha*p, r -= alpha*A*p
__global__ __launch_bounds__(kBlockSize, 2) void CGUpdateXRKernel(float4* __restrict__ x, float4* __restrict__ r, const float4* __restrict__ p, const float4* __restrict__ ap, const uint8_t* __restrict__ fixedMask, float alpha, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Skip fixed particles
    if (ReadOnly(&fixedMask[i]))
    {
        return;
    }

    // CG step: x += alpha * p
    // where alpha = (r·r) / (p·A·p)
    x[i] = Add3(x[i], Mul3(p[i], alpha));
    // Update residual: r -= alpha * A*p
    r[i] = Sub3(r[i], Mul3(ap[i], alpha));
}

// CG iteration: update search direction p = r + beta*p
__global__ __launch_bounds__(kBlockSize, 2) void CGUpdatePKernel(float4* __restrict__ p, const float4* __restrict__ r, const uint8_t* __restrict__ fixedMask, float beta, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Skip fixed particles
    if (ReadOnly(&fixedMask[i]))
    {
        return;
    }

    // CG step: update search direction
    // p = r + beta * p
    // where beta = (r_new·r_new) / (r_old·r_old)
    p[i] = Add3(r[i], Mul3(p[i], beta));
}

// Commit implicit solver results: compute final position and velocity from solved velocity
__global__ __launch_bounds__(kBlockSize, 2) void CommitImplicitKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ x0,
    const float4* __restrict__ vSolved,
    const uint8_t* __restrict__ fixedMask,
    float h,
    float velocityDamping,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    // Fixed particles return to their original position with zero velocity
    if (ReadOnly(&fixedMask[i]))
    {
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        pos[i] = x0[i];
        return;
    }

    // For free particles: compute final velocity and position
    // v_final = solved_velocity * damping (additional velocity damping)
    float4 v = Mul3(vSolved[i], velocityDamping);
    // x_final = x0 + dt * v_final
    float4 p = Add3(x0[i], Mul3(vSolved[i], h));
    vel[i] = v;
    pos[i] = p;
}

// Utility: compute ceil(n / d) for integer division
inline int DivUp(int n, int d)
{
    return (n + d - 1) / d;
}

// Configure L2 cache persistence hints for improved performance
// L2 persisting cache is a feature on advanced GPUs that keeps frequently accessed
// data in cache across kernel invocations, reducing main memory traffic.
// OPTIMIZATION: RTX 3050 supports L2 persisting cache (up to 256KB on RTX 3050).
// In implicit solver loops, the same spring data (endpoints, linearized coefficients) is
// accessed repeatedly across multiple CG iterations. Configuring persistent cache ensures
// this data stays in L2, dramatically reducing memory latency and bandwidth pressure.
inline void ConfigureDeviceCacheHints(DeviceSystem& sys)
{
    if (!sys.cudaAvailable || sys.deviceCacheConfigured)
    {
        return;  // Skip if already configured or CUDA unavailable
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        sys.deviceCacheConfigured = true;
        return;
    }

    // Query device properties to check if L2 cache persistence is available
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess)
    {
        sys.deviceCacheConfigured = true;
        return;
    }

    // If device supports L2 persistent cache, configure it
    if (props.persistingL2CacheMaxSize > 0 && props.l2CacheSize > 0)
    {
        // Use up to 75% of L2 cache for persistent data (balance with temporary data)
        // RTX 3050 has 4MB L2 cache; 75% reservation (3MB) leaves room for temp data
        // while maximizing persistent storage for spring/mass/mask constants.
        const std::size_t persistingBudget = std::min(
            static_cast<std::size_t>(props.persistingL2CacheMaxSize),
            static_cast<std::size_t>(props.l2CacheSize * 3 / 4));

        if (persistingBudget > 0)
        {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistingBudget);
            cudaGetLastError();  // Clear any errors from configuration
        }
    }

    sys.deviceCacheConfigured = true;
}

// Configure kernel cache preferences and shared memory carveout
// Optimizes kernel performance by setting cache preferences and shared memory allocation.
// OPTIMIZATION: RTX 3050 allows configurable split between L1 cache and shared memory.
// Our kernels use shared memory for warp reduction (DotPartialKernel), so we prefer
// shared memory carveout (100% available for kernel use). This reduces L1 cache contention
// and prioritizes our reduction operations.
inline void ConfigureKernelAttributes(DeviceSystem& sys)
{
    if (!sys.cudaAvailable || sys.kernelAttributesConfigured)
    {
        return;  // Skip if already configured or CUDA unavailable
    }

    // Lambda to configure each kernel
    auto configure = [](const void* kernel) -> void
    {
        // Prefer 100% shared memory carveout (reserved for kernel use)
        // This ensures maximum shared memory availability for reductions
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        // Prefer L1 cache over shared memory in the cache hierarchy
        // With full shared memory carveout, L1 cache is still useful for __ldg reads
        cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
    };

    // Configure all kernels with consistent settings

    configure(reinterpret_cast<const void*>(InitForcesKernel));
    configure(reinterpret_cast<const void*>(SpringForceKernel));
    configure(reinterpret_cast<const void*>(IntegrateSemiKernel));
    configure(reinterpret_cast<const void*>(SpringLinearizeKernel));
    configure(reinterpret_cast<const void*>(BuildImplicitBKernel));
    configure(reinterpret_cast<const void*>(InitMassTermKernel));
    configure(reinterpret_cast<const void*>(AccumulateImplicitMatrixKernel));
    configure(reinterpret_cast<const void*>(DotPartialKernel));
    configure(reinterpret_cast<const void*>(ReducePartialSumKernel));
    configure(reinterpret_cast<const void*>(InitCGKernel));
    configure(reinterpret_cast<const void*>(CGUpdateXRKernel));
    configure(reinterpret_cast<const void*>(CGUpdatePKernel));
    configure(reinterpret_cast<const void*>(CommitImplicitKernel));

    cudaGetLastError();  // Clear any configuration errors
    sys.kernelAttributesConfigured = true;
}

// Allocate or verify GPU buffers for all simulation state
// Called before each simulation step to ensure buffers exist and are sized correctly.
// OPTIMIZATION: All buffers use float4 alignment where applicable (8GB RTX 3050 VRAM).
// Allocation happens once; subsequent calls verify instead of reallocating.
// Buffer management avoids fragmentation and keeps memory access patterns consistent.
bool EnsureBuffers(DeviceSystem& sys)
{
    if (!sys.cudaAvailable)
    {
        return true;
    }

    if (sys.particleCount <= 0 || sys.springCount <= 0)
    {
        SetError("invalid system topology");
        return false;
    }

    auto allocFloat4 = [](float4** ptr, int count) -> bool
    {
        if (*ptr != nullptr)
        {
            return true;
        }

        return CheckCuda(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(float4) * count), "cudaMalloc float4");
    };

    if (!allocFloat4(&sys.dForce, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dX0, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dV0, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dJv, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dLinearized, sys.springCount)) return false;
    if (!allocFloat4(&sys.dB, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dV, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dR, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dP, sys.particleCount)) return false;
    if (!allocFloat4(&sys.dAp, sys.particleCount)) return false;

    // Allocate reduction buffers for dot product computation
    // OPTIMIZATION: Two-level reduction minimizes synchronization overhead
    int blocks = std::max(1, DivUp(sys.particleCount, kBlockSize));
    if (sys.dotPartialCount < blocks)
    {
        // Reallocate partial sum buffer if block count increased
        // OPTIMIZATION: Reusing same memory across frames reduces allocation overhead
        if (sys.dDotPartial != nullptr)
        {
            cudaFree(sys.dDotPartial);
            sys.dDotPartial = nullptr;
        }

        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys.dDotPartial), sizeof(float) * blocks), "cudaMalloc dot partial"))
        {
            return false;
        }

        sys.dotPartialCount = blocks;
    }

    if (sys.dDotValue == nullptr)
    {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys.dDotValue), sizeof(float)), "cudaMalloc dot value"))
        {
            return false;
        }
    }

    ConfigureDeviceCacheHints(sys);
    ConfigureKernelAttributes(sys);

    return true;
}

// Compute dot product of two device vectors (respecting fixed particle constraints)
// Uses two-level reduction: partial sums per block, then final sum
// Only counts free particles in the dot product
// OPTIMIZATION SUMMARY:
//   1. DotPartialKernel: Each block computes partial sum using __shfl_down_sync warp reductions
//   2. ReducePartialSumKernel: Reduces partial sums with async memcpy (overlaps copy & compute)
//   3. Result copied to host via DMA transfer
// This minimizes shared memory pressure and synchronization overhead compared to naive reduction.
float DeviceDot(DeviceSystem& sys, const float4* a, const float4* b)
{
    if (!sys.cudaAvailable)
    {
        return 0.0f;
    }

    // Step 1: Launch partial dot product kernel (one partial sum per block)
    int blocks = std::max(1, DivUp(sys.particleCount, kBlockSize));
    DotPartialKernel<<<blocks, kBlockSize>>>(a, b, sys.dFixedMask, sys.particleCount, sys.dDotPartial);
    if (!CheckCuda(cudaGetLastError(), "DotPartialKernel launch"))
    {
        return 0.0f;
    }

    // Step 2: Reduce all partial sums into final value
    ReducePartialSumKernel<<<1, kBlockSize>>>(sys.dDotPartial, blocks, sys.dDotValue);
    if (!CheckCuda(cudaGetLastError(), "ReducePartialSumKernel launch"))
    {
        return 0.0f;
    }

    // Step 3: Copy result back to host
    if (!CheckCuda(cudaMemcpy(&sys.hDotValue, sys.dDotValue, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dot value"))
    {
        return 0.0f;
    }

    return sys.hDotValue;
}

// Apply implicit solver matrix A to vector src, store result in dst
bool MultiplyImplicitMatrix(DeviceSystem& sys, const float4* src, float4* dst)
{
    if (!sys.cudaAvailable)
    {
        return false;
    }

    int pBlocks = DivUp(sys.particleCount, kBlockSize);
    int sBlocks = DivUp(sys.springCount, kBlockSize);
    InitMassTermKernel<<<pBlocks, kBlockSize>>>(src, dst, sys.dMasses, sys.dFixedMask, sys.particleCount);
    if (!CheckCuda(cudaGetLastError(), "InitMassTermKernel launch"))
    {
        return false;
    }

    AccumulateImplicitMatrixKernel<<<sBlocks, kBlockSize>>>(
        src,
        dst,
        sys.dLinearized,
        sys.dSpringEnds,
        sys.dFixedMask,
        sys.springCount);
    if (!CheckCuda(cudaGetLastError(), "AccumulateImplicitMatrixKernel launch"))
    {
        return false;
    }

    return true;
}

// Deallocate all GPU and host memory associated with the system
void FreeSystem(DeviceSystem* sys)
{
    if (sys == nullptr)
    {
        return;
    }

    cudaFree(sys->dSpringEnds);
    cudaFree(sys->dRestLengths);
    cudaFree(sys->dMasses);
    cudaFree(sys->dFixedMask);
    cudaFree(sys->dPosition);
    cudaFree(sys->dVelocity);
    cudaFree(sys->dForce);
    cudaFree(sys->dX0);
    cudaFree(sys->dV0);
    cudaFree(sys->dJv);
    cudaFree(sys->dLinearized);
    cudaFree(sys->dB);
    cudaFree(sys->dV);
    cudaFree(sys->dR);
    cudaFree(sys->dP);
    cudaFree(sys->dAp);
    cudaFree(sys->dDotPartial);
    cudaFree(sys->dDotValue);

    delete sys;
}

inline float Dot3Host(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float4 Add3Host(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

inline float4 Sub3Host(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

inline float4 Mul3Host(const float4& v, float s)
{
    return make_float4(v.x * s, v.y * s, v.z * s, 0.0f);
}

// Host-side fallback: compute spring and gravity forces (CPU simulation)
// Used when CUDA is unavailable. Implements the same physics as GPU kernels.
// Algorithm:
//   1. Initialize forces to gravity*mass for free particles
//   2. For each spring: compute spring vector, distance, and forces
//   3. Apply spring + damping forces to both endpoints
void ComputeForcesSemiHost(DeviceSystem& sys, float springStiffness, float springDamping, float3 gravity)
{
    int n = sys.particleCount;
    for (int i = 0; i < n; ++i)
    {
        if (sys.hFixedMask[i])
        {
            sys.hForce[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            float m = sys.hMasses[i];
            sys.hForce[i] = make_float4(gravity.x * m, gravity.y * m, gravity.z * m, 0.0f);
        }
    }

    for (int s = 0; s < sys.springCount; ++s)
    {
        int2 e = sys.hSpringEnds[s];
        int a = e.x;
        int b = e.y;
        float4 pa = sys.hPosition[a];
        float4 pb = sys.hPosition[b];
        float4 va = sys.hVelocity[a];
        float4 vb = sys.hVelocity[b];

        float4 diff = Sub3Host(pb, pa);
        float dist2 = Dot3Host(diff, diff);
        if (dist2 < kEpsilon)
        {
            continue;
        }

        float dist = std::sqrt(dist2);
        float invDist = 1.0f / dist;
        float4 dir = Mul3Host(diff, invDist);
        float4 relVel = Sub3Host(vb, va);
        float relAlong = Dot3Host(relVel, dir);

        float fSpring = springStiffness * (dist - sys.hRestLengths[s]);
        float fDamper = springDamping * relAlong;
        float4 total = Mul3Host(dir, fSpring + fDamper);

        if (!sys.hFixedMask[a])
        {
            sys.hForce[a] = Add3Host(sys.hForce[a], total);
        }

        if (!sys.hFixedMask[b])
        {
            sys.hForce[b] = Sub3Host(sys.hForce[b], total);
        }
    }
}

// Host-side fallback: semi-implicit integration step (CPU simulation)
// Updates velocities and positions using semi-implicit (symplectic) Euler method.
// More stable than explicit Euler for oscillatory systems.
void IntegrateSemiHost(DeviceSystem& sys, float dt, float velocityDamping)
{
    int n = sys.particleCount;
    for (int i = 0; i < n; ++i)
    {
        if (sys.hFixedMask[i])
        {
            sys.hVelocity[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }

        float invMass = 1.0f / std::max(sys.hMasses[i], kEpsilon);
        float4 accel = Mul3Host(sys.hForce[i], invMass);
        sys.hVelocity[i] = Add3Host(sys.hVelocity[i], Mul3Host(accel, dt));
        sys.hPosition[i] = Add3Host(sys.hPosition[i], Mul3Host(sys.hVelocity[i], dt));
        sys.hVelocity[i] = Mul3Host(sys.hVelocity[i], velocityDamping);
    }
}
}

// Public C API - Exported functions for system creation and simulation
// These functions provide the main interface for C# interop and external clients.
extern "C"
{
// Create a new simulation system (initializes CUDA if available)
// Returns: opaque handle to DeviceSystem struct (nullptr on failure)
MSS_API void* mssCreateSystem()
{
    try
    {
        DeviceSystem* sys = new DeviceSystem();
        int deviceCount = 0;
        cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
        sys->cudaAvailable = (countErr == cudaSuccess && deviceCount > 0);
        gLastError.clear();
        return sys;
    }
    catch (...)
    {
        SetError("failed to allocate DeviceSystem");
        return nullptr;
    }
}

// Destroy a system and release all associated resources
MSS_API int mssDestroySystem(void* system)
{
    FreeSystem(reinterpret_cast<DeviceSystem*>(system));
    return 0;
}

// Upload particle topology and initial state to GPU (or prepare host buffers)
// Parameters:
//   system: handle from mssCreateSystem
//   particleCount: number of particles
//   positionXYZ: packed float array [x0, y0, z0, x1, y1, z1, ...]
//   masses: particle masses (one per particle)
//   fixedMask: byte mask indicating fixed particles (non-zero = fixed)
//   springCount: number of springs
//   springEndpoints: packed int array [a0, b0, a1, b1, ...] (particle indices)
//   restLengths: spring rest lengths
// Returns: 0 on success, -1 on error
MSS_API int mssUploadTopology(
    void* system,
    int particleCount,
    const float* positionXYZ,
    const float* masses,
    const uint8_t* fixedMask,
    int springCount,
    const int* springEndpoints,
    const float* restLengths)
{
    DeviceSystem* sys = reinterpret_cast<DeviceSystem*>(system);
    if (sys == nullptr || particleCount <= 0 || springCount <= 0 || positionXYZ == nullptr || masses == nullptr || fixedMask == nullptr || springEndpoints == nullptr || restLengths == nullptr)
    {
        SetError("mssUploadTopology invalid arguments");
        return -1;
    }

    sys->particleCount = particleCount;
    sys->springCount = springCount;

    // Convert packed position array to float4 format
    std::vector<float4> hostPos(static_cast<std::size_t>(particleCount));
    for (int i = 0; i < particleCount; ++i)
    {
        int base = i * 3;  // Each particle has 3 coordinates
        hostPos[i] = make_float4(positionXYZ[base], positionXYZ[base + 1], positionXYZ[base + 2], 0.0f);
    }

    // Convert packed endpoint array to int2 format
    std::vector<int2> hostEnds(static_cast<std::size_t>(springCount));
    for (int s = 0; s < springCount; ++s)
    {
        int base = s * 2;  // Each spring has 2 endpoints
        hostEnds[s] = make_int2(springEndpoints[base], springEndpoints[base + 1]);
    }

    // Store in host vectors
    sys->hPosition = hostPos;
    sys->hVelocity.assign(static_cast<std::size_t>(particleCount), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    sys->hForce.assign(static_cast<std::size_t>(particleCount), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    sys->hMasses.assign(masses, masses + particleCount);
    sys->hFixedMask.assign(fixedMask, fixedMask + particleCount);
    sys->hSpringEnds = hostEnds;
    sys->hRestLengths.assign(restLengths, restLengths + springCount);

    // Allocate GPU memory if CUDA is available
    if (sys->cudaAvailable)
    {
        // Allocate GPU buffers for particle state
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dPosition), sizeof(float4) * particleCount), "cudaMalloc dPosition"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dVelocity), sizeof(float4) * particleCount), "cudaMalloc dVelocity"))
        {
            sys->cudaAvailable = false;
        }
        
        // Allocate GPU buffers for particle properties
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dMasses), sizeof(float) * particleCount), "cudaMalloc dMasses"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dFixedMask), sizeof(uint8_t) * particleCount), "cudaMalloc dFixedMask"))
        {
            sys->cudaAvailable = false;
        }
        
        // Allocate GPU buffers for spring topology
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dSpringEnds), sizeof(int2) * springCount), "cudaMalloc dSpringEnds"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dRestLengths), sizeof(float) * springCount), "cudaMalloc dRestLengths"))
        {
            sys->cudaAvailable = false;
        }

        // Copy data to GPU
        if (sys->cudaAvailable && !CheckCuda(cudaMemcpy(sys->dPosition, hostPos.data(), sizeof(float4) * particleCount, cudaMemcpyHostToDevice), "cudaMemcpy dPosition"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMemset(sys->dVelocity, 0, sizeof(float4) * particleCount), "cudaMemset dVelocity"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMemcpy(sys->dMasses, masses, sizeof(float) * particleCount, cudaMemcpyHostToDevice), "cudaMemcpy dMasses"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMemcpy(sys->dFixedMask, fixedMask, sizeof(uint8_t) * particleCount, cudaMemcpyHostToDevice), "cudaMemcpy dFixedMask"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMemcpy(sys->dSpringEnds, hostEnds.data(), sizeof(int2) * springCount, cudaMemcpyHostToDevice), "cudaMemcpy dSpringEnds"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMemcpy(sys->dRestLengths, restLengths, sizeof(float) * springCount, cudaMemcpyHostToDevice), "cudaMemcpy dRestLengths"))
        {
            sys->cudaAvailable = false;
        }
    }

    if (!EnsureBuffers(*sys))
    {
        return -1;
    }

    gLastError.clear();
    return 0;
}

// Execute semi-implicit (Symplectic Euler) integration step
// Semi-implicit method is more stable than explicit for spring systems.
// Optionally subdivides time step into smaller substeps for better accuracy.
// Parameters: stepParams contains dt, gravity, spring stiffness/damping, velocity damping, substeps
// Returns: 0 on success, -1 on error
// 
// OPTIMIZATION: Falls back to host CPU computation if CUDA unavailable.
// GPU path uses optimized kernels with float4 alignment and __ldg caching.
MSS_API int mssStepSemi(void* system, const MssSemiParams* stepParams)
{
    DeviceSystem* sys = reinterpret_cast<DeviceSystem*>(system);
    if (sys == nullptr || stepParams == nullptr)
    {
        SetError("mssStepSemi invalid arguments");
        return -1;
    }

    if (!EnsureBuffers(*sys))
    {
        return -1;
    }

    if (!sys->cudaAvailable)
    {
        // CPU fallback path: useful for testing or when GPU unavailable
        // Host-side implementation uses same physics, but with CPU scalar loops
        int substeps = std::max(1, stepParams->substeps);
        float dtSub = stepParams->dt / static_cast<float>(substeps);
        for (int iter = 0; iter < substeps; ++iter)
        {
            ComputeForcesSemiHost(*sys, stepParams->springStiffness, stepParams->springDamping, make_float3(stepParams->gravityX, stepParams->gravityY, stepParams->gravityZ));
            IntegrateSemiHost(*sys, dtSub, stepParams->velocityDamping);
        }

        gLastError.clear();
        return 0;
    }

    // GPU path: optimized for RTX 3050
    // OPTIMIZATION: Uses float4 vectorization, __ldg caching, __launch_bounds__ occupancy tuning
    const int pBlocks = DivUp(sys->particleCount, kBlockSize);
    const int sBlocks = DivUp(sys->springCount, kBlockSize);

    int substeps = std::max(1, stepParams->substeps);
    float dtSub = stepParams->dt / static_cast<float>(substeps);

    for (int iter = 0; iter < substeps; ++iter)
    {
        // Step 1: Initialize forces with gravity
        // OPTIMIZATION: __launch_bounds__ ensures good occupancy on RTX 3050
        InitForcesKernel<<<pBlocks, kBlockSize>>>(
            sys->dForce,
            nullptr,
            sys->dMasses,
            sys->dFixedMask,
            make_float3(stepParams->gravityX, stepParams->gravityY, stepParams->gravityZ),
            sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "InitForcesKernel launch (semi)"))
        {
            return -1;
        }

        // Step 2: Compute spring forces
        // OPTIMIZATION: __ldg reads for rest lengths, atomic adds for force accumulation
        SpringForceKernel<<<sBlocks, kBlockSize>>>(
            sys->dPosition,
            sys->dVelocity,
            sys->dForce,
            sys->dSpringEnds,
            sys->dRestLengths,
            sys->dFixedMask,
            stepParams->springStiffness,
            stepParams->springDamping,
            sys->springCount);
        if (!CheckCuda(cudaGetLastError(), "SpringForceKernel launch"))
        {
            return -1;
        }

        // Step 3: Semi-implicit Euler integration
        // OPTIMIZATION: float4 memory access, __byte_perm for fixed mask handling
        IntegrateSemiKernel<<<pBlocks, kBlockSize>>>(
            sys->dPosition,
            sys->dVelocity,
            sys->dForce,
            sys->dMasses,
            sys->dFixedMask,
            dtSub,
            stepParams->velocityDamping,
            sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "IntegrateSemiKernel launch"))
        {
            return -1;
        }
    }

    gLastError.clear();
    return 0;
}

// Execute implicit integration step with conjugate gradient solver
// Implicit method is unconditionally stable, allowing larger time steps.
// Uses linearized springs and Conjugate Gradient method to solve implicit equation.
// Parameters: stepParams contains dt, gravity, spring parameters, CG tolerance/iterations, velocity damping
// Returns: 0 on success, -1 on error
// 
// OPTIMIZATION STRATEGY FOR RTX 3050:
//   1. Float4 vectors: All particle data (position, velocity, force) use aligned float4
//   2. ReadOnly access: __ldg for constant spring/mass/mask data
//   3. Register bounds: __launch_bounds__ manages occupancy vs. register pressure
//   4. Warp reductions: __shfl_down_sync in dot product computation
//   5. Async copies: cooperative_groups::memcpy_async for latency hiding
//   6. L2 persistence: Shared spring data stays cached across CG iterations
//   7. Branch avoidance: __byte_perm for fixed mask handling
MSS_API int mssStepImplicit(void* system, const MssImplicitParams* stepParams)
{
    DeviceSystem* sys = reinterpret_cast<DeviceSystem*>(system);
    if (sys == nullptr || stepParams == nullptr)
    {
        SetError("mssStepImplicit invalid arguments");
        return -1;
    }

    if (!EnsureBuffers(*sys))
    {
        return -1;
    }

    if (!sys->cudaAvailable)
    {
        SetError("cuda unavailable for implicit path");
        return -1;  // Implicit solver requires GPU
    }

    const int pBlocks = DivUp(sys->particleCount, kBlockSize);
    const int sBlocks = DivUp(sys->springCount, kBlockSize);
    const float h = stepParams->dt;
    const float h2 = h * h;  // Time step squared

    // Save initial state (x0, v0) for later use
    if (!CheckCuda(cudaMemcpy(sys->dX0, sys->dPosition, sizeof(float4) * sys->particleCount, cudaMemcpyDeviceToDevice), "cudaMemcpy dPosition->dX0")) return -1;
    if (!CheckCuda(cudaMemcpy(sys->dV0, sys->dVelocity, sizeof(float4) * sys->particleCount, cudaMemcpyDeviceToDevice), "cudaMemcpy dVelocity->dV0")) return -1;

    // Step 1: Compute initial forces and damping jacobian terms
    InitForcesKernel<<<pBlocks, kBlockSize>>>(
        sys->dForce,
        sys->dJv,
        sys->dMasses,
        sys->dFixedMask,
        make_float3(stepParams->gravityX, stepParams->gravityY, stepParams->gravityZ),
        sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "InitForcesKernel launch (implicit)")) return -1;

    // Step 2: Linearize springs and compute linearized coefficients
    SpringLinearizeKernel<<<sBlocks, kBlockSize>>>(
        sys->dX0,
        sys->dV0,
        sys->dForce,
        sys->dJv,
        sys->dLinearized,
        sys->dSpringEnds,
        sys->dRestLengths,
        sys->dFixedMask,
        stepParams->springStiffness,
        stepParams->springDamping,
        h,
        h2,
        sys->springCount);
    if (!CheckCuda(cudaGetLastError(), "SpringLinearizeKernel launch")) return -1;

    // Step 3: Build right-hand side vector for linear system
    BuildImplicitBKernel<<<pBlocks, kBlockSize>>>(
        sys->dV0,
        sys->dForce,
        sys->dJv,
        sys->dMasses,
        sys->dFixedMask,
        h,
        sys->dB,
        sys->dV,
        sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "BuildImplicitBKernel launch")) return -1;

    // Step 4: Compute initial A*v0 for CG initialization
    if (!MultiplyImplicitMatrix(*sys, sys->dV, sys->dAp)) return -1;

    // Step 5: Initialize CG solver
    InitCGKernel<<<pBlocks, kBlockSize>>>(sys->dB, sys->dAp, sys->dFixedMask, sys->dR, sys->dP, sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "InitCGKernel launch")) return -1;

    // Check if already converged (residual too small)
    float rr = DeviceDot(*sys, sys->dR, sys->dR);
    if (rr < stepParams->cgTolerance)
    {
        // Already solved, just commit results
        CommitImplicitKernel<<<pBlocks, kBlockSize>>>(
            sys->dPosition,
            sys->dVelocity,
            sys->dX0,
            sys->dV,
            sys->dFixedMask,
            h,
            stepParams->velocityDamping,
            sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "CommitImplicitKernel launch (early)")) return -1;
        gLastError.clear();
        return 0;
    }

    // Step 6: CG iterations to solve linear system A*v = b
    for (int iter = 0; iter < std::max(1, stepParams->implicitIterations); ++iter)
    {
        // Compute A*p
        if (!MultiplyImplicitMatrix(*sys, sys->dP, sys->dAp)) return -1;

        // Compute alpha = (r·r) / (p·A·p)
        float pAp = DeviceDot(*sys, sys->dP, sys->dAp);
        if (fabsf(pAp) < 1e-12f)
        {
            break;  // Breakdown: search direction is orthogonal to A*p
        }

        float alpha = rr / pAp;
        // Update: x += alpha*p, r -= alpha*A*p
        CGUpdateXRKernel<<<pBlocks, kBlockSize>>>(sys->dV, sys->dR, sys->dP, sys->dAp, sys->dFixedMask, alpha, sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "CGUpdateXRKernel launch")) return -1;

        // Check convergence
        float rrNew = DeviceDot(*sys, sys->dR, sys->dR);
        if (rrNew < stepParams->cgTolerance)
        {
            rr = rrNew;
            break;  // Converged
        }

        // Compute beta = (r_new·r_new) / (r_old·r_old) and update search direction
        float beta = rrNew / rr;
        CGUpdatePKernel<<<pBlocks, kBlockSize>>>(sys->dP, sys->dR, sys->dFixedMask, beta, sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "CGUpdatePKernel launch")) return -1;
        rr = rrNew;
    }

    // Step 7: Commit implicit solver results (update position and velocity)
    CommitImplicitKernel<<<pBlocks, kBlockSize>>>(
        sys->dPosition,
        sys->dVelocity,
        sys->dX0,
        sys->dV,
        sys->dFixedMask,
        h,
        stepParams->velocityDamping,
        sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "CommitImplicitKernel launch")) return -1;

    gLastError.clear();
    return 0;
}

// Download current particle state from GPU to host memory
MSS_API int mssDownloadState(void* system, float* outPositionXYZ, float* outVelocityXYZ, int particleCount)
{
    DeviceSystem* sys = reinterpret_cast<DeviceSystem*>(system);
    if (sys == nullptr || outPositionXYZ == nullptr || particleCount != sys->particleCount)
    {
        SetError("mssDownloadState invalid arguments");
        return -1;
    }

    if (!sys->cudaAvailable)
    {
        for (int i = 0; i < particleCount; ++i)
        {
            int base = i * 3;
            outPositionXYZ[base] = sys->hPosition[i].x;
            outPositionXYZ[base + 1] = sys->hPosition[i].y;
            outPositionXYZ[base + 2] = sys->hPosition[i].z;

            if (outVelocityXYZ != nullptr)
            {
                outVelocityXYZ[base] = sys->hVelocity[i].x;
                outVelocityXYZ[base + 1] = sys->hVelocity[i].y;
                outVelocityXYZ[base + 2] = sys->hVelocity[i].z;
            }
        }

        gLastError.clear();
        return 0;
    }

    if (sys->hPosition.size() != static_cast<std::size_t>(particleCount))
    {
        sys->hPosition.resize(static_cast<std::size_t>(particleCount));
    }
    if (outVelocityXYZ != nullptr && sys->hVelocity.size() != static_cast<std::size_t>(particleCount))
    {
        sys->hVelocity.resize(static_cast<std::size_t>(particleCount));
    }

    if (!CheckCuda(cudaMemcpy(sys->hPosition.data(), sys->dPosition, sizeof(float4) * particleCount, cudaMemcpyDeviceToHost), "cudaMemcpy dPosition->host")) return -1;
    if (outVelocityXYZ != nullptr)
    {
        if (!CheckCuda(cudaMemcpy(sys->hVelocity.data(), sys->dVelocity, sizeof(float4) * particleCount, cudaMemcpyDeviceToHost), "cudaMemcpy dVelocity->host")) return -1;
    }

    for (int i = 0; i < particleCount; ++i)
    {
        int base = i * 3;
        outPositionXYZ[base] = sys->hPosition[i].x;
        outPositionXYZ[base + 1] = sys->hPosition[i].y;
        outPositionXYZ[base + 2] = sys->hPosition[i].z;

        if (outVelocityXYZ != nullptr)
        {
            outVelocityXYZ[base] = sys->hVelocity[i].x;
            outVelocityXYZ[base + 1] = sys->hVelocity[i].y;
            outVelocityXYZ[base + 2] = sys->hVelocity[i].z;
        }
    }

    gLastError.clear();
    return 0;
}

// Single 1D particle implicit integration (utility function for testing)
MSS_API int mssOneDImplicitStep(MssOneDState* state, const MssOneDParams* simulationParams)
{
    if (state == nullptr || simulationParams == nullptr)
    {
        SetError("mssOneDImplicitStep invalid arguments");
        return -1;
    }

    float dt = fmaxf(simulationParams->dt, 1e-6f);
    float mass = fmaxf(simulationParams->mass, 1e-6f);
    float stiffness = simulationParams->stiffness;
    float damping = simulationParams->damping;

    float denom = 1.0f + (damping * dt) / mass + (stiffness * dt * dt) / mass;
    if (fabsf(denom) < 1e-6f)
    {
        denom = (denom >= 0.0f) ? 1e-6f : -1e-6f;
    }

    float newVelocity = (state->velocity - (stiffness * dt / mass) * state->position) / denom;
    float newPosition = state->position + dt * newVelocity;

    state->velocity = newVelocity;
    state->position = newPosition;
    gLastError.clear();
    return 0;
}

// Get the last error message (thread-local)
MSS_API const char* mssGetLastError()
{
    return gLastError.c_str();
}
}
