// mass_spring_native.h
// 
// C API for CUDA-accelerated mass-spring-damper physics simulation
// Provides both semi-implicit (Symplectic Euler) and implicit (Conjugate Gradient) integrators
// 
// OPTIMIZATION TARGET: NVIDIA RTX 3050
// - Compute Capability 8.6, 128-bit memory bandwidth (448 GB/s)
// - 4MB L2 cache with persistence API for repeated spring lookups
// - 64KB register banks (2 blocks per SM for occupancy targeting)
// - Uses float4 vectorization to match hardware memory bus width
// - Warp-level reductions and async memcpy for latency hiding
// - __ldg caching, __byte_perm for branch-free fixed mask handling
// 
// See README.md for detailed optimization strategies documentation.

#pragma once

#include <cstdint>

#ifdef _WIN32
  #ifdef MASS_SPRING_NATIVE_EXPORTS
    #define MSS_API __declspec(dllexport)
  #else
    #define MSS_API __declspec(dllimport)
  #endif
#else
  #define MSS_API __attribute__((visibility("default")))
#endif

extern "C" {

// Parameters for semi-implicit (Symplectic Euler) integration step
// Semi-implicit method is unconditionally stable for spring systems.
// More accurate than explicit Euler but less stable than full implicit solver.
// Time step can be subdivided via substeps parameter for better accuracy.
// 
// OPTIMIZATION: All scalar parameters fit in registers on RTX 3050.
// No shared memory needed for parameter passing to kernels.
struct MssSemiParams
{
    float dt;                 // Time step size (seconds)
    float springStiffness;    // Spring constant k in Hooke's law: F = -k(x - x_rest)
    float springDamping;      // Damping coefficient c in: F = -c*v
    float gravityX;           // X component of gravity acceleration (m/s²)
    float gravityY;           // Y component of gravity acceleration (m/s²)
    float gravityZ;           // Z component of gravity acceleration (m/s²)
    float velocityDamping;    // Global velocity damping factor (0=no damping, 1=all dissipated)
    int substeps;             // Number of substeps to subdivide dt (>=1 for stability)
};

// Parameters for implicit integration with Conjugate Gradient solver
// Implicit method is unconditionally stable, allowing much larger time steps than semi-implicit.
// Uses linearized springs and solves a linear system via CG to compute velocities.
// More expensive per step but allows larger dt while maintaining stability.
// 
// OPTIMIZATION: CG convergence accelerated via RTX 3050 two-level dot product reduction
// with warp-level shuffles (__shfl_down_sync) and async memcpy for latency hiding.
struct MssImplicitParams
{
    float dt;                    // Time step size (seconds) - can be much larger than semi-implicit
    float springStiffness;       // Spring constant k for linearized springs
    float springDamping;         // Damping coefficient c for linearized springs
    float gravityX;              // X component of gravity acceleration (m/s²)
    float gravityY;              // Y component of gravity acceleration (m/s²)
    float gravityZ;              // Z component of gravity acceleration (m/s²)
    float velocityDamping;       // Global velocity damping factor
    int implicitIterations;      // Number of outer Newton iterations (usually 1-3)
    float cgTolerance;           // CG convergence tolerance (relative residual norm)
};

// 1D mass-spring state for testing implicit integration
// Simplified version of full 3D system for algorithm validation and benchmarking.
// Used in mssOneDImplicitStep for unit testing without full GPU overhead.
struct MssOneDState
{
    float position;   // Current 1D position (x)
    float velocity;   // Current 1D velocity (v = dx/dt)
};

// Parameters for 1D implicit integration step
// Solves 1D mass-spring-damper: m*dv/dt = -k*x - c*v
// Used for testing, validation, and algorithm development.
struct MssOneDParams
{
    float dt;         // Time step size (seconds)
    float mass;       // Mass m (kg)
    float stiffness;  // Spring stiffness k
    float damping;    // Damping coefficient c
};

// Create and initialize GPU simulation system
// Detects CUDA availability, allocates device memory, configures RTX 3050 cache hints.
// OPTIMIZATION: Sets L2 cache persistence (3/4 of 4MB) for spring data and shared memory
// carveout to 100% for reduction kernels.
// Returns opaque DeviceSystem pointer or nullptr on error.
MSS_API void* mssCreateSystem();

// Destroy GPU system and free all device/host memory
// Safe to call multiple times or with nullptr.
MSS_API int mssDestroySystem(void* system);

// Upload simulation topology and initial state to GPU
// Must be called after mssCreateSystem and before stepping simulation.
// Copies particle positions, masses, constraints, and spring connectivity to device.
// 
// OPTIMIZATION: Data layout uses float4 vectorization for float data streams.
// Masses and rest lengths cached via __ldg for efficient repeated reads across springs.
// Fixed particle mask uses __byte_perm for branch-free active/inactive conversion.
MSS_API int mssUploadTopology(
    void* system,
    int particleCount,              // Number of particles in system
    const float* positionXYZ,       // Host array of particle positions (particleCount*3 floats)
    const float* masses,            // Host array of particle masses (particleCount floats)
    const uint8_t* fixedMask,       // Host array of fixed constraints (particleCount bytes, 0=free, !0=fixed)
    int springCount,                // Number of springs in system
    const int* springEndpoints,     // Host array of spring endpoint indices (springCount*2 ints)
    const float* restLengths);      // Host array of spring rest lengths (springCount floats)

// Execute one step of semi-implicit (Symplectic Euler) integration
// More stable than explicit Euler, suitable for moderate time steps (e.g., 0.01-0.05 sec).
// Optionally subdivides dt into substeps for improved accuracy at cost of more force evaluations.
// 
// OPTIMIZATION: Uses __launch_bounds__ for occupancy targeting, float4 alignment for memory
// bandwidth matching, and __ldg for read-only data caching on RTX 3050.
MSS_API int mssStepSemi(void* system, const MssSemiParams* stepParams);

// Execute one step of implicit integration with Conjugate Gradient solver
// Unconditionally stable, allowing much larger time steps than semi-implicit (e.g., 0.1-1.0 sec).
// Solves linearized spring system via preconditioned CG; configure cgTolerance for accuracy/speed tradeoff.
// 
// OPTIMIZATION: Two-level dot product reduction (warp-level + block-level) minimizes synchronization.
// Uses cooperative_groups::memcpy_async for latency hiding between computation phases.
// L2 cache persistence reserves 3MB (75% of 4MB) for spring topology lookups across CG iterations.
MSS_API int mssStepImplicit(void* system, const MssImplicitParams* stepParams);

// Download particle positions and velocities from GPU to host memory
// Call after mssStepSemi or mssStepImplicit to retrieve simulation results.
// Copies device float4 vectors to host in XYZ format (ignoring w component).
MSS_API int mssDownloadState(void* system, float* outPositionXYZ, float* outVelocityXYZ, int particleCount);

// Execute one implicit integration step for 1D mass-spring-damper
// Solves: m*dv/dt = -k*x - c*v via linearization and backsubstitution
// Used for algorithm testing and validation without full GPU overhead.
MSS_API int mssOneDImplicitStep(MssOneDState* state, const MssOneDParams* simulationParams);

// Retrieve last error message from thread-local storage
// Returns empty string if no error occurred.
MSS_API const char* mssGetLastError();

}  // extern "C"
