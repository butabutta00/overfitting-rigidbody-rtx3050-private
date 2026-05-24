#include "mass_spring_native.h"

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
constexpr float kEpsilon = 1e-6f;
constexpr int kBlockSize = 256;

struct DeviceSystem
{
    int particleCount = 0;
    int springCount = 0;

    int2* dSpringEnds = nullptr;
    float* dRestLengths = nullptr;
    float* dMasses = nullptr;
    uint8_t* dFixedMask = nullptr;

    float4* dPosition = nullptr;
    float4* dVelocity = nullptr;
    float4* dForce = nullptr;

    float4* dX0 = nullptr;
    float4* dV0 = nullptr;
    float4* dJv = nullptr;
    float4* dLinearized = nullptr;
    float4* dB = nullptr;
    float4* dV = nullptr;
    float4* dR = nullptr;
    float4* dP = nullptr;
    float4* dAp = nullptr;

    float* dDotPartial = nullptr;
    float* dDotValue = nullptr;
    int dotPartialCount = 0;

    float hDotValue = 0.0f;

    bool cudaAvailable = false;

    std::vector<float4> hPosition;
    std::vector<float4> hVelocity;
    std::vector<float4> hForce;
    std::vector<float> hMasses;
    std::vector<uint8_t> hFixedMask;
    std::vector<int2> hSpringEnds;
    std::vector<float> hRestLengths;
};

thread_local std::string gLastError;

inline void SetError(const std::string& message)
{
    gLastError = message;
}

inline void SetErrorCuda(const char* where, cudaError_t err)
{
    gLastError = std::string(where) + ": " + cudaGetErrorString(err);
}

inline bool CheckCuda(cudaError_t err, const char* where)
{
    if (err == cudaSuccess)
    {
        return true;
    }

    SetErrorCuda(where, err);
    return false;
}

__device__ inline float Dot3(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float4 Add3(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

__device__ inline float4 Sub3(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

__device__ inline float4 Mul3(const float4& v, float s)
{
    return make_float4(v.x * s, v.y * s, v.z * s, 0.0f);
}

__global__ void InitForcesKernel(
    float4* force,
    float4* jv,
    const float* masses,
    const uint8_t* fixedMask,
    float3 gravity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        force[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        float m = masses[i];
        force[i] = make_float4(gravity.x * m, gravity.y * m, gravity.z * m, 0.0f);
    }

    if (jv != nullptr)
    {
        jv[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ __launch_bounds__(kBlockSize, 2) void SpringForceKernel(
    const float4* pos,
    const float4* vel,
    float4* force,
    const int2* springEnds,
    const float* restLengths,
    const uint8_t* fixedMask,
    float springStiffness,
    float springDamping,
    int springCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int2 ends = springEnds[s];
    int a = ends.x;
    int b = ends.y;

    float4 pa = pos[a];
    float4 pb = pos[b];
    float4 va = vel[a];
    float4 vb = vel[b];

    float4 diff = Sub3(pb, pa);
    float dist2 = Dot3(diff, diff);
    if (dist2 < kEpsilon)
    {
        return;
    }

    float dist = sqrtf(dist2);
    float invDist = 1.0f / dist;
    float4 dir = Mul3(diff, invDist);
    float4 relVel = Sub3(vb, va);

    float relAlong = Dot3(relVel, dir);
    float fSpring = springStiffness * (dist - restLengths[s]);
    float fDamper = springDamping * relAlong;
    float4 total = Mul3(dir, fSpring + fDamper);

    if (!fixedMask[a])
    {
        atomicAdd(&force[a].x, total.x);
        atomicAdd(&force[a].y, total.y);
        atomicAdd(&force[a].z, total.z);
    }

    if (!fixedMask[b])
    {
        atomicAdd(&force[b].x, -total.x);
        atomicAdd(&force[b].y, -total.y);
        atomicAdd(&force[b].z, -total.z);
    }
}

__global__ __launch_bounds__(kBlockSize, 2) void IntegrateSemiKernel(
    float4* pos,
    float4* vel,
    const float4* force,
    const float* masses,
    const uint8_t* fixedMask,
    float dt,
    float velocityDamping,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float invMass = 1.0f / fmaxf(masses[i], kEpsilon);
    float4 accel = Mul3(force[i], invMass);

    float4 v = vel[i];
    v = Add3(v, Mul3(accel, dt));
    float4 x = pos[i];
    x = Add3(x, Mul3(v, dt));
    v = Mul3(v, velocityDamping);

    vel[i] = v;
    pos[i] = x;
}

__global__ void CopyStateKernel(const float4* srcPos, const float4* srcVel, float4* dstPos, float4* dstVel, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    dstPos[i] = srcPos[i];
    dstVel[i] = srcVel[i];
}

__global__ __launch_bounds__(kBlockSize, 2) void SpringLinearizeKernel(
    const float4* x0,
    const float4* v0,
    float4* force,
    float4* jv,
    float4* linearized,
    const int2* springEnds,
    const float* restLengths,
    const uint8_t* fixedMask,
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

    int2 ends = springEnds[s];
    int a = ends.x;
    int b = ends.y;

    float4 xa = x0[a];
    float4 xb = x0[b];
    float4 va = v0[a];
    float4 vb = v0[b];

    float4 diff = Sub3(xb, xa);
    float dist2 = Dot3(diff, diff);
    if (dist2 < kEpsilon)
    {
        linearized[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float dist = sqrtf(dist2);
    float invDist = 1.0f / dist;
    float4 dir = Mul3(diff, invDist);
    float4 relVel = Sub3(vb, va);
    float relAlong = Dot3(relVel, dir);

    float fSpring = springStiffness * (dist - restLengths[s]);
    float fDamper = springDamping * relAlong;
    float4 total = Mul3(dir, fSpring + fDamper);

    if (!fixedMask[a])
    {
        atomicAdd(&force[a].x, total.x);
        atomicAdd(&force[a].y, total.y);
        atomicAdd(&force[a].z, total.z);
    }

    if (!fixedMask[b])
    {
        atomicAdd(&force[b].x, -total.x);
        atomicAdd(&force[b].y, -total.y);
        atomicAdd(&force[b].z, -total.z);
    }

    float projADot = Dot3(va, dir);
    float projBDot = Dot3(vb, dir);
    float4 projA = Mul3(dir, projADot);
    float4 projB = Mul3(dir, projBDot);

    float4 jvA = Mul3(Sub3(projB, projA), springDamping);
    float4 jvB = Mul3(Sub3(projA, projB), springDamping);

    if (!fixedMask[a])
    {
        atomicAdd(&jv[a].x, jvA.x);
        atomicAdd(&jv[a].y, jvA.y);
        atomicAdd(&jv[a].z, jvA.z);
    }

    if (!fixedMask[b])
    {
        atomicAdd(&jv[b].x, jvB.x);
        atomicAdd(&jv[b].y, jvB.y);
        atomicAdd(&jv[b].z, jvB.z);
    }

    linearized[s] = make_float4(dir.x, dir.y, dir.z, h * springDamping + h2 * springStiffness);
}

__global__ void BuildImplicitBKernel(
    const float4* v0,
    const float4* force,
    const float4* jv,
    const float* masses,
    const uint8_t* fixedMask,
    float h,
    float4* b,
    float4* v,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        b[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        v[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float m = masses[i];
    float4 rhs = Add3(Mul3(v0[i], m), Mul3(Sub3(force[i], jv[i]), h));
    b[i] = rhs;
    v[i] = v0[i];
}

__global__ void InitMassTermKernel(
    const float4* src,
    float4* dst,
    const float* masses,
    const uint8_t* fixedMask,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        dst[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        dst[i] = Mul3(src[i], masses[i]);
    }
}

__global__ __launch_bounds__(kBlockSize, 2) void AccumulateImplicitMatrixKernel(
    const float4* src,
    float4* dst,
    const float4* linearized,
    const int2* springEnds,
    const uint8_t* fixedMask,
    int springCount)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= springCount)
    {
        return;
    }

    int2 ends = springEnds[s];
    int a = ends.x;
    int b = ends.y;

    float4 lin = linearized[s];
    float coeff = lin.w;
    if (coeff == 0.0f)
    {
        return;
    }

    float4 dir = make_float4(lin.x, lin.y, lin.z, 0.0f);
    float dotA = Dot3(src[a], dir);
    float dotB = Dot3(src[b], dir);
    float4 projA = Mul3(dir, dotA);
    float4 projB = Mul3(dir, dotB);

    if (!fixedMask[a])
    {
        float4 term = Sub3(projA, fixedMask[b] ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : projB);
        atomicAdd(&dst[a].x, coeff * term.x);
        atomicAdd(&dst[a].y, coeff * term.y);
        atomicAdd(&dst[a].z, coeff * term.z);
    }

    if (!fixedMask[b])
    {
        float4 term = Sub3(projB, fixedMask[a] ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : projA);
        atomicAdd(&dst[b].x, coeff * term.x);
        atomicAdd(&dst[b].y, coeff * term.y);
        atomicAdd(&dst[b].z, coeff * term.z);
    }
}

__device__ inline float WarpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }

    return val;
}

__global__ void DotPartialKernel(
    const float4* a,
    const float4* b,
    const uint8_t* fixedMask,
    int count,
    float* partial)
{
    __shared__ float warpSums[8];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (int i = idx; i < count; i += stride)
    {
        if (!fixedMask[i])
        {
            local += Dot3(a[i], b[i]);
        }
    }

    local = WarpReduceSum(local);
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (lane == 0)
    {
        warpSums[warpId] = local;
    }

    __syncthreads();

    float blockSum = 0.0f;
    if (warpId == 0)
    {
        blockSum = (lane < (blockDim.x / warpSize)) ? warpSums[lane] : 0.0f;
        blockSum = WarpReduceSum(blockSum);
        if (lane == 0)
        {
            partial[blockIdx.x] = blockSum;
        }
    }
}

__global__ void ReducePartialSumKernel(const float* partial, int count, float* out)
{
    __shared__ float shared[kBlockSize];
    int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < count; i += blockDim.x)
    {
        local += partial[i];
    }

    shared[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *out = shared[0];
    }
}

__global__ void InitCGKernel(const float4* b, const float4* ap, const uint8_t* fixedMask, float4* r, float4* p, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        r[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        p[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 ri = Sub3(b[i], ap[i]);
    r[i] = ri;
    p[i] = ri;
}

__global__ void CGUpdateXRKernel(float4* x, float4* r, const float4* p, const float4* ap, const uint8_t* fixedMask, float alpha, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        return;
    }

    x[i] = Add3(x[i], Mul3(p[i], alpha));
    r[i] = Sub3(r[i], Mul3(ap[i], alpha));
}

__global__ void CGUpdatePKernel(float4* p, const float4* r, const uint8_t* fixedMask, float beta, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        return;
    }

    p[i] = Add3(r[i], Mul3(p[i], beta));
}

__global__ void CommitImplicitKernel(
    float4* pos,
    float4* vel,
    const float4* x0,
    const float4* vSolved,
    const uint8_t* fixedMask,
    float h,
    float velocityDamping,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    if (fixedMask[i])
    {
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        pos[i] = x0[i];
        return;
    }

    float4 v = Mul3(vSolved[i], velocityDamping);
    float4 p = Add3(x0[i], Mul3(vSolved[i], h));
    vel[i] = v;
    pos[i] = p;
}

inline int DivUp(int n, int d)
{
    return (n + d - 1) / d;
}

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

    int blocks = std::max(1, DivUp(sys.particleCount, kBlockSize));
    if (sys.dotPartialCount < blocks)
    {
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

    return true;
}

float DeviceDot(DeviceSystem& sys, const float4* a, const float4* b)
{
    if (!sys.cudaAvailable)
    {
        return 0.0f;
    }

    int blocks = std::max(1, DivUp(sys.particleCount, kBlockSize));
    DotPartialKernel<<<blocks, kBlockSize>>>(a, b, sys.dFixedMask, sys.particleCount, sys.dDotPartial);
    if (!CheckCuda(cudaGetLastError(), "DotPartialKernel launch"))
    {
        return 0.0f;
    }

    ReducePartialSumKernel<<<1, kBlockSize>>>(sys.dDotPartial, blocks, sys.dDotValue);
    if (!CheckCuda(cudaGetLastError(), "ReducePartialSumKernel launch"))
    {
        return 0.0f;
    }

    if (!CheckCuda(cudaMemcpy(&sys.hDotValue, sys.dDotValue, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dot value"))
    {
        return 0.0f;
    }

    return sys.hDotValue;
}

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

extern "C"
{
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

MSS_API int mssDestroySystem(void* system)
{
    FreeSystem(reinterpret_cast<DeviceSystem*>(system));
    return 0;
}

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

    std::vector<float4> hostPos(static_cast<std::size_t>(particleCount));
    for (int i = 0; i < particleCount; ++i)
    {
        int base = i * 3;
        hostPos[i] = make_float4(positionXYZ[base], positionXYZ[base + 1], positionXYZ[base + 2], 0.0f);
    }

    std::vector<int2> hostEnds(static_cast<std::size_t>(springCount));
    for (int s = 0; s < springCount; ++s)
    {
        int base = s * 2;
        hostEnds[s] = make_int2(springEndpoints[base], springEndpoints[base + 1]);
    }

    sys->hPosition = hostPos;
    sys->hVelocity.assign(static_cast<std::size_t>(particleCount), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    sys->hForce.assign(static_cast<std::size_t>(particleCount), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    sys->hMasses.assign(masses, masses + particleCount);
    sys->hFixedMask.assign(fixedMask, fixedMask + particleCount);
    sys->hSpringEnds = hostEnds;
    sys->hRestLengths.assign(restLengths, restLengths + springCount);

    if (sys->cudaAvailable)
    {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dPosition), sizeof(float4) * particleCount), "cudaMalloc dPosition"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dVelocity), sizeof(float4) * particleCount), "cudaMalloc dVelocity"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dMasses), sizeof(float) * particleCount), "cudaMalloc dMasses"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dFixedMask), sizeof(uint8_t) * particleCount), "cudaMalloc dFixedMask"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dSpringEnds), sizeof(int2) * springCount), "cudaMalloc dSpringEnds"))
        {
            sys->cudaAvailable = false;
        }
        if (sys->cudaAvailable && !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sys->dRestLengths), sizeof(float) * springCount), "cudaMalloc dRestLengths"))
        {
            sys->cudaAvailable = false;
        }

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

    const int pBlocks = DivUp(sys->particleCount, kBlockSize);
    const int sBlocks = DivUp(sys->springCount, kBlockSize);

    int substeps = std::max(1, stepParams->substeps);
    float dtSub = stepParams->dt / static_cast<float>(substeps);

    for (int iter = 0; iter < substeps; ++iter)
    {
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
        return -1;
    }

    const int pBlocks = DivUp(sys->particleCount, kBlockSize);
    const int sBlocks = DivUp(sys->springCount, kBlockSize);
    const float h = stepParams->dt;
    const float h2 = h * h;

    if (!CheckCuda(cudaMemcpy(sys->dX0, sys->dPosition, sizeof(float4) * sys->particleCount, cudaMemcpyDeviceToDevice), "cudaMemcpy dPosition->dX0")) return -1;
    if (!CheckCuda(cudaMemcpy(sys->dV0, sys->dVelocity, sizeof(float4) * sys->particleCount, cudaMemcpyDeviceToDevice), "cudaMemcpy dVelocity->dV0")) return -1;

    InitForcesKernel<<<pBlocks, kBlockSize>>>(
        sys->dForce,
        sys->dJv,
        sys->dMasses,
        sys->dFixedMask,
        make_float3(stepParams->gravityX, stepParams->gravityY, stepParams->gravityZ),
        sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "InitForcesKernel launch (implicit)")) return -1;

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

    if (!MultiplyImplicitMatrix(*sys, sys->dV, sys->dAp)) return -1;

    InitCGKernel<<<pBlocks, kBlockSize>>>(sys->dB, sys->dAp, sys->dFixedMask, sys->dR, sys->dP, sys->particleCount);
    if (!CheckCuda(cudaGetLastError(), "InitCGKernel launch")) return -1;

    float rr = DeviceDot(*sys, sys->dR, sys->dR);
    if (rr < stepParams->cgTolerance)
    {
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

    for (int iter = 0; iter < std::max(1, stepParams->implicitIterations); ++iter)
    {
        if (!MultiplyImplicitMatrix(*sys, sys->dP, sys->dAp)) return -1;

        float pAp = DeviceDot(*sys, sys->dP, sys->dAp);
        if (fabsf(pAp) < 1e-12f)
        {
            break;
        }

        float alpha = rr / pAp;
        CGUpdateXRKernel<<<pBlocks, kBlockSize>>>(sys->dV, sys->dR, sys->dP, sys->dAp, sys->dFixedMask, alpha, sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "CGUpdateXRKernel launch")) return -1;

        float rrNew = DeviceDot(*sys, sys->dR, sys->dR);
        if (rrNew < stepParams->cgTolerance)
        {
            rr = rrNew;
            break;
        }

        float beta = rrNew / rr;
        CGUpdatePKernel<<<pBlocks, kBlockSize>>>(sys->dP, sys->dR, sys->dFixedMask, beta, sys->particleCount);
        if (!CheckCuda(cudaGetLastError(), "CGUpdatePKernel launch")) return -1;
        rr = rrNew;
    }

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

MSS_API const char* mssGetLastError()
{
    return gLastError.c_str();
}
}
