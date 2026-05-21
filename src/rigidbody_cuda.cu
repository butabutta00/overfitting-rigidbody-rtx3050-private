#include "rigidbody_cuda.h"

#include <cuda_runtime.h>
#if defined(RB_CUDA_TC_USE_BF16)
#include <cuda_bf16.h>
#else
#include <cuda_fp16.h>
#endif
#include <mma.h>

#include <cmath>

namespace
{

using namespace nvcuda;

#if !defined(RB_CUDA_TC_USE_FP16) && !defined(RB_CUDA_TC_USE_BF16)
#define RB_CUDA_TC_USE_FP16 1
#endif

#if defined(RB_CUDA_TC_USE_BF16)
using TcScalar = __nv_bfloat16;
__device__ __forceinline__ TcScalar ToTcScalar(float v)
{
    return __float2bfloat16(v);
}
#else
using TcScalar = __half;
__device__ __forceinline__ TcScalar ToTcScalar(float v)
{
    return __float2half_rn(v);
}
#endif

struct DeviceBuffers
{
    float4* position;
    float4* rotation;
    float4* velocity;
    float4* angularVelocity;
    float4* acceleration;
    float4* angularAcceleration;
    float4* deltaVelocity;
    float4* deltaAngularVelocity;
};

__device__ __forceinline__ float3 MakeFloat3(float x, float y, float z)
{
    return make_float3(x, y, z);
}

__device__ __forceinline__ float3 Add3(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 Scale3(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float Dot3(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 NormalizeSafe3(const float3& v)
{
    float len2 = Dot3(v, v);
    float invLen = rsqrtf(fmaxf(len2, 1e-20f));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __forceinline__ float4 QuaternionMultiply(const float4& a, const float4& b)
{
    return make_float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

__device__ __forceinline__ float4 QuaternionNormalize(const float4& q)
{
    float len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    float invLen = rsqrtf(fmaxf(len2, 1e-20f));
    return make_float4(q.x * invLen, q.y * invLen, q.z * invLen, q.w * invLen);
}

__device__ __forceinline__ float4 IntegrateAngularStep(const float4& q, const float3& omegaStep)
{
    float angle = sqrtf(Dot3(omegaStep, omegaStep));
    if (angle <= 1e-12f)
    {
        return q;
    }

    float3 axis = NormalizeSafe3(omegaStep);
    float halfAngle = 0.5f * angle;
    float s = __sinf(halfAngle);
    float c = __cosf(halfAngle);
    float4 delta = make_float4(axis.x * s, axis.y * s, axis.z * s, c);
    return QuaternionNormalize(QuaternionMultiply(delta, q));
}

__device__ __forceinline__ float Select3(float expV, float siV, float vvV, float mExp, float mSi, float mVv)
{
    return expV * mExp + siV * mSi + vvV * mVv;
}

__global__ __launch_bounds__(256, 2)
void AoSToSoAAsync(
    const RbCudaState* __restrict__ input,
    float4* __restrict__ position,
    float4* __restrict__ rotation,
    float4* __restrict__ velocity,
    float4* __restrict__ angularVelocity,
    float4* __restrict__ acceleration,
    float4* __restrict__ angularAcceleration,
    std::size_t count)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    const RbCudaState s = input[i];
    position[i] = make_float4(s.position.x, s.position.y, s.position.z, s.position.w);
    rotation[i] = make_float4(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w);
    velocity[i] = make_float4(s.velocity.x, s.velocity.y, s.velocity.z, s.velocity.w);
    angularVelocity[i] = make_float4(s.angularVelocity.x, s.angularVelocity.y, s.angularVelocity.z, s.angularVelocity.w);
    acceleration[i] = make_float4(s.acceleration.x, s.acceleration.y, s.acceleration.z, s.acceleration.w);
    angularAcceleration[i] = make_float4(s.angularAcceleration.x, s.angularAcceleration.y, s.angularAcceleration.z, s.angularAcceleration.w);
}

__global__ __launch_bounds__(32, 4)
void TensorCoreScaleTcKernel(
    const float4* __restrict__ src,
    float4* __restrict__ out,
    std::size_t count,
    float scale)
{
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const std::size_t tileBase = static_cast<std::size_t>(blockIdx.x) * 16;
    if (tileBase >= count)
    {
        return;
    }

    __shared__ __align__(16) TcScalar matA[16 * 16];
    __shared__ __align__(16) TcScalar matB[16 * 16];
    __shared__ __align__(16) float matC[16 * 16];

    float warpScale = __shfl_sync(0xffffffffu, scale, 0);

    for (int idx = lane; idx < 16 * 16; idx += 32)
    {
        int r = idx / 16;
        int c = idx % 16;
        float a = (r == c) ? warpScale : 0.0f;
        matA[idx] = ToTcScalar(a);
    }

    if (lane < 16)
    {
        std::size_t gi = tileBase + static_cast<std::size_t>(lane);
        float4 s = (gi < count) ? src[gi] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        matB[lane * 16 + 0] = ToTcScalar(s.x);
        matB[lane * 16 + 1] = ToTcScalar(s.y);
        matB[lane * 16 + 2] = ToTcScalar(s.z);
        for (int c = 3; c < 16; ++c)
        {
            matB[lane * 16 + c] = ToTcScalar(0.0f);
        }
    }

    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, TcScalar, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, TcScalar, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    wmma::load_matrix_sync(aFrag, matA, 16);
    wmma::load_matrix_sync(bFrag, matB, 16);
    wmma::fill_fragment(cFrag, 0.0f);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    wmma::store_matrix_sync(matC, cFrag, 16, wmma::mem_row_major);

    if (lane < 16)
    {
        std::size_t gi = tileBase + static_cast<std::size_t>(lane);
        if (gi < count)
        {
            float4 d = out[gi];
            d.x = matC[lane * 16 + 0];
            d.y = matC[lane * 16 + 1];
            d.z = matC[lane * 16 + 2];
            d.w = 0.0f;
            out[gi] = d;
        }
    }
#else
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * 16 + (threadIdx.x & 15);
    if (i < count && (threadIdx.x & 16) == 0)
    {
        float4 s = src[i];
        out[i] = make_float4(s.x * scale, s.y * scale, s.z * scale, 0.0f);
    }
#endif
}

__global__ __launch_bounds__(256, 2)
void IntegrateKernel(
    float4* __restrict__ position,
    float4* __restrict__ rotation,
    float4* __restrict__ velocity,
    float4* __restrict__ angularVelocity,
    const float4* __restrict__ deltaVelocity,
    const float4* __restrict__ deltaAngularVelocity,
    cudaTextureObject_t accelerationTex,
    cudaTextureObject_t angularAccelerationTex,
    std::size_t count,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    float dtWarp = __shfl_sync(0xffffffffu, dt, 0);

    unsigned packedMethod = __byte_perm(static_cast<unsigned>(integrationMethod), 0u, 0x0000);
    int method = static_cast<int>(packedMethod & 0xFFu);

    unsigned expMask = static_cast<unsigned>(-(method == RB_CUDA_EXPLICIT));
    unsigned siMask = static_cast<unsigned>(-(method == RB_CUDA_SEMI_IMPLICIT));
    unsigned vvMask = static_cast<unsigned>(-(method == RB_CUDA_VELOCITY_VERLET));

    float mExp = static_cast<float>(expMask & 1u);
    float mSi = static_cast<float>(siMask & 1u);
    float mVv = static_cast<float>(vvMask & 1u);

    float tMask = static_cast<float>(enableTranslation & 1);
    float rMask = static_cast<float>(enableRotation & 1);

    float4 p4 = position[i];
    float4 q = rotation[i];
    float4 v4 = velocity[i];
    float4 w4 = angularVelocity[i];

    float4 a4 = tex1Dfetch<float4>(accelerationTex, static_cast<int>(i));
    float4 alpha4 = tex1Dfetch<float4>(angularAccelerationTex, static_cast<int>(i));
    float4 dv4 = deltaVelocity[i];
    float4 dw4 = deltaAngularVelocity[i];

    float3 p = MakeFloat3(p4.x, p4.y, p4.z);
    float3 v = MakeFloat3(v4.x, v4.y, v4.z);
    float3 w = MakeFloat3(w4.x, w4.y, w4.z);
    const float3 a = MakeFloat3(a4.x, a4.y, a4.z);
    const float3 alpha = MakeFloat3(alpha4.x, alpha4.y, alpha4.z);
    const float3 dv = MakeFloat3(dv4.x, dv4.y, dv4.z);
    const float3 dw = MakeFloat3(dw4.x, dw4.y, dw4.z);

    float3 pExp = Add3(p, Scale3(v, dtWarp));
    float3 vExp = Add3(v, dv);

    float3 vSi = Add3(v, dv);
    float3 pSi = Add3(p, Scale3(vSi, dtWarp));

    float dt2Half = 0.5f * dtWarp * dtWarp;
    float3 pVv = Add3(p, Add3(Scale3(v, dtWarp), Scale3(a, dt2Half)));
    float3 vVv = Add3(v, dv);

    float3 pSel = make_float3(
        Select3(pExp.x, pSi.x, pVv.x, mExp, mSi, mVv),
        Select3(pExp.y, pSi.y, pVv.y, mExp, mSi, mVv),
        Select3(pExp.z, pSi.z, pVv.z, mExp, mSi, mVv));

    float3 vSel = make_float3(
        Select3(vExp.x, vSi.x, vVv.x, mExp, mSi, mVv),
        Select3(vExp.y, vSi.y, vVv.y, mExp, mSi, mVv),
        Select3(vExp.z, vSi.z, vVv.z, mExp, mSi, mVv));

    p.x = p.x + tMask * (pSel.x - p.x);
    p.y = p.y + tMask * (pSel.y - p.y);
    p.z = p.z + tMask * (pSel.z - p.z);

    v.x = v.x + tMask * (vSel.x - v.x);
    v.y = v.y + tMask * (vSel.y - v.y);
    v.z = v.z + tMask * (vSel.z - v.z);

    float3 wExp = Add3(w, dw);
    float3 wSi = Add3(w, dw);
    float3 wVv = Add3(w, dw);

    float3 stepExp = Scale3(w, dtWarp);
    float3 stepSi = Scale3(wSi, dtWarp);
    float3 wMid = Add3(w, Scale3(alpha, 0.5f * dtWarp));
    float3 stepVv = Scale3(wMid, dtWarp);

    float4 qExp = IntegrateAngularStep(q, stepExp);
    float4 qSi = IntegrateAngularStep(q, stepSi);
    float4 qVv = IntegrateAngularStep(q, stepVv);

    float4 qSel = make_float4(
        Select3(qExp.x, qSi.x, qVv.x, mExp, mSi, mVv),
        Select3(qExp.y, qSi.y, qVv.y, mExp, mSi, mVv),
        Select3(qExp.z, qSi.z, qVv.z, mExp, mSi, mVv),
        Select3(qExp.w, qSi.w, qVv.w, mExp, mSi, mVv));

    float3 wSel = make_float3(
        Select3(wExp.x, wSi.x, wVv.x, mExp, mSi, mVv),
        Select3(wExp.y, wSi.y, wVv.y, mExp, mSi, mVv),
        Select3(wExp.z, wSi.z, wVv.z, mExp, mSi, mVv));

    q.x = q.x + rMask * (qSel.x - q.x);
    q.y = q.y + rMask * (qSel.y - q.y);
    q.z = q.z + rMask * (qSel.z - q.z);
    q.w = q.w + rMask * (qSel.w - q.w);
    q = QuaternionNormalize(q);

    w.x = w.x + rMask * (wSel.x - w.x);
    w.y = w.y + rMask * (wSel.y - w.y);
    w.z = w.z + rMask * (wSel.z - w.z);

    position[i] = make_float4(p.x, p.y, p.z, 0.0f);
    rotation[i] = q;
    velocity[i] = make_float4(v.x, v.y, v.z, 0.0f);
    angularVelocity[i] = make_float4(w.x, w.y, w.z, 0.0f);
}

__global__ __launch_bounds__(256, 2)
void SoAToAoS(
    const float4* __restrict__ position,
    const float4* __restrict__ rotation,
    const float4* __restrict__ velocity,
    const float4* __restrict__ angularVelocity,
    RbCudaState* __restrict__ output,
    std::size_t count)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    RbCudaState s = output[i];
    float4 p = position[i];
    float4 q = rotation[i];
    float4 v = velocity[i];
    float4 w = angularVelocity[i];

    s.position = { p.x, p.y, p.z, p.w };
    s.rotation = { q.x, q.y, q.z, q.w };
    s.velocity = { v.x, v.y, v.z, v.w };
    s.angularVelocity = { w.x, w.y, w.z, w.w };
    output[i] = s;
}

int RunBatch(
    RbCudaState* ioStates,
    std::size_t count,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation)
{
    if (ioStates == nullptr || count == 0)
    {
        return 1;
    }

    const std::size_t stateBytes = sizeof(RbCudaState) * count;
    const std::size_t vecBytes = sizeof(float4) * count;

    RbCudaState* devStates = nullptr;
    DeviceBuffers d = {};

    cudaError_t err = cudaMalloc(&devStates, stateBytes);
    if (err != cudaSuccess) return 2;

    err = cudaMalloc(&d.position, vecBytes);
    if (err != cudaSuccess) return 3;
    err = cudaMalloc(&d.rotation, vecBytes);
    if (err != cudaSuccess) return 4;
    err = cudaMalloc(&d.velocity, vecBytes);
    if (err != cudaSuccess) return 5;
    err = cudaMalloc(&d.angularVelocity, vecBytes);
    if (err != cudaSuccess) return 6;
    err = cudaMalloc(&d.acceleration, vecBytes);
    if (err != cudaSuccess) return 7;
    err = cudaMalloc(&d.angularAcceleration, vecBytes);
    if (err != cudaSuccess) return 8;
    err = cudaMalloc(&d.deltaVelocity, vecBytes);
    if (err != cudaSuccess) return 9;
    err = cudaMalloc(&d.deltaAngularVelocity, vecBytes);
    if (err != cudaSuccess) return 10;

    err = cudaMemcpy(devStates, ioStates, stateBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return 11;

    constexpr int block = 256;
    int grid = static_cast<int>((count + block - 1) / block);

    err = cudaFuncSetAttribute(AoSToSoAAsync, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (err != cudaSuccess) return 12;
    err = cudaFuncSetAttribute(AoSToSoAAsync, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (err != cudaSuccess) return 12;
    AoSToSoAAsync<<<grid, block, block * sizeof(RbCudaState)>>>(
        devStates,
        d.position,
        d.rotation,
        d.velocity,
        d.angularVelocity,
        d.acceleration,
        d.angularAcceleration,
        count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 12;

    int tcGrid = static_cast<int>((count + 15) / 16);
    TensorCoreScaleTcKernel<<<tcGrid, 32>>>(d.acceleration, d.deltaVelocity, count, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 13;

    TensorCoreScaleTcKernel<<<tcGrid, 32>>>(d.angularAcceleration, d.deltaAngularVelocity, count, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 14;

    cudaResourceDesc accelRes = {};
    accelRes.resType = cudaResourceTypeLinear;
    accelRes.res.linear.devPtr = d.acceleration;
    accelRes.res.linear.desc = cudaCreateChannelDesc<float4>();
    accelRes.res.linear.sizeInBytes = vecBytes;

    cudaResourceDesc alphaRes = {};
    alphaRes.resType = cudaResourceTypeLinear;
    alphaRes.res.linear.devPtr = d.angularAcceleration;
    alphaRes.res.linear.desc = cudaCreateChannelDesc<float4>();
    alphaRes.res.linear.sizeInBytes = vecBytes;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t accelerationTex = 0;
    cudaTextureObject_t angularAccelerationTex = 0;

    err = cudaCreateTextureObject(&accelerationTex, &accelRes, &texDesc, nullptr);
    if (err != cudaSuccess) return 15;
    err = cudaCreateTextureObject(&angularAccelerationTex, &alphaRes, &texDesc, nullptr);
    if (err != cudaSuccess) return 16;

    err = cudaFuncSetAttribute(IntegrateKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (err != cudaSuccess) return 17;
    IntegrateKernel<<<grid, block>>>(
        d.position,
        d.rotation,
        d.velocity,
        d.angularVelocity,
        d.deltaVelocity,
        d.deltaAngularVelocity,
        accelerationTex,
        angularAccelerationTex,
        count,
        dt,
        integrationMethod,
        enableTranslation,
        enableRotation);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 17;

    SoAToAoS<<<grid, block>>>(
        d.position,
        d.rotation,
        d.velocity,
        d.angularVelocity,
        devStates,
        count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 18;

    err = cudaMemcpy(ioStates, devStates, stateBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return 19;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return 20;

    cudaDestroyTextureObject(angularAccelerationTex);
    cudaDestroyTextureObject(accelerationTex);

    cudaFree(d.deltaAngularVelocity);
    cudaFree(d.deltaVelocity);
    cudaFree(d.angularAcceleration);
    cudaFree(d.acceleration);
    cudaFree(d.angularVelocity);
    cudaFree(d.velocity);
    cudaFree(d.rotation);
    cudaFree(d.position);
    cudaFree(devStates);
    return 0;
}

}

extern "C" int RbCudaStepSingle(
    RbCudaState* ioState,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation)
{
    return RunBatch(ioState, 1, dt, integrationMethod, enableTranslation, enableRotation);
}

extern "C" int RbCudaStepBatch(
    RbCudaState* ioStates,
    std::size_t count,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation)
{
    return RunBatch(ioStates, count, dt, integrationMethod, enableTranslation, enableRotation);
}
