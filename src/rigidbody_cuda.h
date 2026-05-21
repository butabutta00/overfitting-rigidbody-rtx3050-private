#pragma once

#include <cstddef>

#ifdef _WIN32
  #ifdef RB_CUDA_EXPORTS
    #define RB_CUDA_API __declspec(dllexport)
  #else
    #define RB_CUDA_API __declspec(dllimport)
  #endif
#else
  #define RB_CUDA_API __attribute__((visibility("default")))
#endif

extern "C" {

struct RbCudaFloat4
{
    float x;
    float y;
    float z;
    float w;
};

struct RbCudaState
{
    RbCudaFloat4 position;
    RbCudaFloat4 rotation;
    RbCudaFloat4 velocity;
    RbCudaFloat4 angularVelocity;
    RbCudaFloat4 acceleration;
    RbCudaFloat4 angularAcceleration;
};

enum RbCudaIntegrationMethod
{
    RB_CUDA_EXPLICIT = 0,
    RB_CUDA_SEMI_IMPLICIT = 1,
    RB_CUDA_VELOCITY_VERLET = 2,
};

RB_CUDA_API int RbCudaStepSingle(
    RbCudaState* ioState,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation);

RB_CUDA_API int RbCudaStepBatch(
    RbCudaState* ioStates,
    std::size_t count,
    float dt,
    int integrationMethod,
    int enableTranslation,
    int enableRotation);

}
