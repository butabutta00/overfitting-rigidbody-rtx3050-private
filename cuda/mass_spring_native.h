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

struct MssSemiParams
{
    float dt;
    float springStiffness;
    float springDamping;
    float gravityX;
    float gravityY;
    float gravityZ;
    float velocityDamping;
    int substeps;
};

struct MssImplicitParams
{
    float dt;
    float springStiffness;
    float springDamping;
    float gravityX;
    float gravityY;
    float gravityZ;
    float velocityDamping;
    int implicitIterations;
    float cgTolerance;
};

struct MssOneDState
{
    float position;
    float velocity;
};

struct MssOneDParams
{
    float dt;
    float mass;
    float stiffness;
    float damping;
};

MSS_API void* mssCreateSystem();
MSS_API int mssDestroySystem(void* system);

MSS_API int mssUploadTopology(
    void* system,
    int particleCount,
    const float* positionXYZ,
    const float* masses,
    const uint8_t* fixedMask,
    int springCount,
    const int* springEndpoints,
    const float* restLengths);

MSS_API int mssStepSemi(void* system, const MssSemiParams* stepParams);
MSS_API int mssStepImplicit(void* system, const MssImplicitParams* stepParams);

MSS_API int mssDownloadState(void* system, float* outPositionXYZ, float* outVelocityXYZ, int particleCount);

MSS_API int mssOneDImplicitStep(MssOneDState* state, const MssOneDParams* simulationParams);

MSS_API const char* mssGetLastError();

}
