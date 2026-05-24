#include "mass_spring_native.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
float ParseOrDefault(int argc, char** argv, int index, float defaultValue)
{
    if (index >= argc)
    {
        return defaultValue;
    }

    try
    {
        return std::stof(argv[index]);
    }
    catch (...)
    {
        return defaultValue;
    }
}

int ParseOrDefaultInt(int argc, char** argv, int index, int defaultValue)
{
    if (index >= argc)
    {
        return defaultValue;
    }

    try
    {
        return std::stoi(argv[index]);
    }
    catch (...)
    {
        return defaultValue;
    }
}
}

int main(int argc, char** argv)
{
    int deviceCount = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "CUDA runtime init failed: " << cudaGetErrorString(cudaErr) << '\n';
        return 2;
    }

    std::cout << "CUDA devices: " << deviceCount << '\n';
    if (deviceCount <= 0)
    {
        std::cerr << "No CUDA device detected." << '\n';
        return 3;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
    {
        std::cout << "Using GPU[0]: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << '\n';
    }

    // Positional inputs with defaults for quick validation run:
    // [1] position [2] velocity [3] dt [4] mass [5] stiffness [6] damping [7] steps
    MssOneDState state{};
    state.position = ParseOrDefault(argc, argv, 1, 1.0f);
    state.velocity = ParseOrDefault(argc, argv, 2, 0.0f);

    MssOneDParams params{};
    params.dt = ParseOrDefault(argc, argv, 3, 0.016f);
    params.mass = ParseOrDefault(argc, argv, 4, 1.0f);
    params.stiffness = ParseOrDefault(argc, argv, 5, 120.0f);
    params.damping = ParseOrDefault(argc, argv, 6, 0.2f);

    int steps = ParseOrDefaultInt(argc, argv, 7, 8);
    if (steps < 1)
    {
        steps = 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Input(dummy/default) -> x=" << state.position
              << ", v=" << state.velocity
              << ", dt=" << params.dt
              << ", m=" << params.mass
              << ", k=" << params.stiffness
              << ", c=" << params.damping
              << ", steps=" << steps
              << '\n';

    for (int i = 0; i < steps; ++i)
    {
        int rc = mssOneDImplicitStep(&state, &params);
        if (rc != 0)
        {
            std::cerr << "mssOneDImplicitStep failed: " << mssGetLastError() << '\n';
            return 4;
        }
    }

    std::cout << "Output -> x=" << state.position << ", v=" << state.velocity << '\n';
    return 0;
}
