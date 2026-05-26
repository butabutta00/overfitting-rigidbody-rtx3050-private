#include "mass_spring_native.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace
{
std::unordered_map<std::string, std::string> ParseOptions(int argc, char** argv)
{
    std::unordered_map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i)
    {
        std::string key(argv[i]);
        if (key == "--help" || key == "-h")
        {
            options[key] = "1";
            continue;
        }

        if (key.rfind("--", 0) != 0)
        {
            continue;
        }

        if (i + 1 >= argc)
        {
            continue;
        }

        options[key] = argv[++i];
    }
    return options;
}

float ParseFloatOrDefault(
    const std::unordered_map<std::string, std::string>& options,
    const char* key,
    float defaultValue)
{
    auto it = options.find(key);
    if (it == options.end())
    {
        return defaultValue;
    }

    try
    {
        return std::stof(it->second);
    }
    catch (...)
    {
        return defaultValue;
    }
}

int ParseIntOrDefault(
    const std::unordered_map<std::string, std::string>& options,
    const char* key,
    int defaultValue)
{
    auto it = options.find(key);
    if (it == options.end())
    {
        return defaultValue;
    }

    try
    {
        return std::stoi(it->second);
    }
    catch (...)
    {
        return defaultValue;
    }
}

void PrintHelp()
{
    std::cout << "CUDA 1D Spring-Mass-Damper benchmark options:\n";
    std::cout << "  --position <float>   Initial position (default: 1.0)\n";
    std::cout << "  --velocity <float>   Initial velocity (default: 0.0)\n";
    std::cout << "  --dt <float>         Time-step (default: 0.016)\n";
    std::cout << "  --mass <float>       Mass (default: 1.0)\n";
    std::cout << "  --stiffness <float>  Spring stiffness (default: 120.0)\n";
    std::cout << "  --damping <float>    Spring damping (default: 0.2)\n";
    std::cout << "  --steps <int>        Integration steps (default: 8)\n";
}
}

int main(int argc, char** argv)
{
    auto options = ParseOptions(argc, argv);
    if (options.find("--help") != options.end() || options.find("-h") != options.end())
    {
        PrintHelp();
        return 0;
    }

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

    std::cout << std::fixed << std::setprecision(6);

    MssOneDState state{};
    state.position = ParseFloatOrDefault(options, "--position", 1.0f);
    state.velocity = ParseFloatOrDefault(options, "--velocity", 0.0f);

    MssOneDParams params{};
    params.dt = ParseFloatOrDefault(options, "--dt", 0.016f);
    params.mass = ParseFloatOrDefault(options, "--mass", 1.0f);
    params.stiffness = ParseFloatOrDefault(options, "--stiffness", 120.0f);
    params.damping = ParseFloatOrDefault(options, "--damping", 0.2f);

    int steps = ParseIntOrDefault(options, "--steps", 8);
    if (steps < 1)
    {
        steps = 1;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i)
    {
        int rc = mssOneDImplicitStep(&state, &params);
        if (rc != 0)
        {
            std::cerr << "mssOneDImplicitStep failed: " << mssGetLastError() << '\n';
            return 4;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    const double stepsPerSec = static_cast<double>(steps) / std::max(elapsedMs / 1000.0, 1e-12);
    const float checksum = state.position + state.velocity;

    std::cout << "mass_spring_benchmark\n";
    std::cout << "backend=cuda\n";
    std::cout << "steps=" << steps << '\n';
    std::cout << "position=" << state.position << " velocity=" << state.velocity << '\n';
    std::cout << "dt=" << params.dt << " mass=" << params.mass << " stiffness=" << params.stiffness << " damping=" << params.damping << '\n';
    std::cout << "elapsed_ms=" << elapsedMs << " steps_per_sec=" << stepsPerSec << '\n';
    std::cout << "output_x=" << state.position << " output_v=" << state.velocity << '\n';
    std::cout << "checksum=" << checksum << '\n';
    return 0;
}
