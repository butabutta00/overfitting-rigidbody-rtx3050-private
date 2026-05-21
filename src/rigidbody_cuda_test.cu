#include "rigidbody_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{

struct F3
{
    float x;
    float y;
    float z;
};

struct F4
{
    float x;
    float y;
    float z;
    float w;
};

F3 Add(const F3& a, const F3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

F3 Mul(const F3& a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

float Dot(const F3& a, const F3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

F3 NormalizeSafe(const F3& v)
{
    float len2 = Dot(v, v);
    if (len2 <= 1e-20f)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    float inv = 1.0f / std::sqrt(len2);
    return {v.x * inv, v.y * inv, v.z * inv};
}

F4 QMul(const F4& a, const F4& b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    };
}

F4 QNormalize(const F4& q)
{
    float len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    float inv = 1.0f / std::sqrt(len2 > 1e-20f ? len2 : 1e-20f);
    return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

F4 IntegrateRotation(const F4& q, const F3& step)
{
    float angle = std::sqrt(Dot(step, step));
    if (angle <= 1e-12f)
    {
        return q;
    }
    F3 axis = NormalizeSafe(step);
    float half = 0.5f * angle;
    float s = std::sin(half);
    float c = std::cos(half);
    F4 delta{axis.x * s, axis.y * s, axis.z * s, c};
    return QNormalize(QMul(delta, q));
}

void StepCpu(RbCudaState& s, float dt, int method, int enableTranslation, int enableRotation)
{
    F3 p{s.position.x, s.position.y, s.position.z};
    F4 q{s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w};
    F3 v{s.velocity.x, s.velocity.y, s.velocity.z};
    F3 w{s.angularVelocity.x, s.angularVelocity.y, s.angularVelocity.z};
    F3 a{s.acceleration.x, s.acceleration.y, s.acceleration.z};
    F3 alpha{s.angularAcceleration.x, s.angularAcceleration.y, s.angularAcceleration.z};

    if (method == RB_CUDA_EXPLICIT)
    {
        if (enableTranslation)
        {
            p = Add(p, Mul(v, dt));
            v = Add(v, Mul(a, dt));
        }
        if (enableRotation)
        {
            q = IntegrateRotation(q, Mul(w, dt));
            w = Add(w, Mul(alpha, dt));
        }
    }
    else if (method == RB_CUDA_SEMI_IMPLICIT)
    {
        if (enableTranslation)
        {
            v = Add(v, Mul(a, dt));
            p = Add(p, Mul(v, dt));
        }
        if (enableRotation)
        {
            w = Add(w, Mul(alpha, dt));
            q = IntegrateRotation(q, Mul(w, dt));
        }
    }
    else
    {
        if (enableTranslation)
        {
            p = Add(p, Add(Mul(v, dt), Mul(a, 0.5f * dt * dt)));
            v = Add(v, Mul(a, dt));
        }
        if (enableRotation)
        {
            F3 wMid = Add(w, Mul(alpha, 0.5f * dt));
            q = IntegrateRotation(q, Mul(wMid, dt));
            w = Add(w, Mul(alpha, dt));
        }
    }

    s.position = {p.x, p.y, p.z, 0.0f};
    s.rotation = {q.x, q.y, q.z, q.w};
    s.velocity = {v.x, v.y, v.z, 0.0f};
    s.angularVelocity = {w.x, w.y, w.z, 0.0f};
}

bool AlmostEq(float a, float b, float eps)
{
    return std::fabs(a - b) <= eps;
}

bool CompareStates(const RbCudaState& a, const RbCudaState& b, float eps)
{
    return AlmostEq(a.position.x, b.position.x, eps) &&
           AlmostEq(a.position.y, b.position.y, eps) &&
           AlmostEq(a.position.z, b.position.z, eps) &&
           AlmostEq(a.rotation.x, b.rotation.x, eps) &&
           AlmostEq(a.rotation.y, b.rotation.y, eps) &&
           AlmostEq(a.rotation.z, b.rotation.z, eps) &&
           AlmostEq(a.rotation.w, b.rotation.w, eps) &&
           AlmostEq(a.velocity.x, b.velocity.x, eps) &&
           AlmostEq(a.velocity.y, b.velocity.y, eps) &&
           AlmostEq(a.velocity.z, b.velocity.z, eps) &&
           AlmostEq(a.angularVelocity.x, b.angularVelocity.x, eps) &&
           AlmostEq(a.angularVelocity.y, b.angularVelocity.y, eps) &&
           AlmostEq(a.angularVelocity.z, b.angularVelocity.z, eps);
}

float RandRange(float lo, float hi)
{
    float t = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    return lo + (hi - lo) * t;
}

void FillRandom(RbCudaState& s)
{
    s.position = {RandRange(-3.0f, 3.0f), RandRange(-3.0f, 3.0f), RandRange(-3.0f, 3.0f), 0.0f};
    s.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    s.velocity = {RandRange(-1.0f, 1.0f), RandRange(-1.0f, 1.0f), RandRange(-1.0f, 1.0f), 0.0f};
    s.angularVelocity = {RandRange(-0.5f, 0.5f), RandRange(-0.5f, 0.5f), RandRange(-0.5f, 0.5f), 0.0f};
    s.acceleration = {RandRange(-2.0f, 2.0f), RandRange(-2.0f, 2.0f), RandRange(-2.0f, 2.0f), 0.0f};
    s.angularAcceleration = {RandRange(-1.0f, 1.0f), RandRange(-1.0f, 1.0f), RandRange(-1.0f, 1.0f), 0.0f};
}

int RunCase(std::size_t count, int method)
{
    std::vector<RbCudaState> gpu(count);
    std::vector<RbCudaState> cpu(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        FillRandom(gpu[i]);
        cpu[i] = gpu[i];
    }

    const float dt = 1.0f / 120.0f;
    int status = RbCudaStepBatch(gpu.data(), count, dt, method, 1, 1);
    if (status != 0)
    {
        std::printf("RbCudaStepBatch failed method=%d status=%d\n", method, status);
        return 1;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        StepCpu(cpu[i], dt, method, 1, 1);
    }

    const float eps = 2e-3f;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!CompareStates(gpu[i], cpu[i], eps))
        {
            std::printf("Mismatch method=%d index=%zu\n", method, i);
            return 2;
        }
    }

    return 0;
}

}

int main(int argc, char** argv)
{
    std::srand(7);

    std::size_t count = 1 << 20;
    if (argc > 1)
    {
        count = static_cast<std::size_t>(std::atoll(argv[1]));
    }

    int s1 = RunCase(count, RB_CUDA_SEMI_IMPLICIT);
    int s2 = RunCase(count, RB_CUDA_VELOCITY_VERLET);

    if (s1 == 0 && s2 == 0)
    {
        std::printf("PASS count=%zu methods=semi-implicit,velocity-verlet\n", count);
        return 0;
    }

    std::printf("FAIL\n");
    return 1;
}
