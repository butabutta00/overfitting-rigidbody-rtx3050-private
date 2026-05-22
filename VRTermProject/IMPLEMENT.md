아래 코드가 현재처럼 **강한 스프링 + 감쇠 + 명시적/준명시적 적분** 구조라면, `deltaTime = 0.001`에서는 안정하지만 `0.01`, `0.1`에서 발산하는 것은 자연스러운 현상입니다.

핵심 원인은 다음입니다.

$$
\Delta t
$$

가 커질수록 스프링의 고유진동수

$$
\omega = \sqrt{\frac{k}{m}}
$$

를 시간 적분기가 충분히 해상하지 못하기 때문입니다.

---

# 1. 현재 방식의 안정성 문제

## 1.1 MassSpring 현재 방식

현재 MassSpring은 사실상 다음 형태입니다.

$$
\mathbf{F}_i^n
=
m_i \mathbf{g}
+
\sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}^n
$$

$$
\mathbf{a}_i^n
=
\frac{\mathbf{F}_i^n}{m_i}
$$

$$
\mathbf{v}_i^{n+1}
=
\mathbf{v}_i^n
+
\mathbf{a}_i^n \Delta t
$$

$$
\mathbf{x}_i^{n+1}
=
\mathbf{x}_i^n
+
\mathbf{v}_i^{n+1}\Delta t
$$

$$
\mathbf{v}_i^{n+1}
\leftarrow
0.995\mathbf{v}_i^{n+1}
$$

이는 **semi-implicit Euler**, 또는 **symplectic Euler**입니다.

스프링만 있는 단순계

$$
m\ddot{x} + kx = 0
$$

에 대해 symplectic Euler가 안정하려면 대략

$$
\Delta t \, \omega < 2
$$

즉,

$$
\Delta t < \frac{2}{\omega}
=
2\sqrt{\frac{m}{k}}
$$

이어야 합니다.

감쇠가 있다고 해서 항상 안정해지는 것은 아닙니다. 특히 명시적 감쇠항

$$
-cv
$$

까지 포함하면 큰 시간 간격에서는 오히려 수치적으로 불안정해질 수 있습니다.

---

## 1.2 OneDOF 현재 방식

현재 OneDOF는 다음과 같은 velocity Verlet 변형입니다.

$$
a_0
=
\frac{-kx_0-cv_0}{m}
$$

$$
x_1
=
x_0
+
v_0\Delta t
+
\frac{1}{2}a_0\Delta t^2
$$

$$
v^*
=
v_0
+
a_0\Delta t
$$

$$
a_1
=
\frac{-kx_1-cv^*}{m}
$$

$$
v_1
=
v_0
+
\frac{1}{2}
\left(
a_0+a_1
\right)
\Delta t
$$

이 방식도 결국 힘을 현재 상태에서 평가하는 **명시적 성격**이 강합니다.

따라서

$$
\Delta t = 0.01,\quad 0.1
$$

처럼 커지면 고유진동에 비해 시간 간격이 너무 커져 발산할 수 있습니다.

---

# 2. 목표

요구사항은 다음입니다.

> `deltaTime`이 더 커져도 발산하지 않도록 하는 수치적분 방안을 계산하고 마련하여라.

따라서 명시적 적분 대신 다음 중 하나를 써야 합니다.

1. **Implicit Euler**
2. **Implicit Midpoint**
3. **Newmark-beta method**
4. **Backward Euler with damping**
5. **XPBD / PBD 계열 constraint projection**
6. **substepping**
7. **스프링 힘을 implicit하게 계산하는 안정화 방식**

가장 구현이 쉽고, 큰 $\Delta t$에서도 발산을 막기 좋은 방식은 다음 두 가지입니다.

---

# 3. 권장안 A: OneDOF는 완전 Implicit Euler로 교체

1자유도 스프링-질량-댐퍼는 식을 닫힌 형태로 정확히 풀 수 있습니다.

운동방정식은

$$
m\ddot{x}+c\dot{x}+kx=0
$$

입니다.

Implicit Euler는 다음과 같이 씁니다.

$$
v_{n+1}
=
v_n
+
a_{n+1}\Delta t
$$

$$
x_{n+1}
=
x_n
+
v_{n+1}\Delta t
$$

가속도는 미래 상태에서 계산합니다.

$$
a_{n+1}
=
\frac{-kx_{n+1}-cv_{n+1}}{m}
$$

따라서

$$
v_{n+1}
=
v_n
+
\frac{-kx_{n+1}-cv_{n+1}}{m}
\Delta t
$$

그리고

$$
x_{n+1}
=
x_n
+
v_{n+1}\Delta t
$$

를 대입하면,

$$
v_{n+1}
=
v_n
-
\frac{k\Delta t}{m}
\left(
x_n+v_{n+1}\Delta t
\right)
-
\frac{c\Delta t}{m}v_{n+1}
$$

정리하면,

$$
v_{n+1}
\left(
1
+
\frac{c\Delta t}{m}
+
\frac{k\Delta t^2}{m}
\right)
=
v_n
-
\frac{k\Delta t}{m}x_n
$$

따라서 최종 업데이트 식은 다음입니다.

$$
\boxed{
v_{n+1}
=
\frac{
v_n-\frac{k\Delta t}{m}x_n
}{
1+\frac{c\Delta t}{m}+\frac{k\Delta t^2}{m}
}
}
$$

$$
\boxed{
x_{n+1}
=
x_n
+
v_{n+1}\Delta t
}
$$

또는 분모를 질량 기준으로 정리하면,

$$
\boxed{
v_{n+1}
=
\frac{
mv_n-k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
x_{n+1}
=
x_n+\Delta t v_{n+1}
}
$$

이 방식은 선형 스프링-댐퍼계에 대해 매우 안정적입니다.

큰 $\Delta t$에서도 에너지가 증가하지 않고, 오히려 수치 감쇠가 강해집니다.

---

# 4. OneDOF용 추천 코드

현재 velocity Verlet 부분을 아래 방식으로 바꾸는 것이 좋습니다.

```csharp
float dt = Time.deltaTime;

float denom = mass + damping * dt + stiffness * dt * dt;

float newVelocity = (mass * velocity - stiffness * dt * displacement) / denom;
float newDisplacement = displacement + dt * newVelocity;

velocity = newVelocity;
displacement = newDisplacement;
```

여기서 `displacement`는 평형점 기준 변위 $x$입니다.

외력이 있는 경우, 예를 들어 외력 $F_{\text{ext}}$가 있다면 운동방정식은

$$
m\ddot{x}+c\dot{x}+kx=F_{\text{ext}}
$$

이고 implicit Euler 식은 다음이 됩니다.

$$
\boxed{
v_{n+1}
=
\frac{
mv_n
+
\Delta t F_{\text{ext}}^{n+1}
-
k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
x_{n+1}
=
x_n+\Delta t v_{n+1}
}
$$

코드는 다음과 같습니다.

```csharp
float dt = Time.deltaTime;

float externalForce = 0.0f;

float denom = mass + damping * dt + stiffness * dt * dt;

float newVelocity =
    (mass * velocity + dt * externalForce - stiffness * dt * displacement)
    / denom;

float newDisplacement = displacement + dt * newVelocity;

velocity = newVelocity;
displacement = newDisplacement;
```

---

# 5. 권장안 B: MassSpring에는 implicit spring-damper pair solver 적용

MassSpring 전체를 완전 implicit Euler로 풀려면 전역 선형 시스템을 풀어야 합니다.

$$
\left(
\mathbf{M}
+
\Delta t\mathbf{C}
+
\Delta t^2\mathbf{K}
\right)
\mathbf{v}^{n+1}
=
\mathbf{M}\mathbf{v}^{n}
+
\Delta t \mathbf{F}_{\text{ext}}
-
\Delta t \nabla E(\mathbf{x}^{n})
$$

하지만 Unity C#에서 전체 시스템 행렬을 구성하고 CG solver까지 구현하려면 복잡합니다.

따라서 실용적으로는 각 스프링 쌍에 대해 **상대 운동 방향으로 implicit spring-damper impulse를 적용**하는 방식이 좋습니다.

---

# 6. MassSpring pairwise implicit 방식 유도

스프링 쌍 $i,j$에 대해

$$
\mathbf{d}_{ij}
=
\mathbf{x}_j-\mathbf{x}_i
$$

$$
\ell
=
\|\mathbf{d}_{ij}\|
$$

$$
\hat{\mathbf{n}}
=
\frac{\mathbf{d}_{ij}}{\ell}
$$

$$
C
=
\ell-\ell_0
$$

상대속도는

$$
\mathbf{v}_{rel}
=
\mathbf{v}_j-\mathbf{v}_i
$$

이고 스프링 방향 성분은

$$
v_{rel,n}
=
\mathbf{v}_{rel}\cdot \hat{\mathbf{n}}
$$

입니다.

감소시켜야 할 스프링 방향 운동은 사실상 1자유도 문제입니다.

$$
m_{\text{eff}}\ddot{C}
+
c\dot{C}
+
kC
=
0
$$

여기서 effective mass는

$$
\frac{1}{m_{\text{eff}}}
=
\frac{1}{m_i}
+
\frac{1}{m_j}
$$

즉,

$$
m_{\text{eff}}
=
\frac{1}{
\frac{1}{m_i}+\frac{1}{m_j}
}
$$

입니다.

고정점이 있는 경우 해당 질점의 inverse mass를 0으로 두면 됩니다.

$$
w_i = \frac{1}{m_i},\qquad w_j = \frac{1}{m_j}
$$

$$
w = w_i+w_j
$$

$$
m_{\text{eff}} = \frac{1}{w}
$$

현재 변위 $C$와 방향 속도 $v_{rel,n}$에 대해 implicit Euler를 적용하면,

$$
v_{rel,n}^{n+1}
=
\frac{
m_{\text{eff}}v_{rel,n}^{n}
-
k\Delta t C^n
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
$$

변화시켜야 할 상대속도는

$$
\Delta v_{rel,n}
=
v_{rel,n}^{n+1}
-
v_{rel,n}^{n}
$$

이를 impulse로 표현하면

$$
J
=
m_{\text{eff}}
\Delta v_{rel,n}
$$

입니다.

방향 impulse는

$$
\mathbf{J}
=
J\hat{\mathbf{n}}
$$

각 질점 속도 보정은 다음입니다.

$$
\mathbf{v}_i^{n+1}
\leftarrow
\mathbf{v}_i^{n+1}
-
w_i\mathbf{J}
$$

$$
\mathbf{v}_j^{n+1}
\leftarrow
\mathbf{v}_j^{n+1}
+
w_j\mathbf{J}
$$

단, 부호는 코드에서 정의한 $\mathbf{v}_{rel}=\mathbf{v}_j-\mathbf{v}_i$, $\hat{\mathbf{n}}=(\mathbf{x}_j-\mathbf{x}_i)/\ell$ 기준에 맞춰야 합니다.

위 식은 스프링 방향 상대속도를 implicit하게 안정화합니다.

---

# 7. MassSpring용 pairwise implicit 업데이트 절차

한 프레임에서 다음 순서로 처리합니다.

---

## Step 1. 외력 적용

중력 등 외력만 먼저 속도에 반영합니다.

$$
\mathbf{v}_i
\leftarrow
\mathbf{v}_i
+
\Delta t \mathbf{g}
$$

또는 질량을 포함해,

$$
\mathbf{v}_i
\leftarrow
\mathbf{v}_i
+
\Delta t \frac{\mathbf{F}_{ext,i}}{m_i}
$$

---

## Step 2. 각 스프링에 대해 implicit velocity correction

각 스프링 $(i,j)$에 대해

$$
\mathbf{d}_{ij}
=
\mathbf{x}_j-\mathbf{x}_i
$$

$$
\ell
=
\|\mathbf{d}_{ij}\|
$$

$$
\hat{\mathbf{n}}
=
\frac{\mathbf{d}_{ij}}{\ell}
$$

$$
C
=
\ell-\ell_0
$$

$$
v_{rel,n}
=
(\mathbf{v}_j-\mathbf{v}_i)\cdot \hat{\mathbf{n}}
$$

$$
w_i=\frac{1}{m_i},\qquad w_j=\frac{1}{m_j}
$$

$$
w=w_i+w_j
$$

$$
m_{\text{eff}}=\frac{1}{w}
$$

$$
v_{rel,n}^{new}
=
\frac{
m_{\text{eff}}v_{rel,n}
-
k\Delta t C
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
$$

$$
\Delta v_{rel,n}
=
v_{rel,n}^{new}
-
v_{rel,n}
$$

$$
J
=
m_{\text{eff}}\Delta v_{rel,n}
$$

$$
\mathbf{J}=J\hat{\mathbf{n}}
$$

$$
\mathbf{v}_i
\leftarrow
\mathbf{v}_i
-
w_i\mathbf{J}
$$

$$
\mathbf{v}_j
\leftarrow
\mathbf{v}_j
+
w_j\mathbf{J}
$$

---

## Step 3. 위치 업데이트

$$
\mathbf{x}_i^{n+1}
=
\mathbf{x}_i^n
+
\Delta t \mathbf{v}_i^{n+1}
$$

---

## Step 4. 전역 속도 감쇠

필요하면 기존처럼 약한 전역 감쇠를 적용합니다.

$$
\mathbf{v}_i
\leftarrow
\gamma \mathbf{v}_i
$$

예:

$$
\gamma = 0.995
$$

하지만 $\gamma=0.995$는 프레임당 감쇠라서 $\Delta t$가 바뀌면 물리적 의미가 달라집니다.

더 좋은 방식은 시간 기반 감쇠입니다.

$$
\gamma(\Delta t)
=
e^{-\lambda \Delta t}
$$

예를 들어 기존에 $\Delta t=0.001$에서 $0.995$를 쓰고 있었다면,

$$
0.995=e^{-\lambda 0.001}
$$

따라서

$$
\lambda
=
-\frac{\ln(0.995)}{0.001}
\approx
5.0125
$$

그러면 임의의 $\Delta t$에서

$$
\boxed{
\gamma(\Delta t)
=
e^{-5.0125\Delta t}
}
$$

를 쓰면 됩니다.

예:

$$
\Delta t=0.001
\Rightarrow
\gamma \approx 0.995
$$

$$
\Delta t=0.01
\Rightarrow
\gamma \approx 0.9511
$$

$$
\Delta t=0.1
\Rightarrow
\gamma \approx 0.6058
$$

---

# 8. MassSpring pairwise implicit 코드 예시

아래는 기존 force accumulation 대신 넣을 수 있는 구조입니다.

```csharp
void StepImplicitPairwise(float dt)
{
    // 1. External forces, e.g. gravity
    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity += gravity * dt;
    }

    // 2. Pairwise implicit spring-damper velocity solve
    for (int s = 0; s < springs.Count; s++)
    {
        int i = springs[s].i;
        int j = springs[s].j;

        Particle pi = particles[i];
        Particle pj = particles[j];

        Vector3 xi = pi.position;
        Vector3 xj = pj.position;

        Vector3 vi = pi.velocity;
        Vector3 vj = pj.velocity;

        Vector3 d = xj - xi;
        float length = d.magnitude;

        if (length < 1e-6f)
            continue;

        Vector3 n = d / length;

        float C = length - springs[s].restLength;

        float wi = pi.isFixed ? 0.0f : 1.0f / pi.mass;
        float wj = pj.isFixed ? 0.0f : 1.0f / pj.mass;
        float w = wi + wj;

        if (w <= 0.0f)
            continue;

        float meff = 1.0f / w;

        float vRel = Vector3.Dot(vj - vi, n);

        float k = springs[s].stiffness;
        float c = springs[s].damping;

        float denom = meff + c * dt + k * dt * dt;

        float vRelNew = (meff * vRel - k * dt * C) / denom;

        float deltaVRel = vRelNew - vRel;

        float impulseScalar = meff * deltaVRel;

        Vector3 impulse = impulseScalar * n;

        if (!pi.isFixed)
            vi -= wi * impulse;

        if (!pj.isFixed)
            vj += wj * impulse;

        pi.velocity = vi;
        pj.velocity = vj;

        particles[i] = pi;
        particles[j] = pj;
    }

    // 3. Position update
    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].position += particles[i].velocity * dt;
    }

    // 4. Time-consistent global damping
    float lambda = 5.0125f;
    float gamma = Mathf.Exp(-lambda * dt);

    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity *= gamma;
    }
}
```

---

# 9. 부호 검증

위 방식의 핵심은 다음입니다.

$$
C=\ell-\ell_0
$$

입니다.

만약 스프링이 늘어나면,

$$
C>0
$$

입니다.

늘어난 상태에서는 두 질점이 서로 가까워지도록 상대속도

$$
v_{rel,n}
=
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}
$$

가 감소해야 합니다.

implicit 식은

$$
v_{rel,n}^{new}
=
\frac{
m_{\text{eff}}v_{rel,n}
-
k\Delta t C
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
$$

이므로 $C>0$일 때 $v_{rel,n}^{new}$는 작아집니다.

즉, 두 점이 벌어지는 속도를 줄이거나, 서로 가까워지는 방향으로 바꿉니다.

따라서 방향성이 맞습니다.

---

# 10. 더 강력한 권장안 C: XPBD로 전환

만약 `deltaTime = 0.1`처럼 매우 큰 값에서도 시각적으로 터지지 않는 것을 최우선으로 한다면, 물리적 정확도보다 안정성이 좋은 **XPBD**가 더 적합합니다.

스프링을 힘이 아니라 constraint로 봅니다.

$$
C(\mathbf{x}_i,\mathbf{x}_j)
=
\|\mathbf{x}_j-\mathbf{x}_i\|-\ell_0
$$

XPBD의 compliance를

$$
\alpha = \frac{1}{k}
$$

라고 두고, 시간 보정 compliance는

$$
\tilde{\alpha}
=
\frac{\alpha}{\Delta t^2}
$$

입니다.

constraint multiplier 증분은

$$
\Delta \lambda
=
\frac{
-C-\tilde{\alpha}\lambda
}{
w_i\|\nabla_{\mathbf{x}_i}C\|^2
+
w_j\|\nabla_{\mathbf{x}_j}C\|^2
+
\tilde{\alpha}
}
$$

거리 constraint에서는

$$
\nabla_{\mathbf{x}_i}C=-\hat{\mathbf{n}}
$$

$$
\nabla_{\mathbf{x}_j}C=\hat{\mathbf{n}}
$$

이므로,

$$
\Delta \lambda
=
\frac{
-C-\tilde{\alpha}\lambda
}{
w_i+w_j+\tilde{\alpha}
}
$$

위치 보정은

$$
\Delta \mathbf{x}_i
=
w_i\Delta\lambda
\left(
-\hat{\mathbf{n}}
\right)
$$

$$
\Delta \mathbf{x}_j
=
w_j\Delta\lambda
\hat{\mathbf{n}}
$$

단, 위 부호는 $\hat{\mathbf{n}}=(\mathbf{x}_j-\mathbf{x}_i)/\ell$ 기준입니다.

실무에서는 더 직관적으로 다음처럼 쓰는 경우가 많습니다.

$$
\mathbf{x}_i
\leftarrow
\mathbf{x}_i
+
w_i \Delta\lambda \hat{\mathbf{n}}
$$

$$
\mathbf{x}_j
\leftarrow
\mathbf{x}_j
-
w_j \Delta\lambda \hat{\mathbf{n}}
$$

이때는

$$
\Delta\lambda
=
\frac{
C-\tilde{\alpha}\lambda
}{
w_i+w_j+\tilde{\alpha}
}
$$

처럼 부호 정의를 맞춰야 합니다.

XPBD는 여러 iteration을 돌릴수록 강성이 높아지고 안정해집니다.

---

# 11. XPBD 업데이트 순서

한 time step에서 다음처럼 처리합니다.

## Step 1. 속도에 외력 적용

$$
\mathbf{v}_i
\leftarrow
\mathbf{v}_i+\Delta t\mathbf{g}
$$

## Step 2. 예측 위치 계산

$$
\mathbf{x}_i^*
=
\mathbf{x}_i^n+\Delta t\mathbf{v}_i
$$

## Step 3. constraint projection 반복

각 spring constraint에 대해 여러 번 반복합니다.

$$
C
=
\|\mathbf{x}_j^*-\mathbf{x}_i^*\|-\ell_0
$$

$$
\Delta \lambda
=
\frac{
C-\tilde{\alpha}\lambda
}{
w_i+w_j+\tilde{\alpha}
}
$$

$$
\mathbf{x}_i^*
\leftarrow
\mathbf{x}_i^*
+
w_i\Delta\lambda\hat{\mathbf{n}}
$$

$$
\mathbf{x}_j^*
\leftarrow
\mathbf{x}_j^*
-
w_j\Delta\lambda\hat{\mathbf{n}}
$$

## Step 4. 속도 재계산

$$
\mathbf{v}_i^{n+1}
=
\frac{
\mathbf{x}_i^*-\mathbf{x}_i^n
}{
\Delta t
}
$$

## Step 5. 위치 확정

$$
\mathbf{x}_i^{n+1}
=
\mathbf{x}_i^*
$$

---

# 12. XPBD 코드 예시

```csharp
void StepXPBD(float dt)
{
    int n = particles.Count;

    Vector3[] oldPositions = new Vector3[n];
    Vector3[] predictedPositions = new Vector3[n];

    for (int i = 0; i < n; i++)
    {
        oldPositions[i] = particles[i].position;

        if (particles[i].isFixed)
        {
            predictedPositions[i] = particles[i].position;
            continue;
        }

        particles[i].velocity += gravity * dt;
        predictedPositions[i] = particles[i].position + particles[i].velocity * dt;
    }

    float[] lambdas = new float[springs.Count];

    int solverIterations = 8;

    for (int iter = 0; iter < solverIterations; iter++)
    {
        for (int s = 0; s < springs.Count; s++)
        {
            int i = springs[s].i;
            int j = springs[s].j;

            Vector3 xi = predictedPositions[i];
            Vector3 xj = predictedPositions[j];

            Vector3 d = xj - xi;
            float length = d.magnitude;

            if (length < 1e-6f)
                continue;

            Vector3 nDir = d / length;

            float C = length - springs[s].restLength;

            float wi = particles[i].isFixed ? 0.0f : 1.0f / particles[i].mass;
            float wj = particles[j].isFixed ? 0.0f : 1.0f / particles[j].mass;
            float w = wi + wj;

            if (w <= 0.0f)
                continue;

            float k = springs[s].stiffness;

            float compliance = 1.0f / Mathf.Max(k, 1e-6f);
            float alphaTilde = compliance / (dt * dt);

            float deltaLambda =
                (C - alphaTilde * lambdas[s])
                / (w + alphaTilde);

            lambdas[s] += deltaLambda;

            if (!particles[i].isFixed)
                predictedPositions[i] += wi * deltaLambda * nDir;

            if (!particles[j].isFixed)
                predictedPositions[j] -= wj * deltaLambda * nDir;
        }
    }

    float lambdaDamping = 5.0125f;
    float gamma = Mathf.Exp(-lambdaDamping * dt);

    for (int i = 0; i < n; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity =
            (predictedPositions[i] - oldPositions[i]) / dt;

        particles[i].velocity *= gamma;

        particles[i].position = predictedPositions[i];
    }
}
```

---

# 13. Implicit pairwise vs XPBD 비교

| 방식 | 안정성 | 물리 정확도 | 구현 난이도 | 큰 $\Delta t$ 대응 |
|---|---:|---:|---:|---:|
| 현재 Semi-Implicit Euler | 낮음~중간 | 중간 | 쉬움 | 약함 |
| 현재 Velocity Verlet | 중간 | 중간~높음 | 쉬움 | 약함 |
| OneDOF Implicit Euler | 높음 | 중간 | 매우 쉬움 | 강함 |
| MassSpring Pairwise Implicit | 높음 | 중간 | 중간 | 강함 |
| XPBD | 매우 높음 | 중간 | 중간 | 매우 강함 |
| Full Implicit Euler + CG | 매우 높음 | 높음 | 어려움 | 매우 강함 |

---

# 14. 결론 추천

## OneDOF

현재 velocity Verlet 대신 다음의 **closed-form implicit Euler**를 추천합니다.

$$
\boxed{
v_{n+1}
=
\frac{
mv_n-k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
x_{n+1}
=
x_n+\Delta t v_{n+1}
}
$$

외력이 있으면,

$$
\boxed{
v_{n+1}
=
\frac{
mv_n+\Delta t F_{\text{ext}}-k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
x_{n+1}
=
x_n+\Delta t v_{n+1}
}
$$

---

## MassSpring

목표가 **큰 deltaTime에서도 발산 방지**라면 우선순위는 다음입니다.

### 1순위: XPBD

시각적으로 안정적인 spring-mass를 원하면 XPBD가 가장 안전합니다.

$$
\boxed{
\Delta \lambda
=
\frac{
C-\tilde{\alpha}\lambda
}{
w_i+w_j+\tilde{\alpha}
}
}
$$

$$
\boxed{
\tilde{\alpha}
=
\frac{1}{k\Delta t^2}
}
$$

$$
\boxed{
\mathbf{x}_i
\leftarrow
\mathbf{x}_i+w_i\Delta\lambda\hat{\mathbf{n}}
}
$$

$$
\boxed{
\mathbf{x}_j
\leftarrow
\mathbf{x}_j-w_j\Delta\lambda\hat{\mathbf{n}}
}
$$

---

### 2순위: Pairwise implicit spring-damper

현재 force-based 구조를 최대한 유지하고 싶다면 이 방식이 좋습니다.

$$
\boxed{
v_{rel,n}^{new}
=
\frac{
m_{\text{eff}}v_{rel,n}
-
k\Delta t C
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
\Delta v_{rel,n}
=
v_{rel,n}^{new}-v_{rel,n}
}
$$

$$
\boxed{
\mathbf{J}
=
m_{\text{eff}}\Delta v_{rel,n}\hat{\mathbf{n}}
}
$$

$$
\boxed{
\mathbf{v}_i
\leftarrow
\mathbf{v}_i-w_i\mathbf{J}
}
$$

$$
\boxed{
\mathbf{v}_j
\leftarrow
\mathbf{v}_j+w_j\mathbf{J}
}
$$

---

# 15. 최종 제안

`deltaTime = 0.01`, `0.1`에서도 발산하지 않는 것을 목표로 한다면 다음 조합을 추천합니다.

$$
\boxed{
\text{OneDOF} \rightarrow \text{Implicit Euler}
}
$$

$$
\boxed{
\text{MassSpring} \rightarrow \text{XPBD 또는 Pairwise Implicit}
}
$$

가장 구현 안정성이 높은 조합은 다음입니다.

$$
\boxed{
\text{OneDOF: closed-form Implicit Euler}
}
$$

$$
\boxed{
\text{MassSpring: XPBD, solver iteration } 5\sim 10
}
$$

그리고 기존의 프레임당 감쇠

$$
\mathbf{v}\leftarrow0.995\mathbf{v}
$$

는 반드시 시간 기반 감쇠로 바꾸는 것이 좋습니다.

$$
\boxed{
\mathbf{v}
\leftarrow
e^{-\lambda\Delta t}\mathbf{v}
}
$$

기존 `dt = 0.001`에서 `0.995`와 동일하게 맞추려면,

$$
\boxed{
\lambda
=
-\frac{\ln 0.995}{0.001}
\approx
5.0125
}
$$

따라서,

$$
\boxed{
\mathbf{v}
\leftarrow
e^{-5.0125\Delta t}\mathbf{v}
}
$$

를 사용하면 됩니다.