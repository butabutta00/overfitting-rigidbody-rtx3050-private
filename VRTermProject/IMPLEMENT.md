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


---

가능합니다. **“큰 $\Delta t$에서도 발산하지 않게”** 하려면 XPBD나 implicit 계열이 좋지만, 그중에서도 성능을 더 아끼려면 **전역 implicit 선형계 풀이**나 많은 iteration을 쓰는 XPBD보다, 현재 MassSpring 구조에 가까운 **pairwise implicit / compliant spring projection / quasi-implicit damping** 계열이 적합합니다.

아래는 **수식 결과는 유사하게 유지하면서 성능을 개선하는 방안**입니다.

---

# 1. 핵심 추천

성능과 안정성의 균형을 보면 다음 순서를 추천합니다.

$$
\boxed{
\text{1순위: Pairwise Implicit Spring-Damper 1-pass}
}
$$

$$
\boxed{
\text{2순위: XPBD를 1\sim 3 iteration으로 제한}
}
$$

$$
\boxed{
\text{3순위: 현재 Semi-Implicit Euler + 안정화된 implicit damping}
}
$$

가장 추천하는 것은 **Pairwise Implicit Spring-Damper 1-pass**입니다.

이 방식은 각 스프링마다 닫힌 형태의 1D implicit update만 수행하므로, 전체 행렬을 풀 필요가 없습니다.

즉,

$$
O(N_{\text{spring}})
$$

복잡도를 유지하면서, 현재 명시적 spring force보다 훨씬 안정적입니다.

---

# 2. 기존 명시적 힘 계산의 비용

현재 MassSpring은 스프링마다 대략 다음을 계산합니다.

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
f_s
=
k(\ell-\ell_0)
$$

$$
f_d
=
c
\left(
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}
\right)
$$

$$
\mathbf{F}_{ij}
=
\hat{\mathbf{n}}
(f_s+f_d)
$$

$$
\mathbf{F}_i \mathrel{+}= \mathbf{F}_{ij}
$$

$$
\mathbf{F}_j \mathrel{-}= \mathbf{F}_{ij}
$$

이후 각 질점에 대해

$$
\mathbf{v}_i^{n+1}
=
\mathbf{v}_i^n
+
\frac{\mathbf{F}_i^n}{m_i}\Delta t
$$

$$
\mathbf{x}_i^{n+1}
=
\mathbf{x}_i^n+\mathbf{v}_i^{n+1}\Delta t
$$

입니다.

이 방식은 빠르지만, 큰 $\Delta t$에서 안정성이 약합니다.

---

# 3. 개선안 A — Force Accumulation 제거형 Pairwise Implicit

기존에는 모든 spring force를 누적한 뒤 한 번에 속도를 업데이트했습니다.

개선안은 힘을 저장하지 않고, 각 spring마다 바로 velocity correction을 적용합니다.

## 3.1 수식

스프링 쌍 $(i,j)$에 대해

$$
\mathbf{d}_{ij}
=
\mathbf{x}_j-\mathbf{x}_i
$$

$$
\ell_{ij}
=
\|\mathbf{d}_{ij}\|
$$

$$
\hat{\mathbf{n}}_{ij}
=
\frac{\mathbf{d}_{ij}}{\ell_{ij}}
$$

$$
C_{ij}
=
\ell_{ij}-\ell_{ij}^0
$$

상대속도 방향 성분은

$$
v_{rel}
=
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}_{ij}
$$

inverse mass는

$$
w_i=\frac{1}{m_i},\qquad w_j=\frac{1}{m_j}
$$

$$
w=w_i+w_j
$$

$$
m_{\text{eff}}
=
\frac{1}{w}
$$

입니다.

스프링 방향 운동을 1D implicit Euler로 보면,

$$
m_{\text{eff}}\dot{v}
+
cv
+
kC
=
0
$$

입니다.

Implicit update는 다음처럼 둘 수 있습니다.

$$
v_{rel}^{n+1}
=
\frac{
m_{\text{eff}}v_{rel}^{n}
-
k\Delta t C_{ij}^{n}
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
$$

상대속도 보정량은

$$
\Delta v_{rel}
=
v_{rel}^{n+1}-v_{rel}^{n}
$$

Impulse scalar는

$$
J
=
m_{\text{eff}}\Delta v_{rel}
$$

따라서,

$$
\mathbf{J}
=
J\hat{\mathbf{n}}_{ij}
$$

속도 보정은

$$
\boxed{
\mathbf{v}_i
\leftarrow
\mathbf{v}_i
-
w_i\mathbf{J}
}
$$

$$
\boxed{
\mathbf{v}_j
\leftarrow
\mathbf{v}_j
+
w_j\mathbf{J}
}
$$

마지막에 위치를 업데이트합니다.

$$
\boxed{
\mathbf{x}_i^{n+1}
=
\mathbf{x}_i^n+\Delta t\mathbf{v}_i^{n+1}
}
$$

---

## 3.2 이 방식의 장점

기존과 비교하면 다음 배열을 줄일 수 있습니다.

$$
\mathbf{F}_i
$$

force accumulation buffer를 줄이거나 제거할 수 있습니다.

또한 각 스프링마다 곧바로 velocity correction을 하기 때문에,

$$
\Delta t
$$

가 조금 커져도 명시적 Euler보다 안정적입니다.

연산 복잡도는 여전히

$$
O(N_s)
$$

입니다.

---

# 4. 개선안 B — $m_{\text{eff}}$, restLength, stiffness 계수 사전 계산

매 프레임마다 반복 계산되는 값은 사전 계산할 수 있습니다.

스프링 $(i,j)$마다 질량이 고정되어 있다면,

$$
w_i,\quad w_j,\quad w_i+w_j,\quad m_{\text{eff}}
$$

는 변하지 않습니다.

따라서 미리 저장합니다.

$$
m_{\text{eff},ij}
=
\frac{1}{w_i+w_j}
$$

그리고 매 step에서 분모는

$$
D_{ij}
=
m_{\text{eff},ij}+c_{ij}\Delta t+k_{ij}\Delta t^2
$$

입니다.

$\Delta t$가 고정 step이면 이것도 사전 계산 가능합니다.

$$
\boxed{
D_{ij}^{-1}
=
\frac{1}{
m_{\text{eff},ij}+c_{ij}\Delta t+k_{ij}\Delta t^2
}
}
$$

그럼 매 스프링에서 나눗셈을 제거하고 곱셈으로 바꿀 수 있습니다.

$$
v_{rel}^{n+1}
=
\left(
m_{\text{eff}}v_{rel}^{n}
-
k\Delta t C
\right)
D^{-1}
$$

또는 impulse를 직접 계산하면 더 간단합니다.

$$
J
=
m_{\text{eff}}
\left(
v_{rel}^{n+1}-v_{rel}^{n}
\right)
$$

이를 전개하면,

$$
J
=
m_{\text{eff}}
\left[
\frac{
m_{\text{eff}}v_{rel}
-
k\Delta t C
}{
D
}
-
v_{rel}
\right]
$$

$$
J
=
m_{\text{eff}}
\frac{
m_{\text{eff}}v_{rel}
-
k\Delta t C
-
Dv_{rel}
}{
D}
$$

$$
D
=
m_{\text{eff}}+c\Delta t+k\Delta t^2
$$

이므로,

$$
m_{\text{eff}}v_{rel}
-
Dv_{rel}
=
-
(c\Delta t+k\Delta t^2)v_{rel}
$$

따라서,

$$
\boxed{
J
=
-
m_{\text{eff}}
\frac{
k\Delta t C
+
(c\Delta t+k\Delta t^2)v_{rel}
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

즉,

$$
\boxed{
J
=
-
\beta C
-
\gamma v_{rel}
}
$$

형태로 쓸 수 있습니다.

여기서

$$
\boxed{
\beta
=
\frac{
m_{\text{eff}}k\Delta t
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
\gamma
=
\frac{
m_{\text{eff}}(c\Delta t+k\Delta t^2)
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

따라서 매 스프링에서 필요한 계산은 다음으로 줄어듭니다.

$$
\boxed{
J
=
-\beta C-\gamma v_{rel}
}
$$

$$
\boxed{
\mathbf{J}
=
J\hat{\mathbf{n}}
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

# 5. 최적화된 Pairwise Implicit 최종 식

스프링마다 미리 저장:

$$
w_i=\frac{1}{m_i}
$$

$$
w_j=\frac{1}{m_j}
$$

$$
m_{\text{eff}}=\frac{1}{w_i+w_j}
$$

고정된 $\Delta t$에 대해 미리 저장:

$$
\boxed{
\beta
=
\frac{
m_{\text{eff}}k\Delta t
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
\gamma
=
\frac{
m_{\text{eff}}(c\Delta t+k\Delta t^2)
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

매 step, 매 spring에서 계산:

$$
\mathbf{d}_{ij}
=
\mathbf{x}_j-\mathbf{x}_i
$$

$$
\ell_{ij}
=
\|\mathbf{d}_{ij}\|
$$

$$
\hat{\mathbf{n}}_{ij}
=
\frac{\mathbf{d}_{ij}}{\ell_{ij}}
$$

$$
C_{ij}
=
\ell_{ij}-\ell_{ij}^{0}
$$

$$
v_{rel}
=
(\mathbf{v}_j-\mathbf{v}_i)
\cdot
\hat{\mathbf{n}}_{ij}
$$

$$
\boxed{
J
=
-\beta C_{ij}
-
\gamma v_{rel}
}
$$

$$
\boxed{
\mathbf{J}
=
J\hat{\mathbf{n}}_{ij}
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

이 형태가 가장 실용적입니다.

---

# 6. 코드 구조

```csharp
void StepOptimizedPairwiseImplicit(float dt)
{
    // 1. gravity
    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity += gravity * dt;
    }

    // 2. spring pair velocity correction
    for (int s = 0; s < springs.Count; s++)
    {
        int i = springs[s].i;
        int j = springs[s].j;

        Particle pi = particles[i];
        Particle pj = particles[j];

        Vector3 d = pj.position - pi.position;
        float len = d.magnitude;

        if (len < 1e-6f)
            continue;

        Vector3 n = d / len;

        float C = len - springs[s].restLength;

        Vector3 vRelVec = pj.velocity - pi.velocity;
        float vRel = Vector3.Dot(vRelVec, n);

        float wi = pi.invMass;
        float wj = pj.invMass;

        float J = -springs[s].beta * C - springs[s].gamma * vRel;

        Vector3 impulse = J * n;

        if (wi > 0.0f)
            pi.velocity -= wi * impulse;

        if (wj > 0.0f)
            pj.velocity += wj * impulse;

        particles[i] = pi;
        particles[j] = pj;
    }

    // 3. integrate position
    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].position += particles[i].velocity * dt;
    }

    // 4. optional time-consistent global damping
    float gammaGlobal = Mathf.Exp(-globalDampingLambda * dt);

    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity *= gammaGlobal;
    }
}
```

---

# 7. $\beta, \gamma$ 사전 계산 코드

고정 timestep 또는 timestep이 변경될 때만 계산하면 됩니다.

```csharp
void PrecomputeSpringCoefficients(float dt)
{
    for (int s = 0; s < springs.Count; s++)
    {
        int i = springs[s].i;
        int j = springs[s].j;

        float wi = particles[i].invMass;
        float wj = particles[j].invMass;
        float w = wi + wj;

        if (w <= 0.0f)
        {
            springs[s].beta = 0.0f;
            springs[s].gamma = 0.0f;
            continue;
        }

        float meff = 1.0f / w;

        float k = springs[s].stiffness;
        float c = springs[s].damping;

        float denom = meff + c * dt + k * dt * dt;

        springs[s].beta = meff * k * dt / denom;
        springs[s].gamma = meff * (c * dt + k * dt * dt) / denom;
    }
}
```

---

# 8. 성능 측면에서 더 줄일 수 있는 부분

## 8.1 force 배열 제거

기존:

$$
\mathbf{F}_i
$$

를 매 프레임 초기화하고 누적해야 합니다.

$$
O(N_p)
$$

초기화 비용과 메모리 write가 발생합니다.

개선 방식은 force accumulation 없이 velocity를 바로 보정하므로 메모리 접근이 줄어듭니다.

---

## 8.2 질량 나눗셈 제거

기존에는 매번

$$
\frac{\mathbf{F}}{m}
$$

또는

$$
\frac{1}{m}
$$

계산을 할 수 있습니다.

대신

$$
w_i=\frac{1}{m_i}
$$

를 미리 저장합니다.

$$
\boxed{
\mathbf{a}_i=w_i\mathbf{F}_i
}
$$

또는 impulse 보정에서

$$
\boxed{
\Delta\mathbf{v}_i=w_i\mathbf{J}
}
$$

로 사용합니다.

---

## 8.3 sqrt 최소화는 어렵지만 대체 가능

스프링 방향

$$
\hat{\mathbf{n}}=
\frac{\mathbf{d}}{\|\mathbf{d}\|}
$$

계산에는 sqrt가 필요합니다.

거리 constraint나 spring force는 실제 길이

$$
\ell
$$

이 필요하므로 완전히 제거하기는 어렵습니다.

다만 고성능화에서는 다음 방법을 쓸 수 있습니다.

### 방법 1: fast inverse sqrt 사용

$$
\hat{\mathbf{n}}
=
\mathbf{d}\cdot \frac{1}{\sqrt{\mathbf{d}\cdot\mathbf{d}}}
$$

Unity C#에서는 `Mathf.Sqrt`가 병목이면 Burst/Jobs로 넘기는 게 더 효과적입니다.

---

### 방법 2: small deformation 근사

변형이 작다면 현재 길이 대신 rest length 기준 방향을 쓰는 근사도 가능합니다.

$$
\hat{\mathbf{n}}_{ij}
\approx
\hat{\mathbf{n}}_{ij}^0
$$

$$
C_{ij}
\approx
(\mathbf{x}_j-\mathbf{x}_i)\cdot\hat{\mathbf{n}}_{ij}^0-\ell_{ij}^0
$$

이 경우 sqrt가 사라집니다.

매 spring 계산은 다음이 됩니다.

$$
C_{ij}
=
(\mathbf{x}_j-\mathbf{x}_i)\cdot\hat{\mathbf{n}}_{ij}^0-\ell_{ij}^0
$$

$$
v_{rel}
=
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}_{ij}^0
$$

$$
J
=
-\beta C_{ij}
-
\gamma v_{rel}
$$

$$
\mathbf{J}
=
J\hat{\mathbf{n}}_{ij}^0
$$

이 방식은 매우 빠릅니다.

단점은 큰 변형이나 회전에서는 실제 스프링 방향을 제대로 따라가지 못합니다.

즉, 작은 진동계, 구조물 주변 변형, 거의 선형적인 spring system에는 좋지만, 천/로프처럼 큰 변형에는 부적합할 수 있습니다.

---

# 9. Small Deformation 근사 최종식

초기 상태에서 미리 계산:

$$
\mathbf{d}_{ij}^0
=
\mathbf{x}_j^0-\mathbf{x}_i^0
$$

$$
\ell_{ij}^0
=
\|\mathbf{d}_{ij}^0\|
$$

$$
\hat{\mathbf{n}}_{ij}^0
=
\frac{\mathbf{d}_{ij}^0}{\ell_{ij}^0}
$$

매 step:

$$
\boxed{
C_{ij}
=
(\mathbf{x}_j-\mathbf{x}_i)\cdot\hat{\mathbf{n}}_{ij}^0
-
\ell_{ij}^0
}
$$

$$
\boxed{
v_{rel}
=
(\mathbf{v}_j-\mathbf{v}_i)
\cdot
\hat{\mathbf{n}}_{ij}^0
}
$$

$$
\boxed{
J
=
-\beta C_{ij}
-
\gamma v_{rel}
}
$$

$$
\boxed{
\mathbf{J}
=
J\hat{\mathbf{n}}_{ij}^0
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

이 방식은 sqrt가 없고 dot product 중심이라 매우 빠릅니다.

---

# 10. Small Deformation 코드

```csharp
void StepLinearizedPairwiseImplicit(float dt)
{
    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity += gravity * dt;
    }

    for (int s = 0; s < springs.Count; s++)
    {
        int i = springs[s].i;
        int j = springs[s].j;

        Particle pi = particles[i];
        Particle pj = particles[j];

        Vector3 n0 = springs[s].restDirection;

        Vector3 d = pj.position - pi.position;
        Vector3 v = pj.velocity - pi.velocity;

        float C = Vector3.Dot(d, n0) - springs[s].restLength;
        float vRel = Vector3.Dot(v, n0);

        float J = -springs[s].beta * C - springs[s].gamma * vRel;

        Vector3 impulse = J * n0;

        float wi = pi.invMass;
        float wj = pj.invMass;

        if (wi > 0.0f)
            pi.velocity -= wi * impulse;

        if (wj > 0.0f)
            pj.velocity += wj * impulse;

        particles[i] = pi;
        particles[j] = pj;
    }

    float gammaGlobal = Mathf.Exp(-globalDampingLambda * dt);

    for (int i = 0; i < particles.Count; i++)
    {
        if (particles[i].isFixed)
            continue;

        particles[i].velocity *= gammaGlobal;
        particles[i].position += particles[i].velocity * dt;
    }
}
```

---

# 11. XPBD의 iteration을 줄이는 방법

XPBD를 쓰되 성능을 아끼려면 iteration 수를 줄이고 stiffness를 보정하면 됩니다.

XPBD의 constraint update는

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
\tilde{\alpha}
=
\frac{1}{k\Delta t^2}
$$

입니다.

iteration 수가 적으면 constraint가 덜 만족됩니다.

이때 $k$를 키우기보다 compliance를 줄입니다.

$$
\alpha = \frac{1}{k}
$$

$$
\alpha_{\text{eff}}
=
\frac{\alpha}{N_{\text{iter}}}
$$

또는 경험적으로,

$$
k_{\text{eff}}
=
N_{\text{iter}}k
$$

처럼 쓸 수 있습니다.

하지만 너무 키우면 jitter가 생깁니다.

추천은 다음입니다.

$$
N_{\text{iter}}=2\sim4
$$

$$
\alpha_{\text{eff}}
=
\frac{1}{k N_{\text{iter}}}
$$

그러면 1-pass pairwise implicit보다 조금 더 안정적이고, full XPBD보다 빠릅니다.

---

# 12. OneDOF 성능 개선

OneDOF는 이미 매우 가볍습니다. 그래도 매 step마다 나눗셈을 하지 않으려면 다음을 미리 계산합니다.

Implicit Euler 식:

$$
v_{n+1}
=
\frac{
mv_n-k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
$$

이를

$$
v_{n+1}
=
A v_n
-
B x_n
$$

로 씁니다.

$$
\boxed{
A
=
\frac{m}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
B
=
\frac{k\Delta t}{
m+c\Delta t+k\Delta t^2
}
}
$$

따라서,

$$
\boxed{
v_{n+1}
=
A v_n-Bx_n
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
v_{n+1}
=
\frac{
mv_n+\Delta t F_{\text{ext}}-k\Delta t x_n
}{
m+c\Delta t+k\Delta t^2
}
$$

따라서,

$$
\boxed{
v_{n+1}
=
A v_n
-
B x_n
+
G F_{\text{ext}}
}
$$

$$
\boxed{
G
=
\frac{\Delta t}{
m+c\Delta t+k\Delta t^2
}
}
$$

---

# 13. OneDOF 최적 코드

```csharp
void PrecomputeOneDOF(float dt)
{
    float denom = mass + damping * dt + stiffness * dt * dt;

    coeffA = mass / denom;
    coeffB = stiffness * dt / denom;
    coeffG = dt / denom;
}

void StepOneDOFImplicitFast(float dt)
{
    float externalForce = 0.0f;

    float newVelocity =
        coeffA * velocity
        - coeffB * displacement
        + coeffG * externalForce;

    displacement += dt * newVelocity;
    velocity = newVelocity;
}
```

---

# 14. 성능별 추천 조합

## 정확도와 안정성 균형

$$
\boxed{
\text{MassSpring: Optimized Pairwise Implicit}
}
$$

$$
\boxed{
\text{OneDOF: Precomputed Implicit Euler}
}
$$

특징:

$$
O(N_s+N_p)
$$

$$
\text{발산 억제 강함}
$$

$$
\text{기존 코드 구조와 유사}
$$

---

## 가장 빠른 방식

$$
\boxed{
\text{MassSpring: Linearized Pairwise Implicit}
}
$$

$$
\boxed{
\text{OneDOF: Precomputed Implicit Euler}
}
$$

이 경우 스프링마다 sqrt가 제거됩니다.

대신 큰 회전이나 큰 변형에서는 정확도가 떨어집니다.

---

## 가장 안정적인 방식

$$
\boxed{
\text{MassSpring: XPBD, } 3\sim5 \text{ iterations}
}
$$

$$
\boxed{
\text{OneDOF: Implicit Euler}
}
$$

성능은 조금 더 들지만, 큰 $\Delta t$에서 가장 덜 터집니다.

---

# 15. 최종 추천안

현재 요구가

> 수식 측면에서 비슷한 결과를 가져오면서 조금 더 성능을 개선

이라면 최종적으로 아래 방식을 추천합니다.

---

## MassSpring

기존 스프링 힘

$$
f_s=k(\ell-\ell_0)
$$

$$
f_d=c
\left(
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}
\right)
$$

를 직접 힘으로 누적하지 말고, 아래 impulse 식으로 대체합니다.

$$
\boxed{
J
=
-\beta C
-
\gamma v_{rel}
}
$$

$$
\boxed{
C=\ell-\ell_0
}
$$

$$
\boxed{
v_{rel}=
(\mathbf{v}_j-\mathbf{v}_i)\cdot\hat{\mathbf{n}}
}
$$

$$
\boxed{
\beta
=
\frac{
m_{\text{eff}}k\Delta t
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
\gamma
=
\frac{
m_{\text{eff}}(c\Delta t+k\Delta t^2)
}{
m_{\text{eff}}+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
\mathbf{J}=J\hat{\mathbf{n}}
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

그 후,

$$
\boxed{
\mathbf{x}_i
\leftarrow
\mathbf{x}_i+\Delta t\mathbf{v}_i
}
$$

---

## OneDOF

기존 velocity Verlet 대신 precomputed implicit Euler를 사용합니다.

$$
\boxed{
v_{n+1}
=
A v_n-Bx_n
}
$$

$$
\boxed{
x_{n+1}
=
x_n+\Delta t v_{n+1}
}
$$

$$
\boxed{
A
=
\frac{m}{
m+c\Delta t+k\Delta t^2
}
}
$$

$$
\boxed{
B
=
\frac{k\Delta t}{
m+c\Delta t+k\Delta t^2
}
}
$$

외력이 있으면,

$$
\boxed{
v_{n+1}
=
A v_n-Bx_n+GF_{\text{ext}}
}
$$

$$
\boxed{
G=
\frac{\Delta t}{
m+c\Delta t+k\Delta t^2
}
}
$$

---

이 조합은 기존 결과와 물리적 형태가 비슷하면서도,

$$
\boxed{
\text{force accumulation 제거}
}
$$

$$
\boxed{
\text{나눗셈 사전 계산}
}
$$

$$
\boxed{
\text{implicit 안정성 확보}
}
$$

$$
\boxed{
O(N_s+N_p) \text{ 유지}
}
$$

라는 장점이 있습니다.
