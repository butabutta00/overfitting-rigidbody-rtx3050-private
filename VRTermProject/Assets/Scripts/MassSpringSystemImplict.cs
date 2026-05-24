using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class MassSpringSystemImplict : MonoBehaviour
{
    private enum ComputeBackend
    {
        CSharp = 0,
        NativeCuda = 1
    }

    [Range(0.001f, 0.1f)]
    [Tooltip("타임스텝이 커질수록 시스템이 어떻게 변하는지 관찰하세요.")]
    public float timeStep = 0.01f;

    [Tooltip("K값이 커질수록 천의 강성이 강해지지만 시스템은 불안정해집니다.")]
    public float springStiffness = 500f;

    [Range(5, 80)]
    [Tooltip("Implicit Euler 선형 시스템(CG) 반복 횟수입니다. 값이 클수록 안정적이지만 연산량이 증가합니다.")]
    public int implicitIterations = 24;

    [Header("Fixed Scenario Parameters (Read Only)")]
    [SerializeField] private float particleMass = 0.1f;
    [SerializeField] private float springDamping = 10f;
    [SerializeField] private Vector3 gravity = new Vector3(0, -9.81f, 0);
    [SerializeField] private Collider fixVolume;

    [Header("Physical Simulations")]
    [SerializeField] private float physicsSimulationSkippingCoeff = 3f;
    [SerializeField] private int framesSkippingCompute = 0;
    [SerializeField] private int nextComputeFrameSkip = 0;

    [Header("Compute Backend")]
    [SerializeField] private ComputeBackend computeBackend = ComputeBackend.NativeCuda;
    [SerializeField] private bool autoFallbackToCSharp = true;
    [SerializeField] private bool nativeReady = false;

    // [실시간 Hz 측정용 변수]
    private int fixedUpdateCount = 0;
    private float hzTimer = 0f;
    private float actualPhysicsHz = 0f;

    // 내부 물리 구조체
    private class Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 force;
        public float mass;
        public bool isFixed;
    }

    private class Spring
    {
        public int indexA;
        public int indexB;
        public float restLength;
    }

    private struct LinearizedSpring
    {
        public int indexA;
        public int indexB;
        public Vector3 dir;
        public float coeff;
    }

    private List<Particle> particles = new List<Particle>();
    private List<Spring> springs = new List<Spring>();

    private int[] meshToParticleMap;
    private Mesh workingMesh;
    private Vector3[] visualVertices;
    private int[] meshTriangles;
    private Vector3[] previousSimulatedPositions;
    private Vector3[] currentSimulatedPositions;
    private int interpolationStep;
    private int interpolationStepCount = 1;
    private Vector3[] nativePositions;
    private Vector3[] nativeVelocities;
    private float[] nativeMasses;
    private byte[] nativeFixedMask;
    private int[] nativeSpringEndpoints;
    private float[] nativeRestLengths;
    private MassSpringNativeInterop.SystemHandle nativeSystem;

    private void Start()
    {
        InitializeMeshAndPhysics();
    }

    private void InitializeMeshAndPhysics()
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        workingMesh = Instantiate(mf.sharedMesh);
        workingMesh.MarkDynamic();
        mf.mesh = workingMesh;

        visualVertices = workingMesh.vertices;
        meshTriangles = workingMesh.triangles;
        meshToParticleMap = new int[visualVertices.Length];

        for (int i = 0; i < visualVertices.Length; i++)
        {
            Vector3 worldPos = transform.TransformPoint(visualVertices[i]);
            int existingIndex = FindExistingParticle(worldPos);

            if (existingIndex == -1)
            {
                Particle p = new Particle
                {
                    position = worldPos,
                    velocity = Vector3.zero,
                    force = Vector3.zero,
                    mass = particleMass,
                    isFixed = (fixVolume != null && fixVolume.bounds.Contains(worldPos))
                };
                particles.Add(p);
                meshToParticleMap[i] = particles.Count - 1;
            }
            else
            {
                meshToParticleMap[i] = existingIndex;
            }
        }

        HashSet<long> edgeSet = new HashSet<long>();
        for (int i = 0; i < meshTriangles.Length; i += 3)
        {
            AddSpring(meshToParticleMap[meshTriangles[i]], meshToParticleMap[meshTriangles[i + 1]], edgeSet);
            AddSpring(meshToParticleMap[meshTriangles[i + 1]], meshToParticleMap[meshTriangles[i + 2]], edgeSet);
            AddSpring(meshToParticleMap[meshTriangles[i + 2]], meshToParticleMap[meshTriangles[i]], edgeSet);
        }

        previousSimulatedPositions = new Vector3[particles.Count];
        currentSimulatedPositions = new Vector3[particles.Count];
        for (int i = 0; i < particles.Count; i++)
        {
            Vector3 initialPosition = particles[i].position;
            previousSimulatedPositions[i] = initialPosition;
            currentSimulatedPositions[i] = initialPosition;
        }

        BuildNativeBuffers();
        TryInitializeNativeBackend();
    }

    private void BuildNativeBuffers()
    {
        int particleCount = particles.Count;
        int springCount = springs.Count;

        nativePositions = new Vector3[particleCount];
        nativeVelocities = new Vector3[particleCount];
        nativeMasses = new float[particleCount];
        nativeFixedMask = new byte[particleCount];
        nativeSpringEndpoints = new int[springCount * 2];
        nativeRestLengths = new float[springCount];

        for (int i = 0; i < particleCount; i++)
        {
            Particle particle = particles[i];
            nativePositions[i] = particle.position;
            nativeVelocities[i] = particle.velocity;
            nativeMasses[i] = particle.mass;
            nativeFixedMask[i] = (byte)(particle.isFixed ? 1 : 0);
        }

        for (int i = 0; i < springCount; i++)
        {
            Spring spring = springs[i];
            int baseIndex = i * 2;
            nativeSpringEndpoints[baseIndex] = spring.indexA;
            nativeSpringEndpoints[baseIndex + 1] = spring.indexB;
            nativeRestLengths[i] = spring.restLength;
        }
    }

    private void TryInitializeNativeBackend()
    {
        if (computeBackend != ComputeBackend.NativeCuda)
        {
            nativeReady = false;
            return;
        }

        if (nativeSystem != null)
        {
            nativeSystem.Dispose();
            nativeSystem = null;
        }

        if (!MassSpringNativeInterop.SystemHandle.TryCreate(
                nativePositions,
                nativeMasses,
                nativeFixedMask,
                nativeSpringEndpoints,
                nativeRestLengths,
                out nativeSystem,
                out string error))
        {
            nativeReady = false;
            if (autoFallbackToCSharp)
            {
                computeBackend = ComputeBackend.CSharp;
            }

            Debug.LogWarning($"MassSpringSystemImplict native initialization failed: {error}");
            return;
        }

        nativeReady = true;
    }

    private void AddSpring(int a, int b, HashSet<long> edgeSet)
    {
        if (a == b) return;
        int min = Mathf.Min(a, b);
        int max = Mathf.Max(a, b);
        long edgeKey = ((long)min << 32) | (uint)max;

        if (!edgeSet.Contains(edgeKey))
        {
            edgeSet.Add(edgeKey);
            springs.Add(new Spring
            {
                indexA = a,
                indexB = b,
                restLength = Vector3.Distance(particles[a].position, particles[b].position)
            });
        }
    }

    private int FindExistingParticle(Vector3 pos)
    {
        for (int i = 0; i < particles.Count; i++)
        {
            if (Vector3.SqrMagnitude(particles[i].position - pos) < 0.0001f) return i;
        }
        return -1;
    }

    private void Update()
    {
        var _deltaTimeLogscale = Mathf.Log10(timeStep) + 3f;
        framesSkippingCompute = (int)(_deltaTimeLogscale + 1f + (8 * (Mathf.Pow(2f, -_deltaTimeLogscale * 2))));

        // 사용자가 설정한 timeStep에 맞춰 유니티의 실제 물리 루프 속도를 강제로 동기화
        Time.fixedDeltaTime = timeStep;

        // 실제 초당 물리 업데이트 횟수(Hz) 측정
        hzTimer += Time.deltaTime;
        if (hzTimer >= 1.0f)
        {
            actualPhysicsHz = fixedUpdateCount / hzTimer;
            fixedUpdateCount = 0;
            hzTimer = 0f;
        }
    }

    private void FixedUpdate()
    {
        // 실행 횟수 카운트
        fixedUpdateCount++;

        if (nextComputeFrameSkip <= 0) {
            for (int i = 0; i < particles.Count; i++)
            {
                previousSimulatedPositions[i] = particles[i].position;
            }

            if (nativeReady && computeBackend == ComputeBackend.NativeCuda)
            {
                float effectiveDt = timeStep * Mathf.Max(1, framesSkippingCompute);
                bool ok = nativeSystem.StepImplicit(
                    effectiveDt,
                    springStiffness,
                    springDamping,
                    gravity,
                    0.995f,
                    implicitIterations,
                    1e-8f);

                if (ok && nativeSystem.DownloadState(nativePositions, nativeVelocities))
                {
                    SyncParticlesFromNative();
                }
                else
                {
                    FallbackToCSharp("native implicit step/download failed");
                    IntegrateImplicitEuler();
                }
            }
            else
            {
                IntegrateImplicitEuler();
            }

            for (int i = 0; i < particles.Count; i++)
            {
                currentSimulatedPositions[i] = particles[i].position;
            }

            interpolationStep = 1;
            interpolationStepCount = Mathf.Max(1, framesSkippingCompute + 1);
            nextComputeFrameSkip = framesSkippingCompute;
        } else {
            // 선형보간
            interpolationStep = Mathf.Min(interpolationStep + 1, interpolationStepCount);

            nextComputeFrameSkip--;
        }

        // 2. 시각적 메쉬 갱신
        UpdateVisualMesh();
    }

    private void IntegrateImplicitEuler()
    {
        int count = particles.Count;
        float h = timeStep * Mathf.Max(1, framesSkippingCompute);
        float h2 = h * h;

        Vector3[] x0 = new Vector3[count];
        Vector3[] v0 = new Vector3[count];
        Vector3[] forces = new Vector3[count];
        Vector3[] jvTimesV = new Vector3[count];
        Vector3[] b = new Vector3[count];
        Vector3[] v = new Vector3[count];
        List<LinearizedSpring> linearizedSprings = new List<LinearizedSpring>(springs.Count);

        for (int i = 0; i < count; i++)
        {
            Particle p = particles[i];
            x0[i] = p.position;
            v0[i] = p.velocity;
            v[i] = p.velocity;
            forces[i] = p.isFixed ? Vector3.zero : gravity * p.mass;
            jvTimesV[i] = Vector3.zero;
        }

        foreach (var s in springs)
        {
            int iA = s.indexA;
            int iB = s.indexB;

            Vector3 diff = x0[iB] - x0[iA];
            float dist = diff.magnitude;
            if (dist < 0.0001f)
            {
                continue;
            }

            Vector3 dir = diff / dist;
            Vector3 relVel = v0[iB] - v0[iA];
            float relVelAlongSpring = Vector3.Dot(relVel, dir);

            float fSpring = springStiffness * (dist - s.restLength);
            float fDamper = springDamping * relVelAlongSpring;
            Vector3 totalF = dir * (fSpring + fDamper);

            if (!particles[iA].isFixed) forces[iA] += totalF;
            if (!particles[iB].isFixed) forces[iB] -= totalF;

            Vector3 projVA = dir * Vector3.Dot(v0[iA], dir);
            Vector3 projVB = dir * Vector3.Dot(v0[iB], dir);
            if (!particles[iA].isFixed) jvTimesV[iA] += springDamping * (projVB - projVA);
            if (!particles[iB].isFixed) jvTimesV[iB] += springDamping * (projVA - projVB);

            linearizedSprings.Add(new LinearizedSpring
            {
                indexA = iA,
                indexB = iB,
                dir = dir,
                coeff = h * springDamping + h2 * springStiffness
            });
        }

        for (int i = 0; i < count; i++)
        {
            Particle p = particles[i];
            if (p.isFixed)
            {
                b[i] = Vector3.zero;
                v[i] = Vector3.zero;
                continue;
            }

            b[i] = p.mass * v0[i] + h * (forces[i] - jvTimesV[i]);
        }

        SolveLinearSystemCG(v, b, linearizedSprings);

        for (int i = 0; i < count; i++)
        {
            Particle p = particles[i];
            if (p.isFixed)
            {
                p.force = Vector3.zero;
                p.velocity = Vector3.zero;
                continue;
            }

            p.force = forces[i];
            p.velocity = v[i] * 0.995f;
            p.position = x0[i] + h * v[i];
        }
    }

    private void SolveLinearSystemCG(Vector3[] x, Vector3[] b, List<LinearizedSpring> linearizedSprings)
    {
        int count = particles.Count;
        Vector3[] r = new Vector3[count];
        Vector3[] p = new Vector3[count];
        Vector3[] ap = new Vector3[count];

        MultiplyImplicitMatrix(x, ap, linearizedSprings);

        float rr = 0f;
        for (int i = 0; i < count; i++)
        {
            if (particles[i].isFixed)
            {
                r[i] = Vector3.zero;
                p[i] = Vector3.zero;
                continue;
            }

            r[i] = b[i] - ap[i];
            p[i] = r[i];
            rr += Vector3.Dot(r[i], r[i]);
        }

        if (rr < 1e-12f) return;

        for (int iter = 0; iter < implicitIterations; iter++)
        {
            MultiplyImplicitMatrix(p, ap, linearizedSprings);

            float pAp = 0f;
            for (int i = 0; i < count; i++)
            {
                if (particles[i].isFixed) continue;
                pAp += Vector3.Dot(p[i], ap[i]);
            }

            if (Mathf.Abs(pAp) < 1e-12f) break;

            float alpha = rr / pAp;
            float rrNew = 0f;

            for (int i = 0; i < count; i++)
            {
                if (particles[i].isFixed) continue;
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
                rrNew += Vector3.Dot(r[i], r[i]);
            }

            if (rrNew < 1e-12f) break;

            float beta = rrNew / rr;
            for (int i = 0; i < count; i++)
            {
                if (particles[i].isFixed) continue;
                p[i] = r[i] + beta * p[i];
            }

            rr = rrNew;
        }
    }

    private void MultiplyImplicitMatrix(Vector3[] src, Vector3[] dst, List<LinearizedSpring> linearizedSprings)
    {
        int count = particles.Count;
        for (int i = 0; i < count; i++)
        {
            Particle particle = particles[i];
            dst[i] = particle.isFixed ? Vector3.zero : particle.mass * src[i];
        }

        foreach (var s in linearizedSprings)
        {
            int iA = s.indexA;
            int iB = s.indexB;

            Vector3 projA = s.dir * Vector3.Dot(src[iA], s.dir);
            Vector3 projB = s.dir * Vector3.Dot(src[iB], s.dir);

            bool aDynamic = !particles[iA].isFixed;
            bool bDynamic = !particles[iB].isFixed;

            if (aDynamic)
            {
                dst[iA] += s.coeff * projA;
                if (bDynamic) dst[iA] -= s.coeff * projB;
            }

            if (bDynamic)
            {
                dst[iB] += s.coeff * projB;
                if (aDynamic) dst[iB] -= s.coeff * projA;
            }
        }
    }

    private void UpdateVisualMesh()
    {
        float alpha = Mathf.Clamp01((float)interpolationStep / interpolationStepCount);

        for (int i = 0; i < visualVertices.Length; i++)
        {
            int particleIndex = meshToParticleMap[i];
            Vector3 interpolatedWorldPos = Vector3.Lerp(previousSimulatedPositions[particleIndex], currentSimulatedPositions[particleIndex], alpha);
            visualVertices[i] = transform.InverseTransformPoint(interpolatedWorldPos);
        }
        workingMesh.vertices = visualVertices;
        workingMesh.RecalculateNormals();
        workingMesh.RecalculateBounds();
    }

    private void OnGUI()
    {
        // 정보창 크기를 3줄이 들어가도록 약간 키웠습니다 (높이 85 -> 105)
        GUI.backgroundColor = Color.black;
        GUI.Box(new Rect(10, 10, 290, 105), "Simulation Real-time Info");

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.normal.textColor = Color.white;
        labelStyle.fontSize = 13;

        // 1. 현재 설정된 타임스텝 표시
        GUI.Label(new Rect(20, 35, 270, 20), $"Current Time Step (dt): {timeStep:F4} s", labelStyle);

        // 2. 실제 작동 중인 초당 물리 업데이트 횟수(Hz) 표시
        GUI.Label(new Rect(20, 60, 270, 20), $"Actual Physics Rate: {actualPhysicsHz:F1} Hz", labelStyle);

        // 3. 타임스텝 x 업데이트 레이트 곱한 값 표시
        // 정상 동기화 중이라면 이 값은 항상 1.0 근처에 고정됩니다.
        float productValue = timeStep * actualPhysicsHz;

        // 오차 범위를 고려해 값이 정상적(0.99~1.01)이면 녹색, 프레임 드랍 등이 생기면 황색으로 표시하는 가독성 추가
        if (Mathf.Abs(productValue - 1.0f) < 0.02f)
        {
            labelStyle.normal.textColor = Color.green;
        }
        else
        {
            labelStyle.normal.textColor = Color.yellow;
        }

        GUI.Label(new Rect(20, 85, 270, 20), $"dt x Rate Product: {productValue:F3} (Target: 1.0)", labelStyle);
    }

    private void SyncParticlesFromNative()
    {
        for (int i = 0; i < particles.Count; i++)
        {
            Particle particle = particles[i];
            particle.position = nativePositions[i];
            particle.velocity = nativeVelocities[i];
        }
    }

    private void FallbackToCSharp(string reason)
    {
        if (computeBackend == ComputeBackend.CSharp)
        {
            return;
        }

        nativeReady = false;
        if (autoFallbackToCSharp)
        {
            computeBackend = ComputeBackend.CSharp;
        }

        Debug.LogWarning($"MassSpringSystemImplict switched to C# backend: {reason}. Native error: {MassSpringNativeInterop.GetLastErrorMessage()}");
    }

    private void OnDestroy()
    {
        if (nativeSystem != null)
        {
            nativeSystem.Dispose();
            nativeSystem = null;
        }
    }
}
