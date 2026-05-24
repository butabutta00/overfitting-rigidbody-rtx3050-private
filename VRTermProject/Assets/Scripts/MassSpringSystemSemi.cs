using System;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class MassSpringSystemSemi : MonoBehaviour
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

    [Header("Fixed Scenario Parameters (Read Only)")]
    [SerializeField] private float particleMass = 0.1f;
    [SerializeField] private float springDamping = 10f;
    [SerializeField] private Vector3 gravity = new Vector3(0, -9.81f, 0);
    [SerializeField] private Collider fixVolume;

    [Header("Physical Simulations")]
    [SerializeField] private float physicsSimulationSlicingCoeff = 3f;
    [SerializeField] private float physicsSimulationSliceSize = 0.001f;
    [SerializeField] private float physicsSimulationSubDeltaTime = 0.001f;
    [SerializeField] private int framesSkippingRender = 0;
    [SerializeField] private int nextRenderFrameSkip = 0;

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

    private List<Particle> particles = new List<Particle>();
    private List<Spring> springs = new List<Spring>();

    private int[] meshToParticleMap;
    private Mesh workingMesh;
    private Vector3[] visualVertices;
    private int[] meshTriangles;
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

            Debug.LogWarning($"MassSpringSystemSemi native initialization failed: {error}");
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
        physicsSimulationSliceSize = physicsSimulationSlicingCoeff * (Mathf.Log10(timeStep) + 3f) + 1f;
        physicsSimulationSubDeltaTime = timeStep / physicsSimulationSliceSize;
        framesSkippingRender = Math.Max(0, Mathf.CeilToInt(physicsSimulationSliceSize) - 1);


        // 사용자가 설정한 timeStep에 맞춰 유니티의 실제 물리 루프 속도를 강제로 동기화
        Time.fixedDeltaTime = physicsSimulationSubDeltaTime;

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

        if (nativeReady && computeBackend == ComputeBackend.NativeCuda)
        {
            bool ok = nativeSystem.StepSemi(
                timeStep,
                springStiffness,
                springDamping,
                gravity,
                0.995f,
                1);

            if (ok && nativeSystem.DownloadState(nativePositions, nativeVelocities))
            {
                SyncParticlesFromNative();
            }
            else
            {
                FallbackToCSharp("native step/download failed");
                ComputeStepCSharp();
            }
        }
        else
        {
            ComputeStepCSharp();
        }

        if (nextRenderFrameSkip <= 0) {
            // 4. 시각적 메쉬 갱신
            UpdateVisualMesh();
            nextRenderFrameSkip = framesSkippingRender;
        } else {
            nextRenderFrameSkip--;
        }
    }

    private void ComputeStepCSharp()
    {
        foreach (var p in particles)
        {
            if (p.isFixed) continue;
            p.force = gravity * p.mass;
        }

        foreach (var s in springs)
        {
            Particle pA = particles[s.indexA];
            Particle pB = particles[s.indexB];

            Vector3 diff = pB.position - pA.position;
            float dist = diff.magnitude;
            if (dist < 0.0001f) continue;
            Vector3 dir = diff / dist;

            float fSpring = springStiffness * (dist - s.restLength);
            float fDamper = springDamping * Vector3.Dot(pB.velocity - pA.velocity, dir);

            Vector3 totalF = dir * (fSpring + fDamper);
            if (!pA.isFixed) pA.force += totalF;
            if (!pB.isFixed) pB.force -= totalF;
        }

        Integrate();
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

        Debug.LogWarning($"MassSpringSystemSemi switched to C# backend: {reason}. Native error: {MassSpringNativeInterop.GetLastErrorMessage()}");
    }

    private void Integrate()
    {
        foreach (var p in particles)
        {
            if (p.isFixed) continue;
            Vector3 accel = p.force / p.mass;

        
            p.velocity += accel * timeStep;
            p.position += p.velocity * timeStep;

            p.velocity *= 0.995f;
        }
    }

    private void UpdateVisualMesh()
    {
        for (int i = 0; i < visualVertices.Length; i++)
        {
            visualVertices[i] = transform.InverseTransformPoint(particles[meshToParticleMap[i]].position);
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

    private void OnDestroy()
    {
        if (nativeSystem != null)
        {
            nativeSystem.Dispose();
            nativeSystem = null;
        }
    }
}
