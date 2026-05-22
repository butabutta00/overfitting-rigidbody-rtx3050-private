using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class MassSpringSystemCustomized : MonoBehaviour
{
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

    // [실시간 Hz 측정용 변수]
    private int fixedUpdateCount = 0;
    private float hzTimer = 0f;
    private float actualPhysicsHz = 0f;

    // 내부 물리 구조체
    private class Particle
    {
        public Vector3 position;
        public Vector3 previousPosition;
        public Vector3 predictedPosition;
        public Vector3 velocity;
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
    private float[] springLambdas;

    [Header("XPBD Solver Parameters")]
    [SerializeField, Range(1, 20)] private int solverIterations = 8;
    [SerializeField] private float velocityDampingLambda = 5.0125f;

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
                    previousPosition = worldPos,
                    predictedPosition = worldPos,
                    velocity = Vector3.zero,
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

        springLambdas = new float[springs.Count];
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

        // XPBD 기반 스텝: 큰 timeStep에서도 발산 억제
        StepXPBD(timeStep);

        // 시각적 메쉬 갱신
        UpdateVisualMesh();
    }

    private void StepXPBD(float dt)
    {
        if (particles.Count == 0 || springs.Count == 0) return;

        // 1) 외력 반영 + 예측 위치 계산
        for (int i = 0; i < particles.Count; i++)
        {
            Particle p = particles[i];
            p.previousPosition = p.position;

            if (p.isFixed)
            {
                p.predictedPosition = p.position;
                p.velocity = Vector3.zero;
                continue;
            }

            p.velocity += gravity * dt;
            p.predictedPosition = p.position + p.velocity * dt;
        }

        // 각 타임스텝 시작 시 람다 초기화
        System.Array.Clear(springLambdas, 0, springLambdas.Length);

        // 2) 제약 반복 투영
        float compliance = 1.0f / Mathf.Max(springStiffness, 1e-6f);
        float alphaTilde = compliance / (dt * dt);

        for (int iter = 0; iter < solverIterations; iter++)
        {
            for (int s = 0; s < springs.Count; s++)
            {
                Spring spring = springs[s];
                Particle pA = particles[spring.indexA];
                Particle pB = particles[spring.indexB];

                Vector3 diff = pB.predictedPosition - pA.predictedPosition;
                float dist = diff.magnitude;
                if (dist < 1e-6f) continue;

                Vector3 n = diff / dist;
                float C = dist - spring.restLength;

                float wA = pA.isFixed ? 0f : 1f / pA.mass;
                float wB = pB.isFixed ? 0f : 1f / pB.mass;
                float w = wA + wB;
                if (w <= 0f) continue;

                float deltaLambda = (C - alphaTilde * springLambdas[s]) / (w + alphaTilde);
                springLambdas[s] += deltaLambda;

                if (!pA.isFixed)
                {
                    pA.predictedPosition += wA * deltaLambda * n;
                }

                if (!pB.isFixed)
                {
                    pB.predictedPosition -= wB * deltaLambda * n;
                }
            }
        }

        // 3) 속도/위치 확정 + 시간 기반 감쇠
        float gamma = Mathf.Exp(-velocityDampingLambda * dt);

        for (int i = 0; i < particles.Count; i++)
        {
            Particle p = particles[i];
            if (p.isFixed) continue;

            p.velocity = (p.predictedPosition - p.previousPosition) / dt;
            p.velocity *= gamma;
            p.position = p.predictedPosition;
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
}
