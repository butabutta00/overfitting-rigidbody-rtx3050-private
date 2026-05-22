using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class MassSpringSystem : MonoBehaviour
{
    public enum IntegrationMethod { Explicit, SemiImplicit }

    [Header("Student Task: Stability Settings")]
    [Tooltip("수치 적분 방식을 선택하여 안정성을 비교하세요.")]
    public IntegrationMethod integration = IntegrationMethod.SemiImplicit;

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

        // 1. 힘 초기화 (중력)
        foreach (var p in particles)
        {
            if (p.isFixed) continue;
            p.force = gravity * p.mass;
        }

        // 2. 내부 스프링 연산
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

        // 3. 수치 적분 (Numerical Integration)
        Integrate();

        // 4. 시각적 메쉬 갱신
        UpdateVisualMesh();
    }

    private void Integrate()
    {
        foreach (var p in particles)
        {
            if (p.isFixed) continue;
            Vector3 accel = p.force / p.mass;

            if (integration == IntegrationMethod.Explicit)
            {
                p.position += p.velocity * timeStep;
                p.velocity += accel * timeStep;
            }
            else
            {
                p.velocity += accel * timeStep;
                p.position += p.velocity * timeStep;
            }

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
}