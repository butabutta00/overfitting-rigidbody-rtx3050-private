using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(MeshFilter))]
public class RigidBodyDynamicsLab : MonoBehaviour
{
    public enum IntegrationMethod { Explicit, SemiImplicit, VelocityVerlet }

    [Header("Simulation Settings")]
    public IntegrationMethod integration = IntegrationMethod.SemiImplicit;
    public bool useFixedDeltaTime = true;
    [Range(0.001f, 0.5f)] public float customTimeStep = 0.02f;

    [Header("CUDA Integration")]
    public bool useCudaIntegration = true;

    [Header("Dynamics Toggle")]
    public bool enableTranslation = true;
    public bool enableRotation = true;

    [Header("Mass & Inertia")]
    public float mass = 1.0f;
    public Vector3 inertia = new Vector3(1f, 1f, 1f);

    [Header("Spring-Damper Settings")]
    public float springConstant = 200f;
    public float dampingConstant = 10f;

    [Header("State (Read Only)")]
    public Vector3 velocity;
    public Vector3 angularVelocity;

    private Vector3 forceAccumulator;
    private Vector3 torqueAccumulator;
    private Vector3 inverseInertiaTensor;
    private Vector3 initialPosition;

    private Camera cam;
    private bool isDragging = false;
    private Vector3 localAttachmentVertex;
    private float restLength;
    private float mouseZDistance;
    private Mesh mesh;
    private LineRenderer springLine;

    private bool cudaUnavailable;

    [StructLayout(LayoutKind.Sequential)]
    private struct CudaRigidBodyState
    {
        public Vector4 position;
        public Vector4 rotation;
        public Vector4 velocity;
        public Vector4 angularVelocity;
        public Vector4 acceleration;
        public Vector4 angularAcceleration;
    }

    [DllImport("rigidbody_cuda", EntryPoint = "RbCudaStepSingle")]
    private static extern int RbCudaStepSingleFp(ref CudaRigidBodyState state, float dt, int integrationMethod, int enableTranslation, int enableRotation);

    [DllImport("rigidbody_cuda_bf", EntryPoint = "RbCudaStepSingle")]
    private static extern int RbCudaStepSingleBf(ref CudaRigidBodyState state, float dt, int integrationMethod, int enableTranslation, int enableRotation);

    private static int RbCudaStepSingleBridge(ref CudaRigidBodyState state, float dt, int integrationMethod, int enableTranslation, int enableRotation)
    {
#if RB_CUDA_BF16
        return RbCudaStepSingleBf(ref state, dt, integrationMethod, enableTranslation, enableRotation);
#else
        return RbCudaStepSingleFp(ref state, dt, integrationMethod, enableTranslation, enableRotation);
#endif
    }

    private void Start()
    {
        cam = Camera.main;
        initialPosition = transform.position;
        mesh = GetComponent<MeshFilter>().mesh;
        SetupLineRenderer();
    }

    private void SetupLineRenderer()
    {
        if (!TryGetComponent(out springLine))
        {
            springLine = gameObject.AddComponent<LineRenderer>();
        }
        springLine.startWidth = 0.03f;
        springLine.endWidth = 0.01f;
        springLine.positionCount = 2;
        springLine.material = new Material(Shader.Find("Sprites/Default"));
        springLine.startColor = Color.cyan;
        springLine.endColor = Color.white;
        springLine.enabled = false;
    }

    private void Update()
    {
        HandleMouseInputState();

        if (transform.position.magnitude > 100f)
        {
            Debug.LogWarning("Simulation diverged. Physics state is reset.");
            ResetPhysics();
        }
    }

    private void FixedUpdate()
    {
        if (isDragging)
        {
            ApplySpringForce();
        }

        UpdateInverseInertia();
        float dt = useFixedDeltaTime ? Time.fixedDeltaTime : customTimeStep;

        Vector3 acceleration = forceAccumulator / Mathf.Max(mass, 0.0001f);
        Vector3 angularAcceleration = Vector3.Scale(torqueAccumulator, inverseInertiaTensor);

        bool integratedByCuda = useCudaIntegration && !cudaUnavailable && TryCudaIntegrate(dt, acceleration, angularAcceleration);
        if (!integratedByCuda)
        {
            IntegrateCpu(dt, acceleration, angularAcceleration);
        }

        forceAccumulator = Vector3.zero;
        torqueAccumulator = Vector3.zero;

        velocity *= 0.995f;
        angularVelocity *= 0.995f;
    }

    private void IntegrateCpu(float dt, Vector3 acceleration, Vector3 angularAcceleration)
    {
        switch (integration)
        {
            case IntegrationMethod.Explicit:
                if (enableTranslation)
                {
                    transform.position += velocity * dt;
                    velocity += acceleration * dt;
                }
                if (enableRotation)
                {
                    ApplyRotation(angularVelocity * dt);
                    angularVelocity += angularAcceleration * dt;
                }
                break;

            case IntegrationMethod.SemiImplicit:
                if (enableTranslation)
                {
                    velocity += acceleration * dt;
                    transform.position += velocity * dt;
                }
                if (enableRotation)
                {
                    angularVelocity += angularAcceleration * dt;
                    ApplyRotation(angularVelocity * dt);
                }
                break;

            case IntegrationMethod.VelocityVerlet:
                if (enableTranslation)
                {
                    transform.position += velocity * dt + 0.5f * acceleration * dt * dt;
                    velocity += acceleration * dt;
                }
                if (enableRotation)
                {
                    Vector3 midOmega = angularVelocity + 0.5f * angularAcceleration * dt;
                    ApplyRotation(midOmega * dt);
                    angularVelocity += angularAcceleration * dt;
                }
                break;
        }
    }

    private bool TryCudaIntegrate(float dt, Vector3 acceleration, Vector3 angularAcceleration)
    {
        CudaRigidBodyState state = new CudaRigidBodyState
        {
            position = new Vector4(transform.position.x, transform.position.y, transform.position.z, 0f),
            rotation = new Vector4(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w),
            velocity = new Vector4(velocity.x, velocity.y, velocity.z, 0f),
            angularVelocity = new Vector4(angularVelocity.x, angularVelocity.y, angularVelocity.z, 0f),
            acceleration = new Vector4(acceleration.x, acceleration.y, acceleration.z, 0f),
            angularAcceleration = new Vector4(angularAcceleration.x, angularAcceleration.y, angularAcceleration.z, 0f)
        };

        int integrationMethod = integration == IntegrationMethod.Explicit ? 0 : integration == IntegrationMethod.SemiImplicit ? 1 : 2;

        try
        {
            int status = RbCudaStepSingleBridge(ref state, dt, integrationMethod, enableTranslation ? 1 : 0, enableRotation ? 1 : 0);
            if (status != 0)
            {
                Debug.LogWarning($"CUDA integration failed with status={status}. Falling back to CPU.");
                return false;
            }
        }
        catch (DllNotFoundException)
        {
            cudaUnavailable = true;
            Debug.LogWarning("CUDA plugin not found. Falling back to CPU integration.");
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            cudaUnavailable = true;
            Debug.LogWarning("CUDA plugin entry point not found. Falling back to CPU integration.");
            return false;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"CUDA integration exception: {ex.Message}. Falling back to CPU integration.");
            return false;
        }

        transform.position = new Vector3(state.position.x, state.position.y, state.position.z);
        transform.rotation = new Quaternion(state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w);
        velocity = new Vector3(state.velocity.x, state.velocity.y, state.velocity.z);
        angularVelocity = new Vector3(state.angularVelocity.x, state.angularVelocity.y, state.angularVelocity.z);
        return true;
    }

    private void HandleMouseInputState()
    {
        if (Mouse.current == null) return;

        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            Vector2 mousePos2D = Mouse.current.position.ReadValue();
            Ray ray = cam.ScreenPointToRay(new Vector3(mousePos2D.x, mousePos2D.y, 0f));

            if (Physics.Raycast(ray, out RaycastHit hit) && hit.transform == transform)
            {
                isDragging = true;
                mouseZDistance = cam.WorldToScreenPoint(hit.point).z;
                localAttachmentVertex = FindClosestVertex(transform.InverseTransformPoint(hit.point));

                Vector3 worldVertexPos = transform.TransformPoint(localAttachmentVertex);
                restLength = Vector3.Distance(GetMouseWorldPosition(), worldVertexPos);
                springLine.enabled = true;
            }
        }

        if (Mouse.current.leftButton.wasReleasedThisFrame)
        {
            isDragging = false;
            springLine.enabled = false;
        }
    }

    private void ApplySpringForce()
    {
        Vector3 worldVertexPos = transform.TransformPoint(localAttachmentVertex);
        Vector3 currentMouseWorldPos = GetMouseWorldPosition();

        Vector3 diff = currentMouseWorldPos - worldVertexPos;
        float currentDistance = diff.magnitude;
        Vector3 dir = diff.normalized;

        float displacement = currentDistance - restLength;
        Vector3 springForce = dir * (springConstant * displacement);

        Vector3 r = worldVertexPos - transform.position;
        Vector3 pointVelocity = velocity + Vector3.Cross(angularVelocity, r);
        Vector3 damperForce = -dampingConstant * pointVelocity;

        Vector3 totalForce = springForce + damperForce;

        forceAccumulator += totalForce;
        torqueAccumulator += Vector3.Cross(r, totalForce);

        springLine.SetPosition(0, worldVertexPos);
        springLine.SetPosition(1, currentMouseWorldPos);
    }

    private Vector3 GetMouseWorldPosition()
    {
        Vector2 mousePos2D = Mouse.current.position.ReadValue();
        return cam.ScreenToWorldPoint(new Vector3(mousePos2D.x, mousePos2D.y, mouseZDistance));
    }

    private Vector3 FindClosestVertex(Vector3 hitLocalPoint)
    {
        Vector3[] vertices = mesh.vertices;
        float minDist = float.MaxValue;
        Vector3 closest = Vector3.zero;

        for (int i = 0; i < vertices.Length; i++)
        {
            float dist = Vector3.SqrMagnitude(vertices[i] - hitLocalPoint);
            if (dist < minDist)
            {
                minDist = dist;
                closest = vertices[i];
            }
        }
        return closest;
    }

    private void UpdateInverseInertia()
    {
        float ix = Mathf.Max(inertia.x, 0.0001f);
        float iy = Mathf.Max(inertia.y, 0.0001f);
        float iz = Mathf.Max(inertia.z, 0.0001f);
        inverseInertiaTensor = new Vector3(1f / ix, 1f / iy, 1f / iz);
    }

    private void ResetPhysics()
    {
        transform.position = initialPosition;
        transform.rotation = Quaternion.identity;
        velocity = Vector3.zero;
        angularVelocity = Vector3.zero;
    }

    private void ApplyRotation(Vector3 rotationStep)
    {
        float angle = rotationStep.magnitude * Mathf.Rad2Deg;
        if (angle > 0.0001f)
        {
            Quaternion deltaRot = Quaternion.AngleAxis(angle, rotationStep.normalized);
            transform.rotation = deltaRot * transform.rotation;
        }
    }
}
