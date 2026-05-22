using UnityEngine;
using UnityEngine.InputSystem;

public class OneDOFSpringMassCustomized : MonoBehaviour
{
    [Range(0.001f, 0.2f)]
    [Tooltip("타임스텝이 커질수록 시스템이 어떻게 변하는지 관찰하세요.")]
    public float timeStep = 0.01f;

    [Tooltip("K값이 커질수록 스프링의 강성이 강해지지만 시스템은 불안정해집니다.")]
    public float springStiffness = 200f;

    [Header("Fixed Scenario Parameters (Read Only)")]
    [SerializeField] private float mass = 1.0f;
    [SerializeField] private float springDamping = 1f;

    [Header("1-DOF Scenario Setup")]
    [Tooltip("스프링이 시작되는 고정된 벽(Cube) 오브젝트를 연결하세요.")]
    public Transform anchorTransform;

    [Header("State (Read Only)")]
    public float positionX; // 원점(0) 기준의 변위 x
    public float velocityX; // 속도 v

    private Vector3 initialWorldPosition;
    private bool isDragging = false;
    private Camera mainCam;
    private float mouseZDistance;
    private LineRenderer springLine;

    // 실시간 Hz 측정용 변수
    private int fixedUpdateCount = 0;
    private float hzTimer = 0f;
    private float actualPhysicsHz = 0f;

    private void Start()
    {
        mainCam = Camera.main;
        initialWorldPosition = transform.position;
        positionX = 0f;
        velocityX = 0f;

        SetupLineRenderer();
    }

    private void SetupLineRenderer()
    {
        // LineRenderer가 없으면 자동으로 부착합니다.
        if (!TryGetComponent(out springLine))
        {
            springLine = gameObject.AddComponent<LineRenderer>();
        }

        // 스프링 선 디자인 설정
        springLine.startWidth = 0.04f;
        springLine.endWidth = 0.04f;
        springLine.positionCount = 2;
        springLine.material = new Material(Shader.Find("Sprites/Default"));
        springLine.startColor = Color.cyan;
        springLine.endColor = Color.white;

        // 실시간으로 항상 보여야 하므로 true 설정
        springLine.enabled = true;
    }

    private void Update()
    {
        // 사용자가 설정한 timeStep에 맞춰 유니티의 실제 물리 루프 속도를 강제로 동기화
        Time.fixedDeltaTime = timeStep;

        // 마우스 입력 감지
        HandleMouseInput();

        // 실제 초당 물리 업데이트 횟수(Hz) 측정
        hzTimer += Time.deltaTime;
        if (hzTimer >= 1.0f)
        {
            actualPhysicsHz = fixedUpdateCount / hzTimer;
            fixedUpdateCount = 0;
            hzTimer = 0f;
        }

        // 시각적 위치 갱신 (물리 연산은 x축 변위만 제어)
        transform.position = initialWorldPosition + new Vector3(positionX, 0f, 0f);

        // [핵심 수정] 벽(Anchor)과 물체(Sphere) 사이에 실시간으로 스프링 선을 그립니다.
        if (anchorTransform != null && springLine != null)
        {
            springLine.SetPosition(0, anchorTransform.position);
            springLine.SetPosition(1, transform.position);
        }
    }

    private void FixedUpdate()
    {
        fixedUpdateCount++;

        // 사용자가 마우스로 드래그 중일 때는 물리 연산을 일시 정지
        if (isDragging)
        {
            velocityX = 0f;
            return;
        }

        // Closed-form implicit Euler for m*x'' + c*x' + k*x = 0
        float dt = timeStep;
        float denom = mass + springDamping * dt + springStiffness * dt * dt;
        if (denom < 1e-6f) denom = 1e-6f;

        float newVelocity = (mass * velocityX - springStiffness * dt * positionX) / denom;
        float newPosition = positionX + dt * newVelocity;

        velocityX = newVelocity;
        positionX = newPosition;
    }

    private void HandleMouseInput()
    {
        if (Mouse.current == null) return;

        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            Vector2 mousePos2D = Mouse.current.position.ReadValue();
            Ray ray = mainCam.ScreenPointToRay(new Vector3(mousePos2D.x, mousePos2D.y, 0f));

            if (Physics.Raycast(ray, out RaycastHit hit) && hit.transform == transform)
            {
                isDragging = true;
                mouseZDistance = mainCam.WorldToScreenPoint(transform.position).z;
            }
        }

        if (Mouse.current.leftButton.isPressed && isDragging)
        {
            Vector2 mousePos2D = Mouse.current.position.ReadValue();
            Vector3 mouseWorldPos = mainCam.ScreenToWorldPoint(new Vector3(mousePos2D.x, mousePos2D.y, mouseZDistance));
            positionX = mouseWorldPos.x - initialWorldPosition.x;
        }

        if (Mouse.current.leftButton.wasReleasedThisFrame)
        {
            isDragging = false;
        }
    }

    private void OnGUI()
    {
        // 정보가 한 줄 더 늘어나므로 박스 높이를 키웠습니다 (90 -> 110)
        GUI.backgroundColor = Color.black;
        GUI.Box(new Rect(10, 10, 300, 110), "1-DOF Spring-Mass System Info");

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.normal.textColor = Color.white;
        labelStyle.fontSize = 13;

        // 1. 현재 설정된 타임스텝 표시
        GUI.Label(new Rect(20, 35, 280, 20), $"Time Step (dt): {timeStep:F4} s", labelStyle);

        // 2. 실제 작동 중인 초당 물리 업데이트 횟수(Hz) 표시
        GUI.Label(new Rect(20, 55, 280, 20), $"Physics Rate: {actualPhysicsHz:F1} Hz", labelStyle);

        // 3. 현재 원점으로부터의 변위 표시
        GUI.Label(new Rect(20, 75, 280, 20), $"Displacement (x): {positionX:F4} m", labelStyle);

        // 4. 타임스텝 x 업데이트 레이트 곱한 값 표시
        float productValue = timeStep * actualPhysicsHz;

        // 정상 동기화(1.0 부근)일 때는 녹색, 프레임 드랍 등이 발생하면 황색 표시
        if (Mathf.Abs(productValue - 1.0f) < 0.02f)
        {
            labelStyle.normal.textColor = Color.green;
        }
        else
        {
            labelStyle.normal.textColor = Color.yellow;
        }

        GUI.Label(new Rect(20, 95, 280, 20), $"dt x Rate Product: {productValue:F3} (Target: 1.0)", labelStyle);
    }
}
