using UnityEngine;

public class CameraController : MonoBehaviour
{
    [Header("Zoom Settings")]
    [Tooltip("마우스 휠 감도")]
    public float zoomSpeed = 5.0f;

    [Tooltip("최소 줌 거리 (최대 줌인)")]
    public float minZoom = 2.0f;

    [Tooltip("최대 줌 거리 (최대 줌아웃)")]
    public float maxZoom = 20.0f;

    private Camera cam;

    private void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            cam = Camera.main;
        }
    }

    private void Update()
    {
        float scrollInput = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scrollInput) < 0.0001f) return;

        if (cam.orthographic)
        {
            // 직교 카메라 (2D 스타일 뷰)
            cam.orthographicSize -= scrollInput * zoomSpeed;
            cam.orthographicSize = Mathf.Clamp(cam.orthographicSize, minZoom, maxZoom);
        }
        else
        {
            // 원근 카메라 (3D 뷰): 앞뒤로 이동
            transform.Translate(Vector3.forward * scrollInput * zoomSpeed, Space.Self);

            // 시뮬레이션 대상(원점 부근으로 가정)과의 거리 제한을 두고 싶다면 
            // 필요에 따라 position 값을 조절할 수 있습니다.
        }
    }
}