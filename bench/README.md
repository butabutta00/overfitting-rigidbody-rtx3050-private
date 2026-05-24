간단 사용법 (bench 폴더 기준)

이 폴더에는 두 가지 주요 스크립트가 있습니다:

- `main.py` : CUDA 바이너리를 여러 번 실행해 런타임/출력을 수집하는 배치 벤치마크
- `interactive.py` : 바이너리를 `--interactive` 모드로 실행하고 stdin/stdout으로 연속 시뮬레이션을 드라이브하는 하니스

사전 요구사항
- CUDA 툴킷 및 nvcc가 설치되어 있고, 레포의 빌드 스크립트로 바이너리를 빌드해야 합니다.
	예: `bash ../scripts/build-cuda-standalone-test.sh "86-real;86-virtual"` (환경에 맞게 아키텍처 조정)

예시: `uv run python`로 실행 (레포 루트에서)

1) `interactive.py` 실행 — 연속 시뮬레이션과 파라미터 모드 테스트

```bash
# 레포 루트에서 (binary 경로는 빌드 출력에 맞게 조정)
uv run python bench/interactive.py --binary cuda/build/main --mode hold --steps 50

# K(stiffness)를 사인파로 변조
uv run python bench/interactive.py --binary cuda/build/main --mode sine_k --amp 0.3 --freq 0.5 --steps 200

# K를 그리드로 순회
uv run python bench/interactive.py --binary cuda/build/main --mode grid_k --grid-start 60 --grid-end 180 --grid-steps 7 --steps 70

# 노이즈 모드 (재현성 위해 seed 지정)
uv run python bench/interactive.py --binary cuda/build/main --mode noise_k --scale 0.15 --seed 42 --steps 100
```

2) `main.py` 배치 벤치마크 실행 — 여러 샘플을 측정하여 CSV/플롯 저장

```bash
# 기본 사용 (샘플 10, 워밍업 2)
uv run python bench/main.py --binary cuda/build/main --samples 10 --warmup 2

# 입력 파라미터 지정 예시
uv run python bench/main.py --binary cuda/build/main --samples 20 --warmup 4 --position 1.0 --velocity 0.0 --dt 0.016 --mass 1.0 --stiffness 120.0 --damping 0.2 --steps 8
```

팁
- 바이너리가 macOS에서 빌드되지 않거나 `nvcc`가 없는 경우, CUDA 지원이 있는 머신(Linux + NVIDIA)에서 빌드/테스트하세요.
- `interactive.py`의 모드나 `generate_sequence()`를 수정하면 외력(probe)이나 파라미터 그리드를 쉽게 실험할 수 있습니다.

파일 위치
- 핸들러: [bench/main.py](main.py)
- 인터랙티브 하니스: [bench/interactive.py](interactive.py)

문제가 있으면 어떤 환경(OS, GPU, 빌드 출력)인지 알려주시면 도움 드리겠습니다.
