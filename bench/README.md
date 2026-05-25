간단 사용법 (bench 폴더 기준)

이제 bench Python 코드는 **CUDA vs C# 비교 벤치마크**만 수행합니다.

- `main.py` : CUDA(`cuda/main.cu` 엔트리포인트)와 C#(`cpu_csharp/CpuMassSpringDamper`)를 같은 샘플 수로 반복 실행하여 시간 비교

사전 요구사항
- Python 3
- CUDA 툴킷/nvcc (CUDA 빌드 사용 시)
- .NET SDK (`dotnet`)

기본 실행 (레포 루트에서)

```bash
python3 bench/main.py
```

주요 옵션 예시

```bash
# 샘플/워밍업 조정
python3 bench/main.py --samples 20 --warmup 4

# CUDA/C# 빌드 생략 (이미 빌드된 상태에서)
python3 bench/main.py --skip-cuda-build --skip-csharp-build

# CUDA 입력 파라미터 변경
python3 bench/main.py --cuda-dt 0.016 --cuda-stiffness 120 --cuda-damping 0.2 --cuda-steps 8

# C# 입력 파라미터 변경
python3 bench/main.py --csharp-particles 16384 --csharp-steps 400 --csharp-substeps 2 --csharp-stiffness 120 --csharp-damping 0.2
```

출력
- 콘솔: 샘플별 비교 결과, 요약 통계
- 파일: `bench/results/<timestamp>/cuda_vs_csharp_samples.csv`

CSV 필드
- `cuda_wall_ms`
- `csharp_wall_ms`
- `csharp_reported_ms`
- `wall_speedup_cuda_over_csharp`
- `reported_speedup_cuda_over_csharp`

참고
- `csharp_wall_ms`는 Python 프로세스 외부에서 측정한 실행 시간
- `csharp_reported_ms`는 C# 프로그램 내부에서 출력한 시뮬레이션 루프 시간
