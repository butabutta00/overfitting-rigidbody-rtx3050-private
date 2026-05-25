간단 사용법 (bench 폴더 기준)

이제 bench Python 코드는 **CUDA vs C# 비교 벤치마크**만 수행합니다.

- `main.py` : CUDA(`cuda/main.cu`)와 C#(`cpu_csharp/CpuMassSpringDamper`)를 **동일 인자 규격**으로 실행해 결과/시간 비교

사전 요구사항
- Python 3
- CUDA 툴킷/nvcc (CUDA 빌드 사용 시)
- .NET SDK (`dotnet`)

기본 실행 (레포 루트에서)

```bash
python3 bench/main.py
```

도커 인스턴스 최초 클론 직후 원샷 실행

```bash
bash scripts/run-bench-in-docker.sh -- --samples 10 --warmup 2
```

위 스크립트는 다음 순서로 실행됩니다.
- dotnet 런타임/SDK 설치
- `scripts/build-cuda-standalone-test.sh`로 CUDA 벤치 바이너리 빌드
- `uv run python bench/main.py --skip-cuda-build ...` 실행

벤치 결과 자동 커밋/푸시 옵션

```bash
bash scripts/run-bench-in-docker.sh \
  --commit-push-results \
  --git-username "Bench Runner" \
  --git-email "bench-runner@users.noreply.github.com" \
  -- --samples 10 --warmup 2
```

`--commit-push-results`를 사용하면 최신 `bench/results/<timestamp>` 디렉토리를 커밋하고 push합니다.

주요 옵션 예시

```bash
# 샘플/워밍업 조정
python3 bench/main.py --samples 20 --warmup 4

# CUDA/C# 빌드 생략 (이미 빌드된 상태에서)
python3 bench/main.py --skip-cuda-build --skip-csharp-build

# 공통 입력 파라미터 변경 (CUDA/C# 모두 동일하게 적용)
python3 bench/main.py --position 1.0 --velocity 0.0 --dt 0.016 --mass 1.0 --stiffness 120 --damping 0.2 --steps 200
```

출력
- 콘솔: 샘플별 비교 결과, 요약 통계
- 파일: `bench/results/<timestamp>/cuda_vs_csharp_samples.csv`

CSV 필드
- `cuda_elapsed_ms`, `csharp_elapsed_ms`
- `cuda_output_x`, `cuda_output_v`, `cuda_checksum`
- `csharp_output_x`, `csharp_output_v`, `csharp_checksum`
- `abs_diff_output_x`, `abs_diff_output_v`, `abs_diff_checksum`
- `cuda_wall_ms`, `csharp_wall_ms`
- `wall_speedup_cuda_over_csharp`, `elapsed_speedup_cuda_over_csharp`

참고
- 두 바이너리는 동일 key=value 출력 필드(`elapsed_ms`, `output_x`, `output_v`, `checksum`)를 사용합니다.
- `*_wall_ms`는 Python이 프로세스 외부에서 측정한 실행 시간입니다.
