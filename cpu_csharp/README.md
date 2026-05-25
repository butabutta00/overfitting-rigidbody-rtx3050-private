# CPU C# Spring-Mass-Damper

`cuda/main.cu`의 1D implicit spring-mass-damper 벤치 입출력 규격과 동일하게 동작하는 C# CPU 벤치마크 구현입니다.

## 위치

- 프로젝트: `cpu_csharp/CpuMassSpringDamper`
- 엔트리포인트: `cpu_csharp/CpuMassSpringDamper/Program.cs`

## 실행

```bash
cd cpu_csharp/CpuMassSpringDamper
dotnet run -c Release -- --position 1.0 --velocity 0.0 --dt 0.016 --mass 1.0 --stiffness 120 --damping 0.2 --steps 200
```

## 주요 옵션

- `--position` : 초기 위치
- `--velocity` : 초기 속도
- `--dt` : 타임스텝
- `--mass` : 질량
- `--stiffness` : 스프링 강성
- `--damping` : 스프링 감쇠
- `--steps` : 적분 반복 횟수

출력은 CUDA와 동일한 key=value 필드를 사용합니다:
- `elapsed_ms`
- `output_x`
- `output_v`
- `checksum`

## 벤치 하니스에서 실행 예시 (mattress 모델)

`bench` 디렉터리의 `main.py`를 통해 제공되는 모델 파일을 사용해 C# 구현만 실행할 수 있습니다.

예: `mattress` 모델을 로드하고 C#만 실행

```bash
uv run python main.py --model=mattress --run=csharp
```

예상 출력(일부):

```
Loading model from: /Users/shapelayer/Documents/GitHub/ShapeLayer/overfitting-rigidbody-rtx3050-private/bench/models/mattress.log
Model loaded: 632 particles, 1890 springs
1D equivalent parameters:
	mass=1.056856, stiffness=500.000000, damping=0.200000
	dt=0.001000
```

참고: 도커용 원샷 스크립트에서는 `--run csharp`를 전달하면 CUDA 빌드 단계를 건너뜁니다:

```bash
scripts/run-bench-in-docker.sh --run csharp -- --samples 5 --warmup 1
```
