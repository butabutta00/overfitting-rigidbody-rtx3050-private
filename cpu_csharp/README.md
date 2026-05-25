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
