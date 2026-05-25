# CPU C# Spring-Mass-Damper

`cuda/mass_spring_native.cu`의 semi-implicit spring-mass-damper 연산(중력 초기화 → 스프링 힘 누적 → semi-implicit 적분)을 CPU C#으로 수행하는 벤치마크용 구현입니다.

## 위치

- 프로젝트: `cpu_csharp/CpuMassSpringDamper`
- 엔트리포인트: `cpu_csharp/CpuMassSpringDamper/Program.cs`

## 실행

```bash
cd cpu_csharp/CpuMassSpringDamper
dotnet run -c Release -- --particles 8192 --steps 200 --dt 0.016 --stiffness 120 --damping 0.2
```

## 주요 옵션

- `--particles` : 파티클 개수
- `--steps` : 시뮬레이션 스텝 수
- `--substeps` : 한 스텝 당 substep 수
- `--dt` : 타임스텝
- `--stiffness` : 스프링 강성
- `--damping` : 스프링 감쇠
- `--velocity-damping` : 전역 속도 감쇠 계수
- `--gravity-y` : 중력 Y 성분
- `--mass` : 질량
- `--spacing` : 초기 체인 간격
- `--fixed-first` : 첫 파티클 고정 여부

출력으로 총 실행시간(`elapsed_ms`), 처리량(`steps_per_sec`), 상태 체크섬(`checksum`)을 표시합니다.
