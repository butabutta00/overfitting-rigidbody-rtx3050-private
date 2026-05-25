# Overfitting Mass-Spring-Damper System in RTX 3050

> 단일 플랫폼(좋은 예로 Xbox가 있습니다)으로 수년간 일했던 개발자가 있다고 합시다. 수년이라면 해당 플랫폼에서 통하는 흑마법을 익히기에 충분합니다.
> 
> - 《C++ 최적화》, 커트 건서로스 저(옥찬호 역), 2019.

이 리포지토리에서는 [과제 요구사항](./REQUIREMENTS.md)에 대응하여, Mass-Spring-Damper 시스템의 수치적분 계산을 오직 시연 환경인 RTX3050 아키텍처에 맞춰 과적합하는 내용을 다루고 있습니다.  

## Introduction

C# MonoBehaviour로 구현된 Mass-Spring-Damper 시스템의 수치적분 계산을, 큰 시간 간격(Time-step), 높은 강성(Stiffness)에도 안정적으로 유지되면서, 컴퓨터의 과부화로 시뮬레이션이 끊기지 않는 것을 목표로 구현합니다.  

위 목표를 위해 (1) 시스템의 수치적분 계산과 렌더링 타이밍을 분리, (2) CPU에서 동작하는 C# MonoBehaviour 구현을 GPU에서 동작하는 CUDA C++ 플러그인으로 대체, (3) 이 CUDA 구현을 RTX 3050 아키텍처에 맞춰 최적화하였습니다.  

## Requirements

Hardware Requirements
- NVIDIA RTX 3050 GPU (VARM 8GB) (or higher than [Compute Capability 8.6](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))

Software Requirements
- C++ Compiler
- NVCC
- CMake
- Unity 2022.3.62f3

## Design

### 1. 시스템의 수치적분 계산과 렌더링 타이밍 분리

### 2. RTX 3050 아키텍처에 맞춘 CUDA C++ 플러그인 구현

**`float4` 사용**

RTX 3050 (8GB)의 메모리 대역폭은 128-bit입니다.[^1] 따라서 메모리 접근을 벡터화하기 위해 `float4` 타입을 사용하여 위치, 속도, 힘 데이터를 128-bit 단위로 정렬하여 불러오도록 했습니다.

**`__launch_bounds__`로 레지스터 수 제한**

커널에 Point-wise 연산이 많아 레지스터 사용이 많아질 수 있는데, RTX 3050의 SM당 레지스터 파일 크기는 64K 32-bit 레지스터입니다.[^2] 스레드당 레지스터를 너무 많이 쓰면 Occupancy(점유율)가 떨어져 하드웨어의 병렬성을 충분히 활용하지 못합니다. 따라서 RTX 3050의 병목을 피하기 위해 커널 런치 힌트로 `__launch_bounds__(kBlockSize, 2)`를 사용해 컴파일러가 점유율을 추론해 레지스터를 배치하도록 유도했습니다.  

**`__ldg`로 읽기 전용 데이터 최적화**

질량, 스프링 끝점, rest length, fixed mask등은 커널 내부에서 값이 바뀌는 것을 의도하지 않았습니다. 따라서 이들 값을 읽기 전용 데이터로 간주하여 `__ldg`를 사용하여 캐시 친화적으로 읽도록 했습니다. 이렇게 하여 일반 로드보다 불필요한 캐시 오염을 줄이면서 읽기 전용 데이터를 효율적으로 사용할 수 있습니다.

**`__byte_perm`을 이용한 fixed mask 분기 보조 처리**

고정 파티클 여부를 빠르게 스케일 값으로 바꾸면, 이후 연산에서 branch divergence를 줄이기 쉽습니다. 따라서 `__byte_perm`을 이용하여 fixed mask를 분기 대신 정수 연산으로 정리하도록 했습니다.

**`__shfl_down_sync`로 dot product 부분합 최적화**

dot product 부분합을 shared memory에 의존하지 않고 warp 내부에서 먼저 줄이도록 `__shfl_down_sync`를 사용하여 reduction 병목을 줄이고, 작은 합산을 빠르게 끝내도록 했습니다.

**`cooperative_groups::memcpy_async`로 비동기 복사**

`ReducePartialSumKernel`에서 부분합 배열을 shared memory로 옮길 때 latency를 숨기기 위해 `cooperative_groups::memcpy_async`를 사용하여 계산과 복사를 겹치게 만들어 reduction 단계의 대기 시간을 줄이도록 했습니다.

**L2 persisting cache limit 설정**

반복해서 참조하는 데이터를 L2에 더 오래 남겨 두기 위해 L2 persisting cache limit을 설정했습니다. 같은 스프링/마스/마스크 데이터를 여러 번 읽는 implicit 경로에서 특히 의미가 있습니다.

**shared memory carveout 선호 설정**

reduction과 협조적 복사처럼 shared memory를 자주 쓰는 커널에 유리하도록 캐시/공유메모리 자원 배분을 유도하기 위해 shared memory carveout 선호 설정을 했습니다.


[^1]: GeForce RTX 3050 Graphics Cards, https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/
[^2]: NVIDIA Ampere Tuning Guide, https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy
