# 구현된 항목 정리

이 문서는 현재 코드베이스에 실제로 구현된 기능만 정리한 목록이다. 추측성 기능이나 문서용 설명은 제외했다.

## 1. 시스템 구성

- `mssCreateSystem`, `mssDestroySystem` 기반의 native 시스템 생성/해제
- `mssUploadTopology`를 통한 파티클/스프링 토폴로지 업로드
- 위치, 질량, 고정 마스크, 스프링 연결, rest length를 내부 상태로 저장
- CUDA 사용 가능 여부를 검사하고, 불가능하면 CPU fallback 경로로 동작

## 2. 시뮬레이션 로직

- 준명시적 Euler 기반의 mass-spring 적분(`mssStepSemi`)
- 스프링 힘 계산과 중력 적용
- 고정 파티클 처리
- implicit step 경로(`mssStepImplicit`) 구현
- implicit 선형화, 우변 벡터 구성, conjugate gradient 반복, 최종 상태 반영
- 1자유도 스프링에 대한 닫힌형 implicit step(`mssOneDImplicitStep`)

## 3. CPU fallback

- CUDA가 없을 때 CPU에서 동일한 시뮬레이션을 수행
- 힘 계산(`ComputeForcesSemiHost`)
- 준명시적 적분(`IntegrateSemiHost`)
- CPU 경로에서도 고정 파티클과 damping 처리를 유지

## 4. CUDA 커널 구현

- 초기 힘 설정 커널(`InitForcesKernel`)
- 스프링 힘 계산 커널(`SpringForceKernel`)
- 준명시적 적분 커널(`IntegrateSemiKernel`)
- 상태 복사 커널(`CopyStateKernel`)
- implicit 선형화 커널(`SpringLinearizeKernel`)
- implicit 우변 구성 커널(`BuildImplicitBKernel`)
- 질량 항 초기화 커널(`InitMassTermKernel`)
- implicit 행렬 누적 커널(`AccumulateImplicitMatrixKernel`)
- dot product 부분합 커널(`DotPartialKernel`)
- 부분합 축소 커널(`ReducePartialSumKernel`)
- CG 초기화/갱신 커널(`InitCGKernel`, `CGUpdateXRKernel`, `CGUpdatePKernel`)
- implicit 결과 반영 커널(`CommitImplicitKernel`)

## 5. CUDA 최적화 요소

- `float4` 기반 데이터 배치로 위치/속도/힘을 정렬된 128-bit 단위로 다룸
- `__launch_bounds__(kBlockSize, 2)`로 커널 런치 힌트 지정
- `__ldg` 기반 read-only 접근 경로 사용
- `__byte_perm`을 이용한 fixed mask 분기 보조 처리
- `__shfl_down_sync`를 이용한 warp reduction
- `cooperative_groups::memcpy_async`를 사용한 비동기 복사 경로
- L2 persisting cache limit 설정(`cudaLimitPersistingL2CacheSize`)
- shared memory carveout 선호 설정(`cudaFuncAttributePreferredSharedMemoryCarveout`)

### 적용 이유

- `float4` 배치는 위치/속도/힘을 3차원 벡터로 다루면서도 메모리 접근을 16바이트 단위로 맞추기 위해 사용된다. 이 코드는 입출력과 내부 상태를 모두 벡터형으로 유지하므로, 정렬된 로드/스토리에 유리하다.
- `__launch_bounds__(kBlockSize, 2)`는 커널별 레지스터 사용과 점유율 사이의 균형을 컴파일러에 힌트로 주기 위해 들어간다. 현재 커널은 단순한 point-wise 연산이 많아서, 너무 큰 레지스터 사용으로 워프 수가 줄어드는 상황을 피하려는 목적이다.
- `__ldg`는 읽기 전용 데이터인 질량, 스프링 끝점, rest length, fixed mask를 캐시 친화적으로 읽기 위해 사용된다. 값이 커널 내부에서 바뀌지 않으므로, 일반 로드보다 불필요한 캐시 오염을 줄이기 좋다.
- `__byte_perm`은 fixed mask를 분기 대신 정수 연산으로 정리하기 위해 사용된다. 고정 파티클 여부를 빠르게 스케일 값으로 바꾸면, 이후 연산에서 branch divergence를 줄이기 쉽다.
- `__shfl_down_sync`는 dot product 부분합을 shared memory에 의존하지 않고 warp 내부에서 먼저 줄이기 위해 사용된다. 이 경로는 reduction 병목을 줄이고, 작은 합산을 빠르게 끝내는 데 유리하다.
- `cooperative_groups::memcpy_async`는 `ReducePartialSumKernel`에서 부분합 배열을 shared memory로 옮길 때 latency를 숨기기 위해 사용된다. 계산과 복사를 겹치게 만들어 reduction 단계의 대기 시간을 줄이려는 목적이다.
- L2 persisting cache limit은 반복해서 참조하는 데이터를 L2에 더 오래 남겨 두기 위해 설정된다. 같은 스프링/마스/마스크 데이터를 여러 번 읽는 implicit 경로에서 특히 의미가 있다.
- shared memory carveout 선호 설정은 reduction과 협조적 복사처럼 shared memory를 자주 쓰는 커널에 유리하도록 캐시/공유메모리 자원 배분을 유도하기 위해 들어간다.

## 6. 에러 처리 및 상태 관리

- CUDA API 호출 결과를 `CheckCuda`로 검사
- 마지막 에러 메시지를 `mssGetLastError`로 노출
- 실패 원인을 `gLastError`에 누적

## 7. 현재 코드 기준으로 없는 것

- WMMA/Tensor Core 경로
- PTX inline assembly 직접 작성
- `nvidia-smi`를 통한 클럭 고정 로직
- BF16/TF32를 활용한 별도 mixed precision 경로