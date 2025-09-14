export const nodeInformation = {
  normalization: {
    description: String.raw`![layernorm vs rmsnorm](/img/normalization.png)
  
  ## 정규화의 목적
  - 층 출력을 일정한 범위로 맞춰 **학습 안정성**과 **수렴 속도**를 높임.
  - 초기화/학습률/배치 구성에 대한 **민감도 감소**.
  - 공통 형태:
  $$
  \hat{x} = normalize(x)
  y = γ ⊙ \hat{x} + β
  $$
  여기서 γ, β는 학습 가능한 스케일/시프트.
  
  ## 세 방법의 핵심 차이
  | 방법 | 정규화 축(무엇을 기준으로 보나) | 학습/추론에서 사용하는 통계 | 장점 | 주의사항 |
  |---|---|---|---|---|
  | **BatchNorm** | 채널별로 **배치 전체**(Conv: N,H,W / FC: N) | 학습: 배치 통계 / 추론: 러닝 평균·분산 | CNN에서 수렴/일반화 우수 | 소배치·가변 길이에 취약, 러닝 스탯 관리 필요 |
  | **LayerNorm** | 샘플 **하나의 특징 전체**(대개 마지막 차원) | 항상 현재 샘플 통계 | 배치 크기 영향 적음, RNN/Transformer 적합 | BN 대비 약간의 연산 증가(대개 무시 가능) |
  | **RMSNorm** | LN과 동일 축, **평균 제거 없이** 크기만 | 항상 현재 샘플 통계 | 더 단순·가벼움, 최신 Transformer 다수 | 평균 미제거로 분포 치우침에 민감할 수 있음 |
  
  ## Batch Normalization (BN)
  - **개념**: 같은 채널에서 배치(및 공간) 축의 평균과 분산으로 표준화.
  - **공식(개념)**
  ~~~
  x̂ = (x - μ_B) / sqrt(σ_B^2 + ε)
  y  = γ ⊙ x̂ + β
  ~~~
  학습 시 μ_B, σ_B^2는 현재 배치에서 계산, **추론 시에는 학습 중 적산한 러닝 스탯** 사용.
  - **적합**: 충분한 배치 크기의 **CNN**.

  ## Layer Normalization (LN)
  - **개념**: 샘플 하나의 특징 차원 전체를 사용해 표준화(배치와 무관).
  - **공식(개념)**
  ~~~
  x̂ = (x - μ_sample) / sqrt(σ_sample^2 + ε)
  y  = γ ⊙ x̂ + β
  ~~~
  - **적합**: 배치 크기 변화가 잦은 **RNN/Transformer**. 러닝 스탯 불필요.

  ## RMS Normalization (RMSNorm)
  - **개념**: LN과 동일하지만 평균을 빼지 않고 **RMS(제곱평균제곱근)** 로만 정규화.
  - **공식(개념)**
  ~~~
  rms(x) = sqrt(mean(x^2) + ε)
  y      = γ ⊙ (x / rms(x))           # (옵션) + β
  ~~~
  - **적합**: 단순·경량 구현이 중요한 **Transformer**.

  ## 선택 가이드
  - **CNN + 충분한 배치** → **BatchNorm**  
  - **RNN/Transformer + 소·가변 배치** → **LayerNorm** 또는 **RMSNorm**  
  - 단순·속도 우선 → **RMSNorm**  
  - 보편적 기본값 → **LayerNorm**
  `,
  },
};
