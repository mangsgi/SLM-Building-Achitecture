const BT = '`';
const B = '**';

export const nodeInformation = {
  normalization: {
    description: String.raw`![layernorm vs rmsnorm](/img/normalization.png)
  
  ## 정규화의 목적
  - 층 출력을 일정한 범위로 맞춰 ${B}학습 안정성${B}과 ${B}수렴 속도${B}를 높임.
  - 초기화/학습률/배치 구성에 대한 ${B}민감도 감소${B}.
  - 공통 형태:
  $$
  \hat{x} = normalize(x)
  y = γ ⊙ \hat{x} + β
  $$
  여기서 γ, β는 학습 가능한 스케일/시프트.
  
  ## 세 방법의 핵심 차이
  | 방법 | 정규화 축(무엇을 기준으로 보나) | 학습/추론에서 사용하는 통계 | 장점 | 주의사항 |
  |---|---|---|---|---|
  | ${B}BatchNorm${B} | 채널별로 ${B}배치 전체${B}(Conv: N,H,W / FC: N) | 학습: 배치 통계 / 추론: 러닝 평균·분산 | CNN에서 수렴/일반화 우수 | 소배치·가변 길이에 취약, 러닝 스탯 관리 필요 |
  | ${B}LayerNorm${B} | 샘플 ${B}하나의 특징 전체${B}(대개 마지막 차원) | 항상 현재 샘플 통계 | 배치 크기 영향 적음, RNN/Transformer 적합 | BN 대비 약간의 연산 증가(대개 무시 가능) |
  | ${B}RMSNorm${B} | LN과 동일 축, ${B}평균 제거 없이${B} 크기만 | 항상 현재 샘플 통계 | 더 단순·가벼움, 최신 Transformer 다수 | 평균 미제거로 분포 치우침에 민감할 수 있음 |
  
  ## Batch Normalization (BN)
  - ${B}개념${B}: 같은 채널에서 배치(및 공간) 축의 평균과 분산으로 표준화.
  - ${B}공식(개념)${B}
  $$
  x̂ = (x - μ_B) / sqrt(σ_B^2 + ε)
  y  = γ ⊙ x̂ + β
  $$
  학습 시 μ_B, σ_B^2는 현재 배치에서 계산, ${B}추론 시에는 학습 중 적산한 러닝 스탯${B} 사용.
  - ${B}적합${B}: 충분한 배치 크기의 ${B}CNN${B}.

  ## Layer Normalization (LN)
  - ${B}개념${B}: 샘플 하나의 특징 차원 전체를 사용해 표준화(배치와 무관).
  - ${B}공식(개념)${B}
  $$
  x̂ = (x - μ_sample) / sqrt(σ_sample^2 + ε)
  y  = γ ⊙ x̂ + β
  $$
  - ${B}적합${B}: 배치 크기 변화가 잦은 ${B}RNN/Transformer${B}. 러닝 스탯 불필요.

  ## RMS Normalization (RMSNorm)
  - ${B}개념${B}: LN과 동일하지만 평균을 빼지 않고 ${B}RMS(제곱평균제곱근)${B} 로만 정규화.
  - ${B}공식(개념)${B}
  $$
  rms(x) = sqrt(mean(x^2) + ε)
  y      = γ ⊙ (x / rms(x)) \quad (옵션) + β
  $$
  - ${B}적합${B}: 단순·경량 구현이 중요한 ${B}Transformer${B}.

  ## 선택 가이드
  - ${B}CNN + 충분한 배치${B} → ${B}BatchNorm${B}  
  - ${B}RNN/Transformer + 소·가변 배치${B} → ${B}LayerNorm${B} 또는 ${B}RMSNorm${B}  
  - 단순·속도 우선 → ${B}RMSNorm${B}  
  - 보편적 기본값 → ${B}LayerNorm${B}
  `,
  },
  tokenEmbedding: {
    description: String.raw`![token embedding diagram](/img/token_embedding.png)
  
  ## 토큰(Token)이란?
  - 토큰은 텍스트를 구성하는 가장 작은 단위입니다.  
  - 한 단어 전체일 수도 있고, 단어 조각(subword)이나 문자 하나일 수도 있습니다.  
  - 토큰화(Tokenization) 과정을 통해 긴 문장을 모델이 이해할 수 있는 작은 조각들로 나눕니다.  
  
  예:  
  "나는 밥을 먹었다" → ["나는", "밥을", "먹었다"]
  
  ---
  
  ## Token Embedding이란?
  - 나눠진 토큰을 ${B}숫자 벡터${B}로 바꾸는 과정입니다.  
  - 컴퓨터는 글자를 직접 이해하지 못하기 때문에, 토큰을 고정된 차원의 벡터로 변환합니다.  
  - 이 벡터는 단어의 ${B}의미적 특징${B}을 담고 있어, 비슷한 의미의 단어는 가까운 위치에 놓이게 됩니다.  
  
  비유로 말하면:  
  > 단어들을 지도 위의 좌표로 배치하는 것과 같습니다.  
  > "고양이"와 "강아지"는 같은 동물이라 가까운 위치,  
  > "고양이"와 "민주주의"는 전혀 달라서 멀리 떨어집니다.
  
  ---
  
  ## 동작 방식
  1. ${B}토큰화${B}: 문장을 토큰 단위로 분해 → ["I", "love", "apple", "##s"]  
  2. ${B}토큰 ID 변환${B}: 각 토큰에 숫자 ID 할당 → [101, 202, 305, 401]  
  3. ${B}임베딩 행렬 lookup${B}: ID에 해당하는 벡터를 찾아옴 → [0.1, 0.2, …]  
  4. ${B}결합${B}: 이 벡터에 위치 정보(Positional Embedding)를 더해 Transformer로 전달.  
  
  수식 (개념):  
  $$
  x_i = \text{Embedding}(token\_id_i), \quad 
  z_i = x_i + p_i
  $$
  - $x_i$: 토큰 임베딩 벡터  
  - $p_i$: 위치 임베딩 벡터  
  - $z_i$: 모델 입력으로 쓰이는 최종 벡터  
  
  ---
  
  ## 왜 중요한가?
  - 텍스트를 ${B}수치적 데이터${B}로 바꿔 모델이 이해할 수 있게 함.  
  - 단어 간 ${B}의미적 유사성${B}을 반영하여 더 자연스러운 학습 가능.  
  - 드문 단어도 subword 단위로 나눠 처리할 수 있어 ${B}유연성${B}이 높음.  
  
  ---
  
  ## 우리 프로그램에서
  - 사용자가 입력한 문장은 먼저 ${B}토큰 단위${B}로 쪼개집니다.  
  - 각 토큰은 ${B}숫자 벡터(Token Embedding)${B} 로 변환됩니다.  
  - 이 벡터는 위치 정보와 합쳐져 Transformer 블록으로 전달됩니다.  
  - 따라서 Token Embedding은 우리 모델의 ${B}첫 출발점${B}이자, 텍스트 이해의 기본 단계입니다.
    `,
  },
  positionalEmbedding: {
    description: String.raw`![positional embedding diagram](/img/positional_embedding.png)
  
  ## 포지셔널 임베딩(Positional Embedding)이란?
  - 토큰 임베딩은 단어의 의미를 벡터로 바꾸지만, ${B}순서 정보${B}는 담고 있지 않습니다.  
  - 따라서 문장에서 ${B}"나는 밥을 먹었다"${B} 와 ${B}"밥을 나는 먹었다"${B} 가 같은 벡터로 해석될 수 있습니다.  
  - 이를 막기 위해, 각 토큰에 ${B}위치 정보${B}를 담은 벡터를 더해줍니다.  
  - 이렇게 하면 모델이 문장의 순서와 구조를 이해할 수 있게 됩니다.
  
  ---
  
  ## 어떻게 동작할까?
  1. 입력 문장: "This is an example."  
  2. 토큰화: → ["This", "is", "an", "example", "."]  
  3. 토큰 임베딩: 각 단어를 의미 벡터로 변환  
  4. 포지션 임베딩: 각 위치(1번, 2번, 3번…)에 대응되는 벡터 생성  
  5. ${B}합산${B}: 토큰 벡터 + 포지션 벡터 → 최종 입력  
  
  수식 (개념):  
  $$
  z_i = x_i + p_i
  $$  
  - $x_i$: i번째 토큰의 의미 벡터  
  - $p_i$: i번째 위치의 위치 벡터  
  - $z_i$: Transformer에 들어가는 최종 입력  
  
  ---
  
  ## 절대 vs 상대 포지션
  - ${B}절대 위치 임베딩${B}: "첫 번째, 두 번째…"처럼 위치 자체를 직접 반영  
  - ${B}상대 위치 임베딩${B}: "이 단어와 저 단어가 얼마나 떨어져 있는가?"에 초점을 맞춤  
  - 최신 모델들은 상대 임베딩이나 ${B}RoPE(Rotary Positional Embedding)${B} 같은 변형 방식을 많이 사용합니다.
  
  ---
  
  ## 비유로 이해하기
  - 단어가 ${B}지도 위 점${B}이라면, 포지셔널 임베딩은 ${B}좌표축 눈금${B} 같은 역할을 합니다.  
  - "서울은 부산의 위쪽" 같은 ${B}위치 관계${B}를 알려주는 셈입니다.
  
  ---
  
  ## 우리 프로그램에서는?
  - Token Embedding으로 만든 벡터에 ${B}Positional Embedding${B}을 더해 Transformer 블록에 입력합니다.  
  - 화면에서는 ${B}"토큰 의미 + 위치 정보 = 최종 입력"${B} 구조를 직관적으로 확인할 수 있습니다.  
  - ${B}핵심 파라미터${B}  
    - ctxLength: 생성할 위치 벡터의 길이(모델이 한 번에 볼 수 있는 최대 토큰 수).  
    - embDim: 각 위치 벡터의 차원(토큰 임베딩 차원과 동일).  
    `,
  },
  feedForward: {
    description: String.raw`![feed forward diagram](/img/feed_forward.png)
  
  ## 피드포워드 네트워크(Feed Forward Network, FFN)란?
  - FFN은 FNN(Feedforward Neural Network, 다층 퍼셉트론)의 한 형태로,  
    ${B}입력 → 선형 변환 → 활성화 → 선형 변환 → 출력${B} 구조를 가집니다.  
  - Transformer 블록 안에서는 각 토큰 벡터에 독립적으로 적용되므로 "position-wise FFN"이라고 부릅니다.  
  - Self-Attention이 관계를 본 뒤, FFN은 각 토큰의 의미를 더 풍부하게 다듬어 줍니다.
  
  ---
  
  ## 어떻게 동작할까?
  1. ${B}입력${B}: Self-Attention에서 나온 출력 벡터($x$)를 받습니다.  
  2. ${B}첫 번째 선형 변환 (Fully Connected Layer)${B} → 차원을 확장합니다.  
     - 보통 $4 \times embDim$ 크기까지 늘립니다.  
  3. ${B}활성화 함수(Activation)${B} 적용 (예: GELU, ReLU, SwiGLU 등).  
  4. ${B}두 번째 선형 변환${B} → 다시 원래 차원($embDim$)으로 축소합니다.  
  
  수식 (개념):  
  $$
  FFN(x) = W_2 \cdot f(W_1 \cdot x + b_1) + b_2
  $$  
  
  ---
  
  ## 왜 중요한가?
  - ${B}비선형성 추가${B}: 단순한 선형 변환만 하지 않고 더 복잡한 패턴을 학습할 수 있음.  
  - ${B}표현력 확장${B}: Attention이 단어 간 관계를 본다면, FFN은 그 결과를 더 풍부한 특징으로 바꿔줌.  
  - ${B}독립 처리${B}: 각 토큰은 독립적으로 FFN을 거치므로, "자기 자신을 다듬는 과정"이라고 볼 수 있음.
  - Attention이 "친구 관계"를 살펴보는 과정이라면,  
    FFN은 "내가 가진 능력을 더 키워주는 개인 훈련"에 비유할 수 있습니다.  
  
  ---
  
  ## 우리 프로그램에서는?
  - FFN 레이어는 Transformer Block 안에 기본으로 포함되어 있습니다.  
  - 사용자가 조절할 수 있는 ${B}핵심 파라미터${B}:
    - embDim: 입력/출력 차원 (토큰 벡터 크기)  
    - hiddenDim: 내부 확장 차원 (보통 4 × embDim)  
    - activation: 사용할 활성화 함수 (ReLU, GELU, SwiGLU 등)  
  - 사용자는 ${B}hiddenDim 비율${B}과 ${B}activation 종류${B}를 바꿔가며 성능 차이를 직접 확인할 수 있습니다.
    `,
  },
  dropout: {
    description: String.raw`![dropout diagram](/img/dropout.png)
  
  ## 드롭아웃(Dropout)이란?
  - 학습 중 일부 뉴런(노드)을 ${B}무작위로 꺼버리는 기법${B}입니다.
  - 이렇게 하면 모델이 특정 뉴런이나 연결에 ${B}과도하게 의존하는 것(과적합)${B}을 막아줍니다.
  - 테스트(추론) 단계에서는 모든 뉴런을 사용합니다.
  
  ---
  
  ## 어떻게 동작할까?
  1. 학습 시, 드롭아웃 비율(예: 0.1 → 10%)에 따라 무작위로 뉴런을 비활성화  
  2. 남아 있는 뉴런의 값은 비율에 맞게 ${B}스케일 조정${B} (합이 유지되도록)  
  3. 매 학습 스텝마다 다른 뉴런이 꺼지기 때문에, ${B}여러 네트워크를 앙상블한 효과${B}가 생김  
  
  예시:  
  - 드롭아웃 50% 적용  
  - 원래 뉴런 값: [0.5, 1.2, -0.7, 0.9]  
  - 일부 뉴런 무작위 비활성화 → [0, 1.2, 0, 0.9]  
  - 값 보정 후 → [0, 2.4, 0, 1.8]  
  
  ---
  
  ## 왜 중요한가?
  - ${B}과적합 방지${B}: 학습 데이터에만 특화되지 않고, 새로운 데이터에도 잘 작동하게 만듦  
  - ${B}일반화 성능 향상${B}: 다양한 조합의 뉴런이 학습에 참여 → 더 튼튼한 모델  
  
  ---
  
  ## Transformer에서의 Dropout
  책에서도 설명하듯, GPT 같은 Transformer에서는 드롭아웃을 주로:
  - ${B}어텐션 가중치${B} 계산 후 (Attention Weights)  
  - ${B}Feed Forward 층 출력${B} 후  
  에 적용해 안정성을 높입니다:contentReference[oaicite:0]{index=0}.
  
  ---
  
  ## 우리 프로그램에서는?
  - 사용자가 조절할 수 있는 핵심 파라미터:
    - ${BT}dropoutRate${BT}: 뉴런을 꺼버리는 확률 (0.0 ~ 0.5 범위 권장, 보통 0.1~0.3)
  - 예: ${BT}dropoutRate: 0.1${BT} → 전체 뉴런의 약 10%를 무작위 비활성화  
  - 화면에서 설정 값을 바꿔가며 모델의 ${B}학습 안정성${B}과 ${B}성능 차이${B}를 체험할 수 있습니다.
    `,
  },
  linear: {
    title: 'Linear Output',
    description: String.raw`
  
  ## Linear Layer (선형 변환)이란?
  - 입력 벡터를 가중치 행렬과 곱하고, 편향(bias)을 더해 출력 차원으로 변환하는 가장 기본적인 신경망 레이어입니다.
  - ${B}수식${B}:
  $$
  y = W \cdot x + b
  $$
  - $x$: 입력 벡터  
  - $W$: 가중치 행렬  
  - $b$: 편향  
  - $y$: 출력 벡터  
  
  ---
  
  ## GPT 모델에서의 Linear Output
  - Transformer 블록들을 지난 후, 마지막 단계에서 ${B}내부 벡터(embDim 크기)${B}를 ${B}어휘 집합 크기(vocabSize)${B} 로 변환합니다.
  - 이렇게 나온 벡터는 각 토큰에 대해 "다음 단어가 될 확률 분포"를 만들기 위한 ${B}logits${B}로 사용됩니다 .
  
  ---
  
  ## 왜 중요한가?
  - ${B}출력 공간 연결${B}: 모델 내부 표현을 사람이 읽을 수 있는 단어(토큰) 공간으로 변환.  
  - ${B}확률 예측${B}: Softmax를 통해 각 단어의 확률을 구해 텍스트를 생성.  
  - ${B}학습 가능${B}: W, b는 학습 과정에서 지속적으로 업데이트되어 모델 성능에 직접적으로 기여.  
  
  ---
  
  ## 우리 프로그램에서는?
  - inDim: 입력 차원 (예: 768, 토큰 임베딩 크기)  
  - outDim: 출력 차원 (보통 vocabSize, 예: 50257)  
  - bias: 편향 사용 여부  
  - ${B}Linear Output Layer = "토큰 의미 → 단어 확률" 변환기${B} 역할을 합니다.
    `,
  },
  mhAttention: {
    description: String.raw`![multi head attention diagram](/img/multihead_attention.png)
  
  ## 멀티헤드 어텐션(Multi-Head Attention, MHA)란?
  - ${B}Self-Attention${B}은 단어들 사이의 관계를 찾는 메커니즘입니다.  
  - 하지만 하나의 "관점(헤드)"만 있으면 모든 패턴을 다 잡기 어렵습니다.  
  - MHA는 ${B}여러 개의 어텐션 헤드${B}를 두어 다양한 관점에서 문맥을 해석하고, 이를 합칩니다.
  
  ---
  
  ## 어떻게 동작할까?
  1. 입력 토큰 임베딩 $X$를 세 가지로 선형 변환  
     - Query $Q = XW_Q$, Key $K = XW_K$, Value $V = XW_V$  
  2. $Q, K, V$를 ${B}numHeads${B} 개로 쪼개어 병렬로 어텐션 수행  
  3. 각 헤드별 어텐션 출력 $Z_1, Z_2, …, Z_h$를 얻음  
  4. 이들을 ${B}연결(concatenate)${B} 후, 최종 선형 변환 $W_O$를 적용  
  
  수식 (개념):  
  $$
  \text{MHA}(X) = \text{Concat}(head_1, ..., head_h) W_O
  $$  
  여기서 $head_i = \text{Attention}(Q_i, K_i, V_i)$
  
  ---
  
  ## 왜 중요한가?
  - ${B}다양한 관점 확보${B}: 어떤 헤드는 문법 관계, 다른 헤드는 의미 관계를 잘 포착  
  - ${B}복잡한 문맥 이해${B}: 긴 문장에서 멀리 떨어진 단어 간 연결을 동시에 학습 가능  
  - ${B}Transformer 성능의 핵심${B}: 언어모델이 다양한 패턴을 이해하는 비결
  
  ---
  
  ## 비유로 이해하기
  - 한 사람이 글을 읽으면 한 가지 해석만 가능할 수 있습니다.  
  - 여러 사람이 동시에 각자 관점에서 읽고 의견을 합치는 것 = ${B}멀티헤드 어텐션${B}.
  
  ---
  
  ## 우리 프로그램에서는?
  - 사용자가 설정할 수 있는 ${B}핵심 파라미터${B}:
    - ${BT}numHeads${BT}: 병렬로 사용할 어텐션 헤드 개수  
    - ${BT}embDim${BT}: 임베딩 차원 (전체 벡터 크기)  
    - ${BT}qkvBias${BT}: Query/Key/Value 변환 시 바이어스 사용 여부  
    - ${BT}dropout${BT}: 어텐션 가중치에 적용할 드롭아웃 비율  
  - 헤드 수와 차원 분할 방식을 조정하면서, 모델이 문맥을 더 다양하게 해석하는 효과를 직접 확인할 수 있습니다.
    `,
  },
  transformerBlock: {
    description: String.raw`![transformer block diagram](/img/transformerblock.png)
  
  ## Transformer Block이란?
  - Transformer 아키텍처의 핵심 구성 단위입니다.  
  - 한 블록은 크게 ${B}멀티헤드 어텐션(MHA)${B} 과 ${B}포지션별 피드포워드 네트워크(FFN)${B} 로 이루어져 있습니다.  
  - 각 부분은 ${B}잔차 연결(Residual Connection)${B} 과 ${B}LayerNorm/RMSNorm${B} 으로 감싸져 있어, 깊은 네트워크에서도 안정적으로 학습할 수 있습니다.  

  ---
  
  ## 블록 내부 구조
  1. ${B}입력${B}: 임베딩 + 위치 정보가 합쳐진 토큰 벡터 $X$  
  2. ${B}(정규화 + 잔차) → MHA${B}: 입력을 정규화 후, 멀티헤드 어텐션을 적용 → 출력 $Z$  
  3. ${B}(정규화 + 잔차) → FFN${B}: $Z$를 다시 정규화 후, 피드포워드 네트워크 적용 → 최종 출력 $Y$  
  4. 이렇게 얻은 $Y$는 다음 Transformer Block의 입력으로 전달됩니다.  

  ---
  
  ## 공식 (개념)
  $$
  Z = X + \text{MHA}(\text{Norm}(X))  
  Y = Z + \text{FFN}(\text{Norm}(Z))
  $$

  ---
  
  ## 왜 중요한가?
  - ${B}문맥 이해${B}: MHA가 단어 간 관계를 학습.  
  - ${B}표현력 강화${B}: FFN이 각 단어 표현을 더 풍부하게 다듬음.  
  - ${B}안정성${B}: 정규화와 잔차 연결이 기울기 소실/폭발을 방지하고 깊은 학습 가능.  
  - ${B}반복 구조${B}: Transformer 모델은 수십~수백 개의 블록을 쌓아 올려 강력한 표현력을 얻음.  

  ---
  
  ## 비유로 이해하기
  - ${B}MHA${B}: 교실에서 여러 학생이 동시에 의견을 내며 토론하는 과정.  
  - ${B}FFN${B}: 각 학생이 토론 결과를 자기 생각에 맞게 정리·발전시키는 과정.  
  - ${B}잔차 + 정규화${B}: 토론이 산으로 가지 않도록 정리해주는 선생님 역할.  

  ---
  
  ## 우리 프로그램에서는?
  - Transformer Block은 기본 단위로 여러 개가 반복됩니다.  
  - 사용자가 조절할 수 있는 ${B}핵심 파라미터${B}:
    - ${BT}embDim${BT}: 각 벡터의 크기  
    - ${BT}numHeads${BT}: 어텐션 헤드 개수  
    - ${BT}numOfBlocks${BT}: 블록을 몇 층 쌓을지  
    - ${BT}dropoutRate${BT}: 안정성을 위한 드롭아웃 비율  
    - ${BT}normalization${BT}: LayerNorm / RMSNorm 선택  
  - 이 블록들을 조립해 GPT, BERT 같은 모델을 직접 설계·실험할 수 있습니다.
  `,
  },
  dynamicBlock: {
    description: String.raw`![dynamic block diagram](/img/dynamic_block.png)
  
  ## Dynamic Block이란?
  - ${B}사용자가 직접 구성하는 블록 단위${B}입니다.  
  - 기존 Transformer Block(MHA + FFN + Norm + Residual)처럼 고정된 구조가 아니라,  
    ${B}Attention / FFN / Normalization / Dropout 등 다양한 컴포넌트${B}를 원하는 순서대로 조합할 수 있습니다.  
  - 즉, 실험적으로 ${B}새로운 아키텍처 변형을 시도${B}할 수 있게 하는 유연한 단위입니다.  

  ---

  ## 어떻게 동작할까?
  1. 사용자가 블록 안에 들어갈 레이어들을 선택 (예: Attention → FFN → Norm).  
  2. 선택된 레이어들이 하나의 ${B}연속된 서브 네트워크${B}로 묶임.  
  3. 이 Dynamic Block은 전체 모델 그래프의 일부로 포함되어 학습과 추론에 사용됨.  

  수식 (개념):
  $$
  Y = f_n(\;f_{n-1}(\;...\;f_1(X)\;)\;)
  $$
  여기서 $f_i$는 사용자가 고른 서브 레이어(예: Attention, FFN, Norm 등).

  ---

  ## 왜 중요한가?
  - ${B}유연한 실험${B}: 기존 Transformer Block을 변형해 새로운 구조를 쉽게 시도 가능.  
  - ${B}모듈화 학습${B}: 블록 단위로 성능을 비교하거나 조합을 바꿔볼 수 있음.  
  - ${B}교육 목적${B}: SLM 플랫폼에서 “블록 쌓기”를 통해 Transformer 내부 작동 원리를 직관적으로 이해 가능.  

  ---

  ## 비유로 이해하기
  - Transformer Block이 “레고 기본 블록”이라면,  
  - Dynamic Block은 “레고 커스텀 조립” 같은 개념입니다.  
  - 필요한 조각들을 마음대로 골라서 자신만의 블록을 만드는 과정이죠.  

  ---

  ## 우리 프로그램에서는?
  - Dynamic Block은 ${B}실험 모드${B}에서 특히 유용합니다.  
  - 사용자가 조절할 수 있는 핵심 요소:
    - ${BT}layers[]${BT}: 어떤 레이어들을 포함할지 (MHA, FFN, Norm, Dropout 등).  
    - ${BT}order${BT}: 레이어 적용 순서.  
    - ${BT}params${BT}: 각 서브 레이어별 하이퍼파라미터.  
  - 이 블록을 활용해 기존 Transformer Block과 ${B}성능 비교${B}를 직접 체험할 수 있습니다.
  `,
  },
  residual: {
    description: String.raw`![residual connection diagram](/img/residual.png)
  
  ## Residual Connection (잔차 연결)이란?
  - 신경망이 깊어질수록 ${B}기울기 소실/폭발${B} 문제가 발생할 수 있습니다.  
  - 이를 해결하기 위해, 입력 $x$를 변환 결과 $F(x)$에 그대로 더해주는 구조를 ${B}Residual Connection${B}이라고 합니다.  
  - 즉, 모델이 “변환된 값”뿐 아니라 “원래 입력”도 같이 전달할 수 있게 합니다.  

  ---

  ## 공식 (개념)
  $$
  y = x + F(x)
  $$
  - $x$: 블록 입력  
  - $F(x)$: 블록 내부 연산 (예: Attention, FFN 등)  
  - $y$: 출력 (입력 + 변환 결과)  

  ---

  ## 왜 중요한가?
  - ${B}기울기 소실 방지${B}: 역전파 시 원래 입력 경로가 열려 있어 안정적 학습 가능.  
  - ${B}빠른 수렴${B}: 깊은 네트워크도 쉽게 학습.  
  - ${B}성능 향상${B}: 단순히 층을 쌓는 것보다 훨씬 좋은 결과.  
  - ${B}Transformer, ResNet 공통 핵심 기술${B}.  

  ---

  ## 비유로 이해하기
  - 팀 프로젝트에서 ${B}원래 보고서(x)${B} 가 있고, 팀원이 내용을 수정/추가해서 ${B}개선된 보고서(F(x))${B}를 만듭니다.  
  - 최종 결과물은 “원본 + 수정본”이 합쳐진 형태($x + F(x)$) → 원래 맥락을 잃지 않으면서 발전된 결과를 얻는 것과 같습니다.  

  ---

  ## Transformer에서의 Residual
  - 각 서브 레이어(MHA, FFN) 앞뒤에 Residual Connection이 붙습니다.  
  - 구조:
    $$
    z = x + \text{MHA}(x) \\
    y = z + \text{FFN}(z)
    $$

  - 따라서 정보 손실 없이 안정적으로 여러 블록을 깊게 쌓을 수 있습니다.  

  ---

  ## 우리 프로그램에서는?
  - Residual은 ${B}별도 레이어라기보다는 연결 방식${B}입니다.  
  - 하지만 시각적으로 표시하여, 사용자가 ${B}“원래 입력 + 변환 결과”${B} 구조를 직관적으로 확인할 수 있습니다.  
  - 핵심 포인트:
    - 거의 모든 Transformer Block에서 기본 제공.  
    - 사용자 설정 파라미터는 없음 (내부적으로 자동 적용).  
  - 따라서 Residual은 ${B}안정성과 성능의 숨은 조력자${B} 역할을 합니다.
  `,
  },
  testBlock: {
    description: String.raw`![test block diagram]
  
  ## Test Block이란?
  - ${B}노드 테스트 및 디버깅${B}을 위한 특수한 블록입니다.  
  - 실제 학습 목적보다는, 다양한 노드 타입(Embedding, Attention, FFN 등)을 ${B}실험적으로 조합${B}하고 ${B}동작을 확인${B}하기 위해 사용됩니다.  
  - 즉, “연습장” 역할을 하는 블록으로 이해할 수 있습니다.  

  ---

  ## 언제 사용할까?
  - 새로 만든 노드 타입이 잘 동작하는지 확인할 때.  
  - 여러 블록을 조합했을 때 연결(입출력 차원, 파라미터 등)이 올바른지 검증할 때.  
  - UI에서 ${B}드래그 앤 드롭 / 시각화 / 파라미터 입력${B}이 정상적으로 처리되는지 확인할 때.  

  ---

  ## 특징
  - 내부 구조는 고정되어 있지 않고, ${B}사용자가 원하는 레이어${B}를 자유롭게 삽입 가능.  
  - 주로 다음 요소 테스트에 활용:
    - TokenEmbedding / PositionalEmbedding  
    - Multi-Head Attention  
    - Feed Forward  
    - Normalization / Dropout / Residual  
  - 출력 결과는 학습 성능보다는 ${B}연결 안정성${B}과 ${B}시각화 검증${B}이 목적.  

  ---

  ## 비유로 이해하기
  - Transformer Block이 ${B}완성된 레고 세트${B}라면,  
  - Test Block은 ${B}여러 부품을 맞춰보는 조립 테스트 공간${B}입니다.  
  - 완성품이 아니라, 실험 중간에 구조를 확인하는 데 초점을 둡니다.  

  ---

  ## 우리 프로그램에서는?
  - Test Block은 ${B}부모 노드${B} 역할을 합니다.  
  - 사용자가 다양한 자식 노드(Embedding, Attention, FFN 등)를 붙여서 ${B}노드 간 연결 방식${B}을 시험할 수 있습니다.  
  - 주로 개발/디버깅 단계에서 활용되며,  
    최종 모델 학습 단계에서는 Transformer Block이나 Dynamic Block으로 대체됩니다.
  `,
  },
  gqAttention: {
    description: String.raw`![grouped query attention diagram](/img/gqa.png)
  
  ## Grouped Query Attention (GQA)란?
  - ${B}멀티헤드 어텐션(MHA)${B}의 변형 기법 중 하나입니다.  
  - 일반 MHA에서는 Query, Key, Value의 헤드 수가 동일합니다.  
  - 하지만 GQA에서는 ${B}Query 헤드 수를 크게 하고, Key/Value 헤드는 더 적게 공유${B}합니다.  
  - 이렇게 하면 ${B}계산량과 메모리 사용량을 줄이면서도 성능은 유지${B}할 수 있습니다.  

  ---

  ## 동작 방식
  1. 입력 벡터 $X$를 $Q, K, V$로 변환.  
  2. $Q$는 ${B}많은 수의 헤드${B}로 분할, $K, V$는 ${B}더 적은 그룹${B}으로 분할.  
  3. 여러 Query 헤드가 하나의 Key/Value 그룹을 공유하여 어텐션 계산.  
  4. 최종적으로 모든 Query 결과를 합쳐 출력.  

  수식 (개념):  
  $$
  \text{GQA}(X) = \text{Concat}(\text{Attention}(Q_1, K_1, V_1), \dots, \text{Attention}(Q_h, K_g, V_g)) W_O
  $$
  여기서 $g < h$ (Key/Value 그룹 수 < Query 헤드 수).  

  ---

  ## 왜 중요한가?
  - ${B}효율성${B}: Key/Value 저장 및 계산량을 줄임 → 긴 문맥 처리에 유리.  
  - ${B}성능 유지${B}: Query는 충분히 세분화되어 다양한 관점을 확보.  
  - ${B}최신 모델 적용${B}: LLaMA-2, LLaMA-3 등 최신 SLM/LLM에서 기본적으로 사용.  

  ---

  ## 비유로 이해하기
  - 일반 MHA: ${B}교실에 학생(Q), 참고서(K), 정답지(V)가 모두 1:1로 매칭${B}.  
  - GQA: 학생은 많지만 참고서/정답지는 그룹별로 공유.  
    → 학생 수를 늘려 다양한 질문을 던지되, 참고서는 그룹별로 나눠 보는 구조.  

  ---

  ## 우리 프로그램에서는?
  - GQA는 ${B}MHA의 확장 노드${B}로 선택할 수 있습니다.  
  - 사용자가 조절할 수 있는 핵심 파라미터:
    - ${BT}numHeads${BT}: Query 헤드 개수  
    - ${BT}numKvGroups${BT}: Key/Value 그룹 개수  
    - ${BT}embDim${BT}: 임베딩 차원  
    - ${BT}dropout${BT}: 어텐션 드롭아웃 비율  
  - ${BT}numKvGroups < numHeads${BT} 조건을 지켜야 합니다.  
  - 실험을 통해 MHA 대비 ${B}메모리/속도 효율${B} 차이를 확인할 수 있습니다.
  `,
  },
};
