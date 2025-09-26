// nodeAttributeInformation.ts
const BT = '`';
const B = '**';

export const nodeAttributeInformation = {
  positionalEmbedding: {
    ctxLength: {
      title: 'Context Length',
      description: String.raw`
  모델이 한 번에 처리할 수 있는 ${B}최대 토큰 길이${B}를 의미합니다.  
  입력이 이 길이를 초과하면 잘리거나 ${B}슬라이딩 윈도우(stride)${B} 방식으로 나누어 처리해야 합니다.  
  ## 왜 중요한가?
  - 컨텍스트 길이가 짧으면 긴 문장을 끝까지 이해하지 못함.  
  - 길이가 길수록 더 많은 문맥 정보를 활용할 수 있음.  
  - 하지만 길이가 늘어나면 메모리 사용량과 연산량이 ${B}선형적으로 증가${B}함.  
  ## 대표 모델의 Context Length 비교
  | 모델명      | Context Length | 비고 |
  |-------------|----------------|------|
  | GPT-2       | 1,024 tokens   | 초기 Transformer 기반 LLM |
  | GPT-3       | 2,048 tokens   | 두 배 확장 |
  | GPT-4-turbo | 128,000 tokens | 긴 문맥 처리 강화 |
  | LLaMA-2     | 4,096 tokens   | Meta 공개 모델 |
  | LLaMA-3     | 8,192 tokens   | 최신 공개 사양 |
  
  ## 수식으로 이해하기
  $$
  \text{Input} = [t_1, t_2, \dots, t_{N}] \quad (N \leq \text{ctxLength})
  $$
  
  - $N$: 입력 토큰 개수  
  - $ctxLength$: 모델이 지원하는 최대 토큰 수  
  
  따라서:  
  - $N \leq ctxLength$: 정상 처리  
  - $N > ctxLength$: 초과분은 잘리거나 분할 처리 필요
        `,
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description: `
  토큰을 표현하는 벡터의 ${B}차원 수${B}를 의미합니다.  
  이 값은 모델의 ${B}표현력${B}과 ${B}계산 복잡도${B} 모두에 직접적으로 영향을 줍니다.  
  
  ## 왜 중요한가?
  - 차원이 클수록 더 풍부한 의미와 패턴을 담을 수 있음.
  - 하지만 차원이 커질수록 파라미터 수와 연산량이 증가 → 학습/추론 속도 저하.
  - 너무 작으면 복잡한 언어 패턴을 제대로 표현하지 못함.

  
  ## 대표 모델의 Embedding Dimension 크기
  
  | 모델명      | Embedding Dimension (embDim) | 비고 |
  |-------------|-------------------------------|------|
  | GPT-2 Small | 768                           | 약 1.2억 파라미터 |
  | GPT-2 Medium| 1,024                         | 약 3.5억 파라미터 |
  | GPT-3       | 12,288                        | 약 1,750억 파라미터 |
  | LLaMA-2 7B  | 4,096                         | 70억 파라미터 |
  | LLaMA-2 13B | 5,120                         | 130억 파라미터 |
  
    `,
    },

    posType: {
      title: 'Positional Embedding Type',
      description: String.raw`
  토큰 임베딩에는 ${B}순서 정보가 포함되지 않기 때문에${B}, 위치 정보를 별도로 추가해야 합니다.  
  이때 사용하는 방법이 ${B}Positional Embedding${B}이며, 대표적으로 절대, 상대, RoPE 방식이 있습니다.  
  
  ---
  
  ## 방식별 비교
  
  | 방식              | 개념 | 특징 | 사용 예시 |
  |-------------------|------|------|-----------|
  | 절대 위치 (Absolute) | 각 위치 $i$에 고유한 벡터 $p_i$를 부여 | 단순하고 직관적, 하지만 ${B}길이가 고정됨${B} | GPT-2 |
  | 상대 위치 (Relative) | 두 단어 간 거리 $|i-j|$를 반영 | 긴 문맥 처리에 유리, 일반화 성능 향상 | Transformer-XL, T5 |
  | RoPE (Rotary)       | 각 차원에 회전 변환을 적용해 위치 인코딩 | 효율적이며 긴 컨텍스트 일반화에 강함 | LLaMA, GPT-NeoX |
  
  ---
  
  ## 수식으로 이해하기
  
  1. ${B}절대 위치 임베딩${B}  
  $$
  z_i = x_i + p_i
  $$  
  - $x_i$: i번째 토큰 임베딩  
  - $p_i$: i번째 위치 벡터  
  
  2. ${B}상대 위치 임베딩${B}  
  $$
  \text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T + R}{\sqrt{d_k}}\right)V
  $$  
  - $R$: 위치 차이에 따른 보정 행렬  
  
  3. ${B}RoPE (Rotary Position Embedding)${B}  
  $$
  \text{RoPE}(x, i) = 
  \begin{bmatrix}
  x_{2k} \cos(i \theta_k) - x_{2k+1} \sin(i \theta_k) \\
  x_{2k} \sin(i \theta_k) + x_{2k+1} \cos(i \theta_k)
  \end{bmatrix}
  $$  
  
  - $\theta_k$: 주파수 기반 상수  
  - 벡터를 회전 변환하여 위치 정보를 인코딩  
  
  ---
  
  ## 정리
  - 짧은 문장 처리 → ${B}절대 위치${B}로 충분  
  - 긴 문맥 일반화 필요 → ${B}상대 위치${B} 또는 ${B}RoPE${B} 권장  
  - 최신 모델(예: LLaMA, GPT-4 계열)은 주로 ${B}RoPE${B} 사용
    `,
    },
  },
  tokenEmbedding: {
    vocabSize: {
      title: 'Vocabulary Size',
      description: String.raw`
  모델이 다룰 수 있는 ${B}고유 토큰(token)의 개수${B}를 의미합니다.  
  토큰은 단어, subword, 혹은 문자 단위로 정의되며, ${B}토큰화 방식${B}에 따라 크기가 달라집니다.  

  ---

  ## 왜 중요한가?
  - Vocabulary Size가 클수록 더 많은 단어/표현을 직접 다룰 수 있음.  
  - 하지만 크기가 커질수록 ${B}임베딩 행렬 크기${B}와 ${B}출력 Softmax 계산량${B}이 증가.  
  - 지나치게 작으면 희귀 단어를 처리하기 어렵고, 지나치게 크면 학습/추론 비용이 커짐.  

  ---

  ## 대표 모델의 Vocabulary Size

  | 모델명       | Vocabulary Size | 토크나이저 방식 |
  |--------------|-----------------|-----------------|
  | GPT-2        | 50,257          | BPE (Byte Pair Encoding) |
  | BERT-base    | 30,522          | WordPiece       |
  | LLaMA-2      | 32,000          | SentencePiece   |
  | GPT-4 계열   | ~100,000+       | cl100k_base (BPE 변형) |

  ---

  ## 수식으로 이해하기

  임베딩 행렬:
  $$
  E \in \mathbb{R}^{V \times d}
  $$

  - $V$: Vocabulary Size  
  - $d$: Embedding Dimension  

  토큰 ID $t_i$를 벡터로 변환:
  $$
  x_i = E[t_i] \in \mathbb{R}^{d}
  $$

  출력 단계(Softmax):
  $$
  P(\text{token}=j \mid x) = \frac{\exp(xW_j)}{\sum_{k=1}^{V} \exp(xW_k)}
  $$

  - $W_j$: j번째 토큰에 해당하는 가중치 벡터  
  - 어휘 크기 $V$가 클수록 분모 계산량이 커짐 → 학습/추론 비용 ↑  

  ---

  ## 정리
  - Vocabulary Size는 ${B}토큰 표현력과 연산 효율성의 균형${B}을 맞추는 핵심 파라미터.  
  - 보통 영어 모델은 30K~50K, 최신 다국어/대형 모델은 100K 이상을 사용.
  `,
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description: String.raw`
  토큰을 표현하는 ${B}임베딩 벡터의 차원 수${B}입니다.  
  각 토큰은 $d$ 차원 공간의 벡터로 매핑되며, 이 $d$ 값이 곧 Embedding Dimension 입니다.  
  
  ---
  
  ## 왜 중요한가?
  - ${B}차원이 클수록${B}: 더 많은 의미적/문법적 특징을 담을 수 있어 표현력이 풍부해짐.  
  - ${B}차원이 작을수록${B}: 연산량과 메모리 사용량이 줄어 빠른 학습/추론 가능.  
  - 따라서 모델 설계에서 ${B}표현력 ↔ 효율성${B}의 균형을 맞추는 핵심 요소입니다.  
  
  ---
  
  ## 대표 모델의 Embedding Dimension
  
  | 모델명       | Embedding Dimension (d) | 비고 |
  |--------------|--------------------------|------|
  | GPT-2 Small  | 768                      | 약 1.2억 파라미터 |
  | GPT-2 Medium | 1,024                    | 약 3.5억 파라미터 |
  | GPT-3        | 12,288                   | 초대형 모델, 1,750억 파라미터 |
  | LLaMA-2 7B   | 4,096                    | 중간 규모 |
  | LLaMA-2 13B  | 5,120                    | 대규모 |
  
 
  ---
  
  ## 정리
  - ${B}embDim은 모델 크기와 성능을 직접적으로 결정${B}하는 핵심 값.  
  - 값이 커지면 성능 향상 가능하지만, ${B}계산량과 파라미터 수도 함께 증가${B}.  
  - 보통 중소형 모델은 512~1024, 대형 모델은 4096 이상을 사용.
    `,
    },
  },
  normalization: {
    normType: {
      title: 'Normalization Type',
      description: String.raw`
  신경망 학습을 안정화하고 수렴 속도를 높이기 위해 입력을 정규화하는 방식입니다.  
  대표적으로 ${B}Batch Normalization (BN)${B}, ${B}Layer Normalization (LN)${B}, ${B}RMS Normalization (RMSNorm)${B}이 사용됩니다.  

  ---

  ## 왜 필요한가?
  - 딥러닝에서는 층이 깊어질수록 ${B}분산(variance) 변화${B}와 ${B}기울기 불안정${B} 문제가 발생.  
  - 정규화를 적용하면 층 출력이 일정한 범위를 유지해 ${B}학습 안정성${B}과 ${B}일반화 성능${B}이 개선됨.  

  ---

  ## 방식별 비교

  | 방법         | 정규화 축 (기준)            | 학습/추론 시 통계 | 장점 | 주의사항 |
  |--------------|------------------------------|------------------|------|-----------|
  | BatchNorm    | 채널별, 배치 전체(N, H, W)   | 학습: 배치 통계<br>추론: 러닝 평균/분산 | CNN에서 효과적, 수렴 빠름 | 소배치·가변 길이에 취약 |
  | LayerNorm    | 하나의 샘플 내 특징 전체     | 항상 현재 샘플 통계 | RNN/Transformer에 적합 | 연산량이 BN보다 약간 많음 |
  | RMSNorm      | LN과 동일 축, 평균 제거 X    | 항상 현재 샘플 통계 | 단순/가벼움, 최신 Transformer 다수 사용 | 평균 미제거로 분포 치우침 가능 |

  ---

  ## 공식으로 이해하기

  1. ${B}BatchNorm${B}  
  $$
  \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^{2} + \epsilon}}
  $$

  2. ${B}LayerNorm${B}  
  $$
  \hat{x} = \frac{x - \mu_{\text{sample}}}{\sqrt{\sigma_{\text{sample}}^{2} + \epsilon}}
  $$

  3. ${B}RMSNorm${B}  
  $$
  rms(x) = \sqrt{\operatorname{mean}(x^{2}) + \epsilon}, \quad
  y = \gamma \cdot \frac{x}{rms(x)}
  $$


  ---

  ## 정리
  - ${B}CNN${B} + 충분한 배치 → ${B}BatchNorm${B}  
  - ${B}RNN/Transformer${B} + 소/가변 배치 → ${B}LayerNorm${B}  
  - 단순/속도 우선, 최신 트렌드 → ${B}RMSNorm${B}  
  `,
    },
    eps: {
      title: 'Epsilon',
      description: String.raw`
  수치적 안정성을 위해 ${B}분모에 더해지는 작은 상수${B}입니다.  
  정규화 과정에서 분산(variance)이 0에 가까워지면 분모가 0이 되어 나눗셈이 불가능해지는데,  
  이를 방지하기 위해 $\epsilon$ 값을 더해줍니다.  

  ---

  ## 왜 필요한가?
  - 딥러닝 학습 중 ${B}특정 입력 분산이 0 또는 매우 작아질 수 있음${B}.  
  - 이 경우 나눗셈에서 ${B}0으로 나누기(division by zero) 오류${B} 발생.  
  - $\epsilon$을 더해 안정적으로 계산 가능.  

  ---

  ## 공식에서의 사용 예시

  1. ${B}BatchNorm${B}  
  $$
  \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
  $$

  2. ${B}LayerNorm${B}  
  $$
  \hat{x} = \frac{x - \mu_{sample}}{\sqrt{\sigma_{sample}^2 + \epsilon}}
  $$

  3. ${B}RMSNorm${B}  
  $$
  rms(x) = \sqrt{\text{mean}(x^2) + \epsilon}, \quad
  y = \gamma \cdot \frac{x}{rms(x)}
  $$

  ---

  ## 일반적인 값 범위

  | 적용 모델/라이브러리 | Epsilon 값 |
  |-----------------------|------------|
  | PyTorch BatchNorm     | 1e-5       |
  | TensorFlow LayerNorm  | 1e-12      |
  | GPT-2 / GPT-3         | 1e-5 ~ 1e-6|
  | LLaMA 계열            | 1e-6       |

  ---

  ## 정리
  - $\epsilon$은 학습 파라미터가 아닌 ${B}고정된 상수${B}.  
  - 크기가 너무 크면 정규화 효과가 왜곡되고,  
  - 너무 작으면 여전히 불안정 → 보통 $10^{-5} \sim 10^{-12}$ 사이 값을 사용.
  `,
    },
    inDim: {
      title: 'Input Dimension',
      description: String.raw`
  정규화가 적용되는 ${B}입력 벡터(텐서)의 차원 크기${B}를 의미합니다.  
  일반적으로 이는 ${B}이전 레이어의 출력 차원${B}과 동일해야 합니다.  
  
  ---
  
  ## 왜 중요한가?
  - 정규화는 입력을 특정 차원을 기준으로 평균/분산을 계산해 표준화.  
  - 따라서 올바른 차원 크기를 지정하지 않으면 정규화 대상이 달라져 결과가 잘못됨.  
  - 예: Embedding Dimension이 768인데 inDim을 512로 설정하면 차원 불일치 오류 발생.  
  
  ---
  
  ## 대표 사례
  
  | 레이어 종류          | 출력 차원 (inDim) | 비고 |
  |-----------------------|------------------|------|
  | Token Embedding       | embDim (예: 768) | 토큰 벡터 차원 |
  | Multi-Head Attention  | embDim (예: 768) | 어텐션 출력 차원 |
  | Feed Forward          | embDim (예: 768) | 다시 원래 차원으로 축소 |
  | Transformer Block     | embDim           | 블록 전체적으로 유지 |
  
  ---
  
  ## 수식으로 이해하기
  
  Layer Normalization:
  $$
  \mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad 
  \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
  $$
  
  $$
  \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad
  y_i = \gamma \hat{x_i} + \beta
  $$
  
  - 여기서 $d = inDim$ (입력 벡터의 길이)  
  - 즉, $inDim$이 곧 정규화가 계산되는 ${B}축의 크기${B}  
  
  ---
  
  ## 정리
  - inDim은 ${B}정규화할 대상 벡터의 크기${B}.  
  - 반드시 이전 레이어 출력 차원과 동일해야 함.  
  - 보통 모델 전체에서 ${B}embDim과 동일하게 설정${B}됨.
    `,
    },
  },
  feedForward: {
    actFunc: {
      title: 'Activation Function',
      description: String.raw` 
  Feed Forward Network(FFN)에서 ${B}비선형성(non-linearity)${B}을 추가하는 함수입니다.  
  선형 변환만으로는 표현할 수 없는 복잡한 패턴을 학습하기 위해 사용됩니다.  

  ---

  ## 왜 필요한가?
  - 선형 변환만 있으면 여러 층을 쌓아도 결국 ${B}하나의 선형 변형${B}과 동일.  
  - 활성화 함수를 넣어야만 ${B}비선형 표현${B}이 가능해져 복잡한 언어 패턴을 학습할 수 있음.  

  ---

  ## 대표적인 활성화 함수 비교

  | 함수명 | 수식 | 특징 | Transformer에서 사용 |
  |--------|------|------|----------------------|
  | ReLU   | $f(x) = \max(0, x)$ | 계산 단순, 기울기 소실 방지, 음수 영역 정보 손실 | 초기 Transformer, BERT |
  | GELU   | $f(x) = x \cdot \Phi(x)$ (정규분포 CDF) | 매끄러운 곡선, 작은 음수도 부분적으로 통과 | GPT-2, GPT-3 |
  | SwiGLU | $f(x) = (xW_1) \odot \sigma(xW_2)$ | 게이트 구조, 학습 능력 향상 | PaLM, LLaMA-2 |
  | Tanh   | $f(x) = \tanh(x)$ | [-1, 1] 범위, 고전적 함수 | 거의 사용 안 함 (기울기 소실 문제) |

  ---

  ## 수식 예시

  1. ${B}ReLU${B}
  $$
  f(x) = \max(0, x)
  $$

  2. ${B}GELU${B}
  $$
  f(x) = \frac{1}{2}x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)
  $$

  3. ${B}SwiGLU${B}
  $$
  f(x) = (xW_1) \odot \sigma(xW_2)
  $$

  ---

  ## 정리
  - FFN에서는 ReLU 대신 ${B}GELU${B}가 사실상 표준.  
  - 최신 모델(LLaMA, PaLM 등)은 ${B}SwiGLU${B}로 더 나은 성능 달성.  
  - 어떤 함수를 쓰느냐에 따라 ${B}표현력, 수렴 속도, 성능${B}에 차이가 생김.
  `,
    },
    feedForwardType: {
      title: 'Feed Forward Type',
      description: String.raw`
  Feed Forward Network(FFN)의 ${B}구조적 변형 방식${B}을 의미합니다.  
  Transformer 블록에서 Attention 뒤에 붙는 FFN은 보통 두 가지 주요 형태로 나뉩니다.  

  ---

  ## 1. Standard FFN
  - 가장 기본적인 구조: 선형 → 활성화 → 선형
  - 입력 차원을 확장했다가 다시 축소 (보통 4배 확장 후 원래 크기로 축소)

  ${B}수식${B}:
  $$
  \text{FFN}(x) = W_2 \cdot f(W_1 x + b_1) + b_2
  $$

  - $f$: 활성화 함수 (ReLU, GELU 등)
  - GPT-2, BERT 등 초기 Transformer 모델들이 채택

  ---

  ## 2. Gated FFN (Gated Linear Unit, GLU 변형)
  - 입력을 두 개의 선형 변환으로 분기
  - 하나는 ${B}값(Value)${B}, 다른 하나는 ${B}게이트(Gate, 시그모이드/활성화)${B} 역할
  - 두 출력을 원소별 곱(element-wise product)으로 결합

  ${B}수식${B}:
  $$
  \text{FFN}(x) = (W_1 x) \odot \sigma(W_2 x)
  $$

  - 변형된 형태: ${B}SwiGLU, GeGLU${B} (게이트에 Swish/GELU 적용)
  - PaLM, LLaMA-2, GPT-4 계열 모델에서 널리 사용

  ---

  ## 방식별 비교

  | 유형        | 구조                          | 장점 | 단점 | 사용 예시 |
  |-------------|-------------------------------|------|------|-----------|
  | Standard    | $W_2 f(W_1 x)$                | 단순, 계산 효율 | 표현력이 제한적 | GPT-2, BERT |
  | Gated (GLU) | $(W_1 x) \odot f(W_2 x)$      | 더 강력한 표현력, 성능 ↑ | 파라미터/연산량 증가 | LLaMA, PaLM, GPT-4 |

  ---

  ## 정리
  - ${B}Standard${B}: 단순하고 효율적, 중소형 모델에서 여전히 많이 사용  
  - ${B}Gated (GLU/SwiGLU/GeGLU)${B}: 최신 대규모 모델의 기본 선택, 더 좋은 성능  
  - 연구/실험 단계에서는 두 방식을 비교해 ${B}성능 ↔ 효율성${B}을 직접 확인하는 것이 유용
  `,
    },
    hiddenDim: {
      title: 'Hidden Dimension Size',
      description: String.raw`
  Feed Forward Network(FFN)의 ${B}내부 확장 차원 크기${B}를 의미합니다.  
  입력 차원(embDim)에 비해 몇 배로 확장할지 결정하며,  
  대부분의 Transformer에서는 보통 ${B}4 × embDim${B}으로 설정합니다.  
  
  ---
  
  ## 왜 중요한가?
  - hiddenDim이 클수록 토큰 표현을 더 풍부하게 변환할 수 있어 모델 성능 ↑  
  - 하지만 파라미터 수와 연산량도 선형적으로 증가 → 학습/추론 비용 ↑  
  - 너무 작으면 FFN이 표현할 수 있는 패턴이 제한됨  
  
  ---
  
  ## 구조와 수식
  FFN의 기본 구조:
  $$
  \text{FFN}(x) = W_2 \cdot f(W_1 x + b_1) + b_2
  $$
  
  - $x \in \mathbb{R}^{d}$: 입력 (embDim 차원)  
  - $W_1 \in \mathbb{R}^{d \times h}$: ${B}hiddenDim = h${B}  
  - $W_2 \in \mathbb{R}^{h \times d}$: 다시 embDim으로 축소  
  
  즉:
  - 입력 $d$ → 내부 확장 $h$ → 다시 $d$  
  - 보통 $h = 4d$ (예: embDim=1024 → hiddenDim=4096)
  
  ---
  
  ## 대표 모델의 hiddenDim 설정
  
  | 모델명       | embDim | hiddenDim | 비고 |
  |--------------|--------|-----------|------|
  | GPT-2 Small  | 768    | 3,072     | 약 4× 확장 |
  | BERT-base    | 768    | 3,072     | 4× 확장 |
  | GPT-3        | 12,288 | 49,152    | 4× 확장 |
  | LLaMA-2 7B   | 4,096  | 11,008    | 약 2.7× 확장 (효율성 고려) |
  | PaLM         | 8,192  | 32,768    | 4× 확장 |
  
  ---
  
  ## 정리
  - hiddenDim은 ${B}FFN의 내부 폭(width)${B}을 결정하는 값.  
  - 전통적으로 ${B}4× embDim${B}이 표준이지만,  
  - 최근 모델(LLaMA 계열)은 효율성을 위해 ${B}2.7×~3.5× 비율${B}도 사용.  
  - 성능 ↔ 자원 효율성을 맞추는 핵심 설계 파라미터.
    `,
    },
    bias: {
      title: 'Bias Enabled',
      description: String.raw`
  Feed Forward Network(FFN)나 Linear Layer에서 ${B}편향(bias) 항을 포함할지 여부${B}를 결정합니다.  
  편향은 입력이 모두 0일 때도 출력을 비선형적으로 이동시켜, 모델이 더 유연한 함수를 학습할 수 있게 합니다.  
  
  ---
  
  ## 왜 중요한가?
  - ${B}Bias 포함${B}: 모델이 데이터 분포에 맞게 기준점을 이동시킬 수 있음.  
  - ${B}Bias 제거${B}: 파라미터 수가 줄고, 계산량이 약간 감소.  
  - 큰 규모의 Transformer에서는 Residual + Normalization 구조 덕분에 bias가 꼭 필요하지 않을 수 있음.  
  
  ---
  
  ## 수식으로 이해하기
  
  1. ${B}Bias 있는 경우${B}
  $$
  y = W x + b
  $$
  
  2. ${B}Bias 없는 경우${B}
  $$
  y = W x
  $$
  
  - $W$: 가중치 행렬  
  - $b$: 편향 벡터 (bias)  
  
  ---
  
  ## 모델별 적용 사례
  
  | 모델명       | Linear/FFN Bias | 비고 |
  |--------------|-----------------|------|
  | BERT         | 사용 (bias 포함) | 전통적 Transformer |
  | GPT-2        | 대부분 bias 포함 | 표준 구조 |
  | GPT-3        | 일부 레이어 bias 제거 | 효율화 목적 |
  | LLaMA-2/3    | bias 제거 (대부분) | 대규모 모델, 불필요한 파라미터 축소 |
  
  ---
  
  ## 정리
  - 작은/중간 규모 모델 → bias를 두는 것이 안정적.  
  - 대규모 모델 → bias 항을 제거해도 성능 차이가 거의 없으며, 파라미터 효율 개선 가능.  
  - 따라서 ${B}모델 크기와 아키텍처에 따라 선택적으로 적용${B}.
    `,
    },
  },
  dropout: {
    dropoutRate: {
      title: 'Dropout Rate',
      description: String.raw`
  신경망 학습 시 일부 뉴런(노드)을 ${B}무작위로 비활성화(drop)${B} 하는 비율을 의미합니다.  
  과적합(overfitting)을 방지하고, 다양한 서브 네트워크를 학습하는 효과를 냅니다.  

  ---

  ## 왜 중요한가?
  - ${B}과적합 방지${B}: 특정 뉴런이나 연결에 의존하지 않도록 만듦.  
  - ${B}일반화 성능 향상${B}: 여러 경로를 학습해 새로운 데이터에도 강건함.  
  - ${B}앙상블 효과${B}: 매 학습 스텝마다 다른 뉴런이 꺼져 여러 모델을 학습하는 효과.  

  ---

  ## 수식으로 이해하기

  학습 시:
  $$
  h_i' = \frac{m_i \cdot h_i}{p}
  $$  

  - $h_i$: 원래 뉴런 출력  
  - $m_i \sim \text{Bernoulli}(p)$: 확률 $p=1-\text{dropoutRate}$ 로 활성 유지  
  - $p$: 뉴런이 살아남을 확률  

  추론 시:
  $$
  h_i' = h_i
  $$  
  (즉, 모든 뉴런을 사용하며 스케일링 없음)

  ---

  ## 대표적인 Dropout Rate 값

  | 영역                  | 일반적인 값 | 비고 |
  |-----------------------|-------------|------|
  | Fully Connected Layer | 0.2 ~ 0.5   | 작은 네트워크에서 자주 사용 |
  | Transformer Attention | 0.1         | 안정성을 위해 가볍게 적용 |
  | Large-scale LLM       | 0.0 ~ 0.1   | 데이터가 방대해 과적합 위험 적음 |

  ---

  ## 예시
  - Dropout Rate = 0.1 → 전체 뉴런의 약 10%를 랜덤 비활성화  
  - Dropout Rate = 0.5 → 절반을 비활성화 (작은 네트워크에서 과적합 방지용으로 흔히 사용)

  ---

  ## 정리
  - Dropout Rate는 ${B}0 ~ 1 사이 값${B}.  
  - 값이 너무 크면 학습이 어려워지고, 너무 작으면 과적합 방지 효과가 약함.  
  - 일반적으로 0.1~0.3 범위를 가장 많이 사용.
  `,
    },
  },
  linear: {
    outDim: {
      title: 'Output Dimension',
      description: String.raw`
  Linear Layer(선형 변환)의 ${B}출력 벡터 차원 크기${B}를 의미합니다.  
  이 값은 곧 ${B}다음 레이어 입력 차원${B}과 연결되므로 반드시 호환되어야 합니다.  
  특히 마지막 Linear Layer에서는 ${B}Vocabulary Size${B}와 동일하게 설정해 토큰 확률 분포를 산출합니다.  

  ---

  ## 왜 중요한가?
  - 레이어 간 ${B}차원 불일치 오류${B}를 방지하기 위해 맞춰야 함.  
  - 마지막 Linear에서는 ${B}출력 차원 = 어휘 집합 크기(Vocabulary Size)${B} 가 되어야 Softmax 확률 계산 가능.  
  - 중간 Linear에서는 차원을 확장/축소하여 모델 표현력을 조절할 수 있음.  

  ---

  ## 구조와 수식
  $$
  y = W x + b
  $$

  - $x \in \mathbb{R}^{d_{in}}$: 입력 벡터  
  - $W \in \mathbb{R}^{d_{in} \times d_{out}}$: 가중치 행렬  
  - $b \in \mathbb{R}^{d_{out}}$: 편향  
  - $y \in \mathbb{R}^{d_{out}}$: 출력 벡터  

  여기서 $d_{out} =$ ${B}outDim${B}  

  ---

  ## 대표적 설정 예시

  | 위치            | 입력 차원 ($d_{in}$) | 출력 차원 ($d_{out}$ = outDim) | 비고 |
  |-----------------|----------------------|--------------------------------|------|
  | Transformer 중간 | 768                  | 3072                           | FFN 확장 단계 |
  | Transformer 중간 | 3072                 | 768                            | FFN 축소 단계 |
  | 마지막 Linear    | 768                  | 50,257                         | GPT-2 vocab size |
  | 마지막 Linear    | 4096                 | 32,000                         | LLaMA-2 vocab size |

  ---

  ## 정리
  - ${B}중간 Linear${B}: 차원 확장/축소 역할 (표현력 강화, 연산량 조절).  
  - ${B}출력 Linear${B}: outDim = Vocabulary Size → Softmax로 확률 분포 산출.  
  - outDim은 ${B}다음 레이어 입력과 반드시 일치${B}해야 안전하게 모델이 연결됨.
  `,
    },
    weightTying: {
      title: 'Weight Tying',
      description: String.raw`
  입력 임베딩(Embedding Layer)과 출력 Linear Layer의 ${B}가중치 행렬을 공유(tying)${B} 하는 기법입니다.  
  즉, 두 레이어에서 별도의 행렬을 학습하지 않고, 하나의 행렬을 같이 사용합니다.  

  ---

  ## 왜 사용하는가?
  - ${B}파라미터 수 감소${B}: 대규모 모델에서 수억 개 이상의 파라미터 절약.  
  - ${B}일관성 유지${B}: 입력 임베딩과 출력 로짓 공간이 같은 의미 공간을 공유 → 학습 안정성 증가.  
  - ${B}성능 향상${B}: 실험적으로 perplexity 개선 효과 보고됨 (Press & Wolf, 2017).  

  ---

  ## 수식으로 이해하기

  1. ${B}Weight Tying 없는 경우${B}  
  - 입력 임베딩: $x_i = E[t_i], \quad E \in \mathbb{R}^{V \times d}$  
  - 출력 Linear: $y = W x + b, \quad W \in \mathbb{R}^{d \times V}$  
  - $E$와 $W$는 서로 독립적으로 학습됨.

  2. ${B}Weight Tying 적용 시${B}  
  - $W = E^T$ 로 설정 → 같은 행렬 공유  
  - 출력 단계:
  $$
  \text{logits} = E x
  $$  

  ---

  ## 대표 사례

  | 모델명       | Weight Tying 사용 여부 | 비고 |
  |--------------|------------------------|------|
  | RNN LM (2017)| ✅ 사용                 | 초기 연구에서 효과 입증 |
  | GPT-2        | ✅ 사용                 | 입력/출력 임베딩 공유 |
  | BERT         | ❌ 사용 안 함           | MLM(Masked LM) 특성 |
  | LLaMA-2      | ✅ 사용                 | 메모리/효율성 개선 목적 |

  ---

  ## 장단점

  - ${B}장점${B}  
  - 파라미터 절약 (Embedding + Linear 중복 제거)  
  - 성능 개선 (학습 안정성 ↑)  

  - ${B}단점${B}  
  - 입력/출력 표현을 완전히 동일 공간에 묶어야 해서 유연성이 줄어듦  

  ---

  ## 정리
  - Weight Tying은 ${B}Embedding Layer와 Output Layer를 같은 행렬로 공유하는 방법${B}.  
  - 메모리 절약 + 성능 향상을 동시에 노릴 수 있어, ${B}GPT 계열 등 대부분의 LLM에서 기본적으로 사용${B}됨.
  `,
    },
    bias: {
      title: 'Bias Enabled',
      description: String.raw`
  Linear Layer(선형 변환)에서 ${B}편향(bias) 항을 포함할지 여부${B}를 결정합니다.  
  편향은 출력 값을 일정 방향으로 평행 이동시켜, 데이터 분포에 맞는 ${B}유연한 표현${B}을 가능하게 합니다.  
  
  ---
  
  ## 왜 필요한가?
  - ${B}Bias 포함${B}: 입력이 모두 0이어도, 출력이 0이 아닌 특정 값으로 이동 가능 → 모델 표현력 증가.  
  - ${B}Bias 제거${B}: 파라미터 수가 줄어 메모리/계산 효율 ↑. 대규모 Transformer에서는 Residual + Normalization 구조 덕분에 bias가 거의 필요 없음.  
  
  ---
  
  ## 수식으로 이해하기
  
  1. ${B}Bias 있는 경우${B}
  $$
  y = W x + b
  $$
  
  2. ${B}Bias 없는 경우${B}
  $$
  y = W x
  $$
  
  - $W$: 가중치 행렬  
  - $b$: 편향 벡터 (bias)  
  - $x$: 입력 벡터  
  - $y$: 출력 벡터  
  
  ---
  
  ## 모델별 적용 사례
  
  | 모델명       | Bias 사용 | 비고 |
  |--------------|-----------|------|
  | BERT-base    | ✅ 사용   | 전통적 구조 |
  | GPT-2        | ✅ 사용   | 모든 Linear에 bias 포함 |
  | GPT-3        | ⚠️ 일부 제거 | 대규모 모델 효율화 목적 |
  | LLaMA-2/3    | ❌ 미사용 | 대부분의 Linear에서 bias 제거 |
  
  ---
  
  ## 장단점
  
  - ${B}장점 (포함 시)${B}  
    - 출력 분포를 더 자유롭게 조정 가능  
    - 작은/중간 규모 모델에서 학습 안정성 ↑  
  
  - ${B}단점 (포함 시)${B}  
    - 대규모 모델에서는 효과 미미  
    - 파라미터 수 증가 (Embedding/Linear 크기가 수십억일 경우 영향 큼)  
  
  ---
  
  ## 정리
  - ${B}소규모/중규모 모델${B}: bias를 사용하는 것이 일반적.  
  - ${B}대규모 LLM${B}: 대부분 bias를 제거해도 성능 손실이 거의 없어 효율을 위해 제외.  
  - 따라서 ${BT}bias${BT}는 ${B}모델 크기와 목적${B}에 따라 선택적으로 적용.
    `,
    },
  },
  mhAttention: {
    numHeads: {
      title: 'Number of Heads',
      description: String.raw`
  Multi-Head Attention에서 ${B}병렬로 분리해 학습하는 어텐션 헤드의 개수${B}를 의미합니다.  
  각 헤드는 서로 다른 서브 공간(subspace)에서 Query-Key-Value 연산을 수행하여, 입력의 다양한 관계를 학습합니다.  
  
  ---
  
  ## 왜 여러 개를 쓰는가?
  - ${B}한 개의 어텐션(단일 헤드)${B} → 특정 패턴(예: 문맥적 유사성)만 학습  
  - ${B}여러 개의 어텐션(멀티 헤드)${B} → 병렬적으로 서로 다른 관점(구문, 의미, 위치 관계 등)을 학습  
  - 다양한 표현을 종합해 더 풍부한 문맥 정보를 추출 가능  
  
  ---
  
  ## 수식으로 이해하기
  
  1. 입력 임베딩 차원: $d_{model}$  
  2. 헤드 개수: $h$  
  3. 각 헤드 차원: $d_k = d_{model} / h$
  
  각 헤드에서:
  $$
  \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
  $$
  
  최종 출력:
  $$
  \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
  $$
  
  ---
  
  ## 모델별 Head 개수 예시
  
  | 모델명       | $d_{model}$ | numHeads | head 차원 $d_k$ |
  |--------------|-------------|----------|-----------------|
  | Transformer (Vaswani, 2017) | 512         | 8        | 64              |
  | BERT-base    | 768         | 12       | 64              |
  | GPT-2 small  | 768         | 12       | 64              |
  | GPT-3 175B   | 12,288      | 96       | 128             |
  | LLaMA-2 7B   | 4,096       | 32       | 128             |
  
  ---
  
  ## 정리
  - numHeads는 ${B}병렬 어텐션의 수${B}를 결정하는 값.  
  - 각 헤드 차원은 $d_{model} / h$로 자동 결정 → $d_{model}$은 numHeads로 나누어떨어져야 함.  
  - 헤드 수를 늘리면 표현력은 증가하지만, 연산량과 메모리 사용도 커짐.  
  - 일반적으로 ${B}embDim / headDim ≈ 64${B}가 되도록 설정하는 것이 표준.
    `,
    },

    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: String.raw`
  어텐션 연산에서 ${B}가중치 행렬(Attention Weights)${B}에 적용되는 드롭아웃 비율을 의미합니다.  
  입력 토큰 간 유사도를 확률로 변환한 뒤, 일부 연결을 무작위로 끊어 과적합을 방지합니다.  
  
  ---
  
  ## 왜 필요한가?
  - 어텐션은 모든 토큰 간 관계를 계산하므로 ${B}매우 강력한 표현력${B}을 가짐  
  - 그러나 데이터가 적거나 모델이 크면, 특정 패턴(예: 특정 위치)만 과도하게 학습 → 과적합 위험  
  - Dropout을 적용하면 다양한 연결을 학습하게 되어 ${B}일반화 성능${B}이 향상됨  
  
  ---
  
  ## 수식으로 이해하기
  
  어텐션 가중치:
  $$
  A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
  $$
  
  Dropout 적용 후:
  $$
  A' = \text{Dropout}(A, p)
  $$
  
  최종 출력:
  $$
  \text{Attention}(Q,K,V) = A'V
  $$
  
  여기서 $p = 1 - \text{dropoutRate}$ 는 활성 유지 확률.
  
  ---
  
  ## 대표적인 설정 값
  
  | 모델명       | Attention Dropout | 비고 |
  |--------------|-------------------|------|
  | Transformer (2017) | 0.1 | 논문 기본값 |
  | BERT-base    | 0.1 | HuggingFace 구현 |
  | GPT-2        | 0.1 | OpenAI 기본 |
  | LLaMA-2      | 0.0 | 대규모 데이터셋 덕분에 과적합 위험 낮음 |
  | GPT-3        | 0.0 | 학습 데이터 방대, Dropout 최소화 |
  
  ---
  
  ## 정리
  - Attention Dropout은 ${B}어텐션 가중치 행렬에 적용되는 드롭아웃${B}.  
  - 작은/중간 모델 → 0.1 정도 권장  
  - 대규모 LLM → 0.0 (사실상 필요 없음)  
  - 값이 너무 크면 중요한 연결이 사라져 학습이 불안정해질 수 있음.
    `,
    },

    qkvBias: {
      title: 'QKV Bias',
      description: String.raw`
  쿼리(Query), 키(Key), 값(Value) 벡터를 만들 때 사용하는 ${B}선형 변환(Linear Layer)${B}에  
  편향(bias) 항을 포함할지 여부를 결정합니다.  
  
  - ${B}Bias 포함${B}: 각 변환에 추가 자유도를 주어 작은·중간 규모 모델에서 표현력 향상 가능.  
  - ${B}Bias 제거${B}: 최신 대규모 LLM에서는 성능 차이가 거의 없어, 파라미터 절감을 위해 제거하는 경우가 많음.  
  - 실제로 GPT-3, LLaMA-2/3 같은 모델은 대부분 QKV 변환에서 bias를 사용하지 않습니다.
    `,
    },

    isRoPE: {
      title: 'RoPE Enabled',
      description: String.raw`
  ${B}RoPE(Rotary Positional Embedding)${B} 적용 여부를 결정합니다.  
  
  - RoPE는 기존의 절대/상대 위치 임베딩 대신, ${B}위치 정보를 벡터의 각 차원에 회전(rotary) 방식으로 인코딩${B}합니다.  
  - 장점: 긴 문맥 처리에 강하고, 어텐션 점수 계산에서 자연스럽게 상대적 위치 정보를 반영할 수 있음.  
  - 최신 모델들(LLaMA-2/3, GPT-NeoX 등)에서 기본적으로 사용되는 방식.  
  
  ${B}사용 시 효과${B}  
  - 문맥 길이가 길어질수록 일반 절대 임베딩보다 안정적인 성능.  
  - 상대적 순서 학습이 가능해 다양한 문장 구조 처리에 유리.  
  
  ${B}비사용 시${B}  
  - 전통적인 절대/상대 포지셔널 임베딩을 적용.  
    `,
    },

    ropeBase: {
      title: 'Rope Base',
      description: String.raw`
  ${B}RoPE(Rotary Positional Embedding)${B}에서 사용하는 ${B}기본 각도(θ, theta) 값${B}을 정의합니다.  
  
  RoPE는 위치 정보를 각 차원별로 주기적인 회전 값으로 인코딩하는데,  
  이때 사용하는 주기의 기본값이 ${BT}ropeBase${BT}입니다.  
  
  ---
  
  ## 개념적으로
  - 작은 ${BT}ropeBase${BT}: 더 짧은 주기로 회전 → 짧은 문맥에 민감.  
  - 큰 ${BT}ropeBase${BT}: 더 긴 주기로 회전 → 긴 문맥 정보 보존에 유리.  
  
  ---
  
  ## 일반적인 설정 예시
  
  | 모델       | ropeBase (θ) | 컨텍스트 길이 |
  |------------|--------------|---------------|
  | GPT-NeoX   | 10,000       | 2K            |
  | LLaMA-2    | 10,000,000   | 4K ~ 8K       |
  | LLaMA-3    | 10,000,000   | 8K ~ 16K      |
  
  ---
  
  ## 정리
  - ${BT}ropeBase${BT}는 모델이 ${B}얼마나 긴 문맥까지 안정적으로 처리할 수 있는지${B}와 직결됩니다.  
  - 일반적으로 컨텍스트 길이가 길수록 더 큰 값이 필요합니다.  
    `,
    },
  },
  gqAttention: {
    numHeads: {
      title: 'Number of Heads',
      description: String.raw`
  어텐션을 ${B}여러 개의 헤드${B}(head)로 나누어 병렬 계산할 때,  
  그 헤드의 개수를 정하는 값입니다.  
  
  - 헤드 수가 많을수록: 더 다양한 관점에서 문맥을 해석 가능 (정밀 ↑, 계산량 ↑).  
  - 헤드 수가 적을수록: 계산은 가벼워지지만, 표현력이 제한될 수 있음.  
  
  즉, ${BT}numHeads${BT}는 ${B}모델이 동시에 몇 개의 시선으로 문장을 바라보는지${B}를 결정합니다.
    `,
    },

    ctxLength: {
      title: 'Context Length',
      description: String.raw`
  모델이 한 번에 볼 수 있는 ${B}최대 토큰 수${B}를 의미합니다.  
  
  - 이 값보다 긴 입력은 잘리거나 여러 조각으로 나눠 처리해야 합니다.  
  - 길이가 길수록 더 많은 문맥을 이해할 수 있지만, ${B}메모리 사용량과 계산량이 크게 증가${B}합니다.  
  
  즉, ${BT}ctxLength${BT}는 모델이 ${B}한 번에 기억할 수 있는 문장의 최대 길이${B}를 정하는 값입니다.
    `,
    },

    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: String.raw`
  어텐션 가중치에 적용되는 ${B}드롭아웃 비율${B}입니다.  
  
  - 학습 과정에서 일부 연결을 무작위로 끊어, ${B}과적합을 방지${B}합니다.  
  - 값이 클수록: 더 강하게 무작위화 → 과적합 방지 ↑, 하지만 학습이 불안정할 수 있음.  
  - 값이 작을수록: 안정적인 학습 가능, 하지만 과적합 위험 ↑.  
  
  보통 0.1 ~ 0.3 범위에서 사용합니다.
    `,
    },

    qkvBias: {
      title: 'QKV Bias',
      description: String.raw`
  쿼리(Q), 키(K), 값(V) 벡터를 만들 때 사용하는 ${B}선형 변환(Linear Projection)${B}에  
  ${B}편향(bias) 항을 포함할지 여부${B}를 결정합니다.  
  
  ---
  
  ## 수식으로 이해하기
  입력 $X$에 대해,
  $$
  Q = XW_Q + b_Q,\quad
  K = XW_K + b_K,\quad
  V = XW_V + b_V
  $$
  
  - 여기서 $W_Q, W_K, W_V$는 가중치 행렬  
  - $b_Q, b_K, b_V$는 선택적으로 추가되는 ${B}Bias 벡터${B}  
  - ${BT}qkvBias = true${BT} → 각 변환에 $b$를 포함  
  - ${BT}qkvBias = false${BT} → 순수 선형 변환만 적용
  
  ---
  
  ## 왜 필요한가?
  - ${B}Bias 포함 (true)${B}  
    - 각 Q/K/V가 ${B}0 이외의 기준점(offset)${B}에서 시작 가능  
    - 작은 모델이나 데이터가 제한적인 경우, 표현력 향상에 도움  
  - ${B}Bias 제외 (false)${B}  
    - 불필요한 파라미터를 줄여 ${B}계산 효율성${B} 및 ${B}훈련 안정성${B} 확보  
    - 대규모 LLM(LLaMA-2, GPT-3 등)에서는 보통 bias를 제거함
  
  ---
  
  ## 대표적인 설정 값
  
  | 모델명         | QKV Bias | 비고 |
  |----------------|----------|------|
  | Transformer (2017) | True  | 논문 기본 구조 |
  | BERT-base       | True  | HuggingFace 구현 |
  | GPT-2           | True  | OpenAI 기본 |
  | LLaMA-2         | False | 단순화 + 파라미터 절약 |
  | GPT-3           | False | 초대규모 학습에서 불필요 |
  
  ---
  
  ## 정리
  - ${BT}qkvBias${BT}는 Q/K/V 선형 변환에 ${B}bias 항을 포함할지 여부${B}  
  - 작은/중간 규모 모델 → ${BT}true${BT} (표현력 ↑)  
  - 대규모 모델 (수십억 파라미터) → ${BT}false${BT} (효율/안정성 ↑)  
  - 따라서 모델 규모와 데이터셋 크기에 따라 설정이 달라집니다.
    `,
    },

    isRoPE: {
      title: 'RoPE Enabled',
      description: String.raw`
  RoPE(Rotary Position Embedding)를 사용할지 여부를 결정합니다.  
  RoPE는 ${B}위치 정보를 임베딩에 회전(Rotation) 연산으로 삽입${B}하는 방식입니다.  
  
  ---
  
  ## 기본 개념
  - 전통적인 위치 임베딩(Positional Embedding)  
    → 각 토큰 위치마다 별도의 벡터를 더하거나 학습  
  - RoPE (Su et al., 2021)  
    → ${B}쿼리(Q), 키(K)${B} 벡터에 위치에 따른 ${B}회전 행렬${B}을 적용  
    → 각 차원마다 다른 주파수로 토큰의 상대적 위치를 인코딩  
  
  ---
  
  ## 수식
  입력 쿼리/키 $q, k$에 대해, RoPE 적용 시:
  $$
  q' = R_\theta q,\quad k' = R_\theta k
  $$
  여기서 $R_\theta$는 위치 $i$에 따라 정의된 회전 행렬이며,  
  내적 $q'^T k'$ 계산에 ${B}상대적 위치 차이${B}가 직접 반영됩니다.
  
  ---
  
  ## 장점
  - ${B}상대적 위치 인코딩${B} 가능 → 길이에 무관하게 일반화 ↑  
  - ${B}긴 문맥 처리${B}에 강함 (LLaMA, GPT-NeoX 등 활용)  
  - 추가 파라미터 없음 → 효율적  
  
  ---
  
  ## 언제 쓰나?
  - ${BT}isRoPE = true${BT}  
    - LLaMA, GPT-NeoX, PaLM 등 최신 구조 재현  
    - 긴 컨텍스트 학습/추론에 유리  
  - ${BT}isRoPE = false${BT}  
    - 고전 Transformer (Sinusoidal, Learned Embedding) 재현  
    - 교육/실험용 간단 구조  
  
  ---
  
  ## 정리
  - RoPE는 ${B}위치 정보를 회전 연산으로 삽입하는 기법${B}  
  - 긴 문맥, 대규모 모델에서 사실상 표준  
  - 실험적/교육적 목적 아니면 대부분 ${BT}true${BT}로 설정
    `,
    },

    ropeBase: {
      title: 'RoPE Base',
      description: String.raw`
RoPE에서 사용하는 ${B}기본 주파수 스케일 값(θ, theta)${B}을 정의합니다.  
이 값은 토큰 위치를 임베딩 차원마다 다른 각도로 회전시키는 데 쓰입니다.  

---

## 수식으로 이해하기
RoPE는 각 차원 $2i$에 대해 아래와 같은 각도를 사용합니다:
$$
\theta_i = \text{ropeBase}^{-2i/d_{\text{model}}}
$$

- $d_{model}$ : 임베딩 차원 크기  
- ${BT}ropeBase${BT}가 크면 → 각도 변화가 완만 → ${B}긴 문맥 표현에 유리${B}  
- ${BT}ropeBase${BT}가 작으면 → 각도 변화가 급함 → ${B}짧은 문맥에 집중${B}  

---

## 직관적으로
- 작은 값 (예: 1,000) → 위치 차이가 빨리 커져서 ${B}짧은 범위 관계에 민감${B}  
- 큰 값 (예: 10,000, 100,000) → 위치 차이가 천천히 반영되어 ${B}긴 범위 관계 유지${B}  

---

## 대표적인 설정
- Transformer (Sinusoidal) → 10,000 (기본)  
- GPT-NeoX / LLaMA → 10,000 또는 100,000 (길이 확장 버전)  
- 최근 확장 기법들 (YaRN, NTK RoPE 등) → 1,000,000 이상도 사용  

---

## 정리
- ${BT}ropeBase${BT}는 RoPE에서 ${B}위치 회전 주파수의 기준값${B}  
- 값이 클수록 모델은 ${B}더 긴 문맥${B}을 안정적으로 다룸  
- 보통 10,000을 기본으로 하되, 긴 컨텍스트 모델은 100,000 이상을 사용
  `,
    },

    qkNorm: {
      title: 'QK Normalization',
      description: String.raw`
  ## QK Normalization (Query/Key 정규화)
  
  ${B}QK Normalization${B}은 어텐션에서 사용하는 ${B}Query(Q)와 Key(K) 벡터를 정규화(normalization)할지 여부${B}를 설정하는 값입니다.  
  보통 벡터를 L2 정규화(길이를 1로 맞춤)하여 안정적인 내적 연산을 가능하게 합니다.
  
  ---
  
  ### 특징
  - ${B}True (정규화 적용)${B}
    - Query/Key 벡터 크기를 일정하게 맞춤  
    - 학습 안정성 ↑  
    - 학습률에 덜 민감해지고 수렴이 잘 되는 경우 많음
  - ${B}False (정규화 미적용)${B}
    - 계산이 단순해짐 → 약간의 속도/자원 절약  
    - 하지만 스케일 불안정으로 인해 학습이 흔들릴 수 있음
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}qk_norm${BT}을 True로 설정하면 ${B}어텐션 안정성${B}이 높아집니다.  
  - 특히 ${B}큰 모델${B}이나 ${B}긴 context_length${B}를 다룰 때 효과가 있음  
  - LLaMA-2, LLaMA-3 같은 최신 모델에서도 적극적으로 사용되는 기법입니다.
  
  ---
  
  ### 선택 가이드
  - ${B}작은 모델 / 단순 실험${B} → False (계산 단순화)  
  - ${B}중대형 모델 / 긴 문맥 학습${B} → True (안정성 강화)  
  
  ⚠️ 정규화를 사용하면 일반적으로 학습은 더 안정적이지만,  
  아주 작은 모델에서는 오히려 불필요한 연산 오버헤드가 될 수 있습니다.
    `,
    },

    qkNormEps: {
      title: 'QK Normalization Eps',
      description: String.raw`
  ## QK Normalization Eps

  수치적 안정성을 위해 ${B}분모에 더해지는 작은 상수${B}입니다.  
  정규화 과정에서 분산(variance)이 0에 가까워지면 분모가 0이 되어 나눗셈이 불가능해지는데,  
  이를 방지하기 위해 $\epsilon$ 값을 더해줍니다.  
  `,
    },
  },
  transformerBlock: {
    numOfBlocks: {
      title: 'Number of Blocks',
      description: String.raw`
  트랜스포머 인코더/디코더 블록을 ${B}몇 번 반복할지${B}(=쌓을지)를 지정하는 값입니다.  
  모델의 ${B}깊이${B}(depth)와 ${B}표현력${B}을 직접적으로 결정합니다.  
  
  ---
  
  ## 기본 개념
  - 하나의 Transformer Block =  
    ${B}멀티헤드 어텐션(MHA)${B} + ${B}피드포워드 네트워크(FFN)${B} + ${B}정규화/잔차 연결${B}  
  - ${BT}numOfBlocks${BT}는 이 블록을 ${B}몇 층 쌓을지${B}를 결정  
  
  ---
  
  ## 영향
  - 블록 수 ↑ →  
    - ${B}장점${B}: 더 복잡한 패턴 학습, 성능 ↑  
    - ${B}단점${B}: 연산량/메모리 사용량 급증, 과적합 위험 ↑  
  - 블록 수 ↓ →  
    - ${B}장점${B}: 계산이 빠르고 가볍다  
    - ${B}단점${B}: 복잡한 문맥 관계를 포착하기 어려움  
  
  ---
  
  ## 대표적인 설정
  | 모델명          | 블록 수 | 비고 |
  |-----------------|---------|------|
  | Transformer (2017) | 6 (인코더/디코더 각각) | 논문 기본 |
  | BERT-base        | 12      | 깊이=12, 넓이=768 |
  | GPT-2 small      | 12      | 117M 파라미터 |
  | GPT-3            | 96      | 175B 파라미터 |
  | LLaMA-2 7B       | 32      | 긴 문맥 처리 강화 |
  
  ---
  
  ## 정리
  - ${BT}numOfBlocks${BT} = ${B}모델 깊이${B}를 결정하는 핵심 값  
  - 적을수록 가볍고 빠르지만, 표현력이 제한됨  
  - 많을수록 성능은 좋아지지만, 자원 요구가 크고 학습이 어려워짐  
  - 모델 크기와 사용 환경에 맞춰 적절히 조정해야 함
      `,
    },
  },
  dynamicBlock: {
    numOfBlocks: {
      title: 'Number of Blocks',
      description: String.raw`
  ${B}동적 블록(Dynamic Block)${B} 안에 포함될 하위 블록의 수를 지정합니다.  
  여기서 각 블록은 동일하지 않고, ${B}서로 다른 구성 요소(예: 어텐션, FFN, 정규화 등)를 독립적으로 설정${B}할 수 있습니다.  
  
  ---
  
  ## 기본 개념
  - ${BT}transformerBlock${BT} = 동일한 구조의 블록을 여러 번 반복  
  - ${BT}dynamicBlock${BT} = 각 블록이 독립적 → 서로 다른 하이퍼파라미터/구조 조합 가능  
  
  ---
  
  ## 왜 필요한가?
  - 모델의 특정 구간은 ${B}어텐션 위주${B}, 다른 구간은 ${B}피드포워드 위주${B}로 구성 가능  
  - 실험적으로 ${B}구조 변형${B}을 적용해 성능 차이를 관찰할 수 있음  
  - 교육/연구 목적에서 "블록 조합 실험"에 유용  
  
  ---
  
  ## 영향
  - 블록 수 ↑ → 더 많은 단계에서 구조적 변형을 적용 가능 (유연성 ↑, 계산량 ↑)  
  - 블록 수 ↓ → 단순한 구조, 빠른 계산 (하지만 맞춤형 제어 ↓)  
  
  ---
  
  ## 예시 활용
  - 3개 Dynamic Blocks:  
    1. MHA + FFN  
    2. GQA Attention + RMSNorm  
    3. 단순 FFN Layer  
  
  → 이런 식으로 각 블록이 다른 구성을 가질 수 있음  
  
  ---
  
  ## 정리
  - ${BT}numOfBlocks${BT} = Dynamic Block 안에서 ${B}몇 개의 독립 블록을 둘지${B} 결정  
  - 실험적/맞춤형 아키텍처 설계에 필수  
  - 표준 Transformer 반복 구조와 달리, 유연성이 가장 큰 특징
      `,
    },
  },

  testBlock: {
    testType: {
      title: 'Test Type',
      description: String.raw`
    ${B}테스트 블록(Test Block)${B}의 타입을 지정합니다.  
    주로 모델 설계 과정에서 ${B}새로운 모듈을 시험${B}하거나, ${B}특정 구성 요소의 동작을 검증${B}하기 위해 사용됩니다.  
    
    ---
    
    ## 기본 개념
    - 일반 블록: 모델 학습/추론에서 실제 사용  
    - 테스트 블록: 실험적 목적 → 다양한 ${B}testType${B}을 넣어 동작 확인  
    
    ---
    
    ## 왜 필요한가?
    - 새로운 레이어(예: 커스텀 Attention, 특수 Activation) 실험  
    - 기존 블록 대비 성능/안정성 비교  
    - 디버깅 및 검증용 구조 삽입  
    
    ---
    
    ## 예시 testType
    - ${BT}"dummy"\ → 입력/출력을 그대로 통과시키는 블록  
    - ${BT}linear${BT} → 단순 선형 변환만 적용  
    - ${BT}attention_test${BT} → 어텐션 모듈만 별도로 시험  
    - ${BT}ffn_test${BT} → 피드포워드 네트워크만 독립적으로 시험  
    
    ---
    
    ## 정리
    - ${BT}testType${BT}은 ${B}테스트 블록의 역할/구조${B}를 정의하는 핵심 값  
    - 다양한 타입을 통해 실험적 시나리오를 쉽게 구성 가능  
    - 실제 프로덕션 모델보다는 ${B}연구/디버깅/비교 실험용${B}으로 활용
        `,
    },
  },
};
// … 이하 mhAttention, gqAttention, feedForward, dropout 등 추가
