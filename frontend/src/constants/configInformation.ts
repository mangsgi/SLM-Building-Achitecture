const BT = '`';
const B = '**';

export const fieldInformation = {
  model: {
    title: 'Model Type',
    description: String.raw`![model](/img/model.png)
## Model Type (모델 아키텍처 종류)

이 설정은 ${B}어떤 언어 모델 구조(아키텍처)${B}를 기반으로 학습할지를 선택하는 옵션입니다.  
즉, 우리 프로그램에서 ${B}모델의 뼈대${B}를 고르는 단계라고 이해하면 됩니다.  

---

### 지원되는 모델 종류
- ${B}GPT-2${B}
  - OpenAI에서 공개한 초기 GPT 계열 모델
  - 약 1억 ~ 15억 파라미터 크기까지 다양함
  - 상대적으로 가볍고 학습 속도가 빠름
  - 영어 위주 최적화 → 실험용/교육용으로 적합
- ${B}LLaMA-2${B}
  - Meta에서 공개한 모델
  - SentencePiece 기반 토크나이저 사용
  - 7B, 13B, 70B 등 다양한 크기
  - 다국어 지원 및 긴 문맥 처리 가능
- ${B}LLaMA-3${B}
  - LLaMA-2 후속, 더 긴 컨텍스트(최대 8K~32K 토큰)
  - 효율적 메모리 구조 (Grouped Query Attention, GQA)
  - 최신 GPT-4 계열과 유사한 토크나이저(${BT}cl100k_base${BT})
- ${B}Qwen-3${B}
  - Alibaba에서 공개한 최신 오픈소스 모델
  - 중국어·영어 등 다국어 성능 강화
  - 최신 학습 기법을 적용하여 효율성과 성능 개선

---

### 우리 프로그램에서의 활용
- ${BT}Model Type${BT}을 선택하면,  
  1. ${B}토크나이저${B}가 자동으로 결정됩니다.  
     - GPT-2 → BPE 기반 토크나이저  
     - LLaMA-2 → SentencePiece  
     - LLaMA-3 → GPT-4 계열 토크나이저(${BT}cl100k_base${BT})  
  2. ${B}기본 파라미터 값${B}이 달라집니다.  
     - 예: ${BT}vocab_size${BT}, ${BT}context_length${BT}, ${BT}n_heads${BT}, ${BT}n_blocks${BT} 등  
  3. ${B}학습 구조${B}가 달라집니다.  
     - LLaMA 계열은 QKV bias 제거, GQA 사용  
     - GPT 계열은 단순 Multi-Head Attention 사용

---

### 선택 가이드
- ${B}가볍게 실험/학습 구조 이해${B} → GPT-2 권장  
- ${B}다국어 & 긴 문맥 실험${B} → LLaMA-2  
- ${B}최신 구조, 긴 문맥 + 효율성${B} → LLaMA-3  
- ${B}중국어·영어 다국어 특화${B} → Qwen-3  

⚠️ 주의:  
선택한 모델에 맞는 ${B}토크나이저${B}와 ${B}파라미터${B}를 함께 설정해야 학습이 정상적으로 진행됩니다.  
예를 들어 GPT-2 모델을 선택했는데 LLaMA-2 토크나이저를 쓰면 학습/추론이 불가능합니다.
    `,
  },
  epochs: {
    title: 'Epochs',
    description: String.raw`![epoch](/img/epoch.png)
  ## Epochs (학습 반복 횟수)
  
  ${B}Epoch${B}은 전체 학습 데이터를 몇 번 반복해서 학습할지를 의미합니다.  
  즉, 모델이 데이터셋 전체를 한 바퀴 도는 횟수입니다.
  
  ---
  
  ### 특징
  - ${B}큰 값${B}
    - 데이터셋을 여러 번 학습 → 더 많이 수렴할 가능성 ↑
    - 하지만 너무 크면 ${B}과적합(overfitting)${B} 위험 ↑
  - ${B}작은 값${B}
    - 학습 속도는 빠르지만 충분히 학습하지 못할 수 있음
    - ${B}과소적합(underfitting)${B} 가능성 ↑
  
  ---
  
  ### 우리 프로그램에서의 활용
  - 작은 값(예: 2~5) → 빠른 테스트/디버깅에 적합  
  - 큰 값(예: 50~200) → 정식 학습에 적합, 하지만 시간·GPU 자원 많이 소모  
  
  ---
  
  ### 선택 가이드
  - ${B}빠른 실험${B}: 2 ~ 5  
  - ${B}소규모 데이터셋 학습${B}: 10 ~ 50  
  - ${B}대규모 데이터셋 학습${B}: 50 ~ 200+  
  
  ⚠️ GPU 성능과 학습 데이터 크기에 따라 조정해야 합니다.
      `,
  },
  batch_size: {
    title: 'Batch Size',
    description: String.raw`![batch](/img/batch_size.png)
  ## Batch Size (배치 크기)
  
  ${B}Batch Size${B}는 한 번의 학습 스텝(step)에서 동시에 처리하는 데이터 샘플 개수입니다.  
  
  ---
  
  ### 특징
  - ${B}큰 값${B}
    - 한 번에 많은 샘플을 학습 → ${B}안정적인 손실 감소${B}
    - 하지만 GPU 메모리 사용량 ↑
    - 너무 크면 일반화 성능 저하 가능
  - ${B}작은 값${B}
    - 메모리 절약 가능
    - 업데이트가 자주 일어나 학습이 불안정할 수 있음
    - 하지만 일반화에 유리할 때도 있음
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}Batch Size${BT}는 ${B}GPU VRAM 용량${B}에 크게 좌우됩니다.
    - VRAM이 작으면 → 작은 배치 (예: 2~8)
    - VRAM이 크면 → 큰 배치 (예: 16~64 이상) 가능
  - 같은 모델이라도 ${BT}context_length${BT}가 길면 메모리 사용량이 배로 늘어나므로,  
    ${BT}Batch Size × Context Length${BT}를 고려해 조정해야 합니다.
  
  ---
  
  ### 선택 가이드
  - ${B}테스트/디버깅${B}: 2 ~ 4  
  - ${B}일반 학습 (8GB VRAM 이하)${B}: 2 ~ 8  
  - ${B}고성능 GPU (16GB 이상)${B}: 16 ~ 64  
  
  ⚠️ VRAM 한도를 초과하면 ${B}Out Of Memory (OOM)${B} 에러가 발생합니다.
    `,
  },

  stride: {
    title: 'Stride',
    description: String.raw`
  ## Stride (슬라이딩 간격)
  
  ${B}Stride${B}는 텍스트를 잘라 학습 시, ${B}윈도우(창)를 몇 토큰씩 이동할지${B}를 결정하는 값입니다.  
  즉, ${BT}context_length${BT} 크기만큼 자른 학습 단위를 다음으로 옮길 때 ${B}얼마나 겹치게 할지${B}를 조정하는 옵션입니다.
  
  ---
  
  ### 동작 방식
  - ${BT}context_length = 128, stride = 64${BT}  
    → 128 길이의 창을 만들고, 64 토큰만큼만 건너뛴 뒤 다음 시퀀스를 생성  
  - 이렇게 하면 앞뒤 시퀀스가 ${B}겹쳐지면서 문맥 연결성을 유지${B}할 수 있음  
  
  ---
  
  ### 특징
  - ${B}큰 stride 값${B}
    - 중복이 적음 → 데이터셋 크기 ↓
    - 하지만 문맥 연결이 약해질 수 있음
  - ${B}작은 stride 값${B}
    - 중복이 많음 → 데이터셋 크기 ↑
    - 문맥이 더 매끄럽게 이어짐
    - 학습 안정성이 좋아질 수 있음
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}stride${BT}는 데이터셋을 ${B}얼마나 촘촘히 학습시킬지${B}를 조절합니다.  
  - 작은 stride → 학습 데이터가 많아져 GPU 사용량·시간 ↑  
  - 큰 stride → 빠르게 학습하지만 문맥 이해가 약해질 수 있음
  
  ---
  
  ### 선택 가이드
  - 빠른 테스트: ${B}stride = context_length${B} (겹침 없음)  
  - 일반 학습: ${B}stride = context_length / 2${B} (절반 겹치기)  
  - 긴 문맥 유지 필요: ${B}stride < context_length / 2${B}
  
  ⚠️ ${BT}stride${BT} 값이 너무 작으면 학습 데이터가 불필요하게 커져 학습 시간이 급격히 늘어납니다.
    `,
  },

  dtype: {
    title: 'Data Type',
    description: String.raw`
  ## Data Type (연산 정밀도)
  
  ${B}Data Type${B}은 모델 파라미터와 연산을 어떤 ${B}숫자 정밀도${B}로 저장·계산할지를 결정하는 값입니다.  
  즉, 학습 시 메모리 사용량, 속도, 안정성에 직접적인 영향을 줍니다.
  
  ---
  
  ### 지원되는 타입
  - ${B}fp32 (Float32)${B}
    - 32비트 부동소수점
    - 가장 정확하지만 메모리 사용량이 많고 속도가 느림
    - 안정성은 최고 수준
  - ${B}fp16 (Float16)${B}
    - 16비트 부동소수점
    - 절반 정밀도 → 메모리 절약, 연산 속도 ↑
    - 하지만 underflow/overflow로 학습 불안정 가능
  - ${B}bf16 (BFloat16)${B}
    - 16비트지만 fp32와 비슷한 표현 범위
    - 최신 GPU/TPU에서 권장
    - 속도와 메모리 효율은 fp16 수준, 안정성은 fp32에 가까움
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}dtype${BT}을 변경하면 학습 중 사용하는 ${B}Tensor 데이터 타입${B}이 바뀝니다.
  - 작은 dtype(fp16, bf16)을 쓰면 VRAM 절약과 속도 향상이 가능하지만,  
    환경(GPU 지원 여부)에 따라 학습이 터질 수 있습니다.
  - 특히 ${B}bf16${B}은 최신 NVIDIA GPU(Ampere 이상)나 TPU에서 안정적입니다.
  
  ---
  
  ### 선택 가이드
  - ${B}안전·안정성 최우선${B}: fp32  
  - ${B}속도/메모리 최적화 (지원되는 GPU)${B}: bf16  
  - ${B}속도 최우선, 실험적${B}: fp16  
  
  ⚠️ GPU가 해당 dtype을 지원하지 않으면 오류 발생 → 반드시 환경 확인 필요
    `,
  },

  vocab_size: {
    title: 'Vocabulary Size',
    description: String.raw`
  ## Vocabulary Size (어휘 집합 크기)
  
  ${B}Vocabulary Size${B}는 토크나이저가 사용하는 ${B}전체 단어(토큰) 개수${B}를 의미합니다.  
  즉, 모델이 학습하고 이해할 수 있는 ${B}언어의 단위${B}가 몇 개인지를 결정하는 값입니다.
  
  ---
  
  ### 특징
  - ${B}큰 Vocabulary${B}
    - 더 다양한 단어와 표현을 직접 학습 가능
    - 하지만 모델 파라미터 수와 메모리 사용량 ↑
    - 학습 난이도 ↑
  - ${B}작은 Vocabulary${B}
    - 모델이 가벼워지고 계산 효율 ↑
    - 하지만 희귀 단어는 쪼개져 표현 → 세밀한 표현력이 부족할 수 있음
  
  ---
  
  ### 대표 값
  - GPT-2 → ${B}50,257개${B}
  - LLaMA-2 → ${B}약 32K${B}
  - LLaMA-3 → ${B}약 128K${B}
  
  ---
  
  ### 우리 프로그램에서의 활용
  - 선택한 ${B}모델 타입(Model Type)${B}에 따라 권장 ${BT}vocab_size${BT}가 자동으로 맞춰집니다.  
  - 모델 구조와 토크나이저의 어휘 크기가 다르면 ${B}학습/추론이 불가능${B}합니다.  
    - 예: GPT-2 모델에 LLaMA-2 토크나이저(32K)를 사용하면 오류 발생
  
  ---
  
  ### 선택 가이드
  - GPT-2 실험용 모델 → ${B}50,257${B}  
  - LLaMA-2 → ${B}32K (약 32,000)${B}  
  - LLaMA-3 → ${B}128K (약 128,000)${B}  
  
  ⚠️ ${BT}vocab_size${BT}는 토크나이저와 반드시 맞아야 하며, 잘못 설정하면 학습이 시작되지 않습니다.
    `,
  },

  context_length: {
    title: 'Context Length',
    description: String.raw`
  ## Context Length (문맥 길이)
  
  ${B}Context Length${B}는 모델이 한 번에 처리할 수 있는 최대 토큰(token) 수를 의미합니다.  
  즉, 모델이 동시에 "기억"하고 활용할 수 있는 입력 길이입니다.
  
  ---
  
  ### 특징
  - ${B}짧은 Context Length${B}
    - 긴 문장이나 여러 문단을 한 번에 이해하기 어려움
    - 메모리 사용량 ↓, 학습/추론 속도 ↑
  - ${B}긴 Context Length${B}
    - 더 많은 문맥(대화, 문서, 코드 등)을 한 번에 이해 가능
    - 하지만 연산량과 GPU 메모리 사용량 ↑
  
  ---
  
  ### 대표 값
  - ${B}GPT-2${B} → 1,024 토큰
  - ${B}GPT-3${B} → 2,048 토큰
  - ${B}LLaMA-2${B} → 4,096 토큰
  - ${B}LLaMA-3${B} → 8,192 토큰 (확장 버전은 32K까지 지원)
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}context_length${BT} 값은 학습 데이터 슬라이딩(${BT}stride${BT})과 함께 동작합니다.  
  - 값이 클수록 ${B}Batch Size × Context Length${B} 곱이 커져 VRAM 사용량이 급격히 증가합니다.  
  - 따라서 GPU 성능과 데이터 특성을 고려해 조정해야 합니다.
  
  ---
  
  ### 선택 가이드
  - 짧은 실험/테스트 → 512 ~ 1,024  
  - 일반 학습 → 2,048 ~ 4,096  
  - 긴 문맥 실험 (최신 모델) → 8,192 이상  
  
  ⚠️ 너무 큰 값을 설정하면 GPU 메모리 부족(OOM)이 발생할 수 있습니다.
    `,
  },

  emb_dim: {
    title: 'Embedding Dimension',
    description: String.raw`
  ## Embedding Dimension (임베딩 차원)
  
  ${B}Embedding Dimension${B}은 각 토큰을 벡터로 변환할 때의 ${B}차원 크기${B}를 의미합니다.  
  즉, 모델이 단어(토큰)를 수학적으로 어떤 크기의 벡터 공간에서 표현할지를 결정합니다.
  
  ---
  
  ### 특징
  - ${B}큰 값${B}
    - 더 풍부하고 정교한 의미 표현 가능 → 모델 성능 ↑
    - 하지만 파라미터 수와 연산량, 메모리 사용량 ↑
  - ${B}작은 값${B}
    - 계산 효율적, 메모리 절약
    - 하지만 언어적 표현력과 모델 성능에 한계 ↓
  
  ---
  
  ### 대표 값
  - ${B}GPT-2 small${B} → 768  
  - ${B}LLaMA-2 7B${B} → 4,096  
  - ${B}대형 모델 (수십~수백억 파라미터)${B} → 8,192 이상
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}emb_dim${BT}은 모델의 ${B}표현력과 학습 효율성의 핵심 지표${B}입니다.  
  - 동시에 ${B}n_heads (어텐션 헤드 수)${B}와도 밀접한 관계가 있어,  
    ${BT}emb_dim ÷ n_heads${BT} 값이 정수여야 멀티헤드 어텐션이 정상 동작합니다.
  
  ---
  
  ### 선택 가이드
  - 가벼운 모델 (테스트/학습 구조 이해) → 256 ~ 768  
  - 중간 규모 모델 (실험/연구용) → 1,024 ~ 2,048  
  - 대규모 모델 (정식 학습/프로덕션) → 4,096 이상  
  
  ⚠️ ${BT}emb_dim${BT}이 커질수록 학습 속도와 GPU 메모리 사용량이 급격히 증가합니다.
    `,
  },

  n_heads: {
    title: 'Number of Heads',
    description: String.raw`
  ## Number of Heads (어텐션 헤드 수)
  
  ${B}Number of Heads${B}는 멀티헤드 어텐션(Multi-Head Attention)에서 병렬로 사용하는 ${B}어텐션 헤드의 개수${B}를 의미합니다.  
  즉, 입력 데이터를 서로 다른 관점에서 동시에 해석하도록 분할하는 방식입니다.
  
  ---
  
  ### 특징
  - ${B}많은 헤드 수${B}
    - 여러 관점을 동시에 학습 → 더 다양한 관계 표현 가능
    - 하지만 연산량과 메모리 사용량 ↑
  - ${B}적은 헤드 수${B}
    - 계산량 ↓, 속도 ↑
    - 하지만 복잡한 관계를 충분히 학습하지 못할 수 있음
  
  ---
  
  ### 수학적 제약
  - ${BT}emb_dim ÷ n_heads${BT} 값은 반드시 정수여야 함  
    → 각 헤드가 동일한 차원 크기를 가져야 하기 때문  
  - 예: ${BT}emb_dim = 768, n_heads = 12 → head_dim = 64${BT}
  
  ---
  
  ### 대표 값
  - GPT-2 small → 12 heads  
  - LLaMA-2 7B → 32 heads  
  - 대규모 모델 (수십억 파라미터 이상) → 64+ heads
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}n_heads${BT}는 ${BT}emb_dim${BT}과 직접적으로 연결되며, 모델의 병렬 어텐션 구조를 정의합니다.  
  - 너무 많은 헤드를 선택하면 GPU 연산량이 급증하므로, ${BT}emb_dim${BT} 크기에 맞춰 적절히 조정해야 합니다.
  
  ---
  
  ### 선택 가이드
  - 소규모 실험 모델 → 4 ~ 12 heads  
  - 중간 규모 모델 → 16 ~ 32 heads  
  - 대규모 모델 → 32 ~ 64+ heads  
  
  ⚠️ ${BT}n_heads${BT}는 반드시 ${BT}emb_dim${BT}으로 나누어 떨어져야 하며,  
  나누어떨어지지 않으면 모델이 정상적으로 동작하지 않습니다.
    `,
  },

  n_blocks: {
    title: 'Number of Blocks',
    description: String.raw`
  ## Number of Blocks (Transformer 블록 개수)
  
  ${B}Number of Blocks${B}는 모델 내부에 쌓이는 ${B}Transformer 블록(계층, depth)${B}의 개수를 의미합니다.  
  각 블록은 ${B}어텐션(MHA)${B}과 ${B}FeedForward(FFN)${B} 층으로 구성되어 있으며, 블록 수가 곧 모델의 깊이를 결정합니다.
  
  ---
  
  ### 특징
  - ${B}블록 수가 많을 때${B}
    - 더 깊은 표현 학습 가능 → 복잡한 패턴 이해 ↑
    - 하지만 계산량과 GPU 메모리 사용량 ↑
    - 학습 속도 ↓
  - ${B}블록 수가 적을 때${B}
    - 모델이 가벼워지고 학습/추론 속도 ↑
    - 하지만 표현력이 제한적 → 성능 한계 ↓
  
  ---
  
  ### 대표 값
  - ${B}GPT-2 small${B} → 12 blocks  
  - ${B}GPT-3${B} → 96 blocks  
  - ${B}LLaMA-2 7B${B} → 32 blocks  
  - ${B}LLaMA-3 8B${B} → 48 blocks  
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}n_blocks${BT}는 모델의 ${B}깊이(depth)${B}를 직접 결정하는 핵심 값입니다.  
  - 블록이 많을수록 성능은 좋아지지만, 학습 시간이 길어지고 GPU 요구량이 급격히 늘어납니다.  
  - 실험 단계에서는 적은 블록(예: 2~6)으로 빠르게 테스트 후,  
    본 학습에서는 점차 늘려가는 방식을 추천합니다.
  
  ---
  
  ### 선택 가이드
  - ${B}빠른 테스트/디버깅${B} → 2 ~ 6 blocks  
  - ${B}소규모 모델 학습${B} → 12 ~ 24 blocks  
  - ${B}중대형 모델 학습${B} → 32 ~ 96 blocks  
  
  ⚠️ 블록 수를 늘릴수록 성능은 좋아지지만, 학습 시간이 기하급수적으로 증가하므로 GPU 환경을 고려해야 합니다.
    `,
  },

  drop_rate: {
    title: 'Dropout Rate',
    description: String.raw`
  ## Dropout Rate (드롭아웃 비율)
  
  ${B}Dropout${B}은 학습 시 신경망의 일부 뉴런을 확률적으로 비활성화(0으로 설정)하는 정규화 기법입니다.  
  ${BT}Dropout Rate${BT}은 이때 ${B}꺼지는 뉴런의 비율${B}을 의미합니다.
  
  ---
  
  ### 특징
  - ${B}높은 Dropout Rate (0.3 이상)${B}
    - 많은 뉴런을 끔 → 과적합 방지 효과 ↑
    - 하지만 학습 속도가 느려지고 충분한 학습이 어려울 수 있음
  - ${B}낮은 Dropout Rate (0.1 이하)${B}
    - 모델이 더 많이 학습 가능 → 성능 ↑
    - 하지만 과적합 위험 ↑
  
  ---
  
  ### 일반적인 범위
  - 보통 ${B}0.1 ~ 0.3${B} 사이 값 사용
  - 0.0은 Dropout을 사용하지 않는다는 의미
  
  ---
  
  ### 우리 프로그램에서의 활용
  - 과적합이 발생할 가능성이 높은 ${B}작은 데이터셋${B}에서는 Dropout을 크게 설정 (0.2~0.3)  
  - 데이터셋이 크거나 ${B}모델 용량이 작은 경우${B}에는 Dropout을 줄이거나 꺼도 됨 (0.0~0.1)  
  
  ---
  
  ### 선택 가이드
  - ${B}작은 데이터셋${B} → 0.2 ~ 0.3  
  - ${B}중간 규모 데이터셋${B} → 0.1 ~ 0.2  
  - ${B}큰 데이터셋/대규모 모델${B} → 0.0 ~ 0.1  
  
  ⚠️ Dropout은 ${B}학습 시에만 적용${B}되며, 추론 단계에서는 자동으로 비활성화됩니다.
    `,
  },

  qkv_bias: {
    title: 'QKV Bias',
    description: String.raw`
  ## QKV Bias (Query/Key/Value Bias 사용 여부)
  
  ${B}QKV Bias${B}는 어텐션 메커니즘에서 Query(Q), Key(K), Value(V)를 계산할 때  
  선형 변환(Linear Layer)에 ${B}bias 항(term)${B}을 추가할지 여부를 결정하는 설정입니다.
  
  ---
  
  ### 특징
  - ${B}True (Bias 사용)${B}
    - 선형 변환에 bias 추가 → 더 유연한 표현 학습 가능
    - 작은 데이터셋이나 복잡한 패턴 학습에서 도움이 될 수 있음
  - ${B}False (Bias 제거)${B}
    - 구조 단순화 → 파라미터 수와 연산량 약간 감소
    - 대규모 모델에서는 오히려 안정적이고 효율적
    - LLaMA 계열 모델은 기본적으로 bias를 제거한 구조 사용
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}qkv_bias${BT}는 모델의 ${B}어텐션 구조 세부 설정${B}입니다.  
  - 작은 실험용 모델에서는 True로 설정해도 무방하지만,  
    최신 대규모 모델(LLaMA-2, LLaMA-3 등)은 False를 기본으로 사용합니다.  
  - 값 하나 차이지만, 모델 아키텍처 재현에 중요한 역할을 합니다.
  
  ---
  
  ### 선택 가이드
  - ${B}실험/소규모 모델${B} → True  
  - ${B}대규모 모델(LLaMA 스타일)${B} → False  
  
  ⚠️ 학습 재현성(Reproducibility)을 위해, 사용하는 레퍼런스 모델 구조와 동일하게 맞추는 것이 중요합니다.
    `,
  },

  hidden_dim: {
    title: 'Hidden Dimension',
    description: String.raw`
  ## Hidden Dimension (피드포워드 은닉 차원)
  
  ${B}Hidden Dimension${B}은 Transformer 블록 안의 ${B}FeedForward Network (FFN)${B}에서  
  중간 계층이 갖는 벡터 차원의 크기를 의미합니다.  
  즉, 어텐션으로 얻은 표현을 더 깊게 변환하고 확장하는 역할을 합니다.
  
  ---
  
  ### 특징
  - ${B}큰 hidden_dim${B}
    - 더 많은 비선형 변환 학습 가능 → 모델 표현력 ↑
    - 하지만 파라미터 수와 연산량, 메모리 사용량 ↑
  - ${B}작은 hidden_dim${B}
    - 모델이 가벼워지고 계산 효율 ↑
    - 하지만 복잡한 패턴을 학습하기 어려움
  
  ---
  
  ### 일반적인 설정 규칙
  - 보통 ${BT}hidden_dim = emb_dim × (2 ~ 4)${BT}
  - 예:  
    - LLaMA-2 7B → emb_dim = 4,096 → hidden_dim = 11,008  
    - GPT-2 small → emb_dim = 768 → hidden_dim ≈ 3,072
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}hidden_dim${BT}은 FFN의 ${B}내부 확장 비율${B}을 조절하는 핵심 값입니다.  
  - emb_dim과 함께 모델 용량과 성능을 크게 좌우하므로,  
    ${B}모델 타입(Model Type)${B}에 맞는 비율을 따르는 것이 안전합니다.
  
  ---
  
  ### 선택 가이드
  - ${B}실험/경량 모델${B} → emb_dim × 2  
  - ${B}중간 규모 모델${B} → emb_dim × 3  
  - ${B}대규모 모델${B} → emb_dim × 4  
  
  ⚠️ 너무 크게 설정하면 GPU 메모리 부족(OOM)이 발생할 수 있으므로,  
  항상 ${BT}emb_dim${BT}과 GPU 성능을 고려해 설정해야 합니다.
    `,
  },

  n_kv_groups: {
    title: 'Number of KV Groups',
    description: String.raw`
  ## Number of KV Groups (KV 그룹 개수)
  
  ${B}n_kv_groups${B}는 Grouped Query Attention(GQA)에서  
  ${B}Key/Value(KV) 벡터를 몇 개의 그룹으로 나눌지${B}를 정의하는 값입니다.  
  
  ---
  
  ### GQA(Grouped Query Attention)란?
  - 기존 Multi-Head Attention에서는 ${B}모든 Q, K, V${B}를 1:1로 매칭하여 연산
  - 하지만 GQA에서는 여러 Query 헤드가 ${B}같은 Key/Value 그룹${B}을 공유
  - 메모리 사용량 ↓, 연산 효율 ↑, 긴 컨텍스트 처리에 유리
  
  ---
  
  ### 특징
  - ${B}n_kv_groups가 작을 때${B}
    - 더 많은 Query 헤드가 같은 KV를 공유  
    - 메모리 효율 ↑  
    - 하지만 정보 표현의 다양성이 ↓
  - ${B}n_kv_groups가 클 때${B}
    - KV 공유 범위가 줄어 표현력 ↑  
    - 하지만 메모리 사용량과 연산량 ↑
  
  ---
  
  ### 대표 값
  - ${B}LLaMA-3${B} → 일반적으로 ${BT}n_kv_groups = n_heads ÷ 8${BT}  
  - 예: n_heads = 64 → n_kv_groups = 8
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}n_kv_groups${BT}는 대규모 모델에서 ${B}메모리 효율과 성능 균형${B}을 맞추는 핵심 설정입니다.  
  - 작은 모델(GPT-2 등)에서는 사용되지 않고, LLaMA-3 같은 최신 구조에서만 의미 있음.  
  - 값이 너무 작으면 모델 성능 저하, 너무 크면 GPU 메모리 부담 증가.
  
  ---
  
  ### 선택 가이드
  - ${B}실험/소규모 모델${B} → 기본값 유지 (보통 1, 또는 GQA 미사용)  
  - ${B}중규모 이상 (LLaMA-3)${B} → n_heads ÷ 8  
  - ${B}메모리 절약 최우선${B} → n_heads ÷ 16 (더 큰 공유)  
  
  ⚠️ ${BT}n_kv_groups${BT}는 ${BT}n_heads${BT}와 반드시 연관되어야 하며,  
  무작위 값으로 설정하면 학습이 정상적으로 진행되지 않을 수 있습니다.
    `,
  },

  rope_base: {
    title: 'RoPE Base',
    description: String.raw`
  ## RoPE Base (Rotary Position Embedding 기준값)
  
  ${B}RoPE Base${B}는 Rotary Position Embedding(RoPE)에서  
  위치 정보를 인코딩할 때 사용하는 ${B}기본 주파수 스케일 값${B}입니다.  
  
  ---
  
  ### RoPE란?
  - 기존 Positional Encoding(사인/코사인) 대신,  
    ${B}위치 정보를 회전(rotation) 연산으로 토큰 벡터에 주입${B}하는 방식
  - 긴 문맥에서 더 자연스럽게 위치 관계를 표현할 수 있음
  - GPT-NeoX, LLaMA, Qwen, MPT 등 최신 모델들이 채택
  
  ---
  
  ### 특징
  - 보통 ${B}10,000${B}으로 설정 (기본값)  
  - Base 값이 크면:
    - 긴 거리에서도 구분이 잘 됨 → 긴 문맥 처리에 유리
    - 하지만 세밀한 위치 구분은 다소 떨어짐
  - Base 값이 작으면:
    - 가까운 토큰 사이 구분이 더 정밀
    - 하지만 긴 문맥에서는 표현력이 약해짐
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}rope_base${BT}는 RoPE의 ${B}스케일 기준값${B}으로,  
    모델이 문맥 길이를 어떻게 인식할지에 직접적인 영향을 줍니다.
  - 일반적으로는 ${B}10,000${B}을 그대로 사용하면 충분합니다.  
  - 특별히 긴 문맥(수만 토큰)을 실험할 경우만 조정하는 편입니다.
  
  ---
  
  ### 선택 가이드
  - ${B}일반 학습/실험${B} → 10,000 (기본값)  
  - ${B}긴 문맥 실험 (예: 32K context)${B} → 20,000 이상  
  - ${B}아주 짧은 문맥 실험${B} → 5,000 이하  
  
  ⚠️ 잘못된 값으로 설정하면 RoPE 스케일이 비정상적으로 변해  
  모델이 문맥을 제대로 학습하지 못할 수 있습니다.
    `,
  },

  rope_freq: {
    title: 'RoPE Frequency',
    description: String.raw`
  ## RoPE Frequency (RoPE 주파수 스케일링 값)
  
  ${B}RoPE Frequency${B}는 Rotary Position Embedding(RoPE)에서  
  위치 정보를 인코딩할 때 사용하는 ${B}주파수 스케일링 비율${B}을 의미합니다.  
  
  ---
  
  ### RoPE 복습
  - RoPE는 사인/코사인 파동을 이용해 ${B}위치 정보를 토큰 벡터에 회전 형태로 주입${B}하는 방식
  - ${BT}rope_base${BT}가 기준값이라면, ${BT}rope_freq${BT}는 그 주파수를 ${B}얼마나 확대/축소할지${B} 조절하는 역할
  
  ---
  
  ### 특징
  - ${B}값이 클 때${B}
    - 더 높은 주파수 사용 → 가까운 토큰 간의 ${B}세밀한 위치 차이${B}를 잘 구분
    - 하지만 멀리 떨어진 토큰 간 관계는 표현력이 떨어질 수 있음
  - ${B}값이 작을 때${B}
    - 낮은 주파수 사용 → ${B}넓은 범위의 문맥${B}을 인코딩하는 데 유리
    - 하지만 근접 토큰 사이의 정밀한 구분력이 약해짐
  
  ---
  
  ### 우리 프로그램에서의 활용
  - ${BT}rope_freq${BT}는 모델이 문맥을 바라보는 ${B}스케일의 해상도${B}를 결정합니다.  
  - 보통 기본값을 유지하지만, 긴 문맥 최적화나 특정 실험 목적일 때 조정할 수 있습니다.  
  
  ---
  
  ### 선택 가이드
  - ${B}일반 학습/기본 설정${B} → 1.0 (표준값)  
  - ${B}세밀한 위치 표현 필요 (짧은 문맥 분석)${B} → 2.0 이상  
  - ${B}넓은 문맥 분석 (긴 대화/문서)${B} → 0.5 이하  
  
  ⚠️ ${BT}rope_base${BT}와 ${BT}rope_freq${BT}는 함께 작동하므로,  
  둘의 균형이 맞지 않으면 모델의 위치 인코딩 성능이 저하될 수 있습니다.
    `,
  },

  qk_norm: {
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

  head_dim: {
    title: 'Head Dimension',
    description: String.raw`
  ## Head Dimension (Head 차원)
  
  ${B}Head Dimension${B}은 어텐션에서 사용하는 ${B}Head 차원${B}을 의미합니다.
  
  ### 특징
  - ${B}값이 클 때${B}
    - 더 많은 표현력 가능 → 정밀한 문맥 해석 가능
    - 하지만 더 많은 연산량과 메모리 사용량 ↑
  - ${B}값이 작을 때${B}
    - 더 적은 연산량과 메모리 사용량 ↓
    - 하지만 표현력이 제한될 수 있음
  `,
  },
};
