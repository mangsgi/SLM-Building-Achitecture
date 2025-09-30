const BT = '`';
const B = '**';

export const datasetInformation = {
  tiny_shakespeare: {
    description: String.raw`
# Tiny shakespeare

${B}개요${B}  
- 앤드리j 카르파티의 ${BT}char-rnn${BT} 예제에서 널리 쓰이는 소형 코퍼스. 셰익스피어 희곡 텍스트 약 ${B}1MB(≈1.06 MiB)${B}를 한 파일로 모아 문자/단어 수준 LM 실험에 적합함. 

${B}구성 & 규모${B}  
- 단일 텍스트 파일(희곡 대사 연속본)로 제공. TFDS 기준 스플릿은 train/validation/test가 각각 1개 ${B}“예제”${B}로, 사실상 전체 코퍼스를 학습에 통째로 쓰고 검증·테스트는 사용자가 임의 분할하는 형태로 쓰는 경우가 많음. 

${B}용도${B}  
- 문자 단위/토크나이저 실험, 미니 GPT/나노GPT 튜토리얼, 과적합·샘플링 전략 검증 등 ${B}"작게 돌려보는"${B} 재현 실험에 최적. 예시 노트북과 레포들이 이를 전제로 사용. 
    `,
  },
  openwebtext_100k: {
    description: String.raw`
# OpenWebText-100k

${B}개요${B}  
- OpenAI의 비공개 WebText를 오픈 커뮤니티가 재현한 ${B}OpenWebText${B}에서 ${B}10만 문서 샘플만 추린 슬라이스${B}. 여기서 참조하는 경로는 ${BT}mychen76/openwebtext-100k${BT}로, 허깅페이스에 업로드된 100k 레코드 버전임. 

${B}원본(OpenWebText) 특징${B}  
- 레딧 제출 링크 모음에서 URL을 수집 → 비영어 필터링(fastText) → 중복 제거 등의 정제를 거쳐 구성. 결과는 약 ${B}수십 GB(예: 38GB)${B} 규모의 크롤링 텍스트로 보고됨. 

${B}이 100k 슬라이스의 의의${B}  
- 전체 OpenWebText의 분포를 소규모로 빠르게 실험·프로토타이핑하려는 목적에 적합(메모리/IO 부담 감소). 동일 계열의 10k/100k 서브셋이 벤치·튜토리얼용으로 흔히 쓰임. 
    `,
  },
  tinystories: {
    description: String.raw`
# TinyStories

${B}개요${B}  
- ${B}GPT-3.5/4가 생성한 합성 동화 데이터셋${B}으로, ${B}3–4세 아동이 이해할 수 있는 단어만${B} 사용하여 짧은 이야기(여러 문단)를 담음. “아주 작은 언어모델(SLM)도 유창하고 일관된 영어를 말할 수 있는가?”를 검증하기 위해 고안됨. 

${B}구성 & 규모${B}  
- 공개 버전은 수백만 편의 짧은 이야기와 메타데이터(프롬프트 등)로 제공되며, 학습용 ${BT}TinyStories-train.txt${BT}/검증용 ${BT}tinystories-valid.txt${BT}가 안내됨. 허깅페이스 데이터셋 뷰어 및 전체 번들(tar.gz) 링크가 제공됨. 

${B}특징${B}  
- 어휘 분포를 ${B}의도적으로 단순화${B}해도 문법·일관성·기초 추론을 학습할 수 있음을 보이며, ${B}1~10M 파라미터대 모델${B}에서도 유창한 산출을 확인했다는 보고. 연구·커리큘럼러닝 실험에 자주 사용. 

${B}토크나이저/통계 관점 메모${B}  
- 원 저자 구현에서는 GPT-Neo 토크나이저의 ${B}상위 K 단어만 유지${B}하는 세팅이 소개되며(예: top-10k), 커뮤니티 분석 기준 수백만 문서에 ${B}수억 토큰${B} 규모로 집계되기도 함(버전/전처리에 따라 상이). 
    `,
  },
  c4: {
    description: String.raw`
# C4

${B}개요${B}  
- Common Crawl 기반을 대규모로 **클린징**한 웹 코퍼스(**Colossal Clean Crawled Corpus**)의 공개 판. 구글 T5 연구에서 사용된 C4의 **처리 버전**을 AllenAI가 배포하며, 설정에 따라 ${BT}en${BT}, ${BT}en.noclean${BT}, ${BT}en.noblocklist${BT}, ${BT}realnewslike${BT}, ${BT}multilingual(mC4)${BT} 등 변형이 있음. 

${B}en 설정(영어) 규모${B}  
- TFDS 기준 ${B}≈806.9 GiB${B}, 학습 샘플 ${B}≈3.65억${B}(train), 검증 ${B}≈36만${B} 등 방대한 분량. 웹 스팸/중복/부적절 콘텐츠 제거 등 정제 단계를 거침. 

${B}용도${B}  
- 범용 LM 사전학습에 표준급으로 쓰이며, “대규모 웹텍스트 + 강한 정제”라는 특성 덕분에 자연어 이해/생성 전반의 베이스라인에 적합. ${BT}realnewslike${BT}, ${BT}mC4${BT} 등 변형으로 도메인/다국어 학습도 가능. 
    `,
  },
};
