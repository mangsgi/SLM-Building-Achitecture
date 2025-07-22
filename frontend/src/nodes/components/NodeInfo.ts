// 노드 정보 인터페이스
export interface NodeInfo {
  title: string;
  description: string;
}

// 필드 정보 인터페이스
export interface FieldInfo {
  title: string;
  description: string;
}

// 노드별 필드 정보 타입
export interface NodeFieldInfo {
  [key: string]: FieldInfo;
}

// ✅ 노드 정보 저장소
export const nodeInfo: { [key: string]: NodeInfo } = {
  positionalEmbedding: {
    title: 'Positional Embedding',
    description:
      '입력 시퀀스의 위치 정보를 인코딩하는 레이어입니다. 각 토큰의 위치 정보를 임베딩 벡터에 추가하여 모델이 시퀀스의 순서를 이해할 수 있게 합니다.',
  },
  tokenEmbedding: {
    title: 'Token Embedding',
    description:
      '입력 토큰을 고차원 벡터 공간으로 매핑하는 레이어입니다. 각 토큰을 연속적인 벡터로 변환하여 모델이 처리할 수 있는 형태로 만듭니다.',
  },
  normalization: {
    title: 'Normalization',
    description:
      '레이어의 출력을 정규화하여 학습의 안정성을 높이는 레이어입니다. 주로 Layer Normalization이나 Batch Normalization을 사용합니다.',
  },
  feedForward: {
    title: 'Feed Forward',
    description:
      '각 위치에서 독립적으로 적용되는 완전 연결 신경망 레이어입니다. 주로 두 개의 선형 변환과 그 사이의 활성화 함수로 구성됩니다.',
  },
  dropout: {
    title: 'Dropout',
    description:
      '학습 시 일부 뉴런을 무작위로 비활성화하여 과적합을 방지하는 레이어입니다. 테스트 시에는 모든 뉴런을 사용합니다.',
  },
  linearOutput: {
    title: 'Linear Output',
    description:
      '입력을 선형 변환하는 레이어입니다. 가중치 행렬과 편향을 사용하여 입력을 출력 차원으로 변환합니다.',
  },
  mhAttention: {
    title: 'Multi-Head Attention',
    description:
      '쿼리, 키, 값 벡터를 사용하여 입력 시퀀스의 각 위치 간의 관계를 계산하는 어텐션 메커니즘입니다.',
  },
  transformerBlock: {
    title: 'Transformer Block',
    description:
      '트랜스포머의 핵심 구성 요소로, 멀티헤드 어텐션과 피드포워드 네트워크를 포함합니다. 입력 시퀀스의 각 위치에서 다른 위치의 정보를 참조하여 문맥을 이해하고 처리합니다.',
  },
  dynamicBlock: {
    title: 'Dynamic Block',
    description:
      '동적으로 레이어를 추가할 수 있는 블록입니다. 사용자가 필요에 따라 다양한 레이어를 추가하고 구성할 수 있습니다.',
  },
  residual: {
    title: 'Residual Connection',
    description:
      '입력을 출력에 더하여 그래디언트 소실 문제를 완화하고 학습을 안정화하는 연결입니다.',
  },
  testBlock: {
    title: 'Test Block',
    description:
      '노드 테스트를 위한 부모 노드입니다. 다양한 노드 타입을 테스트하고 구성할 수 있습니다.',
  },
  gqAttention: {
    title: 'Grouped Query Attention',
    description:
      '쿼리, 키, 값 벡터를 그룹화하여 처리하는 어텐션 메커니즘입니다. 메모리 효율성을 높이면서도 성능을 유지합니다.',
  },
};

// ✅ 노드별 필드 정보 저장소
export const nodeFieldInfo: { [key: string]: NodeFieldInfo } = {
  positionalEmbedding: {
    ctxLength: {
      title: 'Context Length',
      description:
        '입력 시퀀스의 최대 길이를 정의합니다. 이 값은 모델이 처리할 수 있는 최대 토큰 수를 결정합니다.',
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description:
        '임베딩 벡터의 차원을 정의합니다. 이 값은 모델의 표현력과 계산 복잡도에 영향을 미칩니다.',
    },
    posType: {
      title: 'Positional Embedding Type',
      description:
        '위치 정보를 인코딩하는 방식을 선택합니다. 각 타입은 서로 다른 장단점을 가지고 있습니다.',
    },
  },
  tokenEmbedding: {
    vocabSize: {
      title: 'Vocabulary Size',
      description:
        '모델이 처리할 수 있는 고유 토큰의 수를 정의합니다. 이는 입력 데이터의 토큰화 방식에 따라 결정됩니다.',
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description:
        '임베딩 벡터의 차원을 정의합니다. 이 값은 모델의 표현력과 계산 복잡도에 영향을 미칩니다.',
    },
  },
  normalization: {
    normType: {
      title: 'Normalization Type',
      description:
        '사용할 정규화 방법을 선택합니다. Layer Normalization, Batch Normalization 등이 있습니다.',
    },
    eps: {
      title: 'Epsilon',
      description:
        '수치적 안정성을 위한 작은 상수입니다. 분모가 0이 되는 것을 방지합니다.',
    },
    inDim: {
      title: 'Input Dimension',
      description:
        '정규화할 입력 텐서의 차원을 지정합니다. 이는 이전 레이어의 출력 차원과 일치해야 합니다.',
    },
  },
  feedForward: {
    actFunc: {
      title: 'Activation Function',
      description:
        '비선형성을 추가하는 활성화 함수를 선택합니다. ReLU, GELU 등이 있습니다.',
    },
    feedForwardType: {
      title: 'Feed Forward Type',
      description:
        '피드포워드 네트워크의 구조를 선택합니다. Standard, Gated 등이 있습니다.',
    },
    numOfFactor: {
      title: 'Number of Factors',
      description:
        '피드포워드 네트워크의 확장 비율을 지정합니다. 입력 차원에 이 값을 곱하여 내부 차원을 결정합니다.',
    },
  },
  dropout: {
    dropoutRate: {
      title: 'Dropout Rate',
      description:
        '비활성화할 뉴런의 비율을 정의합니다. 0에서 1 사이의 값을 가집니다.',
    },
  },
  linearOutput: {
    outDim: {
      title: 'Output Dimension',
      description:
        '출력 벡터의 차원을 정의합니다. 이는 다음 레이어의 입력 차원과 일치해야 합니다. 마지막 레이어로 사용되는 경우 모델의 출력 차원(사전 크기)과 일치해야 합니다.',
    },
  },
  mhAttention: {
    numHeads: {
      title: 'Number of Heads',
      description:
        '병렬로 처리할 어텐션 헤드의 수를 정의합니다. 각 헤드는 서로 다른 표현을 학습합니다.',
    },
    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: '어텐션 가중치에 적용할 드롭아웃 비율을 정의합니다.',
    },
    qkvBias: {
      title: 'QKV Bias',
      description: '쿼리, 키, 값 변환에 편향을 사용할지 여부를 결정합니다.',
    },
  },
  gqAttention: {
    numHeads: {
      title: 'Number of Heads',
      description:
        '병렬로 처리할 어텐션 헤드의 수를 정의합니다. 각 헤드는 서로 다른 표현을 학습합니다.',
    },
    ctxLength: {
      title: 'Context Length',
      description:
        '처리할 수 있는 최대 시퀀스 길이를 정의합니다. 이는 모델의 메모리 사용량과 계산 복잡도에 영향을 미칩니다.',
    },
    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: '어텐션 가중치에 적용할 드롭아웃 비율을 정의합니다.',
    },
    qkvBias: {
      title: 'QKV Bias',
      description: '쿼리, 키, 값 변환에 편향을 사용할지 여부를 결정합니다.',
    },
  },
  transformerBlock: {
    numLayers: {
      title: 'Number of Layers',
      description:
        '트랜스포머 블록 내의 레이어 수를 지정합니다. 각 레이어는 멀티헤드 어텐션과 피드포워드 네트워크를 포함합니다.',
    },
  },
  dynamicBlock: {
    numOfBlocks: {
      title: 'Number of Blocks',
      description:
        '동적 블록 내에 포함될 블록의 수를 지정합니다. 각 블록은 독립적으로 구성될 수 있습니다.',
    },
  },
  testBlock: {
    testType: {
      title: 'Test Type',
      description:
        '테스트 블록의 타입을 지정합니다. 다양한 테스트 시나리오를 구성할 수 있습니다.',
    },
  },
};
