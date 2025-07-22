// 모든 노드에 공통으로 필요한 속성을 포함하는 Base interface
export interface BaseNodeData {
  label: string; // Node 이름
  hideHandles?: boolean; // 노드 핸들 숨김 여부
  isTarget?: boolean; // 모델 타겟 노드 여부
  inDim?: number; // 입력 차원
  outDim?: number; // 출력 차원
  [key: string]: unknown; // Index Signature
}

// Layer의 특성에 따라 BaseNodeData를 확장한 interface
export interface TokenEmbeddingData extends BaseNodeData {
  vocabSize: number;
  embDim: number;
}

export interface PositionalEmbeddingData extends BaseNodeData {
  ctxLength: number;
  posType: string;
  vocabSize: number;
  embDim: number;
}

export interface FeedForwardData extends BaseNodeData {
  hiddenDim: number;
  feedForwardType: string;
  actFunc: string;
}

export interface DropoutData extends BaseNodeData {
  dropoutRate: number;
}

export interface NormalizationData extends BaseNodeData {
  normType: string;
}

export interface ResidualData extends BaseNodeData {
  source: string;
}

export interface MHAttentionData extends BaseNodeData {
  dropoutRate: number;
  ctxLength: number;
  numHeads: number;
  qkvBias?: false;
}

export interface GQAttentionData extends BaseNodeData {
  dropoutRate: number;
  ctxLength: number;
  numHeads: number;
  qkvBias?: false;
}

export interface GPT2TransformerBlockData extends BaseNodeData {
  numLayers: number;
  // 레이어 이름별 true/false 값 딕셔너리 추가
}

export interface TransformerBlockData extends BaseNodeData {
  numOfBlocks: number;
}

export interface TestBlockData extends BaseNodeData {
  testType: number;
  // sdpAttention?: SDPAttentionData | null;
}
