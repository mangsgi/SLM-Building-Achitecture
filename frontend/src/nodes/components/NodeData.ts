// 모든 노드에 공통으로 필요한 속성을 포함하는 Base interface
export interface BaseNodeData {
  label: string; // Node 이름
  openModal?: (nodeData: BaseNodeData) => void;
  hideHandles?: boolean;
  isTarget?: boolean;
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
  posEmbeddingType: string;
  vocabSize: number;
  embDim: number;
}

export interface FeedForwardData extends BaseNodeData {
  numOfFactor: number;
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

export interface SDPAttentionData extends BaseNodeData {
  dropoutRate: number;
  ctxLength: number;
  numHeads: number;
  qkvBias?: false;
}

export interface TransformerBlockData extends BaseNodeData {
  numLayers: number;
  // 레이어 이름별 true/false 값 딕셔너리 추가
}

export interface DynamicBlockData extends BaseNodeData {
  numLayers: number;
}

export interface TestBlockData extends BaseNodeData {
  numHeads: number;
  sdpAttention?: SDPAttentionData | null;
}
