// 모든 노드에 공통으로 필요한 속성을 포함하는 Base interface
export interface BaseNodeData {
  label: string; // Node 이름
  openModal?: (nodeData: BaseNodeData) => void;
  hideHandles?: boolean;
  inDim?: number; // 입력 차원
  outDim?: number; // 출력 차원
  [key: string]: unknown; // Index Signature, 미리 명시하지 않은 추가 속성들이 있을 수 있음을 나타냄, 서로 다른 추가 데이터를 포함할 수 있도록 유연성을 제공
}

//  Layer의 특성에 따라 BaseNodeData를 확장한 interface
export interface TokenEmbeddingData extends BaseNodeData {
  vocabSize: number;
  embDim: number;
}

export interface PositionalEmbeddingData extends BaseNodeData {
  ctxLength: number;
  embDim: number;
  posEmbeddingType: string;
}

export interface FeedForwardData extends BaseNodeData {
  numOfFactor: number;
  actFunc: string;
}

export interface DropoutData extends BaseNodeData {
  dropoutRate: number;
}

export interface SDPAttentionData extends BaseNodeData {
  dropoutRate: number;
  ctxLength: number;
}

export interface MaskedMHABlockData extends BaseNodeData {
  numHeads: number;
  sdpAttention?: SDPAttentionData | null;
}

export interface TransformerBlockData extends BaseNodeData {
  dropout1?: DropoutData | null;
  feedForward?: FeedForwardData | null;
  layerNorm2?: BaseNodeData | null;
  dropout2?: DropoutData | null;
  maskedMHA?: MaskedMHABlockData | null;
  layerNorm1?: BaseNodeData | null;
  numLayers: number;
}
