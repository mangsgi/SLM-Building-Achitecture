// HTML Select 태그의 value와 label 리스트를 지정하기 위해 사용
export interface Option {
  value: string;
  label: string;
}

// 모든 노드에 공통으로 필요한 속성을 포함하는 Base interface
export interface BaseNodeData {
  id?: string;
  label: string;
  [key: string]: unknown; // Index Signature, 미리 명시하지 않은 추가 속성들이 있을 수 있음을 나타냄, 서로 다른 추가 데이터를 포함할 수 있도록 유연성을 제공
}

/* 
  Layer의 특성에 따라 Node Data 확장 interface
*/
export interface TokenEmbeddingData extends BaseNodeData {
  vocabSize: number;
  embDim: number;
}

export interface PositionalEmbeddingData extends BaseNodeData {
  ctxLength: number;
  embDim: number;
  posEmbeddingType: string;
}

export interface LayerNormData extends BaseNodeData {
  inDim: number;
  outDim: number;
}
