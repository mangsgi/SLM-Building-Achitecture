// -------------------------------------------
// -------------- Node Heights ---------------
// -------------------------------------------

// 기본 노드 높이
export const NODE_HEIGHTS = {
  tokenEmbedding: 175,
  positionalEmbedding: 241,
  normalization: 109,
  feedForward: 240,
  dropout: 109,
  linear: 174,
  sdpAttention: 371,
  gqAttention: 371,
} as const;

// 기본 높이값 (노드 타입이 정의되지 않은 경우 사용)
export const DEFAULT_NODE_HEIGHT = 43;

// 노드 간 간격
export const NODE_GAP = 10;

// 블록 노드의 시작 y 위치
export const BLOCK_START_Y = 110;
