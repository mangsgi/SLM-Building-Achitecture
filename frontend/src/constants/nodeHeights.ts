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
  mhAttention: 241,
  gqAttention: 241,
} as const;

// 기본 높이값 (노드가 접혀있을 때의 높이)
export const DEFAULT_NODE_HEIGHT = 43;
// 기본 블록 노드 높이 (블록 노드 내 자식 노드가 없을 때의 높이)
export const DEFAULT_BLOCK_NODE_HEIGHT = 90;
// 노드 간 간격 (블록 노드 내 자식 노드 간 간격)
export const NODE_GAP = 10;

// 블록 노드의 시작 y 위치
export const BLOCK_START_Y = 110;
