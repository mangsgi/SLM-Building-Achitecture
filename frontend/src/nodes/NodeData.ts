export interface NodeData {
  id?: string; // 노드 id (없을 수도 있음)
  label: string;
  inDim: number;
  outDim: number;
  [key: string]: unknown;
}

export interface MultiHeadAttentionData extends NodeData {
  numHeads: number;
}

export interface DropoutData extends NodeData {
  dropoutRate: number;
}
