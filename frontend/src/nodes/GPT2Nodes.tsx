import React from 'react'; // useState
// import { useReactFlow } from 'reactflow';

export interface NodeData {
  id?: string; // id가 포함되어 있어야 함
  label: string;
  inDim: number;
  outDim: number;
  [key: string]: unknown;
}

/* ────── Attention ────── */
export interface MultiHeadAttentionData extends NodeData {
  numHeads: number;
}

export const MaskedMultiHeadAttention: React.FC<{
  data: MultiHeadAttentionData;
}> = ({ data: { label, inDim, outDim, numHeads } }) => (
  <div className="node border p-2 m-2 bg-white">
    <h3 className="font-bold">{label}</h3>
    <p>Input Dim: {inDim}</p>
    <p>Output Dim: {outDim}</p>
    <p>Heads: {numHeads}</p>
  </div>
);

/* ────── Feed Forward ────── */
export const FeedForward: React.FC<{ data: NodeData }> = ({
  data: { label, inDim, outDim },
}) => (
  <div className="node border p-2 m-2 bg-white">
    <h3 className="font-bold">{label}</h3>
    <p>Input Dim: {inDim}</p>
    <p>Output Dim: {outDim}</p>
  </div>
);

/* ────── Regularization ────── */
export interface DropoutData extends NodeData {
  dropoutRate: number;
}

export const Dropout: React.FC<{ data: DropoutData }> = ({
  data: { label, inDim, outDim, dropoutRate },
}) => (
  <div className="node border p-2 m-2 bg-white">
    <h3 className="font-bold">{label}</h3>
    <p>Input Dim: {inDim}</p>
    <p>Output Dim: {outDim}</p>
    <p>Rate: {dropoutRate}</p>
  </div>
);

/* ────── Output ────── */
export const LinearOutputLayer: React.FC<{ data: NodeData }> = ({
  data: { label, inDim, outDim },
}) => (
  <div className="node border p-2 m-2 bg-white">
    <h3 className="font-bold">{label}</h3>
    <p>Input Dim: {inDim}</p>
    <p>Output Dim: {outDim}</p>
  </div>
);
