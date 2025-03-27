import React, { useState, useEffect } from 'react';
import { useReactFlow } from 'reactflow';
import { BlockWrapper } from './NodeWrapper';
import { NodeTitle } from './NodeComponents';
import { SDPAttentionData, MaskedMHABlockData } from './NodeData';
import SDPAttentionLayer from './SDPAttention';
import NodeSlot from './NodeSlot';

const MaskedMHABlock: React.FC<{ data: MaskedMHABlockData }> = ({ data }) => {
  const { setNodes } = useReactFlow();
  const [sdpAttention, setSdpAttention] = useState<SDPAttentionData | null>(
    data.sdpAttention || null,
  );

  // 블록 내부 변수: n_heads, inDim, outDim
  const [nHeads, setNHeads] = useState<number>(data.numHeads || 0);
  const [blockInDim, setBlockInDim] = useState<number | undefined>(data.inDim);
  const [blockOutDim, setBlockOutDim] = useState<number | undefined>(
    data.outDim,
  );

  // 내부 노드의 값이 변경되면 부모 노드 데이터도 업데이트
  useEffect(() => {
    if (sdpAttention) {
      const newInDim = sdpAttention.inDim;
      const newOutDim = sdpAttention.outDim;
      const candidate = sdpAttention.n_heads;
      const newNHeads = typeof candidate === 'number' ? candidate : nHeads;
      setBlockInDim(newInDim);
      setBlockOutDim(newOutDim);
      setNHeads(newNHeads);

      if (data.id) {
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === data.id) {
              return {
                ...node,
                data: {
                  ...node.data,
                  inDim: newInDim,
                  outDim: newOutDim,
                  n_heads: newNHeads,
                },
              };
            }
            return node;
          }),
        );
      }
    } else {
      setBlockInDim(undefined);
      setBlockOutDim(undefined);
      setNHeads(0);
    }
  }, [sdpAttention, data.id, setNodes, nHeads]);

  return (
    <BlockWrapper hideHandles={true}>
      <NodeTitle>{data.label}</NodeTitle>
      <div className="text-sm mb-2">
        <div>Heads: {nHeads}</div>
        <div>inDim: {blockInDim ?? '-'}</div>
        <div>outDim: {blockOutDim ?? '-'}</div>
      </div>
      <div className="flex flex-col items-center gap-2 mt-2 w-full">
        <NodeSlot<SDPAttentionData>
          slotLabel="SDPAttention (드래그 앤 드롭)"
          data={sdpAttention}
          onChange={(nd) => setSdpAttention(nd)}
          nodeComponent={SDPAttentionLayer}
          allowedTypes={['sdpAttention']}
        />
      </div>
    </BlockWrapper>
  );
};

export default MaskedMHABlock;
