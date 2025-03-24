import React, { useCallback, useState, useEffect } from 'react';
import { useReactFlow } from 'reactflow';
import NodeWrapper from './NodeWrapper';
import { NodeTitle } from './NodeComponents';
import { BaseNodeData, SDPAttentionData, MaskedMHABlockData } from './NodeData';
import SDPAttentionLayer from './SDPAttention';

interface MaskedMHABlockProps {
  data: MaskedMHABlockData;
  onChange?: (newData: MaskedMHABlockData) => void;
}

interface SlotProps {
  allowedType: string; // 여기서는 "sdpAttention"만 허용
  slotLabel: string;
  data: BaseNodeData | null;
  onChange: (newData: BaseNodeData | null) => void;
}

const Slot: React.FC<SlotProps> = ({
  allowedType,
  slotLabel,
  data,
  onChange,
}) => {
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const raw = e.dataTransfer.getData('application/reactflow');
      if (!raw) return;
      const dropped = JSON.parse(raw);

      if (dropped.nodeType !== allowedType) {
        alert(`이 슬롯에는 ${allowedType} 노드만 드롭 가능합니다!`);
        return;
      }

      const newData: BaseNodeData = {
        id: `${dropped.nodeType}-${Date.now()}`,
        label: dropped.label || dropped.nodeType,
        ...dropped,
      };

      onChange(newData);
    },
    [allowedType, onChange],
  );

  let content = (
    <div className="italic text-gray-400 text-sm">
      {slotLabel} (드래그 앤 드롭)
    </div>
  );
  if (data) {
    // allowedType === "sdpAttention"인 경우에만 SPDAttentionLayer를 렌더링
    switch (allowedType) {
      case 'sdpAttention':
        content = (
          <SDPAttentionLayer
            data={data as SDPAttentionData}
            onChange={(nd: SDPAttentionData) => onChange(nd)}
          />
        );
        break;
      default:
        content = <div>Unknown node type</div>;
        break;
    }
  }

  return (
    <div
      className="my-2 p-2 w-full bg-transparent border-dashed border-2 border-gray-200 rounded"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {content}
    </div>
  );
};

const MaskedMHABlock: React.FC<MaskedMHABlockProps> = ({ data }) => {
  const { setNodes } = useReactFlow();

  // 단 하나의 슬롯 상태: SPDAttention
  const [sdpAttention, setSdpAttention] = useState<SDPAttentionData | null>(
    data.sdpAttention || null,
  );

  // 블록 자체의 inDim/outDim 업데이트:
  // sdpAttention 슬롯의 inDim을 블록의 inDim, outDim을 블록의 outDim으로 사용
  useEffect(() => {
    if (sdpAttention) {
      const newInDim = sdpAttention.inDim;
      // 가정: sdpAttention에 outDim 필드가 존재함
      const newOutDim = (sdpAttention as { outDim: number }).outDim;
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
                },
              };
            }
            return node;
          }),
        );
      }
    }
  }, [sdpAttention, data.id, setNodes]);

  return (
    <NodeWrapper hideHandles={true}>
      <NodeTitle>{data.label}</NodeTitle>
      <div className="flex flex-col items-center gap-2 mt-2 w-56">
        <Slot
          allowedType="sdpAttention"
          slotLabel="SPDAttention"
          data={sdpAttention}
          onChange={(nd) => setSdpAttention(nd as SDPAttentionData)}
        />
      </div>
    </NodeWrapper>
  );
};

export default MaskedMHABlock;
