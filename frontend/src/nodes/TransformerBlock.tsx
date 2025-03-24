import React, { useCallback, useState, useEffect } from 'react';
import { useReactFlow, Edge } from 'reactflow';

import NodeWrapper from './NodeWrapper';
import { NodeTitle } from './NodeComponents';
import {
  BaseNodeData,
  DropoutData,
  FeedForwardData,
  MaskedMHABlockData,
  TransformerBlockData,
} from './NodeData';
import DropoutLayer from './Dropout';
import FeedForwardLayer from './FeedForward';
import LayerNormLayer from './LayerNorm';
import MaskedMHABlock from './MaskedMHABlock';

interface TransformerBlockProps {
  data: TransformerBlockData;
}

// Slot 컴포넌트: 특정 노드 타입만 드롭받아서 상태를 저장/편집
interface SlotProps {
  allowedType: string; // e.g. 'dropout', 'feedForward', 'layerNorm', 'maskedMultiHeadAttention'
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
      e.stopPropagation(); // 이벤트 전파 차단 추가

      const raw = e.dataTransfer.getData('application/reactflow');
      if (!raw) return;
      const dropped = JSON.parse(raw);

      // dropped.nodeType이 slot이 허용하는 타입인지 확인
      if (dropped.nodeType !== allowedType) {
        alert(`이 슬롯에는 ${allowedType} 노드만 드롭 가능합니다!`);
        return;
      }

      // 기존 NodeData에서 필요한 필드만 추출
      // Sidebar에서 넘어오는 nodeData 구조에 맞춰서 필요 시 수정
      const newData: BaseNodeData = {
        id: `${dropped.nodeType}-${Date.now()}`,
        label: dropped.label || dropped.nodeType,
        ...dropped, // inDim, outDim, dropoutRate 등
      };

      onChange(newData);
    },
    [allowedType, onChange],
  );

  // 노드가 없으면 "드롭 영역" 표시, 있으면 해당 노드의 Editor를 렌더링
  let content = (
    <div className="italic text-gray-400 text-sm">
      {slotLabel} (드래그 앤 드롭)
    </div>
  );
  if (data) {
    // allowedType에 따라 Editor를 스위치
    switch (allowedType) {
      case 'dropout':
        content = (
          <DropoutLayer
            data={data as DropoutData}
            onChange={(nd) => onChange(nd)}
          />
        );
        break;
      case 'feedForward':
        content = (
          <FeedForwardLayer
            data={data as FeedForwardData}
            onChange={(nd) => onChange(nd)}
          />
        );
        break;
      case 'layerNorm':
        content = (
          <LayerNormLayer
            data={data as BaseNodeData}
            onChange={(nd: BaseNodeData) => onChange(nd)}
          />
        );
        break;
      case 'maskedMHABlock':
        content = (
          <MaskedMHABlock
            data={data as MaskedMHABlockData}
            onChange={(nd: MaskedMHABlockData) => onChange(nd)}
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

// 최종 TransformerBlock 컴포넌트
const TransformerBlock: React.FC<TransformerBlockProps> = ({ data }) => {
  const { getEdges, setEdges, setNodes, getNode } = useReactFlow();

  // 6개 슬롯 useState
  const [dropout1, setDropout1] = useState<DropoutData | null>(
    data.dropout1 || null,
  );
  const [feedForward, setFeedForward] = useState<FeedForwardData | null>(
    data.feedForward || null,
  );
  const [layerNorm2, setLayerNorm2] = useState<BaseNodeData | null>(
    data.layerNorm2 || null,
  );
  const [dropout2, setDropout2] = useState<DropoutData | null>(
    data.dropout2 || null,
  );
  const [maskedMHA, setMaskedMHA] = useState<MaskedMHABlockData | null>(
    data.maskedMHA || null,
  );
  const [layerNorm1, setLayerNorm1] = useState<BaseNodeData | null>(
    data.layerNorm1 || null,
  );

  // const [numLayers, setNumLayers] = useState<number>(data.numLayers);

  // 프로그램적으로 내부 슬롯들을 순서대로 연결하는 Internal Edge 생성
  // 내부 edge 업데이트: 이전 내부 edge와 비교하여 변경된 경우에만 업데이트
  useEffect(() => {
    const newInternalEdges: Edge[] = [];
    // Top down: Dropout1 ← FeedForward ← LayerNorm2 ← Dropout2 ← MaskedMHA ← LayerNorm1
    const slotNodes = [
      dropout1,
      feedForward,
      layerNorm2,
      dropout2,
      maskedMHA,
      layerNorm1,
    ];
    for (let i = 0; i < slotNodes.length - 1; i++) {
      const source = slotNodes[i];
      const target = slotNodes[i + 1];
      // source/target이 not-null일 때 Edge를 push
      if (source && target && source.id && target.id) {
        newInternalEdges.push({
          id: `internal-${data.id}-${source.id}-${target.id}`,
          source: source.id,
          target: target.id,
          type: 'default',
          style: { stroke: 'transparent' },
        });
      }
    }

    // 기존 글로벌 edge 중 이 TransformerBlock에 속하는 internal edge를 제거 후 새로 추가
    const existingEdges = getEdges();
    const filteredEdges = existingEdges.filter(
      (edge) => !edge.id.startsWith(`internal-${data.id}-`),
    );
    setEdges([...filteredEdges, ...newInternalEdges]);
  }, [
    dropout1,
    feedForward,
    layerNorm2,
    dropout2,
    maskedMHA,
    layerNorm1,
    data.id,
    getEdges,
    setEdges,
  ]);

  // TransformerBlock 자체의 inDim/outDim 업데이트:
  // 첫 슬롯(dropout1)의 inDim을 블록의 inDim, 마지막 슬롯(layerNorm1)의 outDim을 블록의 outDim으로 사용
  useEffect(() => {
    if (dropout1 && layerNorm1 && dropout1.inDim !== undefined) {
      const newInDim = dropout1.inDim;
      const newOutDim = (layerNorm1 as { outDim: number }).outDim;
      const currentNode = getNode(data.id!);
      if (
        currentNode &&
        (currentNode.data.inDim !== newInDim ||
          currentNode.data.outDim !== newOutDim)
      ) {
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
  }, [dropout1, layerNorm1, data.id, setNodes, getNode]);

  // numLayers 업데이트
  // useEffect(() => {
  //   if (data.id) {
  //     setNodes((nds) =>
  //       nds.map((node) => {
  //         if (node.id === data.id) {
  //           return {
  //             ...node,
  //             data: {
  //               ...node.data,
  //               numLayers: numLayers,
  //             },
  //           };
  //         }
  //         return node;
  //       }),
  //     );
  //   }
  // }, [numLayers, data.id, setNodes]);

  return (
    <NodeWrapper hideHandles={true}>
      <NodeTitle>{data.label}</NodeTitle>
      {/* 그림에 나온 순서대로 6개 슬롯 배치 (필요시 CSS로 정렬) */}
      <div className="flex flex-col items-center gap-2 mt-2 w-56">
        {/* Dropout 1 */}
        <Slot
          allowedType="dropout"
          slotLabel="Dropout 1"
          data={dropout1}
          onChange={(nd) => setDropout1(nd as DropoutData)}
        />

        {/* FeedForward */}
        <Slot
          allowedType="feedForward"
          slotLabel="FeedForward"
          data={feedForward}
          onChange={(nd) => setFeedForward(nd as FeedForwardData)}
        />

        {/* LayerNorm 2 */}
        <Slot
          allowedType="layerNorm"
          slotLabel="LayerNorm 2"
          data={layerNorm2}
          onChange={(nd) => setLayerNorm2(nd as BaseNodeData)}
        />

        {/* Dropout 2 */}
        <Slot
          allowedType="dropout"
          slotLabel="Dropout 2"
          data={dropout2}
          onChange={(nd) => setDropout2(nd as DropoutData)}
        />

        {/* Masked Multi-Head Attention */}
        <Slot
          allowedType="maskedMHABlock"
          slotLabel="Masked MHA"
          data={maskedMHA}
          onChange={(nd) => setMaskedMHA(nd as MaskedMHABlockData)}
        />

        {/* LayerNorm 1 */}
        <Slot
          allowedType="layerNorm"
          slotLabel="LayerNorm 1"
          data={layerNorm1}
          onChange={(nd) => setLayerNorm1(nd as BaseNodeData)}
        />
      </div>
    </NodeWrapper>
  );
};

export default TransformerBlock;
