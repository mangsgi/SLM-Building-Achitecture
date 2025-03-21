import React, { useCallback, useState } from 'react';
import NodeWrapper from './NodeWrapper';
import { NodeTitle } from './NodeComponents';

import {
  BaseNodeData,
  DropoutData,
  FeedForwardData,
  LayerNormData,
  MaskedMHAData,
} from './NodeData';

// TransformerBlock이 가지는 6개 슬롯
interface TransformerBlockData {
  label?: string;
  dropout1?: DropoutData | null;
  feedForward?: FeedForwardData | null;
  layerNorm2?: LayerNormData | null;
  dropout2?: DropoutData | null;
  maskedMHA?: MaskedMHAData | null;
  layerNorm1?: LayerNormData | null;
}

interface TransformerBlockProps {
  data: TransformerBlockData;
}

// 각 노드 편집 UI를 간단히 구현
// ------------------------------

// 1) Dropout Editor
const DropoutEditor: React.FC<{
  data: DropoutData;
  onChange: (newData: DropoutData) => void;
}> = ({ data, onChange }) => {
  const [editMode, setEditMode] = useState(false);
  const [inDim, setInDim] = useState(data.inDim.toString());
  const [outDim, setOutDim] = useState(data.outDim.toString());
  const [dropoutRate, setDropoutRate] = useState(data.dropoutRate.toString());

  const handleSave = () => {
    onChange({
      ...data,
      inDim: Number(inDim),
      outDim: Number(outDim),
      dropoutRate: Number(dropoutRate),
    });
    setEditMode(false);
  };

  return (
    <div className="p-2 border bg-white rounded w-full">
      <div className="font-bold">{data.label || 'Dropout'}</div>
      {editMode ? (
        <div className="mt-1">
          <label className="text-sm">inDim</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            value={inDim}
            onChange={(e) => setInDim(e.target.value)}
          />
          <label className="text-sm">outDim</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            value={outDim}
            onChange={(e) => setOutDim(e.target.value)}
          />
          <label className="text-sm">dropoutRate</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            step="0.01"
            value={dropoutRate}
            onChange={(e) => setDropoutRate(e.target.value)}
          />
          <button
            className="mt-2 bg-green-100 px-2 py-1 rounded text-sm"
            onClick={handleSave}
          >
            Save
          </button>
        </div>
      ) : (
        <div className="mt-1 text-sm">
          <p>inDim: {data.inDim}</p>
          <p>outDim: {data.outDim}</p>
          <p>dropoutRate: {data.dropoutRate}</p>
          <button
            className="mt-2 bg-blue-100 px-2 py-1 rounded text-sm"
            onClick={() => setEditMode(true)}
          >
            Edit
          </button>
        </div>
      )}
    </div>
  );
};

// 4) Masked Multi-Head Attention Editor
const MaskedMHAEditor: React.FC<{
  data: MaskedMHAData;
  onChange: (newData: MaskedMHAData) => void;
}> = ({ data, onChange }) => {
  const [editMode, setEditMode] = useState(false);
  const [inDim, setInDim] = useState(data.inDim.toString());
  const [outDim, setOutDim] = useState(data.outDim.toString());
  const [numHeads, setNumHeads] = useState(data.numHeads.toString());

  const handleSave = () => {
    onChange({
      ...data,
      inDim: Number(inDim),
      outDim: Number(outDim),
      numHeads: Number(numHeads),
    });
    setEditMode(false);
  };

  return (
    <div className="p-2 border bg-gray-200 rounded w-full">
      <div className="font-bold">
        {data.label || 'Masked Multi-Head Attention'}
      </div>
      {editMode ? (
        <div className="mt-1 text-sm">
          <label>inDim</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            value={inDim}
            onChange={(e) => setInDim(e.target.value)}
          />
          <label>outDim</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            value={outDim}
            onChange={(e) => setOutDim(e.target.value)}
          />
          <label>numHeads</label>
          <input
            className="border rounded p-1 w-full text-sm"
            type="number"
            value={numHeads}
            onChange={(e) => setNumHeads(e.target.value)}
          />
          <button
            className="mt-2 bg-green-100 px-2 py-1 rounded text-sm"
            onClick={handleSave}
          >
            Save
          </button>
        </div>
      ) : (
        <div className="mt-1 text-sm">
          <p>inDim: {data.inDim}</p>
          <p>outDim: {data.outDim}</p>
          <p>numHeads: {data.numHeads}</p>
          <button
            className="mt-2 bg-blue-100 px-2 py-1 rounded text-sm"
            onClick={() => setEditMode(true)}
          >
            Edit
          </button>
        </div>
      )}
    </div>
  );
};

// ----------------------------------------------------

// Slot 컴포넌트: 특정 노드 타입만 드롭받아서 상태를 저장/편집
interface SlotProps {
  allowedType: string; // 예: 'dropout', 'feedForward', 'layerNorm', 'maskedMultiHeadAttention'
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
          <DropoutEditor
            data={data as DropoutData}
            onChange={(nd) => onChange(nd)}
          />
        );
        break;
      case 'feedForward':
        // content = (
        //   <FeedForwardEditor
        //     data={data as FeedForwardData}
        //     onChange={(nd) => onChange(nd)}
        //   />
        // );
        break;
      case 'layerNorm':
        // content = (
        //   <LayerNormEditor
        //     data={data as LayerNormData}
        //     onChange={(nd) => onChange(nd)}
        //   />
        // );
        break;
      case 'maskedMultiHeadAttention':
        content = (
          <MaskedMHAEditor
            data={data as MaskedMHAData}
            onChange={(nd) => onChange(nd)}
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

// ----------------------------------------------------

// 최종 TransformerBlock 컴포넌트
const TransformerBlock: React.FC<TransformerBlockProps> = ({ data }) => {
  // 6개 슬롯 상태
  const [dropout1, setDropout1] = useState<DropoutData | null>(
    data.dropout1 || null,
  );
  const [feedForward, setFeedForward] = useState<FeedForwardData | null>(
    data.feedForward || null,
  );
  const [layerNorm2, setLayerNorm2] = useState<LayerNormData | null>(
    data.layerNorm2 || null,
  );
  const [dropout2, setDropout2] = useState<DropoutData | null>(
    data.dropout2 || null,
  );
  const [maskedMHA, setMaskedMHA] = useState<MaskedMHAData | null>(
    data.maskedMHA || null,
  );
  const [layerNorm1, setLayerNorm1] = useState<LayerNormData | null>(
    data.layerNorm1 || null,
  );

  return (
    <NodeWrapper>
      <NodeTitle>{data.label || 'Transformer Block'}</NodeTitle>
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
          onChange={(nd) => setLayerNorm2(nd as LayerNormData)}
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
          allowedType="maskedMultiHeadAttention"
          slotLabel="Masked MHA"
          data={maskedMHA}
          onChange={(nd) => setMaskedMHA(nd as MaskedMHAData)}
        />

        {/* LayerNorm 1 */}
        <Slot
          allowedType="layerNorm"
          slotLabel="LayerNorm 1"
          data={layerNorm1}
          onChange={(nd) => setLayerNorm1(nd as LayerNormData)}
        />
      </div>
    </NodeWrapper>
  );
};

export default TransformerBlock;
