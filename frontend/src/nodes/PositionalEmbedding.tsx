import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { PositionalEmbeddingData } from './NodeData'; //, Option
import { LayerWrapper } from './NodeWrapper';
import {
  NodeTitle,
  ReadField,
  EditField,
  EditSelectField,
} from './NodeComponents';
import NodeActionPanel from './NodeActionPanel';
import NodeInfoModal from './NodeInfoModal';

const posTypeOptions: string[] = [
  'Learned Positional Embedding',
  'Sinusoidal Positional Embedding',
  'Relative Positional Embedding',
  'Rotary Positional Embedding',
];

export const PositionalEmbeddingLayer: React.FC<{
  data: PositionalEmbeddingData;
}> = ({ data: initialData }) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // PositionalEmbeddingData 상태변수 저장
  const [ctxLengthStr, setCtxLengthStr] = useState<string>(
    initialData.ctxLength !== undefined ? initialData.ctxLength.toString() : '',
  );
  const [embDimStr, setEmbDimStr] = useState<string>(
    initialData.embDim !== undefined ? initialData.embDim.toString() : '',
  );
  const [posType, setPosType] = useState<string>(
    initialData.posEmbeddingType !== undefined
      ? initialData.posEmbeddingType
      : posTypeOptions[0],
  );

  // Edit 버튼 클릭
  const handleEditClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(false);

    const newContextLength =
      ctxLengthStr === '' ? initialData.ctxLength : Number(ctxLengthStr);
    const newEmbDim = embDimStr === '' ? initialData.embDim : Number(embDimStr);

    // 노드 데이터 업데이트
    if (initialData.id) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ctxLength: newContextLength,
                embDim: newEmbDim,
                posEmbeddingType: posType,
              },
            };
          }
          return node;
        }),
      );
    }
  };

  // Delete 버튼 클릭
  const handleDeleteClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (initialData.id) {
      setNodes((nds) => nds.filter((node) => node.id !== initialData.id));
    }
  };

  // 정보 아이콘 클릭
  const handleInfoClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setIsInfoOpen(true);
    // 여기에 추가 동작을 구현하세요.
  };

  return (
    <LayerWrapper>
      <div className="relative group">
        <NodeTitle>{initialData.label}</NodeTitle>
        {editMode ? (
          <div>
            <EditField
              label="Context Length:"
              id="ctxLengthInput"
              name="ctxLength"
              value={ctxLengthStr}
              placeholder="Enter context length"
              onChange={setCtxLengthStr}
            />
            <EditField
              label="Embedding dimension size:"
              id="embDimSize"
              name="embDim"
              value={embDimStr}
              placeholder="Enter embedding dimension"
              onChange={setEmbDimStr}
            />
            <EditSelectField
              label="Positional Embedding Type:"
              id="posTypeSelect"
              name="posType"
              value={posType}
              onChange={setPosType}
              options={posTypeOptions}
            />
          </div>
        ) : (
          <div>
            <ReadField label="Context Length:" value={ctxLengthStr} />
            <ReadField label="Embedding Dimension Size:" value={embDimStr} />
            <ReadField label="Positional Embedding Type:" value={posType} />
          </div>
        )}
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
      </div>

      {/* InfoModal은 여기서 렌더링 */}
      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {initialData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default PositionalEmbeddingLayer;
