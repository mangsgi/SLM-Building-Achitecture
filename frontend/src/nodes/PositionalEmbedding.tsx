import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { PositionalEmbeddingData, Option } from './NodeData';
import NodeWrapper from './NodeWrapper';
import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
  EditSelectField,
} from './NodeComponents';

const posTypeOptions: Option[] = [
  {
    value: 'LearnedPositionalEmbedding',
    label: 'Learned Positional Embedding',
  },
  {
    value: 'SinusoidalPositionalEmbedding',
    label: 'Sinusoidal Positional Embedding',
  },
  {
    value: 'RelativePositionalEmbedding',
    label: 'Relative Positional Embedding',
  },
  { value: 'RotaryPositionalEmbedding', label: 'Rotary Positional Embedding' },
];

export const TokenEmbeddingLayer: React.FC<{
  data: PositionalEmbeddingData;
}> = ({ data: initialData }) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [ctxLengthStr, setCtxLengthStr] = useState<string>(
    initialData.ctxLength.toString(),
  );
  const [embDimStr, setEmbDimStr] = useState<string>(
    initialData.embDim.toString(),
  );
  const [posType, setPosType] = useState<string>(
    initialData.posEmbeddingType || 'LearnedPositionalEmbedding',
  );

  // Edit 버튼 클릭
  const handleEditClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    console.log('Edit button clicked');
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    const newContextLength =
      ctxLengthStr === '' ? initialData.ctxLength : Number(ctxLengthStr);
    const newEmbDim = embDimStr === '' ? initialData.embDim : Number(embDimStr);

    setEditMode(false);

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

  return (
    <NodeWrapper>
      <NodeTitle>{initialData.label}</NodeTitle>
      {editMode ? (
        <div>
          <EditField
            label="Context Length:"
            id="ctxLengthInput"
            name="ctxLength"
            value={ctxLengthStr}
            placeholder="Enter Context Length"
            onChange={setCtxLengthStr}
          />
          <EditField
            label="Embedding Dimension Size:"
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
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Context Length:" value={ctxLengthStr} />
          <ReadField label="Embedding Dimension Size:" value={embDimStr} />
          <ReadField
            label="Positional Embedding Type:"
            value={
              posTypeOptions.find((option) => option.value === posType)
                ?.label || posType
            }
          />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default TokenEmbeddingLayer;
