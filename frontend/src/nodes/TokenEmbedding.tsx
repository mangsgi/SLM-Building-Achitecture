import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { TokenEmbeddingData } from './NodeData';
import NodeWrapper from './NodeWrapper';

export const TokenEmbeddingLayer: React.FC<{ data: TokenEmbeddingData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const [vocabSizeStr, setVocabSizeStr] = useState<string>(
    initialData.vocabSize !== undefined ? initialData.vocabSize.toString() : '',
  );
  const [embDimStr, setEmbDimStr] = useState<string>(
    initialData.embDim !== undefined ? initialData.embDim.toString() : '',
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
    const newVocabSize =
      vocabSizeStr === '' ? initialData.vocabSize : Number(vocabSizeStr);
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
                vocabSize: newVocabSize,
                embDim: newEmbDim,
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
            label="Vocabulary Size:"
            id="vocabSizeInput"
            name="vocabSize"
            value={vocabSizeStr}
            placeholder="Enter Vocabulary Size"
            onChange={setVocabSizeStr}
          />
          <EditField
            label="Embedding Dimension Size:"
            id="embDimSize"
            name="embDim"
            value={embDimStr}
            placeholder="Enter embedding dimension"
            onChange={setEmbDimStr}
          />
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Vocabulary Size:" value={vocabSizeStr} />
          <ReadField label="Embedding Dimension Size:" value={embDimStr} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default TokenEmbeddingLayer;
