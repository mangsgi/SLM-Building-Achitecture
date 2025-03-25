import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './NodeComponents';
import { TokenEmbeddingData } from './NodeData';
import { LayerWrapper } from './NodeWrapper';
import NodeActionPanel from './NodeActionPanel';

export const TokenEmbeddingLayer: React.FC<{ data: TokenEmbeddingData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  // TokenEmbeddingData 상태변수 저장
  const [vocabSizeStr, setVocabSizeStr] = useState<string>(
    initialData.vocabSize !== undefined ? initialData.vocabSize.toString() : '',
  );
  const [embDimStr, setEmbDimStr] = useState<string>(
    initialData.embDim !== undefined ? initialData.embDim.toString() : '',
  );

  // Edit 버튼 클릭
  const handleEditClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    // 이벤트 버블링을 중단해서 부모 컴포넌트로 전파되지 않도록 함
    e.stopPropagation();
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(false);

    const newVocabSize =
      vocabSizeStr === '' ? initialData.vocabSize : Number(vocabSizeStr);
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
    console.log('Info icon clicked');
    // 여기에 추가 동작을 구현하세요.
  };

  return (
    <LayerWrapper>
      <div className="relative group">
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
          </div>
        ) : (
          <div>
            <ReadField label="Vocabulary Size:" value={vocabSizeStr} />
            <ReadField label="Embedding Dimension Size:" value={embDimStr} />
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
    </LayerWrapper>
  );
};

export default TokenEmbeddingLayer;
