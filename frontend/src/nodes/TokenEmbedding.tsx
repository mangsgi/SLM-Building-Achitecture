import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './NodeComponents';
import { TokenEmbeddingData } from './NodeData';
import { LayerWrapper } from './NodeWrapper';
import NodeActionPanel from './NodeActionPanel';
import NodeInfoModal from './NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

export const TokenEmbeddingLayer: React.FC<{ data: TokenEmbeddingData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // TokenEmbeddingData 상태변수 저장
  const [vocabSizeStr, setVocabSizeStr] = useState<string>(
    initialData.vocabSize !== undefined ? initialData.vocabSize.toString() : '',
  );
  const [embDimStr, setEmbDimStr] = useState<string>(
    initialData.embDim !== undefined ? initialData.embDim.toString() : '',
  );

  // Save 버튼에 들어갈 Custom Save
  const customSave = () => {
    const newVocabSize =
      vocabSizeStr === '' ? initialData.vocabSize : Number(vocabSizeStr);
    const newEmbDim = embDimStr === '' ? initialData.embDim : Number(embDimStr);

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

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<TokenEmbeddingData>({
    initialData,
    setNodes,
    setEditMode,
    customSave,
  });

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

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {initialData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default TokenEmbeddingLayer;
