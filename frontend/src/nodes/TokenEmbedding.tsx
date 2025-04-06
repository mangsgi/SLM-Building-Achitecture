import React, { useState } from 'react';
import { useReactFlow, NodeProps } from 'reactflow';

import { NodeTitle } from './components/Components';
import { TokenEmbeddingData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const getFields = (data: TokenEmbeddingData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Vocabulary Size:',
    name: 'vocabSize',
    value: data.vocabSize?.toString() || '',
    placeholder: 'Enter vocabulary size',
  },
  {
    type: 'number',
    label: 'Embedding Dimension Size:',
    name: 'embDim',
    value: data.embDim?.toString() || '',
    placeholder: 'Enter embedding dimension',
  },
];

interface TokenEmbeddingLayerProps {
  id: string;
}

export const TokenEmbeddingLayer: React.FC<
  NodeProps<TokenEmbeddingLayerProps>
> = ({ id }) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as TokenEmbeddingData;

  // input 값 변경 시, 노드의 data에 직접 업데이트트
  const handleFieldChange = (
    field: keyof TokenEmbeddingData,
    value: string,
  ) => {
    const newValue = field === 'label' ? value : Number(value);
    setNodes((nds) =>
      nds.map((nodeItem) => {
        if (nodeItem.id === id) {
          return {
            ...nodeItem,
            data: {
              ...nodeItem.data,
              [field]: newValue,
            },
          };
        }
        return nodeItem;
      }),
    );
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
  } = useCommonNodeActions<TokenEmbeddingData>({
    id,
    setNodes,
    setEditMode,
    setIsCollapsed,
  });

  return (
    <LayerWrapper hideHandles={currentData.hideHandles}>
      <div className="relative group">
        <NodeTitle onClick={handleNodeClick}>{currentData.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
        {/* Collapse가 아닐 때만 필드 보여줌 */}
        {!isCollapsed && (
          <FieldRenderer
            fields={getFields(currentData)}
            editMode={editMode}
            onChange={(name: string, value: string) =>
              handleFieldChange(name as keyof TokenEmbeddingData, value)
            }
          />
        )}
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {currentData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default TokenEmbeddingLayer;
