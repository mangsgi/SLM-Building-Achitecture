import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { PositionalEmbeddingData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const posTypeOptions: string[] = [
  'Learned Positional Embedding',
  'Sinusoidal Positional Embedding',
  'Relative Positional Embedding',
  'Rotary Positional Embedding',
];

const getFields = (data: PositionalEmbeddingData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Context Length:',
    name: 'ctxLength',
    value: data.ctxLength?.toString() || '',
    placeholder: 'Enter context length',
  },
  {
    type: 'number',
    label: 'Embedding Dimension Size:',
    name: 'embDim',
    value: data.embDim?.toString() || '',
    placeholder: 'Enter embedding dimension',
  },
  {
    type: 'select',
    label: 'Positional Embedding Type:',
    name: 'posType',
    value: (data.posType || '') as string,
    options: posTypeOptions,
  },
];

interface PositionalEmbeddingLayerProps {
  id: string;
}

export const PositionalEmbeddingLayer: React.FC<
  PositionalEmbeddingLayerProps
> = ({ id }) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const handleNodeClick = () => {
    setIsCollapsed((prev) => !prev);
  };

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as PositionalEmbeddingData;

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (
    field: keyof PositionalEmbeddingData,
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
  } = useCommonNodeActions<PositionalEmbeddingData>({
    currentData,
    setNodes,
    setEditMode,
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
              handleFieldChange(name as keyof PositionalEmbeddingData, value)
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

export default PositionalEmbeddingLayer;
