import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { PositionalEmbeddingData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';
import { nodeInfo, nodeFieldInfo } from './components/NodeInfo';

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
    info: nodeFieldInfo.positionalEmbedding.ctxLength,
  },
  {
    type: 'number',
    label: 'Embedding Dimension Size:',
    name: 'embDim',
    value: data.embDim?.toString() || '',
    placeholder: 'Enter embedding dimension',
    info: nodeFieldInfo.positionalEmbedding.embDim,
  },
  {
    type: 'select',
    label: 'Positional Embedding Type:',
    name: 'posType',
    value: data.posType || 'Learned Positional Embedding',
    options: posTypeOptions,
    info: nodeFieldInfo.positionalEmbedding.posType,
  },
];

interface PositionalEmbeddingLayerProps {
  id: string;
}

export const PositionalEmbeddingLayer: React.FC<
  PositionalEmbeddingLayerProps
> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as PositionalEmbeddingData;

  // input 값 변경 시, 노드의 data에 직접 업데이트 + string 처리 for select
  const handleFieldChange = (
    field: keyof PositionalEmbeddingData,
    value: string,
  ) => {
    const stringFields: (keyof PositionalEmbeddingData)[] = [
      'label',
      'posType',
    ];
    const newValue = stringFields.includes(field) ? value : Number(value);
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
  } = useCommonNodeActions<PositionalEmbeddingData>({
    id,
    setNodes,
    setEditMode,
    setIsCollapsed,
    setEdges,
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
            onInfoClick={(info) => {
              // FlowCanvas의 필드 정보 모달을 열기 위한 이벤트 발생
              const event = new CustomEvent('fieldInfo', { detail: info });
              window.dispatchEvent(event);
            }}
          />
        )}
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">
          {nodeInfo.positionalEmbedding.title}
        </h3>
        <p className="text-sm">{nodeInfo.positionalEmbedding.description}</p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default PositionalEmbeddingLayer;
