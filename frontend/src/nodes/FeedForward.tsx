import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { FeedForwardData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const getFields = (data: FeedForwardData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Input Dimension:',
    name: 'inDim',
    value: data.inDim?.toString() || '',
    placeholder: 'Enter input dimension',
  },
  {
    type: 'number',
    label: 'Number of Factor:',
    name: 'numOfFactor',
    value: data.numOfFactor?.toString() || '',
    placeholder: 'Enter number of factor',
  },
  {
    type: 'select',
    label: 'Activation Function Type:',
    name: 'actFunc',
    value: data.actFunc || '',
    options: actFuncTypeOptions,
  },
];

interface FeedForwardLayerProps {
  id: string;
}

const actFuncTypeOptions: string[] = ['ReLU', 'GELU', 'SwiGLU'];

export const FeedForwardLayer: React.FC<FeedForwardLayerProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as FeedForwardData;

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof FeedForwardData, value: string) => {
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
  } = useCommonNodeActions<FeedForwardData>({
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
              handleFieldChange(name as keyof FeedForwardData, value)
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

export default FeedForwardLayer;
