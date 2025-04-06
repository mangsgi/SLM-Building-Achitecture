import React, { useState } from 'react';
import { useReactFlow, NodeProps } from 'reactflow';

import { NodeTitle } from './components/Components';
import { SDPAttentionData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const getFields = (data: SDPAttentionData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Input Dimension:',
    name: 'inDim',
    value: data.inDim?.toString() || '',
    placeholder: 'Enter input dimension',
  },
  {
    type: 'number',
    label: 'Output Dimension:',
    name: 'outDim',
    value: data.outDim?.toString() || '',
    placeholder: 'Enter output dimension',
  },
  {
    type: 'number',
    label: 'Dropout Rate:',
    name: 'dropoutRate',
    value: data.dropoutRate?.toString() || '',
    placeholder: 'Enter dropout rate',
  },
  {
    type: 'number',
    label: 'Context Length:',
    name: 'ctxLength',
    value: data.ctxLength?.toString() || '',
    placeholder: 'Enter context length',
  },
];

interface SDPAttentionLayerProps {
  id: string;
}

export const SDPAttentionLayer: React.FC<NodeProps<SDPAttentionLayerProps>> = ({
  id,
}) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const handleNodeClick = () => {
    setIsCollapsed((prev) => !prev);
  };

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as SDPAttentionData;

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof SDPAttentionData, value: string) => {
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
  } = useCommonNodeActions<SDPAttentionData>({
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
              handleFieldChange(name as keyof SDPAttentionData, value)
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

export default SDPAttentionLayer;
