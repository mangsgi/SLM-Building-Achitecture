import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { NormalizationData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const normTypeOptions: string[] = ['Layer Normalization', 'RMS Normalization'];

const getFields = (data: NormalizationData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Input Dimension:',
    name: 'inDim',
    value: data.inDim?.toString() || '',
    placeholder: 'Enter input dimension',
  },
  {
    type: 'select',
    label: 'Normalization Type:',
    name: 'normType',
    value: data.normType || 'Layer Normalization',
    options: normTypeOptions,
  },
];

interface NormalizationLayerProps {
  id: string;
}

export const NormalizationLayer: React.FC<NormalizationLayerProps> = ({
  id,
}) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as NormalizationData;

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof NormalizationData, value: string) => {
    const stringFields: (keyof NormalizationData)[] = ['label', 'normType'];
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
  } = useCommonNodeActions<NormalizationData>({
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
              handleFieldChange(name as keyof NormalizationData, value)
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
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {currentData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default NormalizationLayer;
