import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/FieldComponents';
import { PositionalEmbeddingData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import FieldRenderer from './components/FieldRenderer';
import { nodeInfo } from './components/NodeInfo';
import { nodeRegistry } from './components/nodeRegistry';

interface PositionalEmbeddingLayerProps {
  id: string;
}

export const PositionalEmbeddingLayer: React.FC<
  PositionalEmbeddingLayerProps
> = ({ id }) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const typedData = node.type as string;

  // input 값 변경 시, 노드의 data에 직접 업데이트 + string 처리 for select
  const handleFieldChange = (
    field: keyof PositionalEmbeddingData,
    value: string,
  ) => {
    const stringFields = nodeRegistry.get(typedData)?.stringFields ?? [];
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
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
    handleInfoClick,
    handleLockToggle,
  } = useCommonNodeActions<PositionalEmbeddingData>({ id, setEditMode });

  return (
    <LayerWrapper hideHandles={node.data.hideHandles}>
      <div className="relative group">
        <NodeTitle onClick={handleNodeClick}>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          isLocked={node.data.isLocked}
          onInfo={() => handleInfoClick(nodeInfo.positionalEmbedding)}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
          onLockToggle={handleLockToggle}
        />
        {/* isCollapsed가 false일 때만 필드 보여줌 */}
        {!node.data.isCollapsed && (
          <FieldRenderer
            fields={nodeRegistry.get(typedData)?.getFields(node.data) ?? []}
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
    </LayerWrapper>
  );
};

export default PositionalEmbeddingLayer;
