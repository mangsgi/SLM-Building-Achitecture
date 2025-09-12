import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/FieldComponents';
import { MHAttentionData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import FieldRenderer from './components/FieldRenderer';
import { nodeInfo } from './components/NodeInfo';
import { nodeRegistry } from './components/nodeRegistry';
import { calculateNodeHeight } from '../constants/nodeHeights';
import { repositionSiblings } from './components/useCommonNodeActions';

interface MHAttentionLayerProps {
  id: string;
}

export const MHAttentionLayer: React.FC<MHAttentionLayerProps> = ({ id }) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const typedData = node.type as string;

  // input 값 변경 시, 노드의 data에 직접 업데이트 + string 처리 for select
  const handleFieldChange = (field: keyof MHAttentionData, value: string) => {
    const stringFields = nodeRegistry.get(typedData)?.stringFields ?? [];

    let newValue: string | number | boolean;
    if (value === 'true' || value === 'false') {
      newValue = value === 'true';
    } else if (stringFields.includes(field as string)) {
      newValue = value;
    } else {
      newValue = Number(value);
    }

    setNodes((nds) => {
      // 1. 현재 노드의 데이터와 높이를 먼저 업데이트합니다.
      let updatedNodes = nds.map((nodeItem) => {
        if (nodeItem.id === id) {
          const updatedData = { ...nodeItem.data, [field]: newValue };
          const updatedNode = { ...nodeItem, data: updatedData };

          // isRoPE 필드가 변경될 때만 높이를 재계산합니다.
          if (field === 'isRoPE') {
            return {
              ...updatedNode,
              height: calculateNodeHeight(updatedNode),
            };
          }
          return updatedNode;
        }
        return nodeItem;
      });

      // 2. 현재 노드에 부모가 있다면, 형제 노드들의 위치를 재조정합니다.
      const currentNode = updatedNodes.find((n) => n.id === id);
      if (currentNode && currentNode.parentNode && field === 'isRoPE') {
        updatedNodes = repositionSiblings(updatedNodes, currentNode.parentNode);
      }

      return updatedNodes;
    });
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
    handleInfoClick,
    handleLockToggle,
  } = useCommonNodeActions<MHAttentionData>({ id, setEditMode });

  return (
    <LayerWrapper hideHandles={node.data.hideHandles}>
      <div className="relative group">
        <NodeTitle onClick={handleNodeClick}>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          isLocked={node.data.isLocked}
          onInfo={() => handleInfoClick(nodeInfo.mhAttention)}
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
              handleFieldChange(name as keyof MHAttentionData, value)
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

export default MHAttentionLayer;
