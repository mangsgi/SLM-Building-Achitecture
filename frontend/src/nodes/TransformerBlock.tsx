import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useNodes } from 'reactflow';

import { BlockWrapper } from './components/BlockWrapper';
import { NodeTitle } from './components/FieldComponents';
import { TransformerBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import FieldRenderer from './components/FieldRenderer';
import { nodeInfo } from './components/NodeInfo';
import { nodeRegistry } from './components/nodeRegistry';
import {
  NODE_GAP,
  DEFAULT_NODE_HEIGHT,
  DEFAULT_BLOCK_NODE_HEIGHT,
} from '../constants/nodeHeights';

interface TransformerBlockLayerProps {
  id: string;
}

const TransformerBlock: React.FC<NodeProps<TransformerBlockLayerProps>> = ({
  id,
}) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const typedData = node.type as string;

  const allNodes = useNodes();
  const childNodes = useMemo(() => {
    return allNodes.filter((n) => n.parentNode === id);
  }, [allNodes, id]);

  // 자식 노드들이 차지하는 순수 영역의 높이 계산
  const childrenAreaHeight = useMemo(() => {
    // 자식이 없으면 플레이스홀더를 위한 높이 반환
    if (childNodes.length === 0) {
      return DEFAULT_BLOCK_NODE_HEIGHT;
    }
    // 자식이 있으면, 자식들의 총 높이 + 자식들 사이의 간격의 합을 반환
    const totalChildrenSize = childNodes.reduce(
      (acc, node) => acc + (node.height ?? DEFAULT_NODE_HEIGHT),
      DEFAULT_NODE_HEIGHT,
    );
    const totalGaps = (childNodes.length - 1) * NODE_GAP;
    return totalChildrenSize + totalGaps;
  }, [childNodes]);

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (
    field: keyof TransformerBlockData,
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
    handleInfoClick,
    handleLockToggle,
  } = useCommonNodeActions<TransformerBlockData>({
    id,
    setEditMode,
  });

  return (
    <BlockWrapper
      childrenAreaHeight={childrenAreaHeight}
      isTarget={node.data.isTarget}
    >
      <div className="relative group">
        <NodeTitle>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          isLocked={node.data.isLocked}
          onInfo={() => handleInfoClick(nodeInfo.transformerBlock)}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
          onLockToggle={handleLockToggle}
        />
        <FieldRenderer
          fields={nodeRegistry.get(typedData)?.getFields(node.data) ?? []}
          editMode={editMode}
          onChange={(name: string, value: string) =>
            handleFieldChange(name as keyof TransformerBlockData, value)
          }
          onInfoClick={(info) => {
            const event = new CustomEvent('fieldInfo', { detail: info });
            window.dispatchEvent(event);
          }}
        />
        {childNodes.length === 0 && (
          <div className="border-dashed border-2 text-center text-gray-500 italic p-4">
            여기에 노드를 드롭하세요
          </div>
        )}
      </div>
    </BlockWrapper>
  );
};

export default TransformerBlock;
