import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useStore } from 'reactflow';

import { BlockWrapper } from './components/BlockWrapper';
import { NodeTitle } from './components/FieldComponents';
import { TransformerBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import FieldRenderer from './components/FieldRenderer';
import { nodeInfo } from './components/nodeInfo';
import { nodeRegistry } from './components/nodeRegistry';
import { NODE_GAP, DEFAULT_NODE_HEIGHT } from '../constants/nodeHeights';

interface TransformerBlockLayerProps {
  id: string;
}

const TransformerBlock: React.FC<NodeProps<TransformerBlockLayerProps>> = ({
  id,
}) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const typedData = node.type as string;

  // 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();
  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);
  const childNodesHeight = useMemo(() => {
    return childNodes.reduce(
      (acc, node) => NODE_GAP + acc + (node.height ?? DEFAULT_NODE_HEIGHT),
      40,
    );
  }, [childNodes]);

  // ✅ input 값 변경 시, 노드의 data에 직접 업데이트
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

  // ✅ 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    isLocked,
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleInfoClick,
    handleLockToggle,
  } = useCommonNodeActions<TransformerBlockData>({
    id,
    setNodes,
    setEditMode,
    setEdges,
  });

  return (
    <BlockWrapper
      childNodesHeight={childNodesHeight}
      isTarget={node.data.isTarget}
    >
      <div className="relative group">
        <NodeTitle>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          isLocked={isLocked}
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
        {childNodesHeight === 40 && (
          <div className="border-dashed border-2 text-center text-gray-500 italic">
            여기에 노드를 드롭하세요
          </div>
        )}
      </div>
    </BlockWrapper>
  );
};

export default TransformerBlock;
