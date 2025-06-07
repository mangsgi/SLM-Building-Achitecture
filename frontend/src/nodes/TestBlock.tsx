import React, { useState, useMemo } from 'react';
import { useReactFlow, useStore } from 'reactflow';

import { NodeTitle } from './components/Components';
import { BlockWrapper } from './components/BlockWrapper';
import { TestBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer from './components/FieldRenderer';
import { nodeInfo } from './components/nodeInfo';
import { nodeRegistry } from './components/nodeRegistry';

interface TestBlockProps {
  id: string;
}

export const TestBlock: React.FC<TestBlockProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const typedData = node.type as string;

  // ✅ 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();
  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);
  const childNodesHeight = useMemo(() => {
    return childNodes.reduce((acc, node) => 10 + acc + (node.height ?? 0), 0);
  }, [childNodes]);

  // ✅ input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof TestBlockData, value: string) => {
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
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleInfoClick,
  } = useCommonNodeActions<TestBlockData>({
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
          onInfo={() => handleInfoClick(nodeInfo.testBlock)}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
        <FieldRenderer
          fields={nodeRegistry.get(typedData)?.getFields(node.data) ?? []}
          editMode={editMode}
          onChange={(name: string, value: string) =>
            handleFieldChange(name as keyof TestBlockData, value)
          }
          onInfoClick={(info) => {
            const event = new CustomEvent('fieldInfo', { detail: info });
            window.dispatchEvent(event);
          }}
        />
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">
          {nodeInfo.testBlock.title}
        </h3>
        <p className="text-sm">{nodeInfo.testBlock.description}</p>
      </NodeInfoModal>
    </BlockWrapper>
  );
};

export default TestBlock;
