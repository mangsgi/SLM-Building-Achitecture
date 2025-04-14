import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useStore } from 'reactflow';

import { NodeTitle } from './components/Components';
import { BlockWrapper } from './components/BlockWrapper';
import { TestBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

interface TestBlockProps {
  id: string;
}

const getFields = (data: TestBlockData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Number of Layers:',
    name: 'numOfLayers',
    value: data.numLayers?.toString() || '',
    placeholder: 'Enter the number of layers',
  },
];

const TestBlock: React.FC<NodeProps<TestBlockProps>> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as TestBlockData;

  // 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();
  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);
  const childNodesHeight = useMemo(() => {
    return childNodes.reduce((acc, node) => 10 + acc + (node.height ?? 0), 0);
  }, [childNodes]);

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof TestBlockData, value: string) => {
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
  } = useCommonNodeActions<TestBlockData>({
    id,
    setNodes,
    setEditMode,
    setEdges,
  });

  return (
    <BlockWrapper childNodesHeight={childNodesHeight}>
      <div className="relative group">
        <NodeTitle>{currentData.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
        <FieldRenderer
          fields={getFields(currentData)}
          editMode={editMode}
          onChange={(name: string, value: string) =>
            handleFieldChange(name as keyof TestBlockData, value)
          }
        />
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {currentData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </BlockWrapper>
  );
};

export default TestBlock;
