import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useStore } from 'reactflow';

import { BlockWrapper } from './components/BlockWrapper';
import { NodeTitle } from './components/Components';
import { DynamicBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

interface DynamicBlockLayerProps {
  id: string;
}

const getFields = (data: DynamicBlockData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Number of Layers:',
    name: 'numOfLayers',
    value: data.numLayers?.toString() || '',
    placeholder: 'Enter the number of layers',
  },
];

// Dynamic Block 컴포넌트는 내부에 노드를 드롭하면 수직으로 정렬하고, 노드 사이에 자동 edge를 표시
const DynamicBlock: React.FC<NodeProps<DynamicBlockLayerProps>> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as DynamicBlockData;

  // 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();
  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);
  const childNodesHeight = useMemo(() => {
    return childNodes.reduce((acc, node) => 10 + acc + (node.height ?? 20), 20);
  }, [childNodes]);

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof DynamicBlockData, value: string) => {
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
  } = useCommonNodeActions<DynamicBlockData>({
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
            handleFieldChange(name as keyof DynamicBlockData, value)
          }
        />
        {childNodesHeight === 20 && (
          <div className="border-dashed border-2 text-center text-gray-500 italic">
            여기에 노드를 드롭하세요
          </div>
        )}
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

export default DynamicBlock;
