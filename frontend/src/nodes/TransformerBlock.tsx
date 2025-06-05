import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useStore } from 'reactflow';

import { BlockWrapper } from './components/BlockWrapper';
import { NodeTitle } from './components/Components';
import { TransformerBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';
import { nodeInfo, nodeFieldInfo } from './components/nodeInfo';
import { NODE_GAP, DEFAULT_NODE_HEIGHT } from '../constants/nodeHeights';

interface TransformerBlockLayerProps {
  id: string;
}

const getFields = (data: TransformerBlockData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Number of Blocks:',
    name: 'numOfBlocks',
    value: data.numOfBlocks?.toString() || '',
    placeholder: 'Enter the number of blocks',
    info: nodeFieldInfo.dynamicBlock.numOfBlocks,
  },
];

// Dynamic Block 컴포넌트는 내부에 노드를 드롭하면 수직으로 정렬하고, 노드 사이에 자동 edge를 표시
const TransformerBlock: React.FC<NodeProps<TransformerBlockLayerProps>> = ({
  id,
}) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as TransformerBlockData;

  // 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();
  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);
  const childNodesHeight = useMemo(() => {
    return childNodes.reduce(
      (acc, node) => NODE_GAP + acc + (node.height ?? DEFAULT_NODE_HEIGHT),
      20,
    );
  }, [childNodes]);

  // ✅ input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (
    field: keyof TransformerBlockData,
    value: string,
  ) => {
    const stringFields: (keyof TransformerBlockData)[] = ['label'];
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
  const { handleDeleteClick, handleEditClick, handleSaveClick } =
    useCommonNodeActions<TransformerBlockData>({
      id,
      setNodes,
      setEditMode,
      setEdges,
    });

  // ✅ 노드 정보 클릭 핸들러 오버라이드
  const handleInfoClick = () => {
    const event = new CustomEvent('nodeInfo', {
      detail: nodeInfo.transformerBlock,
    });
    window.dispatchEvent(event);
  };

  return (
    <BlockWrapper
      childNodesHeight={childNodesHeight}
      isTarget={currentData.isTarget}
    >
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
