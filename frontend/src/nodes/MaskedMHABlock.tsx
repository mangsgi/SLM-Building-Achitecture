import React, { useState, useMemo } from 'react';
import { useReactFlow, NodeProps, useStore } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './components/Components';
import { BlockWrapper } from './components/BlockWrapper';
import { MaskedMHABlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

interface MaskedMHABlockProps {
  id: string;
}

const MaskedMHABlock: React.FC<NodeProps<MaskedMHABlockProps>> = ({ id }) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as MaskedMHABlockData;

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
  const handleFieldChange = (
    field: keyof MaskedMHABlockData,
    value: string,
  ) => {
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
  } = useCommonNodeActions<MaskedMHABlockData>({
    id,
    setNodes,
    setEditMode,
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
        {editMode ? (
          <div>
            <EditField
              label="Number of Heads:"
              id="numOfHeadsInput"
              name="numOfHeads"
              value={
                currentData.numHeads !== undefined
                  ? currentData.numHeads.toString()
                  : ''
              }
              placeholder="Enter the number of heads"
              onChange={(value) => handleFieldChange('numHeads', value)}
            />
          </div>
        ) : (
          <div>
            <ReadField
              label="Number of Heads:"
              value={
                currentData.numHeads !== undefined
                  ? currentData.numHeads.toString()
                  : ''
              }
            />
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

export default MaskedMHABlock;
