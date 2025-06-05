import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { ResidualData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './useCommonNodeActions';
import { nodeInfo } from './components/nodeInfo';

interface ResidualLayerProps {
  id: string;
}

export const ResidualLayer: React.FC<ResidualLayerProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as ResidualData;

  // ✅ 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const { handleDeleteClick, handleEditClick, handleSaveClick } =
    useCommonNodeActions<ResidualData>({
      id,
      setNodes,
      setEditMode,
      setEdges,
    });

  // ✅ 노드 정보 클릭 핸들러 오버라이드
  const handleInfoClick = () => {
    const event = new CustomEvent('nodeInfo', {
      detail: nodeInfo.residual,
    });
    window.dispatchEvent(event);
  };

  return (
    <LayerWrapper hideHandles={currentData.hideHandles} isResidual={true}>
      <div className="relative group">
        <NodeTitle>{currentData.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
      </div>
    </LayerWrapper>
  );
};

export default ResidualLayer;
