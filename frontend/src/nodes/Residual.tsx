import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/FieldComponents';
import { ResidualData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import { nodeInfo } from './components/nodeInfo';

interface ResidualLayerProps {
  id: string;
}

export const ResidualLayer: React.FC<ResidualLayerProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;

  // ✅ 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleInfoClick,
  } = useCommonNodeActions<ResidualData>({
    id,
    setNodes,
    setEditMode,
    setEdges,
  });

  return (
    <LayerWrapper hideHandles={node.data.hideHandles} isResidual={true}>
      <div className="relative group">
        <NodeTitle>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={() => handleInfoClick(nodeInfo.residual)}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
      </div>
    </LayerWrapper>
  );
};

export default ResidualLayer;
