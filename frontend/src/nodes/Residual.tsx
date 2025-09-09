import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/FieldComponents';
import { ResidualData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import { useCommonNodeActions } from './components/useCommonNodeActions';
import { nodeInfo } from './components/NodeInfo';

interface ResidualLayerProps {
  id: string;
}

export const ResidualLayer: React.FC<ResidualLayerProps> = ({ id }) => {
  const { getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleInfoClick,
    handleLockToggle,
  } = useCommonNodeActions<ResidualData>({
    id,
    setEditMode,
  });

  return (
    <LayerWrapper hideHandles={node.data.hideHandles} isResidual={true}>
      <div className="relative group">
        <NodeTitle>{node.data.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          isLocked={node.data.isLocked}
          onInfo={() => handleInfoClick(nodeInfo.residual)}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
          onLockToggle={handleLockToggle}
        />
      </div>
    </LayerWrapper>
  );
};

export default ResidualLayer;
