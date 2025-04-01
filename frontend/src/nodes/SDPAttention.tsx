import React, { useState } from 'react';
import { useReactFlow, NodeProps } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './components/Components';
import { SDPAttentionData } from './components/NodeData';
import { LayerWrapper } from './components/NodeWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

interface SDPAttentionLayerProps {
  id: string;
}

export const SDPAttentionLayer: React.FC<NodeProps<SDPAttentionLayerProps>> = ({
  id,
}) => {
  const { setNodes, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false); // ← 추가

  const handleNodeClick = () => {
    setIsCollapsed((prev) => !prev);
  };

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as SDPAttentionData;

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (field: keyof SDPAttentionData, value: string) => {
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
  } = useCommonNodeActions<SDPAttentionData>({
    initialData: currentData,
    setNodes,
    setEditMode,
  });

  return (
    <LayerWrapper
      hideHandles={currentData.hideHandles}
      onClick={handleNodeClick}
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
        {/* Collapse가 아닐 때만 필드 보여줌 */}
        {!isCollapsed && (
          <>
            {editMode ? (
              <div>
                <EditField
                  label="Input Dimension:"
                  id="inDimInput"
                  name="inDim"
                  value={
                    currentData.inDim !== undefined
                      ? currentData.inDim.toString()
                      : ''
                  }
                  placeholder="Enter input dimension"
                  onChange={(value) => handleFieldChange('inDim', value)}
                />
                <EditField
                  label="Output Dimension:"
                  id="outDimInput"
                  name="outDim"
                  value={
                    currentData.outDim !== undefined
                      ? currentData.outDim.toString()
                      : ''
                  }
                  placeholder="Enter output dimension"
                  onChange={(value) => handleFieldChange('outDim', value)}
                />
                <EditField
                  label="Dropout Rate:"
                  id="dropoutRateInput"
                  name="dropoutRate"
                  value={
                    currentData.dropoutRate !== undefined
                      ? currentData.dropoutRate.toString()
                      : ''
                  }
                  placeholder="Enter dropout rate"
                  onChange={(value) => handleFieldChange('dropoutRate', value)}
                />
                <EditField
                  label="Context Length:"
                  id="ctxLengthInput"
                  name="ctxLength"
                  value={
                    currentData.ctxLength !== undefined
                      ? currentData.ctxLength.toString()
                      : ''
                  }
                  placeholder="Enter context length"
                  onChange={(value) => handleFieldChange('ctxLength', value)}
                />
              </div>
            ) : (
              <div>
                <ReadField
                  label="Input Dimension:"
                  value={
                    currentData.inDim !== undefined
                      ? currentData.inDim.toString()
                      : ''
                  }
                />
                <ReadField
                  label="Output Dimension:"
                  value={
                    currentData.outDim !== undefined
                      ? currentData.outDim.toString()
                      : ''
                  }
                />
                <ReadField
                  label="Dropout Rate:"
                  value={
                    currentData.dropoutRate !== undefined
                      ? currentData.dropoutRate.toString()
                      : ''
                  }
                />
                <ReadField
                  label="Context Length:"
                  value={
                    currentData.ctxLength !== undefined
                      ? currentData.ctxLength.toString()
                      : ''
                  }
                />
              </div>
            )}
          </>
        )}
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {currentData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default SDPAttentionLayer;
