import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './NodeComponents';
import { BaseNodeData } from './NodeData';
import { LayerWrapper } from './NodeWrapper';
import NodeActionPanel from './NodeActionPanel';
import NodeInfoModal from './NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

interface LayerNormLayerProps {
  data: BaseNodeData;
  onChange?: (newData: BaseNodeData) => void;
}

export const LayerNormLayer: React.FC<LayerNormLayerProps> = ({
  data: initialData,
  onChange,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // BaseNodeData 상태변수 저장
  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
  );

  // Save 버튼에 들어갈 Custom Save
  const customSave = () => {
    const newInDim = inDimStr === '' ? initialData.inDim : Number(inDimStr);

    if (initialData.id) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                inDim: newInDim,
              },
            };
          }
          return node;
        }),
      );
    }
    // Block 안에 있는 노드 데이터 업데이트
    if (onChange) {
      onChange({
        ...initialData,
        inDim: newInDim,
      });
    }
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<BaseNodeData>({
    initialData,
    setNodes,
    setEditMode,
    customSave,
  });

  return (
    <LayerWrapper>
      <div className="relative group">
        <NodeTitle>{initialData.label}</NodeTitle>
        {editMode ? (
          <div>
            <EditField
              label="Input Dimension:"
              id="inDimInput"
              name="inDim"
              value={inDimStr}
              placeholder="Enter Input Dimension"
              onChange={setInDimStr}
            />
          </div>
        ) : (
          <div>
            <ReadField label="Input Dimension:" value={inDimStr} />
          </div>
        )}
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
        <p className="text-sm">
          여기에 {initialData.label} 노드에 대한 추가 정보를 입력하세요.
        </p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default LayerNormLayer;
