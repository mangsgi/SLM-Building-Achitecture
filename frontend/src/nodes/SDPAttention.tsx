import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './NodeComponents';
import { SDPAttentionData } from './NodeData';
import { LayerWrapper } from './NodeWrapper';
import NodeActionPanel from './NodeActionPanel';
import NodeInfoModal from './NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

interface SPDAttentionLayerProps {
  data: SDPAttentionData;
  onChange?: (newData: SDPAttentionData) => void;
}

export const SPDAttentionLayer: React.FC<SPDAttentionLayerProps> = ({
  data: initialData,
  onChange,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // SDPAttentionData 상태변수 저장
  const [dropoutRateStr, setDropoutRateStr] = useState<string>(
    initialData.dropoutRate !== undefined
      ? initialData.dropoutRate.toString()
      : '',
  );
  const [ctxLengthStr, setCtxLengthStr] = useState<string>(
    initialData.ctxLength !== undefined ? initialData.ctxLength.toString() : '',
  );
  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
  );
  const [outDimStr, setOutDimStr] = useState<string>(
    initialData.outDim !== undefined ? initialData.outDim.toString() : '',
  );

  // Save 버튼에 들어갈 Custom Save
  const customSave = () => {
    const newDropoutRate =
      dropoutRateStr === '' ? initialData.dropoutRate : Number(dropoutRateStr);
    const newCtxLength =
      ctxLengthStr === '' ? initialData.ctxLength : Number(ctxLengthStr);

    if (initialData.id) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                dropoutRate: newDropoutRate,
                ctxLength: newCtxLength,
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
        dropoutRate: newDropoutRate,
        ctxLength: newCtxLength,
      });
    }
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<SDPAttentionData>({
    initialData,
    setNodes,
    setEditMode,
    customSave,
  });

  return (
    <LayerWrapper hideHandles={editMode /* 편집 시 핸들 숨김 */}>
      <div className="relative group">
        <NodeTitle>{initialData.label}</NodeTitle>
        {editMode ? (
          <div>
            <EditField
              label="Input Dimension:"
              id="inDimInput"
              name="inDim"
              value={inDimStr}
              placeholder="Enter input dimension"
              onChange={setInDimStr}
            />
            <EditField
              label="Output Dimension:"
              id="outDimInput"
              name="outDim"
              value={outDimStr}
              placeholder="Enter output dimension"
              onChange={setOutDimStr}
            />
            <EditField
              label="Dropout Rate:"
              id="dropoutRateInput"
              name="dropoutRate"
              value={dropoutRateStr}
              placeholder="Enter dropout rate"
              onChange={setDropoutRateStr}
            />
            <EditField
              label="Context Length:"
              id="ctxLengthInput"
              name="ctxLength"
              value={ctxLengthStr}
              placeholder="Enter context length"
              onChange={setCtxLengthStr}
            />
          </div>
        ) : (
          <div>
            <ReadField label="Input Dimension:" value={inDimStr} />
            <ReadField label="Output Dimension:" value={outDimStr} />
            <ReadField label="Dropout Rate:" value={dropoutRateStr} />
            <ReadField label="Context Length:" value={ctxLengthStr} />
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

export default SPDAttentionLayer;
