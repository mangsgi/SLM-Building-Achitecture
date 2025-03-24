import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { SDPAttentionData } from './NodeData';
import NodeWrapper from './NodeWrapper';

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

  // Edit 버튼 클릭
  const handleEditClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    console.log('Edit button clicked');
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    const newDropoutRate =
      dropoutRateStr === '' ? initialData.dropoutRate : Number(dropoutRateStr);
    const newCtxLength =
      ctxLengthStr === '' ? initialData.ctxLength : Number(ctxLengthStr);

    setEditMode(false);

    // 노드 데이터 업데이트
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

  return (
    <NodeWrapper>
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
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Input Dimension:" value={inDimStr} />
          <ReadField label="Output Dimension:" value={outDimStr} />
          <ReadField label="Dropout Rate:" value={dropoutRateStr} />
          <ReadField label="Context Length:" value={ctxLengthStr} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default SPDAttentionLayer;
