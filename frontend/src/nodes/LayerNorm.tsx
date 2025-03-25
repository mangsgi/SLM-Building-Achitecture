import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { BaseNodeData } from './NodeData';
import { NodeWrapper } from './NodeWrapper';

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

  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
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
    const newInDim = inDimStr === '' ? initialData.inDim : Number(inDimStr);

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
            placeholder="Enter Input Dimension"
            onChange={setInDimStr}
          />
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Input Dimension:" value={inDimStr} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default LayerNormLayer;
