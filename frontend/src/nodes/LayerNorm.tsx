import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { LayerNormData } from './NodeData';
import NodeWrapper from './NodeWrapper';

export const LayerNormLayer: React.FC<{ data: LayerNormData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim.toString(),
  );
  const [outDimStr, setOutDimStr] = useState<string>(
    initialData.outDim.toString(),
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
    const newOutDim = outDimStr === '' ? initialData.outDim : Number(outDimStr);

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
                outDim: newOutDim,
              },
            };
          }
          return node;
        }),
      );
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
          <EditField
            label="Output Dimension:"
            id="outDimInput"
            name="outDim"
            value={outDimStr}
            placeholder="Enter Output Dimension"
            onChange={setOutDimStr}
          />
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Input Dimension:" value={inDimStr} />
          <ReadField label="Output Dimension:" value={outDimStr} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default LayerNormLayer;
