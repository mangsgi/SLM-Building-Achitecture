import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { BaseNodeData } from './NodeData';
import NodeWrapper from './NodeWrapper';

export const LinearLayer: React.FC<{ data: BaseNodeData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined
      ? initialData.inDim.toString()
      : '정의되지 않았습니다.',
  );
  const [outDimStr, setOutDimStr] = useState<string>(
    initialData.outDim !== undefined
      ? initialData.outDim.toString()
      : '정의되지 않았습니다.',
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

export default LinearLayer;
