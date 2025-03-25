import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { FeedForwardData } from './NodeData';
import { NodeWrapper } from './NodeWrapper';
import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
  EditSelectField,
} from './NodeComponents';

interface FeedForwardDataProps {
  data: FeedForwardData;
  onChange?: (newData: FeedForwardData) => void;
}

const actFuncTypeOptions: string[] = ['ReLU', 'GELU', 'SwiGLU'];

export const FeedForwardLayer: React.FC<FeedForwardDataProps> = ({
  data: initialData,
  onChange,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
  );
  const [numOfFactorStr, setNumOfFactorStr] = useState<string>(
    initialData.numOfFactor !== undefined
      ? initialData.numOfFactor.toString()
      : '',
  );
  const [actFunc, setActFunc] = useState<string>(
    initialData.actFunc !== undefined
      ? initialData.actFunc
      : actFuncTypeOptions[0],
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
    const newActFunc = actFunc === '' ? initialData.actFunc : actFunc;
    const newNumOfFactor =
      numOfFactorStr === '' ? initialData.numOfFactor : Number(numOfFactorStr);

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
                actFunc: newActFunc,
                numOfFactor: newNumOfFactor,
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
        actFunc: newActFunc,
        numOfFactor: newNumOfFactor,
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
            label="Number of Factor:"
            id="numOfFactorInput"
            name="numOfFactor"
            value={numOfFactorStr}
            placeholder="Enter Number of factor"
            onChange={setNumOfFactorStr}
          />
          <EditSelectField
            label="Activation Function Type:"
            id="actFuncSelect"
            name="actFunc"
            value={actFunc}
            onChange={setActFunc}
            options={actFuncTypeOptions}
          />
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          <ReadField label="Input Dimension:" value={inDimStr} />
          <ReadField label="Number of Factor:" value={numOfFactorStr} />
          <ReadField label="Activation Function Type:" value={actFunc} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default FeedForwardLayer;
