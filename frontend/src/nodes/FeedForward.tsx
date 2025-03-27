import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  EditSelectField,
} from './NodeComponents';
import { FeedForwardData } from './NodeData';
import { LayerWrapper } from './NodeWrapper';
import NodeActionPanel from './NodeActionPanel';
import NodeInfoModal from './NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

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
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // FeedForwardData 상태변수 저장
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

  // Save 버튼에 들어갈 Custom Save
  const customSave = () => {
    const newInDim = inDimStr === '' ? initialData.inDim : Number(inDimStr);
    const newActFunc = actFunc === '' ? initialData.actFunc : actFunc;
    const newNumOfFactor =
      numOfFactorStr === '' ? initialData.numOfFactor : Number(numOfFactorStr);

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

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<FeedForwardData>({
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
          </div>
        ) : (
          <div>
            <ReadField label="Input Dimension:" value={inDimStr} />
            <ReadField label="Number of Factor:" value={numOfFactorStr} />
            <ReadField label="Activation Function Type:" value={actFunc} />
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

export default FeedForwardLayer;
