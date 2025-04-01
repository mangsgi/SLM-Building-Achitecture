import React, { useState, useEffect } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle, ReadField, EditField } from './components/Components';
import { DropoutData } from './components/NodeData';
import { LayerWrapper } from './components/NodeWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';

interface DropoutLayerProps {
  data: DropoutData;
  onChange?: (newData: DropoutData) => void;
}

export const DropoutLayer: React.FC<DropoutLayerProps> = ({
  data: initialData,
  onChange,
}) => {
  const { setNodes, getEdges, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  // DropoutData 상태변수 저장
  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
  );
  const [dropoutRateStr, setDropoutRateStr] = useState<string>(
    initialData.dropoutRate !== undefined
      ? initialData.dropoutRate.toString()
      : '',
  );

  // Edge가 연결될 때 Source Node의 Input Dimension을 받아오는 useEffect
  useEffect(() => {
    // 이 노드로 들어오는 엣지가 있는지 찾는다
    const edges = getEdges();
    const incomingEdge = edges.find((edge) => edge.target === initialData.id);
    if (!incomingEdge) return;

    // 소스 노드를 찾는다
    const sourceNode = getNode(incomingEdge.source);
    if (!sourceNode || !sourceNode.data) return;

    // 소스 노드가 outDim을 가지고 있으면, 그 값을 inDim에 반영
    const sourceOutDim = sourceNode.data.inDim;
    console.log({ sourceOutDim });
    if (typeof sourceOutDim === 'number') {
      // 이미 inDim이 동일하면 굳이 업데이트 X
      if (inDimStr === sourceOutDim.toString()) return;

      // 로컬 state 업데이트
      setInDimStr(String(sourceOutDim));

      // 글로벌 노드 데이터도 업데이트
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                inDim: sourceOutDim,
              },
            };
          }
          return node;
        }),
      );
    }
  }, [getEdges, getNode, setNodes, inDimStr, initialData.id]);

  // Save 버튼에 들어갈 Custom Save
  const customSave = () => {
    const newInDim = inDimStr === '' ? initialData.inDim : Number(inDimStr);
    const newDropoutRate =
      dropoutRateStr === '' ? initialData.dropoutRate : Number(dropoutRateStr);

    if (initialData.id) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                inDim: newInDim,
                dropoutRate: newDropoutRate,
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
        dropoutRate: newDropoutRate,
      });
    }
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<DropoutData>({
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
              label="Dropout Rate:"
              id="dropoutRateInput"
              name="dropoutRate"
              value={dropoutRateStr}
              placeholder="Enter dropout rate"
              onChange={setDropoutRateStr}
            />
          </div>
        ) : (
          <div>
            <ReadField label="Dropout Rate:" value={dropoutRateStr} />
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

export default DropoutLayer;
