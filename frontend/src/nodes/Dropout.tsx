import React, { useState, useEffect } from 'react';
import { useReactFlow } from 'reactflow';

import {
  NodeTitle,
  ReadField,
  EditField,
  ActionButton,
} from './NodeComponents';
import { DropoutData } from './NodeData';
import NodeWrapper from './NodeWrapper';

export const DropoutLayer: React.FC<{ data: DropoutData }> = ({
  data: initialData,
}) => {
  const { setNodes, getEdges, getNode } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);

  const [inDimStr, setInDimStr] = useState<string>(
    initialData.inDim !== undefined ? initialData.inDim.toString() : '',
  );
  const [dropoutRateStr, setDropoutRateStr] = useState<string>(
    initialData.dropoutRate.toString(),
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
    const newDropoutRate =
      dropoutRateStr === '' ? initialData.dropoutRate : Number(dropoutRateStr);

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
                dropoutRate: newDropoutRate,
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
          {/* <EditField
            label="Input Dimension:"
            id="inDimInput"
            name="inDim"
            value={inDimStr}
            placeholder="Enter input dimension"
            onChange={setInDimStr}
          /> */}
          <EditField
            label="Dropout Rate:"
            id="dropoutRateInput"
            name="dropoutRate"
            value={dropoutRateStr}
            placeholder="Enter dropout rate"
            onChange={setDropoutRateStr}
          />
          <ActionButton onClick={handleSaveClick} className="bg-green-200">
            Save
          </ActionButton>
        </div>
      ) : (
        <div>
          {/* <ReadField label="Input Dimension:" value={inDimStr} /> */}
          <ReadField label="Dropout Rate:" value={dropoutRateStr} />
          <ActionButton onClick={handleEditClick} className="bg-blue-200">
            Edit
          </ActionButton>
        </div>
      )}
    </NodeWrapper>
  );
};

export default DropoutLayer;
