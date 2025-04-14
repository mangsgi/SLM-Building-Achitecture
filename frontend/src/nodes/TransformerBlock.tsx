import React, {
  useMemo,
  useState,
  useEffect,
  useRef,
  useLayoutEffect,
} from 'react';
import { useReactFlow, useStore } from 'reactflow';

import { NodeTitle } from './components/Components';
import { BlockWrapper } from './components/BlockWrapper';
import { TransformerBlockData } from './components/NodeData';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import NodeSlot from './components/NodeSlot';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';

const getFields = (data: TransformerBlockData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Number of Layers:',
    name: 'numOfLayers',
    value: data.numLayers?.toString() || '',
    placeholder: 'Enter the number of layers',
  },
];

interface TransformerBlockProps {
  id: string;
}

const TransformerBlock: React.FC<TransformerBlockProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as TransformerBlockData;

  // 6개 슬롯 useState
  // const [residual1, setResidual1] = useState<boolean>(false);
  // const [dropout1, setDropout1] = useState<boolean>(false);
  const [feedForward, setFeedForward] = useState<boolean>(false);
  const [layerNorm2, setLayerNorm2] = useState<boolean>(false);
  // const [residual2, setResidual2] = useState<boolean>(false);
  // const [dropout2, setDropout2] = useState<boolean>(false);
  const [maskedMHA, setMaskedMHA] = useState<boolean>(false);
  const [layerNorm1, setLayerNorm1] = useState<boolean>(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const [measuredHeight, setMeasuredHeight] = useState<number>(0);
  useLayoutEffect(() => {
    if (contentRef.current) {
      const rect = contentRef.current.getBoundingClientRect();
      setMeasuredHeight(rect.height);
    }
    console.log(measuredHeight);
  }, [
    // residual1,
    feedForward,
    layerNorm2,
    // dropout2,
    maskedMHA,
    layerNorm1,
    editMode,
  ]);
  // 자식 노드와 자식 노드의 높이 합 저장
  const getNodes = useStore((state) => state.getNodes);
  const nodes = getNodes();

  const childNodes = useMemo(() => {
    return nodes.filter((n) => n.parentNode === id);
  }, [nodes]);

  // const childNodesHeight = useMemo(() => {
  //   return childNodes.reduce((acc, node) => 10 + acc + (node.height ?? 0), 0);
  // }, [childNodes]);

  // const SLOT_HEIGHT = 60;

  // const activeSlots = [
  //   !residual1,
  //   !feedForward,
  //   !layerNorm2,
  //   !dropout2,
  //   !maskedMHA,
  //   !layerNorm1,
  // ].filter(Boolean).length;

  // const computedHeight = childNodes.length
  //   ? childNodes.reduce((acc, node) => acc + (node.height ?? 0) + 10, 0)
  //   : activeSlots * SLOT_HEIGHT;

  useEffect(() => {
    childNodes.forEach((node) => {
      if (node.type === 'dropout') {
        // setDropout1(true);
      } else if (node.type === 'feedForward') {
        setFeedForward(true);
      } else if (node.type === 'layerNorm' && !layerNorm1) {
        setLayerNorm1(true);
      } else if (node.type === 'layerNorm' && !layerNorm2) {
        setLayerNorm2(true);
      } else if (node.type === 'maskedMHABlock') {
        setMaskedMHA(true);
      }
    });
  }, [childNodes]);

  // input 값 변경 시, 노드의 data에 직접 업데이트
  const handleFieldChange = (
    field: keyof TransformerBlockData,
    value: string,
  ) => {
    const newValue = field === 'label' ? value : Number(value);
    setNodes((nds) =>
      nds.map((nodeItem) => {
        if (nodeItem.id === id) {
          return {
            ...nodeItem,
            data: {
              ...nodeItem.data,
              [field]: newValue,
            },
          };
        }
        return nodeItem;
      }),
    );
  };

  // 공통 액션 핸들러를 커스텀 훅을 통해 생성
  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  } = useCommonNodeActions<TransformerBlockData>({
    id,
    setNodes,
    setEditMode,
    setEdges,
  });

  return (
    <BlockWrapper childNodesHeight={measuredHeight}>
      <div className="relative group" ref={contentRef}>
        <NodeTitle>{currentData.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
        {/* Collapse가 아닐 때만 필드 보여줌 */}
        <FieldRenderer
          fields={getFields(currentData)}
          editMode={editMode}
          onChange={(name: string, value: string) =>
            handleFieldChange(name as keyof TransformerBlockData, value)
          }
        />
        {/* 그림에 나온 순서대로 6개 슬롯 배치 */}
        <div className="flex flex-col items-center w-56">
          {/* {!residual1 && (
            <NodeSlot allowedType="dropout" slotLabel="Dropout 1" />
          )} */}
          {!feedForward && (
            <NodeSlot allowedType="feedForward" slotLabel="FeedForward" />
          )}
          {!layerNorm2 && (
            <NodeSlot allowedType="layerNorm" slotLabel="LayerNorm 2" />
          )}
          {/* {!dropout2 && (
            <NodeSlot allowedType="dropout" slotLabel="Dropout 2" />
          )} */}
          {!maskedMHA && (
            <NodeSlot allowedType="maskedMHABlock" slotLabel="Masked MHA" />
          )}
          {!layerNorm1 && (
            <NodeSlot allowedType="layerNorm" slotLabel="LayerNorm 1" />
          )}
        </div>

        <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
          <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
          <p className="text-sm">
            여기에 {currentData.label} 노드에 대한 추가 정보를 입력하세요.
          </p>
        </NodeInfoModal>
      </div>
    </BlockWrapper>
  );
};

export default TransformerBlock;
