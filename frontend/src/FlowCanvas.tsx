import React, { useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Connection,
  Edge,
  Node,
} from 'reactflow';
import 'reactflow/dist/style.css';

import TokenEmbeddingLayer from './nodes/TokenEmbedding';
import PositionalEmbeddingLayer from './nodes/PositionalEmbedding';
import LayerNormLayer from './nodes/LayerNorm';
import FeedForwardLayer from './nodes/FeedForward';
import DropoutLayer from './nodes/Dropout';
import LinearLayer from './nodes/Linear';
import SDPAttentionLayer from './nodes/SDPAttention';
import MaskedMHABlock from './nodes/MaskedMHABlock';
import TransformerBlock from './nodes/TransformerBlock';
import DynamicBlock from './nodes/DynamicBlock';

function FlowCanvas() {
  // ReactFlow에서 각 노드와 엣지 상태 저장
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // onDrop 시 드롭된 노드의 정확한 위치를 계산하기 위해 DOM 요소 참조
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // nodeTypes를 한 번만 생성하도록 하여, 렌더링 시 불필요한 재생성을 방지
  const nodeTypes = useMemo(
    () => ({
      tokenEmbedding: TokenEmbeddingLayer,
      positionalEmbedding: PositionalEmbeddingLayer,
      layerNorm: LayerNormLayer,
      feedForward: FeedForwardLayer,
      dropout: DropoutLayer,
      linear: LinearLayer,
      sdpAttention: SDPAttentionLayer,
      maskedMHABlock: MaskedMHABlock,
      transformerBlock: TransformerBlock,
      dynamicBlock: DynamicBlock,
    }),
    [],
  );

  // useCallback을 통해 핸들러나 콜백 함수나 의존성 배열을 메모이제이션하여 불필요한 재생성을 방지
  // 노드 간 연결 이벤트 핸들러
  const onConnect = useCallback(
    // 전달된 params를 기반으로 addEdge 헬퍼 함수를 사용해 현재 edges 상태에 새로운 edge를 추가
    (params: Edge<unknown> | Connection) =>
      setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges],
  );

  // Drag된 요소가 Canvas 위로 올라오는 이벤트 핸들러
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    // 사용자에게 이동 동작임을 시각적으로 표시
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Sidebar의 Node가 Canvas에 Drop되는 이벤트 처리
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      // Data 파싱
      const dataString = event.dataTransfer.getData('application/reactflow');
      if (!dataString) return;

      const parsedData = JSON.parse(dataString);
      const { nodeType, label, ...props } = parsedData;

      // Drop 위치 계산
      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const id = `${nodeType}-${+new Date()}`;
      const newNode: Node = {
        id,
        type: nodeType,
        position,
        data: { id, label, ...props },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes],
  );

  return (
    <div ref={reactFlowWrapper} className="w-full h-full bg-gray-50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        snapToGrid={true} // 노드의 부드러운 이동 적용
        snapGrid={[3, 3]}
        nodeTypes={nodeTypes} // 전달된 nodeTypes 매핑 적용
      >
        <Controls />
      </ReactFlow>
    </div>
  );
}

export default FlowCanvas;
