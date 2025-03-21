import React, { useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  // MiniMap,
  Controls,
  Connection,
  Edge,
  Node,
  // MarkerType, { type: MarkerType.ArrowClosed }
} from 'reactflow';
import 'reactflow/dist/style.css';

import TokenEmbeddingLayer from './nodes/TokenEmbedding';
import PositionalEmbeddingLayer from './nodes/PositionalEmbedding';
import LayerNormLayer from './nodes/LayerNorm';
import FeedForwardLayer from './nodes/FeedForward';
import DynamicBlock from './nodes/DynamicBlock';
import DropoutLayer from './nodes/Dropout';
import LinearLayer from './nodes/Linear';

function FlowCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const nodeTypes = useMemo(
    () => ({
      tokenEmbedding: TokenEmbeddingLayer,
      positionalEmbedding: PositionalEmbeddingLayer,
      layerNorm: LayerNormLayer,
      dynamicBlock: DynamicBlock,
      feedForward: FeedForwardLayer,
      dropout: DropoutLayer,
      linear: LinearLayer,
      // transformerBlock: TransformerBlock,
      // maskedMultiHeadAttention: GPT2Nodes.MaskedMultiHeadAttention,
    }),
    [],
  );

  // Node Connect 이벤트 처리
  const onConnect = useCallback(
    (params: Edge<unknown> | Connection) =>
      setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Sidebar의 Node를 Canvas에 Drop 이벤트 처리
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const dataString = event.dataTransfer.getData('application/reactflow');
      if (!dataString) return;

      const parsedData = JSON.parse(dataString);
      const { nodeType, label, ...props } = parsedData;

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const id = `${nodeType}-${+new Date()}`;
      const newNode: Node = {
        id,
        type: nodeType,
        position,
        data: { id, label, ...props }, // id를 data에도 포함
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
        snapToGrid={true}
        snapGrid={[3, 3]}
        nodeTypes={nodeTypes} // 전달된 nodeTypes 매핑 적용
      >
        <Controls />
        {/* <MiniMap /> */}
      </ReactFlow>
    </div>
  );
}

export default FlowCanvas;
