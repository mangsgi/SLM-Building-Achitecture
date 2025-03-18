import React, { useCallback, useRef } from 'react';
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  // MiniMap,
  Controls,
  Connection,
  Edge,
  Node,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

function FlowCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback(
    (params: Edge<unknown> | Connection) =>
      setEdges((eds) =>
        addEdge(
          { ...params, markerEnd: { type: MarkerType.ArrowClosed } },
          eds,
        ),
      ),
    [setEdges],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const nodeType = event.dataTransfer.getData('application/reactflow');
      if (!nodeType) return;

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: Node = {
        id: `${nodeType}-${+new Date()}`,
        type: 'default',
        position,
        data: { label: `${nodeType} Node` },
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
      >
        <Controls />
        {/* <MiniMap /> */}
      </ReactFlow>
    </div>
  );
}

export default FlowCanvas;
