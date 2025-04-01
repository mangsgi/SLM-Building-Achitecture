import React, {
  useCallback,
  useRef,
  useMemo,
  useState,
  // useEffect,
} from 'react';
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Connection,
  Edge,
  Node,
  NodeDragHandler,
  NodeMouseHandler,
  ReactFlowInstance,
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
import NodeInfoModal from './nodes/components/NodeInfoModal';
import { BaseNodeData } from './nodes/components/NodeData';

function FlowCanvas() {
  // ReactFlow에서 각 노드와 엣지 상태 저장
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Drag 중인 Node가 목표할 Node 설정
  const [target, setTarget] = useState<Node<BaseNodeData, string> | null>(null);
  // const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  // const [nodeName, setNodeName] = useState('');

  // Drag된 객체 지정
  const dragRef = useRef<Node | null>(null);

  // onDrop 시 드롭된 노드의 정확한 위치를 계산하기 위해 DOM 요소 참조
  const [reactFlowInstance, setReactFlowInstance] =
    useState<ReactFlowInstance | null>(null);
  // const reactFlowWrapper = useRef(null);

  // 노드 정보 modal을 위한 상태변수 저장
  const [globalModalData, setGlobalModalData] = useState<BaseNodeData | null>(
    null,
  );
  const showModal = (nodeData: BaseNodeData) => {
    setGlobalModalData(nodeData);
  };

  // nodeTypes 매핑
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

  // Node Click 시
  const onNodeClick: NodeMouseHandler = (_, node) => {
    if (node && node.id) {
      // setClickedNodeId(node.id);
      // setNodeName(node.data.label);
      console.log(node);
    }
  };

  // 사용자가 특정 노드를 클릭했을 때나 노드 이름을 변경했을 때 해당 노드의 label을 업데이트
  // useEffect(() => {
  //   setNodes((nds) =>
  //     nds.map((node) => {
  //       if (node.id === clickedNodeId) {
  //         return {
  //           ...node,
  //           data: {
  //             ...node.data,
  //             label: nodeName,
  //           },
  //         };
  //       }

  //       return node;
  //     }),
  //   );
  // }, [clickedNodeId, nodeName, setNodes]);

  // 노드 간 연결 이벤트 핸들
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

  // Node Drag가 시작되었을 때, Drag된 객체를 지정
  const onNodeDragStart: NodeDragHandler = (_, node) => {
    dragRef.current = node;
  };

  // Sidebar의 Node가 Canvas에 Drop되는 이벤트 처리
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      // Data 파싱
      const dataString = event.dataTransfer.getData('application/reactflow');
      if (!dataString) return;
      console.log(dataString);
      const parsedData = JSON.parse(dataString);
      const { nodeType, id, label, ...props } = parsedData;

      // Node 위치 지정
      if (!reactFlowInstance) return;
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      console.log(position.x, position.y);

      const newNode: Node = {
        id,
        type: nodeType,
        position,
        data: { id, label, openModal: showModal, ...props },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance],
  );

  // Drag 중인 Node가 중심에 위치한 Node를 target Node로 설정
  const onNodeDrag: NodeDragHandler = (_, node) => {
    // Node의 X 중심 좌표와 Y 중심 좌표 계산
    const centerX = node.position.x + (node.width ?? 0) / 2;
    const centerY = node.position.y + (node.height ?? 0) / 2;

    // Node의 중간 부분이 위치해있는 곳의 Node 찾기
    const targetNode = nodes.find(
      (n) =>
        centerX > n.position.x &&
        centerX < n.position.x + (n.height ?? 0) &&
        centerY > n.position.y &&
        centerY < n.position.y + (n.height ?? 0) &&
        n.type === 'maskedMHABlock' &&
        n.id !== node.id,
    );

    console.log(target);
    if (targetNode) {
      setTarget(targetNode as Node<BaseNodeData, string>);
    } else {
      setTarget(null);
    }
  };

  // Drag를 멈췄을 때 부모 자녀 관계 설정
  const onNodeDragStop: NodeDragHandler = (_, node) => {
    console.log(node, target);
    setNodes((nodes) =>
      nodes.map((n) => {
        if (n.id === node.id && target) {
          // target의 자식 노드들을 찾아서 total height 계산
          const targetChildren = nodes.filter(
            (n) => n.parentNode === target?.id && n.id != node.id,
          );
          const totalHeight = targetChildren.reduce(
            (sum, child) => 10 + sum + (child.height ?? 0),
            0,
          );
          if (node.type !== 'maskedMHABlock') {
            n.data = { ...n.data };
            n.parentNode = target?.id;
            n.position = { x: 10, y: 110 + totalHeight }; // 노드의 위치 지정 **in 부모 Node**
            n.extent = 'parent'; // Node의 이동반경을 부모 Node 안으로 제한
            n.draggable = false; // Node가 Drag 되지 않음
            n.data.hideHandles = true;
          }
        } else if (n.id === target?.id) {
          n.data = { ...n.data };
        }
        return n;
      }),
    );
    setTarget(null);
    dragRef.current = null;
  };

  return (
    <div className="w-full h-full bg-gray-50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance} // Viewport가 초기화될 때 콜백 함수
        onDrop={onDrop}
        onDragOver={onDragOver}
        snapToGrid={true} // 노드의 부드러운 이동 적용
        snapGrid={[3, 3]}
        nodeTypes={nodeTypes} // 전달된 nodeTypes 매핑 적용
        onNodeDragStart={onNodeDragStart}
        onNodeDrag={onNodeDrag}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={onNodeClick}
      >
        <Controls />
      </ReactFlow>

      {/* 전역 모달: globalModalData가 있으면 전체 화면 모달을 표시 */}
      {globalModalData && (
        <NodeInfoModal isOpen={true} onClose={() => setGlobalModalData(null)}>
          <h3 className="text-lg font-semibold mb-2">Node 정보</h3>
          <p className="text-sm">
            {globalModalData.label} 노드에 대한 추가 정보입니다.
          </p>
          {/* 노드별 추가 정보 렌더링 */}
        </NodeInfoModal>
      )}
    </div>
  );
}

export default FlowCanvas;
