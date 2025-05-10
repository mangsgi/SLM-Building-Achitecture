import React, {
  useCallback,
  useRef,
  useMemo,
  useState,
  useEffect,
  useContext,
} from 'react';
import ReactFlow, {
  useReactFlow,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Connection,
  NodeDragHandler,
  NodeMouseHandler,
  // ReactFlowInstance,
} from 'reactflow';
import type { Edge, Node } from 'reactflow';

import 'reactflow/dist/style.css';

import TokenEmbeddingLayer from './nodes/TokenEmbedding';
import PositionalEmbeddingLayer from './nodes/PositionalEmbedding';
import NormalizationLayer from './nodes/Normalization';
import FeedForwardLayer from './nodes/FeedForward';
import DropoutLayer from './nodes/Dropout';
import LinearLayer from './nodes/Linear';
import SDPAttentionLayer from './nodes/SDPAttention';
import TestBlock from './nodes/TestBlock';
import TransformerBlock from './nodes/TransformerBlock';
import DynamicBlock from './nodes/DynamicBlock';
import ResidualLayer from './nodes/Residual';
import NodeInfoModal from './nodes/components/NodeInfoModal';
import { BaseNodeData } from './nodes/components/NodeData';
import { defaultConfig } from './Config';
import ButtonEdge from './ButtonEdge';
import { flowContext } from './store/ReactFlowContext';

const edgeTypes = { buttonEdge: ButtonEdge };

// Config로부터 Data를 받아 nodeType에 따라 node에 데이터 적용하는 함수
function getNodeDataByType(
  nodeType: string,
  config: typeof defaultConfig,
  baseData: BaseNodeData,
): BaseNodeData {
  const data = { ...baseData, inDim: config.emb_dim, outDim: config.emb_dim };
  switch (nodeType) {
    case 'tokenEmbedding':
      return {
        ...data,
        vocabSize: config.vocab_size,
        embDim: config.emb_dim,
      };
    case 'positionalEmbedding':
      return {
        ...data,
        ctxLength: config.context_length,
        embDim: config.emb_dim,
      };
    case 'linear':
      return {
        ...data,
        outDim: config.vocab_size,
      };
    case 'dropout':
      return {
        ...data,
        dropoutRate: config.drop_rate,
      };
    case 'sdpAttention':
      return {
        ...data,
        ctxLength: config.context_length,
        dropoutRate: config.drop_rate,
        numHeads: config.n_heads,
        qkvBias: config.qkv_bias,
      };
    case 'transformerBlock':
      return {
        ...data,
      };
    case 'dynamicBlock':
      return {
        ...data,
        numLayers: config.n_layers,
      };
    default:
      return data;
  }
}

// Canvas 메인 함수
const FlowCanvas = ({
  config,
  flowDataRef,
}: {
  config: typeof defaultConfig;
  flowDataRef: React.MutableRefObject<{ nodes: Node[]; edges: Edge[] }>;
}) => {
  // ReactFlow에서 각 노드와 엣지 상태 저장
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { getEdges } = useReactFlow();
  // onDrop 시 드롭된 노드의 정확한 위치를 계산하기 위해 DOM 요소 참조 & ReactFlowInstance 저장
  const { reactFlowInstance, setReactFlowInstance } = useContext(flowContext);

  useEffect(() => {
    flowDataRef.current = { nodes, edges };
  }, [nodes, edges]);

  // Config 값이 변경될 때마다 Node의 Data 업데이트
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        return {
          ...node,
          data: getNodeDataByType(node.type || '', config, node.data),
        };
      }),
    );
  }, [config, setNodes]);

  // Drag 중인 Node가 목표할 Node 설정
  const [target, setTarget] = useState<Node<BaseNodeData, string> | null>(null);

  // Drag된 객체 지정
  const dragRef = useRef<Node | null>(null);

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
      normalization: NormalizationLayer,
      feedForward: FeedForwardLayer,
      dropout: DropoutLayer,
      linear: LinearLayer,
      sdpAttention: SDPAttentionLayer,
      testBlock: TestBlock,
      transformerBlock: TransformerBlock,
      dynamicBlock: DynamicBlock,
      residual: ResidualLayer,
    }),
    [],
  );
  const allowedTypes = ['testBlock', 'dynamicBlock'];

  // 노드 간 연결 이벤트 핸들
  const onConnect = useCallback(
    // 전달된 params를 기반으로 addEdge 헬퍼 함수를 사용해 현재 edges 상태에 새로운 edge를 추가
    (params: Edge<unknown> | Connection) => {
      const newEdge = {
        ...params,
        type: 'buttonEdge',
        id: `${params.source}-${params.sourceHandle}-${params.target}-${params.targetHandle}`,
      };
      console.log('Connecting Node via Handle: ', newEdge);

      // sourceHandle이 'residual'일 때 residual 노드의 data.source 업데이트
      setNodes((nodes) =>
        nodes.map((node) => {
          if (
            node.id === newEdge.source &&
            newEdge.sourceHandle === 'residual' &&
            node.type === 'residual'
          ) {
            return {
              ...node,
              data: {
                ...node.data,
                source: newEdge.target, // 연결된 노드를 저장
              },
            };
          }
          return node;
        }),
      );

      setEdges((eds) => {
        return addEdge(newEdge, eds);
      });
    },
    [setEdges],
  );

  // Node Click 시 이벤트 핸들러
  const onNodeClick: NodeMouseHandler = (_, node) => {
    if (node && node.id) {
      // setClickedNodeId(node.id);
      // setNodeName(node.data.label);
      console.log(node);
    }
  };

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

      const parsedData = JSON.parse(dataString);
      const { nodeType, id, label, ...props } = parsedData;

      // Node 위치 지정
      if (!reactFlowInstance) return;
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: Node = {
        id,
        type: nodeType,
        position,
        data: getNodeDataByType(nodeType, config, {
          id,
          label,
          openModal: showModal,
          ...props,
        }),
      };
      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, config],
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
        n.type?.includes('Block') &&
        n.id !== node.id,
    );

    console.log(
      `target: ${targetNode},\nnode x좌표: ${centerX},\nnode y좌표: ${centerY},\ntarget x좌표: ${targetNode?.position.x},\ntarget y좌표: ${targetNode?.position.y}`,
    );

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
        // target이 존재할 경우 Node 부모 설정 여부 결정
        if (n.id === node.id && target) {
          // target의 자식 노드들을 찾아서 total height 계산
          const targetChildren = nodes.filter(
            (n) => n.parentNode === target?.id && n.id != node.id,
          );
          const totalHeight = targetChildren.reduce(
            (sum, child) => 10 + sum + (child.height ?? 0),
            0,
          );
          // target이 Block이고 Node가 Block이 아닐 때
          if (
            !node.type?.includes('Block') &&
            target.type &&
            allowedTypes.includes(target.type)
          ) {
            n.data = { ...n.data };
            n.parentNode = target?.id;
            n.position = { x: 10, y: 110 + totalHeight }; // 노드의 위치 지정 **in 부모 Node**
            n.extent = 'parent'; // Node의 이동반경을 부모 Node 안으로 제한
            n.draggable = false; // Node가 Drag 되지 않음
            n.data.hideHandles = true; // Edge Handle 부분 숨기기

            // Node의 기존 Edge 모두 삭제
            const relatedEdges = getEdges().filter(
              (e) => e.source === node.id || e.target === node.id,
            );
            if (relatedEdges.length > 0) {
              setEdges((edges) =>
                edges.filter(
                  (e) => e.source !== node.id && e.target !== node.id,
                ),
              );
            }
            // Node도 Block일 경우
          } else if (
            node.type?.includes('Block') &&
            target.type &&
            allowedTypes.includes(target.type)
          ) {
            console.log("Block can't includes Block Type.");
          }
          // target과 Node가 같으면 그냥 둠
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
        edgeTypes={edgeTypes}
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
};

export default FlowCanvas;
