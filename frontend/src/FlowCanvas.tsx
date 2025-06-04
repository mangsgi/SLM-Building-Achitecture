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
import GQAttentionLayer from './nodes/GQAttention';
// import TestBlock from './nodes/TestBlock';
import GPT2TransformerBlock from './nodes/GPT2TransformerBlock';
import TransformerBlock from './nodes/TransformerBlock';
import ResidualLayer from './nodes/Residual';
import NodeInfoModal from './nodes/components/NodeInfoModal';
import { BaseNodeData } from './nodes/components/NodeData';
import { defaultConfig } from './Config';
import ButtonEdge from './ButtonEdge';
import { flowContext } from './store/ReactFlowContext';
import { getNodeComponent } from './nodes/components/nodeRegistry';

const edgeTypes = { buttonEdge: ButtonEdge };

// ✅ Config로부터 Data를 받아 nodeType에 따라 node에 데이터 적용하는 함수
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
    case 'gpt2TransformerBlock':
      return {
        ...data,
      };
    case 'transformerBlock':
      return {
        ...data,
        numOfBlocks: config.n_blocks,
      };
    default:
      return data;
  }
}

interface FlowCanvasProps {
  config: typeof defaultConfig;
  flowDataRef: React.MutableRefObject<{ nodes: Node[]; edges: Edge[] }>;
}

// ✅ Canvas 메인 함수
export const FlowCanvas: React.FC<FlowCanvasProps> = ({
  config,
  flowDataRef,
}: FlowCanvasProps) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { getEdges } = useReactFlow();
  // onDrop 시 드롭된 노드의 정확한 위치를 계산하기 위해 DOM 요소 참조 & ReactFlowInstance 저장
  const { reactFlowInstance, setReactFlowInstance } = useContext(flowContext);

  useEffect(() => {
    flowDataRef.current = { nodes, edges };
  }, [nodes, edges]);

  // ✅ Config 값이 변경될 때마다 Node의 Data 업데이트
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
      gqAttention: GQAttentionLayer,
      testBlock: getNodeComponent('testBlock'),
      gpt2TransformerBlock: GPT2TransformerBlock,
      transformerBlock: TransformerBlock,
      residual: ResidualLayer,
    }),
    [],
  );
  const allowedTypes = [
    'testBlock',
    'transformerBlock',
    'gpt2TransformerBlock',
  ];

  // ✅ 노드 간 연결 이벤트 핸들
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
            newEdge.sourceHandle === 'residual-source' &&
            node.type === 'residual'
          ) {
            return {
              ...node,
              data: {
                ...node.data,
                source: newEdge.target,
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

  // ✅ Node Click 시 이벤트 핸들러
  const onNodeClick: NodeMouseHandler = (_, node) => {
    if (node && node.id) {
      // setClickedNodeId(node.id);
      // setNodeName(node.data.label);
      console.log(node);
    }
  };

  // ✅ Drag된 요소가 Canvas 위로 올라오는 이벤트 핸들러
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    // 사용자에게 이동 동작임을 시각적으로 표시
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // ✅ Node Drag가 시작되었을 때, Drag된 객체를 지정
  const onNodeDragStart: NodeDragHandler = (_, node) => {
    dragRef.current = node;
  };

  // ✅ Sidebar의 Node가 Canvas에 Drop되는 이벤트 처리
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
      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left + 160,
        y: event.clientY - reactFlowBounds.top + 50,
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

  // ✅ Drag 중인 Node가 중심에 위치한 Node를 target Node로 설정
  const onNodeDrag: NodeDragHandler = (_, node) => {
    // Node의 X 중심 좌표와 Y 중심 좌표 계산
    const centerX = (node.position.x + (node.width ?? 0)) / 2;
    const centerY = (node.position.y + (node.height ?? 0)) / 2;

    // 이전 타겟 노드의 isTarget 초기화
    if (target) {
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id === target.id) {
            return {
              ...n,
              data: { ...n.data, isTarget: false },
            };
          }
          return n;
        }),
      );
    }

    // Node의 중간 부분이 위치해있는 곳의 부모 Node 찾기
    const targetNode = nodes.find(
      (n) =>
        centerX > n.position.x &&
        centerX < n.position.x + (n.width ?? 0) &&
        centerY > n.position.y &&
        centerY < n.position.y + (n.height ?? 0) &&
        n.type?.includes('Block') &&
        n.id !== node.id,
    );

    console.log('node.position.x: ', node.position.x);
    console.log('node.position.y: ', node.position.y);
    console.log('node.width: ', node.width);
    console.log('node.height: ', node.height);
    console.log('targetNode.position.x: ', targetNode?.position.x);
    console.log('targetNode.position.y: ', targetNode?.position.y);
    console.log('targetNode.width: ', targetNode?.width);
    console.log('targetNode.height: ', targetNode?.height);

    if (targetNode) {
      // 타겟 노드의 isTarget 설정
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id === targetNode.id) {
            return {
              ...n,
              data: { ...n.data, isTarget: true },
            };
          }
          return n;
        }),
      );
      setTarget(targetNode as Node<BaseNodeData, string>);
    } else {
      setTarget(null);
    }
  };

  // ✅ Node 드래그가 끝났을 때
  const onNodeDragStop: NodeDragHandler = (_, node) => {
    // 이전 타겟 노드의 isTarget 초기화
    if (target) {
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id === target.id) {
            return {
              ...n,
              data: { ...n.data, isTarget: false },
            };
          }
          return n;
        }),
      );
    }

    // target이 존재할 경우에만 부모-자식 관계 설정
    if (target) {
      // Node의 부모 설정 여부 결정
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id === node.id) {
            // target이 Block이고 Node가 Block이 아닐 때 자식으로 처리하고 위치 지정
            if (
              !node.type?.includes('Block') &&
              target.type &&
              allowedTypes.includes(target.type)
            ) {
              // target의 자식 노드들을 찾아서 total height 계산
              const totalHeight = nodes.reduce((sum, child) => {
                if (child.parentNode === target?.id && child.id !== node.id) {
                  return sum + (child.height ?? 0) + 10;
                }
                return sum;
              }, 0);
              n.data = { ...n.data };
              n.parentNode = target?.id;
              n.position = { x: 10, y: 110 + totalHeight }; // Node의 위치 지정 in 부모 Node
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
            } else if (
              node.type?.includes('Block') &&
              target.type &&
              allowedTypes.includes(target.type)
            ) {
              console.log("Block can't includes Block Type.");
            }
          }
          return n;
        }),
      );
    }

    setTarget(null);
    dragRef.current = null;
  };

  // ✅ 노드 정보 modal을 위한 상태변수 저장
  const [globalModalData, setGlobalModalData] = useState<BaseNodeData | null>(
    null,
  );
  const showModal = (nodeData: BaseNodeData) => {
    setGlobalModalData(nodeData);
  };

  // ✅ 필드 정보 모달 상태 관리
  const [fieldInfoModal, setFieldInfoModal] = useState<{
    isOpen: boolean;
    info: { title: string; description: string } | null;
  }>({
    isOpen: false,
    info: null,
  });

  // ✅ 필드 정보 모달을 열기 위한 이벤트 리스너
  useEffect(() => {
    const handleFieldInfo = (event: CustomEvent) => {
      setFieldInfoModal({
        isOpen: true,
        info: event.detail,
      });
    };

    window.addEventListener('fieldInfo', handleFieldInfo as EventListener);
    return () => {
      window.removeEventListener('fieldInfo', handleFieldInfo as EventListener);
    };
  }, []);

  return (
    <div className="relative h-full bg-gray-50" ref={reactFlowWrapper}>
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
        defaultEdgeOptions={{
          zIndex: 20, // 모든 edge에 zIndex 높게 부여
        }}
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

      {/* 필드 정보 모달 */}
      <NodeInfoModal
        isOpen={fieldInfoModal.isOpen}
        onClose={() => setFieldInfoModal({ isOpen: false, info: null })}
      >
        {fieldInfoModal.info && (
          <>
            <h3 className="text-lg font-semibold mb-2">
              {fieldInfoModal.info.title}
            </h3>
            <p className="text-sm">{fieldInfoModal.info.description}</p>
          </>
        )}
      </NodeInfoModal>
    </div>
  );
};

export default FlowCanvas;
