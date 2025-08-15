import React, { useCallback, useRef, useState } from 'react';
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  MiniMap,
  Controls,
  Connection,
  Edge,
  Node,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface ModelConfig {
  model: string;
  epochs: number;
  batch_size: number;
  vocab_size: number;
  context_length: number;
  emb_dim: number;
  n_heads: number;
  n_blocks: number;
  drop_rate: number;
  qkv_bias: boolean;
  dtype: string;
}

interface LayerData {
  id: string;
  label: string;
  inDim?: number;
  outDim?: number;
  vocabSize?: number;
  embDim?: number;
  ctxLength?: number;
  dropoutRate?: number;
  numOfFactor?: number;
  source?: string;
  numHeads?: number;
  qkvBias?: boolean;
  numOfBlocks?: number;
  numKvGroups?: number;
}

interface LayerNode {
  type: string;
  data: LayerData;
  children?: LayerNode[];
}

interface CompleteModelRequest {
  config: ModelConfig;
  model: LayerNode[];
  dataset: string;
  modelName: string;
}

function FlowCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    model: "gpt-2",
    epochs: 10,
    batch_size: 8,
    vocab_size: 50257,
    context_length: 1024,
    emb_dim: 768,
    n_heads: 12,
    n_blocks: 12,
    drop_rate: 0.1,
    qkv_bias: false,
    dtype: "bf16"
  });
  const [modelName, setModelName] = useState("my-slm-model");
  const [dataset, setDataset] = useState("Dataset 4");

  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // 노드 연결 이벤트 처리 (타입에 unknown 사용)
  const onConnect = useCallback(
    (params: Edge<unknown> | Connection) =>
      setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  // 캔버스 위에 드래그 중일 때
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // 드롭 시 새로운 노드 생성
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

  // 노드를 모델 구조로 변환하는 함수
  const convertNodesToModelStructure = (): LayerNode[] => {
    const sortedNodes = [...nodes].sort((a, b) => {
      // Y 좌표로 정렬 (위에서 아래로)
      return a.position.y - b.position.y;
    });

    return sortedNodes.map(node => {
      const nodeType = node.data.label?.replace(' Node', '').toLowerCase();
      const layerData: LayerData = {
        id: node.id,
        label: node.data.label || node.id,
        inDim: modelConfig.emb_dim,
        outDim: modelConfig.emb_dim,
      };

             // 노드 타입에 따른 추가 데이터 설정
       switch (nodeType) {
         case 'tokenembedding':
           layerData.vocabSize = modelConfig.vocab_size;
           layerData.embDim = modelConfig.emb_dim;
           break;
         case 'positionalembedding':
           layerData.embDim = modelConfig.emb_dim;
           layerData.ctxLength = modelConfig.context_length;
           break;
         case 'dropout':
           layerData.dropoutRate = modelConfig.drop_rate;
           break;
         case 'sdpattention':
         case 'flashattention':
         case 'gqaattention':
           layerData.numHeads = modelConfig.n_heads;
           layerData.ctxLength = modelConfig.context_length;
           layerData.dropoutRate = modelConfig.drop_rate;
           layerData.qkvBias = modelConfig.qkv_bias;
           if (nodeType === 'gqaattention') {
             layerData.numKvGroups = Math.floor(modelConfig.n_heads / 2); // 기본값
           }
           break;
         case 'feedforward':
           layerData.numOfFactor = 4;
           break;
         case 'transformerblock':
           layerData.numOfBlocks = modelConfig.n_blocks;
           break;
         case 'linear':
           layerData.outDim = modelConfig.vocab_size;
           break;
       }

      return {
        type: nodeType || 'unknown',
        data: layerData
      };
    });
  };

  // 모델 구조 검증
  const validateModelStructure = async () => {
    try {
      const modelStructure = convertNodesToModelStructure();
      const requestData: CompleteModelRequest = {
        config: modelConfig,
        model: modelStructure,
        dataset,
        modelName
      };

      const response = await fetch('http://localhost:8000/validate-complete-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const result = await response.json();
      
      if (response.ok) {
        alert('✅ 모델 구조 검증 성공!\n' + JSON.stringify(result, null, 2));
      } else {
        alert('❌ 모델 구조 검증 실패!\n' + result.detail);
      }
    } catch (error) {
      alert('❌ 요청 실패: ' + error);
    }
  };

  // 모델 생성
  const buildModel = async () => {
    try {
      const modelStructure = convertNodesToModelStructure();
      const requestData: CompleteModelRequest = {
        config: modelConfig,
        model: modelStructure,
        dataset,
        modelName
      };

      const response = await fetch('http://localhost:8000/build-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const result = await response.json();
      
      if (response.ok) {
        alert('✅ 모델 생성 성공!\n' + JSON.stringify(result, null, 2));
      } else {
        alert('❌ 모델 생성 실패!\n' + result.detail);
      }
    } catch (error) {
      alert('❌ 요청 실패: ' + error);
    }
  };

  return (
    // Tailwind 클래스: 80% 너비, 전체 높이, 어두운 배경
    <div ref={reactFlowWrapper} className="w-4/5 h-full bg-gray-900 relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        <Controls />
        <MiniMap />
      </ReactFlow>
      
      {/* 모델 설정 및 액션 버튼 */}
      <div className="absolute top-4 right-4 bg-gray-800 p-4 rounded-lg text-white">
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">모델 이름:</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="w-full px-3 py-1 bg-gray-700 rounded text-sm"
          />
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">데이터셋:</label>
          <input
            type="text"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full px-3 py-1 bg-gray-700 rounded text-sm"
          />
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">임베딩 차원:</label>
          <input
            type="number"
            value={modelConfig.emb_dim}
            onChange={(e) => setModelConfig(prev => ({...prev, emb_dim: parseInt(e.target.value)}))}
            className="w-full px-3 py-1 bg-gray-700 rounded text-sm"
          />
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">어텐션 헤드 수:</label>
          <input
            type="number"
            value={modelConfig.n_heads}
            onChange={(e) => setModelConfig(prev => ({...prev, n_heads: parseInt(e.target.value)}))}
            className="w-full px-3 py-1 bg-gray-700 rounded text-sm"
          />
        </div>
        
        <div className="space-y-2">
          <button
            onClick={validateModelStructure}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm"
          >
            모델 구조 검증
          </button>
          <button
            onClick={buildModel}
            className="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded text-sm"
          >
            모델 생성
          </button>
        </div>
      </div>
    </div>
  );
}

export default FlowCanvas;
