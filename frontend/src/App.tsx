import { ReactFlowProvider, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState, useEffect } from 'react';
import type { Edge, Node } from 'reactflow';
import { useNavigate } from 'react-router-dom';

import CanvasHamburgerButton from './ui-component/CanvasHamburgerButton';
import ConfigButton from './ui-component/ConfigButton';
import SendModelButton from './ui-component/SendModelButton';
import Sidebar from './Sidebar';
import Config from './Config';
import {
  ModelConfig,
  GPT2Config,
  Llama2Config,
  Llama3Config,
  modelConfigs,
  Qwen3Config,
} from './constants/modelConfigs';
import FlowCanvas from './FlowCanvas';
import { ReactFlowContext } from './store/ReactFlowContext';
import Header from './ui-component/Header';
import ModelButton from './ui-component/TestModelButton';
import { referenceModels } from './constants/referenceModels';
import Modal from './ui-component/Modal'; // Modal 컴포넌트 import 추가

// 모델을 구성하는 노드 타입
export interface ModelNode {
  type?: string;
  data: {
    id: string;
    label: string;
    [key: string]: unknown;
  };
  children?: ModelNode[]; // Block 노드일 경우에만
}

// 백엔드에 보낼 모델 JSON 파일 구성 함수
async function buildModelJSON(
  nodes: Node[],
  edges: Edge[],
  config: Record<string, any>,
): Promise<ModelNode[]> {
  // emb_dim 짝수 유효성 검사 (Config)
  if (config.emb_dim && Number(config.emb_dim) % 2 !== 0) {
    throw new Error(
      `Config의 Embedding Dimension(emb_dim)은 짝수여야 합니다. 현재 값: ${config.emb_dim}`,
    );
  }

  // emb_dim 짝수 유효성 검사 (Nodes)
  for (const node of nodes) {
    if (node.data.embDim && Number(node.data.embDim) % 2 !== 0) {
      throw new Error(
        `노드 '${node.data.label}'의 Embedding Dimension(embDim)은 짝수여야 합니다. 현재 값: ${node.data.embDim}`,
      );
    }
  }

  // Llama3 GQA 유효성 검사
  if ('n_kv_groups' in config) {
    const gqaNodes = nodes.filter((n) => n.type === 'gqAttention');
    for (const node of gqaNodes) {
      const numHeads = Number(node.data.numHeads);
      const nKvGroups = Number(config.n_kv_groups);
      if (numHeads % nKvGroups !== 0) {
        throw new Error(
          `GQA 노드 '${node.data.label}'의 numHeads(${numHeads})는 config의 n_kv_groups(${nKvGroups})로 나누어 떨어져야 합니다.`,
        );
      }
    }
  }

  // TransformerBlock의 Head Dimension (head_dim) 유효성 검사
  for (const node of nodes) {
    if (!node.data.isLocked && node.type === 'transformerBlock') {
      const embDim = Number(config.emb_dim);
      const numHeads = Number(node.data.numHeads);

      if (!embDim || !numHeads) continue; // 필요한 값이 없으면 건너뜀

      if (embDim % numHeads !== 0) {
        throw new Error(
          `TransformerBlock '${node.data.label}'의 Embedding Dimension(${embDim})은 Number of Heads(${numHeads})로 나누어 떨어져야 합니다.`,
        );
      }

      const headDim = embDim / numHeads;
      if (headDim % 2 !== 0) {
        throw new Error(
          `TransformerBlock '${node.data.label}'의 Head Dimension(emb_dim / numHeads)은 짝수여야 합니다. 현재 값: ${headDim}`,
        );
      }
    }
  }

  // 1. 노드 맵 생성
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  // console.log('🔍 Nodes:', nodes);
  // console.log('🔍 Edges:', edges);

  // 2. in-degree 계산 (Residual Edge 제외)
  const inDegree = new Map<string, number>();
  nodes.forEach((n) => inDegree.set(n.id, 0));
  edges.forEach((edge) => {
    // Residual 연결을 위한 엣지는 in-degree 계산에서 제외
    if (
      edge.sourceHandle === 'residual-source' ||
      edge.targetHandle === 'residual-target'
    ) {
      return; // 이 엣지는 건너뜁니다.
    }
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  });

  // 3. 인접 리스트 생성
  const adj = new Map<string, string[]>();
  edges.forEach((edge) => {
    if (!adj.has(edge.source)) adj.set(edge.source, []);
    adj.get(edge.source)?.push(edge.target);
  });

  // 4. DFS 수행 함수
  const visited = new Set<string>();

  function dfs(nodeId: string): ModelNode[] {
    if (visited.has(nodeId)) return [];
    visited.add(nodeId);

    const node = nodeMap.get(nodeId);
    if (!node) return [];

    const { type, data } = node;
    const result: ModelNode = {
      type,
      data: { ...data },
    };

    // Node에서 필요없는 데이터 제거
    delete result.data.openModal;
    delete result.data.hideHandles;
    delete result.data.isCollapsed;
    delete result.data.isTarget;
    delete result.data.isLocked;
    delete (result.data as any).label;

    // Block 노드이면 children도 탐색
    const isBlock = type?.includes('Block');
    if (isBlock) {
      result.children = [];
      // Block 내부 자식 노드 순서 보장
      result.children = nodes
        .filter((n) => n.parentNode === nodeId)
        .sort((a, b) => (a.position.y || 0) - (b.position.y || 0))
        .map((child) => {
          const childData = { ...child.data };
          delete childData.openModal;
          delete childData.hideHandles;
          delete childData.isCollapsed;
          delete childData.isTarget;
          delete childData.isLocked;
          return {
            type: child.type,
            data: childData,
          };
        });
    }

    const results: ModelNode[] = [result];

    // 일반 노드인 경우에도 & Block을 다 순회하고 다음 노드 DFS
    const nextIds = adj.get(nodeId) || [];
    for (const nextId of nextIds) {
      results.push(...dfs(nextId));
    }

    return results;
  }

  // 5. 진입점에서부터 DFS 실행
  // 5-1. 루트 노드 찾기 (루트 노드는 진입점이 하나인 노드)
  const rootNodes = Array.from(inDegree.entries()).filter(
    ([id, deg]) => deg === 0 && !nodeMap.get(id)?.parentNode,
  );

  // 5-2. 예외 처리
  if (rootNodes.length !== 1) {
    throw new Error(
      `⚠ 모델 구성 오류: 시작 노드가 ${rootNodes.length}개 존재합니다. 하나의 루트 노드만 있어야 합니다.`,
    );
  }

  // 5-3. DFS 실행
  const model: ModelNode[] = [];
  for (const [nodeId] of rootNodes) {
    const dfsResult = dfs(nodeId);
    model.push(...dfsResult);
  }

  // console.log('📦 Generated Model JSON:', model);

  return model;
}

// 메인 컴포넌트
function App() {
  // Sideber와 Config 토글을 위한 상태 변수
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);

  // 로컬 스토리지에서 모델 타입을 불러와 config 상태를 초기화
  const [config, setConfig] = useState<ModelConfig>(() => {
    const savedModelType = localStorage.getItem('modelType');
    if (savedModelType === 'gpt-2') {
      return { ...modelConfigs['GPT-2'], model: 'gpt-2' } as GPT2Config;
    } else if (savedModelType === 'llama2') {
      return { ...modelConfigs['Llama2'], model: 'llama2' } as Llama2Config;
    } else if (savedModelType === 'llama3') {
      return { ...modelConfigs['Llama3'], model: 'llama3' } as Llama3Config;
    } else if (savedModelType === 'qwen3') {
      return { ...modelConfigs['Qwen3'], model: 'qwen3' } as Qwen3Config;
    }

    return { ...modelConfigs['GPT-2'], model: 'gpt-2' } as GPT2Config;
  });

  const navigate = useNavigate();

  // 오류 모달 상태 추가
  const [errorModal, setErrorModal] = useState<{
    isOpen: boolean;
    message: string;
  }>({ isOpen: false, message: '' });

  // 로컬 스토리지에서 상태를 불러오거나 기본값으로 초기화
  const initialFlowState = () => {
    try {
      const savedState = localStorage.getItem('canvasState');
      if (savedState) {
        const { nodes, edges } = JSON.parse(savedState);
        // 노드와 엣지에 대한 기본 유효성 검사
        if (Array.isArray(nodes) && Array.isArray(edges)) {
          return { nodes, edges };
        }
      }
    } catch (error) {
      console.error('저장된 캔버스 상태를 불러오는 데 실패했습니다:', error);
    }
    // 저장된 상태가 없거나 유효하지 않으면 기본값 반환
    return { nodes: [], edges: [] };
  };

  const [nodes, setNodes, onNodesChange] = useNodesState(
    initialFlowState().nodes,
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    initialFlowState().edges,
  );

  // 캔버스 상태를 로컬 스토리지에 저장
  useEffect(() => {
    try {
      const canvasState = JSON.stringify({ nodes, edges });
      localStorage.setItem('canvasState', canvasState);
    } catch (error) {
      console.error('캔버스 상태를 저장하는 데 실패했습니다:', error);
    }
  }, [nodes, edges]);

  // config가 변경될 때마다 모델 타입을 로컬 스토리지에 저장
  useEffect(() => {
    if (config && config.model) {
      localStorage.setItem('modelType', config.model);
    }
  }, [config]);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleConfig = () => setIsConfigOpen((prev) => !prev);

  const loadReferenceModel = async (
    modelName: 'GPT-2' | 'Llama2' | 'Llama3' | 'Qwen3',
  ) => {
    const model = referenceModels[modelName];
    if (!model || !model.nodes || !model.edges || model.nodes.length === 0) {
      setErrorModal({
        isOpen: true,
        message: `Reference model '${modelName}' is empty. Please add nodes and edges to src/constants/referenceModels.ts`,
      });
      return;
    }
    setNodes(model.nodes);
    setEdges(model.edges);
  };

  // 모델 전송 함수
  const handleSendModel = async () => {
    // 모델 다운로드 (Reference 생성 시 주석 해제)
    // const flowState = { nodes, edges };
    // const jsonString = JSON.stringify(flowState, null, 2);
    // const blob = new Blob([jsonString], { type: 'application/json' });
    // const url = URL.createObjectURL(blob);
    // const link = document.createElement('a');
    // link.href = url;
    // link.download = 'reactflow-state.json';
    // document.body.appendChild(link);
    // link.click();
    // document.body.removeChild(link);
    // URL.revokeObjectURL(url);

    try {
      const model = await buildModelJSON(nodes, edges, config);

      if (!model.length) {
        console.warn('모델 생성 실패 또는 구성 오류로 인해 이동 중단됨.');
        return;
      }

      navigate('/canvas/dataset', { state: { model, config } });
    } catch (e: any) {
      setErrorModal({ isOpen: true, message: e.message });
    }
  };

  const handleTestModelClick = () => {
    navigate('/test');
  };

  return (
    <div className="flex flex-col w-full h-screen">
      {/* Header 영역 */}
      <Header>
        <div className="flex items-center gap-4">
          <ModelButton onClick={handleTestModelClick} text="Test Model" />
          <SendModelButton onClick={handleSendModel} text="Select Dataset" />
        </div>
      </Header>
      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-grow relative min-h-0">
        <ReactFlowProvider>
          <ReactFlowContext>
            <div
              className={`absolute top-2 z-10 flex items-center gap-2 transition-all duration-300 ease-in-out ${
                isSidebarOpen ? 'left-[170px] ml-2' : 'left-4' // 250px -> 170px
              }`}
              onClick={toggleSidebar}
            >
              <CanvasHamburgerButton />
            </div>
            <div
              className={`transition-all duration-300 ease-in-out overflow-hidden ${
                isSidebarOpen ? 'w-[220px]' : 'w-0' // 250px -> 220px
              }`}
            >
              <Sidebar loadReferenceModel={loadReferenceModel} />
            </div>

            <div className="flex-1 h-full relative">
              <FlowCanvas
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                setNodes={setNodes}
                setEdges={setEdges}
                config={config}
              />
            </div>

            {/* Config 토글 버튼 */}
            <div
              onClick={toggleConfig}
              className={`absolute top-2 z-10 transition-all duration-300 ease-in-out ${
                isConfigOpen ? 'right-[220px] mr-2' : 'right-2'
              }`}
              aria-label="Toggle Config"
            >
              <ConfigButton />
            </div>
            <div
              className={`absolute top-0 right-0 h-1/2 bg-white shadow-lg transition-all duration-300 ease-in-out overflow-hidden ${
                isConfigOpen ? 'w-[220px]' : 'w-0'
              }`}
            >
              <Config config={config} setConfig={setConfig} />
            </div>
          </ReactFlowContext>
        </ReactFlowProvider>
      </div>

      {/* 오류 표시를 위한 Modal 컴포넌트 */}
      <Modal
        isOpen={errorModal.isOpen}
        onClose={() => setErrorModal({ isOpen: false, message: '' })}
        title="모델 구성 오류"
      >
        <p className="text-sm text-gray-600">{errorModal.message}</p>
      </Modal>
    </div>
  );
}

export default App;
