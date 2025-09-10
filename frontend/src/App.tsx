import { ReactFlowProvider, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState, useEffect } from 'react';
import type { Edge, Node } from 'reactflow';
import { useNavigate } from 'react-router-dom';

import CanvasHamburgerButton from './ui-component/CanvasHamburgerButton';
import ConfigButton from './ui-component/ConfigButton';
import SendModelButton from './ui-component/SendModelButton';
import Sidebar from './Sidebar';
import Config, { defaultConfig } from './Config';
import FlowCanvas from './FlowCanvas';
import { ReactFlowContext } from './store/ReactFlowContext';
import Header from './ui-component/Header';
import ModelButton from './ui-component/TestModelButton';
import { referenceNodes, referenceEdges } from './constants/referenceModels';
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
  console.log('🔍 Nodes:', nodes);
  console.log('🔍 Edges:', edges);

  // 2. in-degree 계산
  const inDegree = new Map<string, number>();
  nodes.forEach((n) => inDegree.set(n.id, 0));
  edges.forEach((edge) => {
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
  // 5-1. 루트 노드 찾기
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

  console.log('📦 Generated Model JSON:', model);

  return model;
}

// 메인 컴포넌트
function App() {
  // Sideber와 Config 토글을 위한 상태 변수
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);
  const [config, setConfig] = useState<Record<string, any>>(defaultConfig);
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

  // Save flow state to local storage whenever it changes
  useEffect(() => {
    try {
      const canvasState = JSON.stringify({ nodes, edges });
      localStorage.setItem('canvasState', canvasState);
    } catch (error) {
      console.error('캔버스 상태를 저장하는 데 실패했습니다:', error);
    }
  }, [nodes, edges]);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleConfig = () => setIsConfigOpen((prev) => !prev);

  const loadReferenceModel = async () => {
    if (!referenceNodes || !referenceEdges || referenceNodes.length === 0) {
      setErrorModal({
        isOpen: true,
        message:
          'Reference model is empty. Please add nodes and edges to src/constants/reference-model.ts',
      });
      return;
    }
    setNodes(referenceNodes);
    setEdges(referenceEdges);
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
            {/* 사이드바가 열린 경우 Sidebar 랜더링*/}
            <div
              className={`transition-all duration-300 ease-in-out ${isSidebarOpen ? 'w-[250px]' : 'w-0'}`}
            >
              <Sidebar loadReferenceModel={loadReferenceModel} />
            </div>
            {isConfigOpen && (
              <Config
                onToggle={toggleConfig}
                config={config}
                setConfig={setConfig}
              />
            )}

            {/* flex-1으로 FlowCanvas가 화면에서 가능한 많은 공간을 차지할 수 있도록 처리 */}
            <div className="flex-1 h-full">
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

            {/* 상단 왼쪽 버튼들 */}
            <div
              className={`absolute top-2 z-10 flex items-center gap-2 transition-all duration-300 ease-in-out ${
                isSidebarOpen ? 'left-[250px] ml-2' : 'left-4'
              }`}
            >
              <div onClick={toggleSidebar}>
                <CanvasHamburgerButton />
              </div>
            </div>

            {/* Config가 닫힌 경우 우측 상단에 토글 버튼 */}
            {!isConfigOpen && (
              <div
                onClick={toggleConfig}
                className="absolute top-4 right-4 z-10"
                aria-label="Open Config"
              >
                <ConfigButton />
              </div>
            )}
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
