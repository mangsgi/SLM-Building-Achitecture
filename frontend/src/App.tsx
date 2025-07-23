import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState, useRef } from 'react';
import type { Edge, Node } from 'reactflow';
import { useNavigate } from 'react-router-dom';

import CanvasHamburgerButton from './ui-component/CanvasHamburgerButton';
import ConfigButton from './ui-component/ConfigButton';
import NextButton from './ui-component/NextButton';
import Sidebar from './Sidebar';
import Config, { defaultConfig } from './Config';
import FlowCanvas from './FlowCanvas';
import { ReactFlowContext } from './store/ReactFlowContext';

// 모델을 구성하는 노드 타입
export interface ModelNode {
  type?: string;
  data: {
    id: string;
    label: string;
    [key: string]: unknown; // 필요하다면 더 세부적으로 정의 가능
  };
  children?: ModelNode[]; // Block 노드일 경우에만
}

// ✅ 백엔드에 보낼 모델 JSON 파일 구성 함수
async function buildModelJSON(
  nodes: Node[],
  edges: Edge[],
  config: Record<string, any>,
): Promise<ModelNode[]> {
  // emb_dim 짝수 유효성 검사 (Config)
  if (config.emb_dim && Number(config.emb_dim) % 2 !== 0) {
    alert(
      `Config의 Embedding Dimension(emb_dim)은 짝수여야 합니다. 현재 값: ${config.emb_dim}`,
    );
    return [];
  }

  // emb_dim 짝수 유효성 검사 (Nodes)
  for (const node of nodes) {
    if (node.data.embDim && Number(node.data.embDim) % 2 !== 0) {
      alert(
        `노드 '${node.data.label}'의 Embedding Dimension(embDim)은 짝수여야 합니다. 현재 값: ${node.data.embDim}`,
      );
      return [];
    }
  }

  // Llama3 GQA 유효성 검사
  if ('n_kv_groups' in config) {
    const gqaNodes = nodes.filter((n) => n.type === 'gqAttention');
    for (const node of gqaNodes) {
      const numHeads = Number(node.data.numHeads);
      const nKvGroups = Number(config.n_kv_groups);
      if (numHeads % nKvGroups !== 0) {
        alert(
          `GQA 노드 '${node.data.label}'의 numHeads(${numHeads})는 config의 n_kv_groups(${nKvGroups})로 나누어 떨어져야 합니다.`,
        );
        return [];
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
        alert(
          `TransformerBlock '${node.data.label}'의 Embedding Dimension(${embDim})은 Number of Heads(${numHeads})로 나누어 떨어져야 합니다.`,
        );
        return [];
      }

      const headDim = embDim / numHeads;
      if (headDim % 2 !== 0) {
        alert(
          `TransformerBlock '${node.data.label}'의 Head Dimension(emb_dim / numHeads)은 짝수여야 합니다. 현재 값: ${headDim}`,
        );
        return [];
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
    delete result.data.isTarget;
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
    alert(
      `⚠ 모델 구성 오류: 시작 노드가 ${rootNodes.length}개 존재합니다. 하나의 루트 노드만 있어야 합니다.`,
    );
    return [];
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

// ✅ 메인 컴포넌트
function App() {
  // Sideber와 Config 토글을 위한 상태 변수
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);
  const [config, setConfig] = useState<Record<string, any>>(defaultConfig);
  const navigate = useNavigate();

  // ✅ FlowCanvas에 전달할 데이터 참조 객체
  const flowDataRef = useRef<{ nodes: Node[]; edges: Edge[] }>({
    nodes: [],
    edges: [],
  });

  // ✅ 콜백함수를 인자로 전달하는 Setter를 호출하는 토글 함수 정의
  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleConfig = () => setIsConfigOpen((prev) => !prev);

  // ✅ 모델 전송 함수
  const handleSendModel = async () => {
    const { nodes, edges } = flowDataRef.current;
    const model = await buildModelJSON(nodes, edges, config);

    if (!model.length) {
      console.warn('모델 생성 실패 또는 구성 오류로 인해 이동 중단됨.');
      return;
    }

    navigate('/canvas/dataset', { state: { model, config } });
  };

  return (
    <div className="flex flex-col w-full h-screen">
      {/* Header 영역 */}
      <header className="bg-white p-4 shadow flex justify-between items-center">
        <h1 className="text-2xl font-semibold text-left">
          Building Your Own SLM
        </h1>
        <NextButton onClick={handleSendModel} text="Select Dataset" />
      </header>
      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-grow relative min-h-0">
        <ReactFlowProvider>
          <ReactFlowContext>
            {/* 사이드바가 열린 경우 Sidebar 랜더링*/}
            {isSidebarOpen && <Sidebar onToggle={toggleSidebar} />}
            {isConfigOpen && (
              <Config
                onToggle={toggleConfig}
                config={config}
                setConfig={setConfig}
              />
            )}

            {/* flex-1으로 FlowCanvas가 화면에서 가능한 많은 공간을 차지할 수 있도록 처리 */}
            <div className="flex-1 h-full">
              <FlowCanvas config={config} flowDataRef={flowDataRef} />
            </div>

            {/* 상단 왼쪽 버튼들 */}
            <div
              className={`absolute top-2 z-10 flex items-center gap-2 ${
                isSidebarOpen ? 'left-[250px] ml-2' : 'left-4'
              }`}
            >
              {!isSidebarOpen && (
                <div onClick={toggleSidebar}>
                  <CanvasHamburgerButton />
                </div>
              )}
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
    </div>
  );
}

export default App;
