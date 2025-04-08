import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState, useRef } from 'react';
import type { Edge, Node } from 'reactflow';

import CanvasHamburgerButton from './ui-component/CanvasHamburgerButton';
import ConfigButton from './ui-component/ConfigButton';
import SendModelButton from './ui-component/SendModelButton';
import Sidebar from './Sidebar';
import Config, { defaultConfig } from './Config';
import FlowCanvas from './FlowCanvas';

function App() {
  // Sideber와 Config 토글을 위한 상태 변수
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);
  // 모델 Config 변수
  const [config, setConfig] = useState(defaultConfig);

  const flowDataRef = useRef<{ nodes: Node[]; edges: Edge[] }>({
    nodes: [],
    edges: [],
  });

  // 콜백함수를 인자로 전달하는 Setter를 호출하는 토글 함수 정의
  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleConfig = () => setIsConfigOpen((prev) => !prev);

  return (
    <div className="flex flex-col w-full h-screen">
      {/* Header 영역 */}
      <header className="bg-white p-4 shadow">
        <h1 className="text-2xl font-semibold text-left">
          나만의 작은 언어 모델 만들기
        </h1>
      </header>
      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-grow relative min-h-0">
        <ReactFlowProvider>
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
            className={`absolute top-4 z-10 flex items-center ${
              isSidebarOpen ? 'left-[16.6667%]' : 'left-4'
            }`}
          >
            {!isSidebarOpen && (
              <div onClick={toggleSidebar} aria-label="Open Sidebar">
                <CanvasHamburgerButton />
              </div>
            )}
            <SendModelButton
              onClick={() => {
                const { nodes, edges } = flowDataRef.current;
                console.log('📤 Nodes:', nodes);
                console.log('📤 Edges:', edges);
              }}
            />
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
        </ReactFlowProvider>
      </div>
    </div>
  );
}

export default App;
