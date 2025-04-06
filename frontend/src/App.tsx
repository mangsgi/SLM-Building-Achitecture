import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState } from 'react';

import CanvasHamburgerIcon from './ui-component/CanvasHamburgerIcon';
import DocumentWithGearIcon from './ui-component/DocumentWithGearIcon';
import Sidebar from './Sidebar';
import Config from './Config';
import FlowCanvas from './FlowCanvas';

function App() {
  // Sideber 토글을 위한 상태 변수
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);

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
      <div className="flex flex-grow relative">
        <ReactFlowProvider>
          {/* 사이드바가 열린 경우 Sidebar 랜더링*/}
          {isSidebarOpen && <Sidebar onToggle={toggleSidebar} />}
          {isConfigOpen && <Config onToggle={toggleConfig} />}
          {/* flex-1으로 FlowCanvas가 화면에서 가능한 많은 공간을 차지할 수 있도록 처리 */}
          <div className="flex-1 h-full">
            <FlowCanvas />
          </div>
          {/* 사이드바가 닫힌 경우 Canvas 위에 토글 버튼 오버레이 */}
          {!isSidebarOpen && (
            <button
              onClick={toggleSidebar}
              className="absolute top-4 left-4 z-10 p-2 bg-green-100 rounded focus:outline-none shadow"
              aria-label="Open Sidebar"
            >
              <CanvasHamburgerIcon />
            </button>
          )}
          {/* Config가 닫힌 경우 우측 상단에 토글 버튼 */}
          {!isConfigOpen && (
            <button
              onClick={toggleConfig}
              className="absolute top-4 right-4 z-10 p-2 bg-blue-100 rounded focus:outline-none shadow"
              aria-label="Open Config"
            >
              <DocumentWithGearIcon />
            </button>
          )}
        </ReactFlowProvider>
      </div>
    </div>
  );
}

export default App;
