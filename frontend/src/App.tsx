import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { useState } from 'react';

import CanvasHamburgerIcon from './ui-component/CanvasHamburgerIcon';
import Sidebar from './Sidebar';
import FlowCanvas from './FlowCanvas';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);

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
          {isSidebarOpen && <Sidebar onToggle={toggleSidebar} />}
          {/* FlowCanvas를 감싸는 래퍼에 flex-1 추가 */}
          <div className="flex-1">
            <FlowCanvas />
          </div>
          {/* 사이드바가 닫힌 경우 캔버스 위에 토글 버튼 오버레이 */}
          {!isSidebarOpen && (
            <button
              onClick={toggleSidebar}
              className="absolute top-4 left-4 z-10 p-2 bg-green-100 rounded focus:outline-none shadow"
              aria-label="Open Sidebar"
            >
              <CanvasHamburgerIcon />
            </button>
          )}
        </ReactFlowProvider>
      </div>
    </div>
  );
}

export default App;
