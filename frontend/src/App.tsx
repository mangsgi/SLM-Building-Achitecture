import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';

import Sidebar from './Sidebar';
import FlowCanvas from './FlowCanvas';

function App() {
  return (
    // Tailwind 클래스: flex 컨테이너, 전체 너비/높이
    <div className="flex w-full h-screen">
      <ReactFlowProvider>
        <Sidebar />
        <FlowCanvas />
      </ReactFlowProvider>
    </div>
  );
}

export default App;
