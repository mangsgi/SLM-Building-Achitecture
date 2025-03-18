import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';

interface SidebarProps {
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onToggle }) => {
  // 드래그 시작 시 dataTransfer에 노드 타입 저장
  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
  ) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="w-1/5 h-full bg-green-100 p-4">
      <div className="flex items-center justify-end">
        <button
          onClick={onToggle}
          className="p-2 focus:outline-none"
          aria-label="Toggle Sidebar"
        >
          {/* 햄버거 아이콘 */}
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
      </div>
      <h2 className="text-xl font-bold mb-4">레이어 목록</h2>
      <SidebarNodeItem
        nodeType="Transformer"
        label="Transformer Block"
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="Linear"
        label="Linear Layer"
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="Embedding"
        label="Embedding Layer"
        onDragStart={onDragStart}
      />
    </aside>
  );
};

export default Sidebar;
