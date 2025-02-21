import React from 'react';

const Sidebar = () => {
  // 드래그 시작 시 dataTransfer에 노드 타입 저장
  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
  ) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    // Tailwind 클래스: 20% 너비, 전체 높이, 어두운 배경, 흰색 글자, 내부 여백
    <aside className="w-1/5 h-full bg-gray-800 text-white p-4">
      <h2 className="text-xl font-bold mb-4">레이어 목록</h2>
      <div
        className="my-2 p-2 bg-gray-600 rounded cursor-grab hover:bg-gray-500"
        draggable
        onDragStart={(event) => onDragStart(event, 'Transformer')}
      >
        Transformer Block
      </div>
      <div
        className="my-2 p-2 bg-gray-600 rounded cursor-grab hover:bg-gray-500"
        draggable
        onDragStart={(event) => onDragStart(event, 'Linear')}
      >
        Linear Layer
      </div>
      <div
        className="my-2 p-2 bg-gray-600 rounded cursor-grab hover:bg-gray-500"
        draggable
        onDragStart={(event) => onDragStart(event, 'Embedding')}
      >
        Embedding Layer
      </div>
    </aside>
  );
};

export default Sidebar;
