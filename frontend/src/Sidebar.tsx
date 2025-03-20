import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';
import CanvasHamburgerIcon from './ui-component/CanvasHamburgerIcon';

interface SidebarProps {
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onToggle }) => {
  // 드래그 시작 시 dataTransfer에 노드 타입 저장
  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    data: string,
  ) => {
    event.dataTransfer.setData('application/reactflow', data);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="w-1/5 h-full bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold mb-4">레이어 목록</h2>
        <button
          onClick={onToggle}
          className="p-2 focus:outline-none"
          aria-label="Toggle Sidebar"
        >
          <CanvasHamburgerIcon />
        </button>
      </div>
      {/* GPT-2 노드 항목들 */}
      <SidebarNodeItem
        nodeType="tokenEmbedding"
        label="Token Embedding Layer"
        nodeData={{ inDim: 50000, outDim: 768 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="positionalEmbedding"
        label="Positional Embedding Layer"
        nodeData={{ inDim: 512, outDim: 768 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="maskedMultiHeadAttention"
        label="Masked Multi-Head Attention"
        nodeData={{ inDim: 768, outDim: 768, numHeads: 12 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="layerNorm1"
        label="LayerNorm 1"
        nodeData={{ inDim: 768, outDim: 768 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="feedForward"
        label="Feed Forward"
        nodeData={{ inDim: 768, outDim: 3072 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dropout"
        label="Dropout"
        nodeData={{ inDim: 3072, outDim: 3072, dropoutRate: 0.1 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="layerNorm2"
        label="LayerNorm 2"
        nodeData={{ inDim: 768, outDim: 768 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="finalLayerNorm"
        label="Final LayerNorm"
        nodeData={{ inDim: 768, outDim: 768 }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="linearOutput"
        label="Linear Output Layer"
        nodeData={{ inDim: 768, outDim: 50000 }}
        onDragStart={onDragStart}
      />
    </aside>
  );
};

export default Sidebar;
