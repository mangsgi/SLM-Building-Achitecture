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
        nodeData={{
          label: 'Token Embedding Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="positionalEmbedding"
        nodeData={{
          label: 'Positional Embedding Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="linear"
        nodeData={{ label: 'Linear Layer' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="layerNorm"
        nodeData={{
          label: 'Layer Normalization Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="feedForward"
        nodeData={{
          label: 'Feed Forward Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dropout"
        nodeData={{ label: 'Dropout Layer' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="sdpAttention"
        nodeData={{
          label: 'Scaled Dot-Product Attention',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="maskedMHABlock"
        nodeData={{ label: 'Masked Multi-Head Attention' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="transformerBlock"
        nodeData={{
          label: 'Trasnformer Block',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dynamicBlock"
        nodeData={{
          label: 'Dynamic Block',
        }}
        onDragStart={onDragStart}
      />
    </aside>
  );
};

export default Sidebar;
