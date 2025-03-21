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
          vocabSize: 50267,
          embDim: 768,
          label: 'Token Embedding Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="positionalEmbedding"
        nodeData={{
          ctxLength: 512,
          embDim: 768,
          label: 'Positional Embedding Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="linear"
        nodeData={{ inDim: 768, outDim: 50267, label: 'Linear Layer' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="layerNorm"
        nodeData={{
          inDim: 768,
          label: 'Layer Normalization Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="feedForward"
        nodeData={{
          inDim: 768,
          numOfFactor: 3072,
          actFunc: 'GELU',
          label: 'Feed Forward Layer',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dropout"
        nodeData={{ dropoutRate: 0.1, label: 'Dropout Layer' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dynamicBlock"
        nodeData={{
          label: 'Dynamic Block',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="maskedMultiHeadAttention"
        label="Masked Multi-Head Attention"
        nodeData={{ inDim: 768, outDim: 768, numHeads: 12 }}
        onDragStart={onDragStart}
      />
    </aside>
  );
};

export default Sidebar;
