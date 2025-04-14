import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';
import CanvasHamburgerIcon from './ui-component/CanvasHamburgerButton';
import { BaseNodeData } from './nodes/components/NodeData';

interface SidebarProps {
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onToggle }) => {
  // Drag 이벤트 핸들러 함수
  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
    nodeData: BaseNodeData,
  ) => {
    const id = `${nodeType}-${+new Date()}`;
    const dataString = JSON.stringify({ nodeType, id, ...nodeData });
    event.dataTransfer.setData('application/reactflow', dataString);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="w-1/6 h-full shadow z-10 bg-white p-4 overflow-y-auto">
      {/* Sidebar Header 영역 */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Node 목록</h2>
        <div onClick={onToggle}>
          <CanvasHamburgerIcon />
        </div>
      </div>
      {/* 노드 항목 영역 */}
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
        nodeType="normalization"
        nodeData={{
          label: 'Normalization',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="feedForward"
        nodeData={{
          label: 'Feed Forward',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="dropout"
        nodeData={{ label: 'Dropout' }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType="residual"
        nodeData={{
          label: 'Residual',
        }}
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
      <SidebarNodeItem
        nodeType="testBlock"
        nodeData={{ label: 'testBlock' }}
        onDragStart={onDragStart}
      />
    </aside>
  );
};

export default Sidebar;
