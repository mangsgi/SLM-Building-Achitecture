import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';
import CanvasHamburgerIcon from './ui-component/CanvasHamburgerButton';
import { BaseNodeData } from './nodes/components/NodeData';
import { nodeRegistry } from './nodes/components/nodeRegistry';

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
    <aside className="w-[250px] h-full shadow z-10 bg-white px-4 py-2 overflow-y-auto">
      {/* Sidebar Header 영역 */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Node List</h2>
        <div onClick={onToggle}>
          <CanvasHamburgerIcon />
        </div>
      </div>
      {/* 노드 항목 영역 */}
      <SidebarNodeItem
        nodeType={nodeRegistry.get('tokenEmbedding')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('tokenEmbedding')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('positionalEmbedding')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('positionalEmbedding')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('linearOutput')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('linearOutput')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('feedForward')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('feedForward')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('normalization')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('normalization')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('dropout')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('dropout')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('residual')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('residual')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('mhAttention')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('mhAttention')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('gqAttention')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('gqAttention')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('transformerBlock')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('transformerBlock')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      <SidebarNodeItem
        nodeType={nodeRegistry.get('testBlock')?.type ?? ''}
        nodeData={{
          label: nodeRegistry.get('testBlock')?.label ?? '',
        }}
        onDragStart={onDragStart}
      />
      {/* <SidebarNodeItem
        nodeType="gpt2TransformerBlock"
        nodeData={{
          label: 'GPT-2 Transformer Block',
        }}
        onDragStart={onDragStart}
      /> */}
    </aside>
  );
};

export default Sidebar;
