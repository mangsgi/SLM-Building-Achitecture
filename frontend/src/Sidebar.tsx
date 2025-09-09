import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';
import { BaseNodeData } from './nodes/components/NodeData';
import { nodeRegistry } from './nodes/components/nodeRegistry';

interface SidebarProps {
  loadReferenceModel: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ loadReferenceModel }) => {
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
    <aside className="h-full shadow z-10 bg-white px-4 py-2 overflow-y-auto transition-transform duration-300 ease-in-out flex flex-col">
      {/* Sidebar Header 영역 */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Node List</h2>
      </div>

      {/* 노드 항목 영역 */}
      <div className="flex-grow">
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
          nodeType={nodeRegistry.get('linear')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('linear')?.label ?? '',
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
      </div>

      {/* 레퍼런스 모델 로드 버튼 */}
      <div className="mt-auto pt-4 border-t border-gray-200">
        <button
          onClick={loadReferenceModel}
          className="w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Load Reference Model
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
