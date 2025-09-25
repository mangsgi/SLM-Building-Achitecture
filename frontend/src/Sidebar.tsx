import React from 'react';
import SidebarNodeItem from './SidebarNodeItem';
import { BaseNodeData } from './nodes/components/NodeData';
import { nodeRegistry } from './nodes/components/nodeRegistry';

interface SidebarProps {
  loadReferenceModel: (
    modelName: 'GPT-2' | 'Llama2' | 'Llama3' | 'Qwen3',
  ) => void;
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
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('positionalEmbedding')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('positionalEmbedding')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('linear')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('linear')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('feedForward')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('feedForward')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('normalization')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('normalization')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('dropout')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('dropout')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('residual')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('residual')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('mhAttention')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('mhAttention')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('gqAttention')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('gqAttention')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        <SidebarNodeItem
          nodeType={nodeRegistry.get('transformerBlock')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('transformerBlock')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        />
        {/* <SidebarNodeItem
          nodeType={nodeRegistry.get('testBlock')?.type ?? ''}
          nodeData={{
            label: nodeRegistry.get('testBlock')?.label ?? '',
            inDim: 0,
            outDim: 0,
          }}
          onDragStart={onDragStart}
        /> */}
      </div>

      {/* 레퍼런스 모델 로드 버튼 */}
      <div className="mt-auto pt-4 border-t border-gray-200">
        <p className="text-sm font-medium text-gray-600 mb-2 text-center">
          Load Reference Model
        </p>
        <div className="flex gap-1">
          <button
            onClick={() => loadReferenceModel('GPT-2')}
            className="flex-1 px-2 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            GPT-2
          </button>
          <button
            onClick={() => loadReferenceModel('Llama2')}
            className="flex-1 px-2 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Llama2
          </button>
          <button
            onClick={() => loadReferenceModel('Llama3')}
            className="flex-1 px-2 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Llama3
          </button>
        </div>
        <div className="flex gap-1 mt-2">
          <button
            onClick={() => loadReferenceModel('Qwen3')}
            className="w-1/3 px-2 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Qwen3
          </button>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
