import React, { useCallback } from 'react';
import { BaseNodeData } from './NodeData';

interface NodeSlotProps<T extends BaseNodeData> {
  slotLabel: string;
  data: T | null;
  onChange: (newData: T | null) => void;
  // 노드 렌더링 컴포넌트를 prop으로 전달합니다.
  nodeComponent: React.FC<{ data: T; onChange: (data: T) => void }>;
  // 옵션: 드롭 가능한 노드 타입들을 배열로 전달할 수 있습니다.
  allowedTypes?: string[];
}

const NodeSlot = <T extends BaseNodeData>({
  slotLabel,
  data,
  onChange,
  nodeComponent: NodeComponent,
  allowedTypes,
}: NodeSlotProps<T>) => {
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();

      const raw = e.dataTransfer.getData('application/reactflow');
      if (!raw) return;
      const dropped = JSON.parse(raw);
      if (allowedTypes && !allowedTypes.includes(dropped.nodeType)) {
        alert(`이 슬롯에는 ${allowedTypes.join(', ')} 노드만 드롭 가능합니다!`);
        return;
      }

      const newData: BaseNodeData = {
        id: `${dropped.nodeType}-${Date.now()}`,
        label: dropped.label || dropped.nodeType,
        ...dropped,
      };
      onChange(newData as T);
    },
    [onChange, allowedTypes],
  );

  let content;
  if (data) {
    content = (
      <div className="node-slot-inner border rounded shadow">
        {/* 노드 컨텐츠 영역 */}
        <div className="node-content p-2">
          <NodeComponent data={data} onChange={onChange} />
        </div>
      </div>
    );
  } else {
    content = (
      <div className="node-slot-placeholder italic text-gray-400 text-sm p-2">
        {slotLabel} (드래그 앤 드롭)
      </div>
    );
  }

  return (
    <div
      className="node-slot-container my-2 p-2 w-full bg-transparent border-dashed border-2 border-gray-200 rounded"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {content}
    </div>
  );
};

export default NodeSlot;
