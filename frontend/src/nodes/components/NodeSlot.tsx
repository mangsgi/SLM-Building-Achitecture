import React, { useCallback } from 'react';

interface NodeSlotProps {
  slotLabel: string;
  allowedType: string;
}

const NodeSlot: React.FC<NodeSlotProps> = ({ slotLabel, allowedType }) => {
  // 드래그 오버 시 기본 동작 방지 및 dropEffect 설정
  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    },
    [],
  );

  // 노드 슬롯 내에서 드롭 이벤트를 처리하여 onChange를 통해 데이터를 업데이트하고,
  // 이벤트가 FlowCanvas의 onDrop까지 버블링되지 않도록 함
  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const dataString = event.dataTransfer.getData('application/reactflow');
      if (!dataString) return;
      let parsedData;
      try {
        parsedData = JSON.parse(dataString);
      } catch (error) {
        console.error('드롭된 데이터 파싱 오류:', error);
        return;
      }
      // allowedTypes가 정의되어 있다면 해당 타입만 허용
      if (allowedType && !allowedType.includes(parsedData.nodeType)) {
        console.warn(
          `드롭된 노드 타입 '${parsedData.nodeType}'은(는) 이 슬롯에서 허용되지 않습니다.`,
        );
        return;
      }
    },
    [allowedType],
  );

  return (
    <div
      className="node-slot-container p-2 h-12 w-full bg-transparent border-dashed border-2 border-gray-200 rounded"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className="node-slot-placeholder italic text-gray-400 text-sm p-2">
        {slotLabel} (드래그 앤 드롭)
      </div>
    </div>
  );
};

export default NodeSlot;
