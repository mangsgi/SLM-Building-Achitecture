import React, { useState, useCallback } from 'react';

import { BlockWrapper } from './NodeWrapper';
import { NodeTitle } from './NodeComponents';

interface LayerData {
  id: string;
  nodeType: string;
  label: string;
  // 필요에 따라 추가 필드 가능
}

interface DynamicBlockProps {
  data: {
    label?: string;
  };
}

// Dynamic Block 컴포넌트는 내부에 노드를 드롭하면 수직으로 정렬하고, 노드 사이에 자동 edge를 표시
const DynamicBlock: React.FC<DynamicBlockProps> = ({ data }) => {
  const [layers, setLayers] = useState<LayerData[]>([]);

  // 내부 영역에 노드 드롭 시 호출되는 핸들러
  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    const dataString = event.dataTransfer.getData('application/reactflow');
    if (!dataString) return;

    const nodeData = JSON.parse(dataString);
    const newLayer: LayerData = {
      id: `${nodeData.nodeType}-${Date.now()}`,
      nodeType: nodeData.nodeType,
      label: nodeData.label || nodeData.nodeType,
    };

    setLayers((prev) => [...prev, newLayer]);
  }, []);

  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    },
    [],
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="w-full h-full"
    >
      <BlockWrapper>
        <NodeTitle>{data.label || 'Transformer Block'}</NodeTitle>
        <div className="transformer-block-inner flex flex-col items-center mt-2">
          {layers.length === 0 && (
            <div className="text-gray-500 italic">여기에 노드를 드롭하세요</div>
          )}
          {layers.map((layer, index) => (
            <div key={layer.id} className="w-full flex flex-col items-center">
              <div className="p-2 border rounded bg-gray-100 w-full text-center">
                {layer.label}
              </div>
              {index < layers.length - 1 && (
                <div
                  className="connector my-1"
                  style={{
                    width: '2px',
                    height: '20px',
                    backgroundColor: '#ccc',
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </BlockWrapper>
    </div>
  );
};

export default DynamicBlock;
