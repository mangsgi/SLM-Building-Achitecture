import React from 'react';

// 차원 정보를 담은 JSON 문자열 datatransfer
interface NodeData {
  inDim: number;
  outDim: number;
  [key: string]: unknown;
}

interface SidebarNodeItemProps {
  nodeType: string;
  label: string;
  nodeData: NodeData;
  onDragStart: (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
  ) => void;
}

const SidebarNodeItem: React.FC<SidebarNodeItemProps> = ({
  nodeType,
  label,
  nodeData,
  onDragStart,
}) => {
  const dataString = JSON.stringify({ nodeType, label, ...nodeData });

  return (
    <div
      className="my-2 p-2 rounded cursor-grab hover:bg-green-100"
      draggable
      onDragStart={(event) => onDragStart(event, dataString)}
    >
      {label}
    </div>
  );
};

export default SidebarNodeItem;
