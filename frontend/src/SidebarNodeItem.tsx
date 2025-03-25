import React from 'react';

import { BaseNodeData } from './nodes/NodeData';

interface SidebarNodeItemProps {
  nodeType: string;
  nodeData: BaseNodeData;
  // onDragStart를 부모(Sidebar)에서 실행할 수 있도록 Data 전달
  onDragStart: (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
  ) => void;
}

const SidebarNodeItem: React.FC<SidebarNodeItemProps> = ({
  nodeType,
  nodeData,
  onDragStart,
}) => {
  const dataString = JSON.stringify({ nodeType, ...nodeData });

  return (
    <div
      className="my-2 p-2 rounded cursor-grab hover:bg-green-100"
      draggable
      onDragStart={(event) => onDragStart(event, dataString)}
    >
      {nodeData.label}
    </div>
  );
};

export default SidebarNodeItem;
