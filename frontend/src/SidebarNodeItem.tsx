import React from 'react';

import { BaseNodeData } from './nodes/components/NodeData';

interface SidebarNodeItemProps {
  nodeType: string;
  nodeData: BaseNodeData;
  onDragStart: (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
    nodeData: BaseNodeData,
  ) => void;
}

const SidebarNodeItem: React.FC<SidebarNodeItemProps> = ({
  nodeType,
  nodeData,
  onDragStart,
}) => {
  return (
    <div
      className="my-2 p-2 rounded cursor-grab hover:bg-gray-100"
      draggable
      onDragStart={(event) => onDragStart(event, nodeType, nodeData)}
    >
      {nodeData.label}
    </div>
  );
};

export default SidebarNodeItem;
