import React from 'react';

interface SidebarNodeItemProps {
  nodeType: string;
  label: string;
  onDragStart: (
    event: React.DragEvent<HTMLDivElement>,
    nodeType: string,
  ) => void;
}

const SidebarNodeItem: React.FC<SidebarNodeItemProps> = ({
  nodeType,
  label,
  onDragStart,
}) => {
  return (
    <div
      className="my-2 p-2 bg-green-200 rounded cursor-grab hover:bg-gray-500"
      draggable
      onDragStart={(event) => onDragStart(event, nodeType)}
    >
      {label}
    </div>
  );
};

export default SidebarNodeItem;
