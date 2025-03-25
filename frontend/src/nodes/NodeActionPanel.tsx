import React from 'react';
import { FiInfo, FiEdit2, FiSave, FiTrash2 } from 'react-icons/fi';

interface NodeActionPanelProps {
  editMode: boolean;
  onInfo: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onEdit: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onSave: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onDelete: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

const NodeActionPanel: React.FC<NodeActionPanelProps> = ({
  editMode,
  onInfo,
  onEdit,
  onSave,
  onDelete,
}) => {
  return (
    <div className="absolute -right-14 -top-4 transform opacity-0 group-hover:opacity-100 transition-opacity duration-200">
      <div className="p-2 rounded-md flex flex-col items-center space-y-1">
        <button
          onClick={onInfo}
          className="bg-white hover:bg-green-100 text-black p-2 rounded focus:outline-none focus:ring-0"
        >
          <FiInfo size={16} />
        </button>
        <button
          onClick={editMode ? onSave : onEdit}
          className="bg-white hover:bg-green-100 text-black p-2 rounded focus:outline-none focus:ring-0"
        >
          {editMode ? <FiSave size={16} /> : <FiEdit2 size={16} />}
        </button>
        <button
          onClick={onDelete}
          className="bg-white hover:bg-green-100 text-black p-2 rounded focus:outline-none focus:ring-0"
        >
          <FiTrash2 size={16} />
        </button>
      </div>
    </div>
  );
};

export default NodeActionPanel;
