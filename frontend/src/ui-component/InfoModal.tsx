import React from 'react';

interface InfoModalProps {
  title: string;
  description: string;
  onClose: () => void;
}

const InfoModal: React.FC<InfoModalProps> = ({
  title,
  description,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">{title}</h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
        <p className="text-gray-600">{description}</p>
      </div>
    </div>
  );
};

export default InfoModal;
