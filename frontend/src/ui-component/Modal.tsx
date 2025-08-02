import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
  if (!isOpen) return null;

  return (
    // 전체 화면을 덮는 오버레이. 클릭 시 onClose 실행.
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30"
      onClick={onClose}
    >
      {/* 모달 내용: 클릭 이벤트 전파를 막아 오버레이 클릭으로 닫히지 않도록 함 */}
      <div
        className="bg-white rounded-md shadow-lg p-4 w-80"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
};

export default Modal;
