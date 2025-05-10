// X 모양의 아이콘 버튼 컴포넌트
import React from 'react';

const XIconButton: React.FC<{
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}> = ({ onClick }) => {
  return (
    <button
      className="w-[20px] h-[20px] flex justify-center items-center p-0 bg-[#eee] border border-white rounded-full cursor-pointer hover:bg-[#000000] hover:text-[#eee] hover:shadow-[0_0_6px_2px_rgba(0,0,0,0.08)]"
      onClick={onClick}
    >
      <svg
        width="8"
        height="8"
        viewBox="0 0 8 8"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M1.5 1.5L6.5 6.5M6.5 1.5L1.5 6.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
};
export default XIconButton;
