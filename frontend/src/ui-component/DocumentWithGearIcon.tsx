import React from 'react';

const DocumentWithGearIcon: React.FC = () => {
  return (
    <svg
      className="w-6 h-6"
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {/* 문서 테두리 (오른쪽 선은 매우 짧게!) */}
      <path d="M10 4h20l8 8v10" /> {/* 오른쪽 선 길이 줄임 */}
      <path d="M10 4v40h12" /> {/* 왼쪽과 하단 일부만 */}
      {/* 접힌 모서리 선 */}
      <path d="M30 4v8h8" />
      {/* 기어 아이콘 */}
      <g transform="translate(28, 32)">
        <circle cx="6" cy="6" r="3" fill="none" />
        <path
          d="
          M6 0v2 
          M6 10v2 
          M0 6h2 
          M10 6h2 
          M2 2l1.4 1.4 
          M9.6 9.6L8.2 8.2 
          M2 10l1.4-1.4 
          M9.6 2.4L8.2 3.8
        "
        />
      </g>
    </svg>
  );
};

export default DocumentWithGearIcon;
