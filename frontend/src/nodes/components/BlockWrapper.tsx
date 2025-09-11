import { FC, ReactNode } from 'react';
import { Handle, Position } from 'reactflow';

interface BlockWrapperProps {
  children: ReactNode;
  childrenAreaHeight: number; // 자식 영역의 순수 높이
  isTarget?: boolean;
}

export const BlockWrapper: FC<BlockWrapperProps> = ({
  children,
  childrenAreaHeight,
  isTarget = false,
}) => {
  // 블록 노드의 헤더(제목 등) 영역의 기본 높이와, 자식 영역의 상/하단 여백
  const HEADER_HEIGHT = 60;
  const PADDING_Y = 20;

  // 최종 높이 계산 = 헤더 높이 + 자식 영역 높이 + 상/하단 여백
  const totalHeight = HEADER_HEIGHT + childrenAreaHeight + PADDING_Y;

  return (
    <div
      className={`block-wrapper p-2 bg-white border-2 rounded shadow ${
        isTarget ? 'border-blue-400' : 'border-gray-300 hover:border-green-300'
      }`}
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '320px',
        height: `${totalHeight}px`,
        zIndex: 1,
        isolation: 'isolate',
      }}
    >
      {/* 상단 핸들 */}
      <Handle
        type="target"
        position={Position.Top}
        id="tgt"
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          top: '-5px',
          transform: 'translate(-50%, 0)',
          zIndex: 2,
        }}
      />

      <div style={{ position: 'relative', zIndex: 10 }}>{children}</div>

      {/* 하단 핸들 */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="src"
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          bottom: '-5px',
          transform: 'translate(-50%, 0)',
          zIndex: 2,
        }}
      />
    </div>
  );
};
