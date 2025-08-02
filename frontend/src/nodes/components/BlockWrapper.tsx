import { FC, ReactNode } from 'react';
import { Handle, Position } from 'reactflow';

interface BlockWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
  childNodesHeight: number;
  isTarget?: boolean;
}

export const BlockWrapper: FC<BlockWrapperProps> = ({
  children,
  childNodesHeight,
  isTarget = false,
}) => {
  return (
    <div
      className={`block-wrapper p-2 bg-white border-2 rounded shadow ${
        isTarget ? 'border-blue-400' : 'border-gray-300 hover:border-green-300'
      }`}
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '320px',
        height: `${100 + childNodesHeight}px`,
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
