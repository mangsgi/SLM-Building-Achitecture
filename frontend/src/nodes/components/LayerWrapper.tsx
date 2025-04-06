import { FC, ReactNode } from 'react';
import { Handle, Position } from 'reactflow';

interface LayerWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
}

export const LayerWrapper: FC<LayerWrapperProps> = ({
  children,
  hideHandles = false,
}) => {
  const handleStyle: React.CSSProperties = hideHandles
    ? { opacity: 0, pointerEvents: 'none' as const }
    : { pointerEvents: 'none' as const, zIndex: 1 };

  return (
    <div
      className="z-10 p-2 block-wrapper bg-white border-2 border-gray-300 rounded shadow hover:border-green-100"
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '300px',
      }}
    >
      {/* 상단 핸들 */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          left: '50%',
          top: '-6px',
          transform: 'translate(-50%, 0)',
          ...handleStyle,
        }}
      />

      {children}

      {/* 하단 핸들 */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          left: '50%',
          bottom: '-5px',
          transform: 'translate(-50%, 0)',
          ...handleStyle,
        }}
      />
    </div>
  );
};
