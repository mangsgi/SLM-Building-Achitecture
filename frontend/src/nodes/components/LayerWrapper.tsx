import { FC, ReactNode } from 'react';
import { Handle, Position } from 'reactflow';

interface LayerWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
  isResidual?: boolean;
}

export const LayerWrapper: FC<LayerWrapperProps> = ({
  children,
  hideHandles = false,
  isResidual = false,
}) => {
  const handleStyle: React.CSSProperties = hideHandles
    ? { opacity: 0, pointerEvents: 'none' as const }
    : { pointerEvents: 'auto' as const, zIndex: 12 };

  return (
    <div
      className="z-10 p-2 layer-wrapper bg-white border-2 border-gray-300 rounded shadow hover:border-green-100"
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '300px',
        zIndex: 11,
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
          zIndex: 12,
          ...handleStyle,
        }}
      />

      <div style={{ position: 'relative', zIndex: 11 }}>{children}</div>

      {/* 오른쪽 핸들 (Residual용) */}
      <Handle
        type={isResidual ? 'source' : 'target'}
        position={Position.Right}
        id="residual"
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          right: '-5px',
          transform: 'translate(0, -50%)',
          zIndex: 12,
        }}
      />

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
          zIndex: 12,
          ...handleStyle,
        }}
      />
    </div>
  );
};
