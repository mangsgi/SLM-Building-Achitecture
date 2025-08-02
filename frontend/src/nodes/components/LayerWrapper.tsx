import { FC, ReactNode } from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { isValidConnection } from '../../ButtonEdge';

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
  const { getEdges } = useReactFlow();

  const handleStyle: React.CSSProperties = hideHandles
    ? { opacity: 0, pointerEvents: 'none' as const }
    : { pointerEvents: 'auto' as const, zIndex: 12 };

  return (
    <div
      className="z-10 p-2 layer-wrapper bg-white border-2 border-gray-300 rounded shadow hover:border-green-300"
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
        isValidConnection={(params) => isValidConnection(getEdges(), params)}
      />

      <div style={{ position: 'relative', zIndex: 11 }}>{children}</div>

      {/* 오른쪽 핸들 (Residual용) */}
      {isResidual && (
        <Handle
          type="source"
          position={Position.Left}
          id="residual-source"
          style={{
            background: '#ccc',
            width: '10px',
            height: '10px',
            left: '-5px',
            top: '25%',
            transform: 'translate(0, -50%)',
            zIndex: 12,
          }}
          isValidConnection={(params) => isValidConnection(getEdges(), params)}
        />
      )}
      <Handle
        type={'target'}
        position={Position.Left}
        id="residual-target"
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          left: '-5px',
          top: isResidual ? '75%' : '50%',
          transform: 'translate(0, -50%)',
          zIndex: 12,
        }}
        isValidConnection={(params) => isValidConnection(getEdges(), params)}
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
        isValidConnection={(params) => isValidConnection(getEdges(), params)}
      />
    </div>
  );
};
