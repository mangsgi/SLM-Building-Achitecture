import { FC, ReactNode, MouseEventHandler } from 'react';
import { Handle, Position } from 'reactflow';

interface BlockWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
  childNodesHeight?: number;
}

export const BlockWrapper: FC<BlockWrapperProps> = ({
  children,
  childNodesHeight = 0,
}) => {
  return (
    <div
      className="block-wrapper p-2 bg-white border-2 border-gray-300 rounded shadow hover:border-green-100"
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '320px',
        height: `${130 + childNodesHeight}px`,
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
          bottom: '-6px',
          transform: 'translate(-50%, 0)',
        }}
      />
    </div>
  );
};

interface LayerWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
  onClick?: MouseEventHandler<HTMLDivElement>;
}

export const LayerWrapper: FC<LayerWrapperProps> = ({
  children,
  hideHandles = false,
  onClick,
}) => {
  const handleStyle: React.CSSProperties = hideHandles
    ? { opacity: 0, pointerEvents: 'none' as const }
    : { pointerEvents: 'none' as const, zIndex: 1 };

  return (
    <div
      className="block-wrapper p-2 bg-white border-2 border-gray-300 rounded shadow hover:border-green-100"
      onClick={onClick}
      style={{
        pointerEvents: 'auto',
        position: 'relative',
        width: '300px', // 고정 너비
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
