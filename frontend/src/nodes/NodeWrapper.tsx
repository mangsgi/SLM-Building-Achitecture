import { FC, ReactNode } from 'react';
import { Handle, Position } from 'reactflow';

interface NodeWrapperProps {
  children: ReactNode;
  hideHandles?: boolean;
}

/* 노드의 공통 테두리, 배경, 핸들(상단/하단)을 렌더링하는 컨테이너 컴포넌트 */
const NodeWrapper: FC<NodeWrapperProps> = ({
  children,
  hideHandles = false,
}) => {
  const handleStyle = hideHandles ? { opacity: 0 } : {};
  return (
    <div
      className="node p-2 bg-white border-2 border-transparent hover:border-green-100"
      style={{ pointerEvents: 'auto', position: 'relative' }}
    >
      {/* 상단 Handle */}
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

      {/* ReactNode  */}
      {children}

      {/* 하단 Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#ccc',
          width: '10px',
          height: '10px',
          left: '50%',
          bottom: '0px',
          transform: 'translate(-50%, 50%)',
          ...handleStyle,
        }}
      />
    </div>
  );
};

export default NodeWrapper;
