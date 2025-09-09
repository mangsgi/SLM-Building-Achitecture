import { getBezierPath, EdgeText, EdgeProps } from 'reactflow';
import type { Edge, Connection } from 'reactflow';
import { useDispatch } from 'react-redux';
import { useContext } from 'react';
import { SET_DIRTY } from './store/actions';
import { flowContext } from './store/ReactFlowContext';
import XIconButton from './ui-component/XIconButton';

// Edge 연결 가능 여부 확인
export function isValidConnection(
  edges: Edge[],
  connection: Connection,
): boolean {
  return !edges.some(
    (edge) =>
      (edge.source === connection.source &&
        edge.sourceHandle === connection.sourceHandle) ||
      (edge.target === connection.target &&
        edge.targetHandle === connection.targetHandle) ||
      connection.source === connection.target,
  );
}

const foreignObjectSize = 40;

const ButtonEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  data,
  markerEnd,
}: EdgeProps) => {
  const [edgePath, edgeCenterX, edgeCenterY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const { deleteEdge } = useContext(flowContext);
  const dispatch = useDispatch();

  // Edge 클릭 시 삭제
  const onEdgeClick = (
    evt: React.MouseEvent<HTMLButtonElement>,
    edgeId: string,
  ) => {
    evt.stopPropagation();
    deleteEdge(edgeId);
    dispatch({ type: SET_DIRTY });
    // console.log('edgeId: ', edgeId);
  };

  return (
    <>
      <path
        id={id}
        style={style}
        className="z-20 react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      {data && data.label && (
        <EdgeText
          x={sourceX + 10}
          y={sourceY + 10}
          label={data.label}
          labelStyle={{ fill: 'black' }}
          labelBgStyle={{ fill: 'transparent' }}
          labelBgPadding={[2, 4]}
          labelBgBorderRadius={2}
        />
      )}
      <foreignObject
        width={foreignObjectSize}
        height={foreignObjectSize}
        x={edgeCenterX - foreignObjectSize / 2}
        y={edgeCenterY - foreignObjectSize / 2}
        className="flex items-center justify-center bg-transparent z-12"
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
        <div className="w-full h-full flex items-center justify-center">
          <XIconButton
            onClick={(event) => {
              onEdgeClick(event, id);
            }}
          />
        </div>
      </foreignObject>
    </>
  );
};

export default ButtonEdge;
