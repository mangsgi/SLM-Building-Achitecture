import { createContext, useState } from 'react';
import { useDispatch } from 'react-redux';
import { SET_DIRTY } from './actions';
import { ReactFlowInstance } from 'reactflow';

// flowContext 타입 정의
interface FlowContextType {
  reactFlowInstance: ReactFlowInstance | null;
  setReactFlowInstance: React.Dispatch<
    React.SetStateAction<ReactFlowInstance | null>
  >;
  deleteEdge: (edgeId: string) => void;
}

// flowContext 초기값 설정
const initialValue: FlowContextType = {
  reactFlowInstance: null,
  setReactFlowInstance: () => {},
  deleteEdge: () => {},
};

// flowContext 생성
export const flowContext = createContext<FlowContextType>(initialValue);

// flowContext를 제공하는 컴포넌트
export const ReactFlowContext = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const dispatch = useDispatch();
  const [reactFlowInstance, setReactFlowInstance] =
    useState<ReactFlowInstance | null>(null);

  const deleteEdge = (edgeId: string) => {
    reactFlowInstance?.setEdges(
      reactFlowInstance?.getEdges().filter((edge) => edge.id !== edgeId),
    );
    dispatch({ type: SET_DIRTY });
    console.log('edgeId: ', edgeId);
  };

  return (
    <flowContext.Provider
      value={{
        reactFlowInstance: reactFlowInstance ?? null,
        setReactFlowInstance,
        deleteEdge,
      }}
    >
      {children}
    </flowContext.Provider>
  );
};
