import { MouseEvent } from 'react';
import { Node } from 'reactflow';
import { BaseNodeData } from './NodeData';

// NodaData 템플릿 적용
interface UseCommonNodeActionsParams<T extends BaseNodeData> {
  initialData: T;
  setNodes: (updater: (nds: Node<T>[]) => Node<T>[]) => void;
  setEditMode: React.Dispatch<React.SetStateAction<boolean>>;
  customSave: () => void;
}

// 노드별 공통 로직 Custom Hook으로 구현
export function useCommonNodeActions<T extends BaseNodeData>({
  initialData,
  setNodes,
  setEditMode,
  customSave,
}: UseCommonNodeActionsParams<T>) {
  // Delete 버튼 클릭
  const handleDeleteClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (initialData.id) {
      setNodes((nds) => nds.filter((node) => node.id !== initialData.id));
    }
  };

  // 정보 버튼 클릭 -> Modal 랜더링
  const handleInfoClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (initialData.openModal) {
      initialData.openModal(initialData);
    }
  };

  // Edit 버튼 클릭
  const handleEditClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(false);
    // Save 관련 데이터 업데이트는 노드별 customSave 콜백에서 처리
    customSave();
  };

  return {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
  };
}
