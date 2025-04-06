import { MouseEvent, useState, useEffect } from 'react';
import { Node } from 'reactflow';
import { BaseNodeData } from './components/NodeData';

// NodaData 템플릿 적용
interface UseCommonNodeActionsParams<T extends BaseNodeData> {
  id: string;
  setNodes: (updater: (nds: Node<T>[]) => Node<T>[]) => void;
  setEditMode: React.Dispatch<React.SetStateAction<boolean>>;
  setIsCollapsed?: React.Dispatch<React.SetStateAction<boolean>>;
}

// 노드별 공통 로직 Custom Hook으로 구현
export function useCommonNodeActions<T extends BaseNodeData>({
  id,
  setNodes,
  setEditMode,
  setIsCollapsed,
}: UseCommonNodeActionsParams<T>) {
  // Layer Node 별 펼쳐져 있을 때 높이
  const defaultHeightMap: Record<string, number> = {
    tokenEmbedding: 174,
    positionalEmbedding: 240,
    layerNorm: 109,
    feedForward: 240,
    dropout: 109,
    linear: 174,
    sdpAttention: 306,
  };

  // useEffect 사용을 위한 Node별 isCollapsed useState
  const [collapseTrigger, setCollapseTrigger] = useState<boolean | null>(false);

  useEffect(() => {
    if (collapseTrigger === null) return;
    setNodes((nds) => {
      const targetNode = nds.find((n) => n.id === id);
      if (!targetNode?.parentNode || !targetNode.type) return nds;
      const parentId = targetNode.parentNode;

      const newHeight = collapseTrigger
        ? 43
        : defaultHeightMap[targetNode.type];

      // 해당 노드 height 변경
      const updatedNodes = nds.map((node) => {
        if (node.id === id) {
          return { ...node, height: newHeight };
        }
        return node;
      });

      // 형제 노드 정렬
      const siblings = updatedNodes
        .filter((n) => n.parentNode === parentId)
        .sort((a, b) => a.position.y - b.position.y);

      let yOffset = 110;
      const reordered = siblings.map((n) => {
        const updated = {
          ...n,
          position: { ...n.position, y: yOffset },
        };
        yOffset += (n.height ?? 40) + 10;
        return updated;
      });

      return nds.filter((n) => n.parentNode !== parentId).concat(reordered);
    });

    // !collapseTrigger을 인자로 사용 시 무한 루프가 되므로 null로 설정
    setCollapseTrigger(null);
    // Node id별로 collapseTrigger 구분
  }, [collapseTrigger, id, setNodes]);

  // Node Click 시 !isCollapsed 후, 자식 노드 위치 변경을 위한 useEffect 실행
  const handleNodeClick = () => {
    if (!setIsCollapsed) return;
    setIsCollapsed((prev) => {
      const newState = !prev;
      setCollapseTrigger(newState);
      return newState;
    });
  };

  // Delete 버튼 클릭 시 노드 삭제 및 부모 존재 시 남은 노드들 위치 조정
  const handleDeleteClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();

    setNodes((nds) => {
      const nodeToDelete = nds.find((n) => n.id === id);
      if (!nodeToDelete) {
        console.warn('삭제 대상 노드를 찾지 못했습니다:', id);
        return nds;
      }

      // 부모 노드를 삭제하는 경우: 자식 노드도 같이 삭제
      if (!nodeToDelete.parentNode) {
        // 이 노드가 부모인 모든 자식 노드 찾기
        const childIds = nds
          .filter((n) => n.parentNode === id)
          .map((n) => n.id);

        // 부모 노드와 자식 노드를 모두 삭제
        return nds.filter((n) => n.id !== id && !childIds.includes(n.id));
      }

      // 삭제할 노드의 부모 노드가 존재하면
      if (!nodeToDelete.parentNode) {
        // 이 노드가 부모인 모든 자식 노드 찾기
        const childIds = nds
          .filter((n) => n.parentNode === id)
          .map((n) => n.id);

        // 부모 노드와 자식 노드를 모두 삭제
        return nds.filter((n) => n.id !== id && !childIds.includes(n.id));
      }

      // 일반 노드(부모 있음) 삭제 시 위치 재정렬
      const parentId = nodeToDelete.parentNode;

      // 나를 제외한 자식 노드 찾기
      const remainingChildren = nds
        .filter((n) => n.parentNode === parentId && n.id !== id)
        .sort((a, b) => a.position.y - b.position.y);

      // 위치 변경
      let yOffset = 110;
      const updatedChildren = remainingChildren.map((child) => {
        const updated = {
          ...child,
          position: { ...child.position, y: yOffset },
        };
        yOffset += (child.height ?? 40) + 10;
        return updated;
      });

      return nds
        .filter((n) => n.id !== id && n.parentNode !== parentId)
        .concat(updatedChildren);
    });
  };

  // 정보 버튼 클릭 -> Modal 랜더링
  const handleInfoClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setNodes((nds) => {
      const targetNode = nds.find((node) => node.id === id);
      const data = targetNode?.data as BaseNodeData;
      if (data?.openModal) {
        data.openModal(data);
      }
      return nds;
    });
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
  };

  return {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
  };
}
