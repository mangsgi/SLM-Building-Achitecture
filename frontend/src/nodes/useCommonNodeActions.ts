import { MouseEvent, useState, useEffect } from 'react';
import type { Node, Edge } from 'reactflow';
import { BaseNodeData } from './components/NodeData';
import { NODE_HEIGHTS, DEFAULT_NODE_HEIGHT } from '../constants/nodeHeights';
import { NodeInfo } from './components/nodeInfo';

// NodaData 템플릿 적용
interface UseCommonNodeActionsParams<T extends BaseNodeData> {
  id: string;
  setNodes: (updater: (nds: Node<T>[]) => Node<T>[]) => void;
  setEditMode: React.Dispatch<React.SetStateAction<boolean>>;
  setIsCollapsed?: React.Dispatch<React.SetStateAction<boolean>>;
  setEdges: (updater: (eds: Edge[]) => Edge[]) => void;
}

// 노드별 공통 로직 Custom Hook으로 구현
export function useCommonNodeActions<T extends BaseNodeData>({
  id,
  setNodes,
  setEditMode,
  setIsCollapsed,
  setEdges,
}: UseCommonNodeActionsParams<T>) {
  // Layer Node 별 펼쳐져 있을 때 높이
  const defaultHeightMap = NODE_HEIGHTS;
  const [collapseTrigger, setCollapseTrigger] = useState<boolean | null>(false);

  // ✅ 특정 Node id의 collapseTrigger 변경 시 실행되어 형제 Node의 위치 조정
  useEffect(() => {
    if (collapseTrigger === null) return;
    setNodes((nds) => {
      const targetNode = nds.find((n) => n.id === id);
      if (!targetNode?.parentNode || !targetNode.type) return nds;
      const parentId = targetNode.parentNode;

      const newHeight = collapseTrigger
        ? 43
        : (defaultHeightMap[targetNode.type as keyof typeof defaultHeightMap] ??
          DEFAULT_NODE_HEIGHT);

      // 해당 Node height 변경
      const updatedNodes = nds.map((node) => {
        if (node.id === id) {
          return { ...node, height: newHeight };
        }
        return node;
      });

      // 형제 Node 정렬
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

  // ✅ 노드 정보 클릭 핸들러 오버라이드
  const handleInfoClick = (info: NodeInfo) => {
    const event = new CustomEvent('nodeInfo', {
      detail: info,
    });
    window.dispatchEvent(event);
  };

  // ✅ Node Click 시 !isCollapsed 후, 자식 노드 위치 변경을 위한 useEffect 실행
  const handleNodeClick = () => {
    if (!setIsCollapsed) return;
    setIsCollapsed((prev) => {
      const newState = !prev;
      setCollapseTrigger(newState);
      return newState;
    });
  };

  // ✅ Delete 버튼 클릭 시 노드 삭제 및 부모 존재 시 남은 노드들 위치 조정
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

      // Edge 삭제
      setEdges((eds) =>
        eds.filter((edge) => edge.source !== id && edge.target !== id),
      );

      return nds
        .filter((n) => n.id !== id && n.parentNode !== parentId)
        .concat(updatedChildren);
    });
  };

  // ✅ Edit 버튼 클릭
  const handleEditClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(true);
  };

  // ✅ Save 버튼 클릭
  const handleSaveClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setEditMode(false);
    // Save 관련 데이터 업데이트는 노드별 customSave 콜백에서 처리
  };

  return {
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
    handleInfoClick,
  };
}
