import { MouseEvent } from 'react';
import { useReactFlow, type Node, type Edge } from 'reactflow';
import { BaseNodeData } from './NodeData';
import {
  calculateNodeHeight,
  DEFAULT_NODE_HEIGHT,
  BLOCK_START_Y,
  NODE_GAP,
} from '../../constants/nodeHeights';
import { NodeInfo } from './NodeInfo';

// NodaData 템플릿 적용
interface UseCommonNodeActionsParams {
  id: string;
  setEditMode: React.Dispatch<React.SetStateAction<boolean>>;
}

// 부모 노드 내의 자식 노드(형제)들의 위치를 재정렬하는 유틸리티 함수
export const repositionSiblings = (nodes: Node[], parentId: string): Node[] => {
  const siblings = nodes
    .filter((n) => n.parentNode === parentId)
    .sort((a, b) => a.position.y - b.position.y);

  let yOffset = BLOCK_START_Y;
  const reorderedSiblings = siblings.map((sibling) => {
    const correctHeight = sibling.data.isCollapsed
      ? DEFAULT_NODE_HEIGHT
      : calculateNodeHeight(sibling);

    const updatedSibling = {
      ...sibling,
      position: { ...sibling.position, y: yOffset },
      height: correctHeight,
    };
    yOffset += correctHeight + NODE_GAP;
    return updatedSibling;
  });

  return nodes
    .filter((n) => n.parentNode !== parentId)
    .concat(reorderedSiblings);
};

// 노드별 공통 로직 Custom Hook으로 구현
export function useCommonNodeActions<T extends BaseNodeData>({
  id,
  setEditMode,
}: UseCommonNodeActionsParams) {
  const { setNodes, setEdges, getNodes, getEdges } = useReactFlow<T, Edge>();

  // 노드 정보 클릭 핸들러
  const handleInfoClick = (info: NodeInfo) => {
    const event = new CustomEvent('nodeInfo', {
      detail: info,
    });
    window.dispatchEvent(event);
  };

  // Node Click 시 isCollapsed 상태를 반전시키고, 노드 높이와 형제 노드 위치를 재조정
  const handleNodeClick = () => {
    setNodes((nds) => {
      const targetNode = nds.find((n) => n.id === id);
      if (!targetNode || !targetNode.data) return nds;

      const newIsCollapsed = !targetNode.data.isCollapsed;

      // 1. 대상 노드의 isCollapsed 상태와 높이를 업데이트
      let updatedNodes = nds.map((n) => {
        if (n.id === id) {
          const newHeight = newIsCollapsed
            ? DEFAULT_NODE_HEIGHT
            : calculateNodeHeight(n);
          return {
            ...n,
            data: { ...n.data, isCollapsed: newIsCollapsed },
            height: newHeight,
          };
        }
        return n;
      });

      // 2. 부모가 있는 경우, 형제 노드 위치 재조정
      if (targetNode.parentNode) {
        updatedNodes = repositionSiblings(updatedNodes, targetNode.parentNode);
      }

      return updatedNodes;
    });
  };

  // Delete 버튼 클릭 시 노드 삭제 및 부모 존재 시 남은 노드들 위치 조정
  const handleDeleteClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();

    const allNodes = getNodes();
    const allEdges = getEdges();
    const nodeToDelete = allNodes.find((n) => n.id === id);
    if (!nodeToDelete) return;

    let nodesToKeep: Node<T>[];
    let edgesToKeep: Edge[];

    // 부모 노드를 삭제하는 경우: 자식 노드도 같이 삭제
    if (!nodeToDelete.parentNode) {
      const childIds = allNodes
        .filter((n) => n.parentNode === id)
        .map((n) => n.id);
      const allIdsToDelete = [id, ...childIds];

      nodesToKeep = allNodes.filter(
        (n) => !allIdsToDelete.includes(n.id),
      ) as Node<T>[];
      edgesToKeep = allEdges.filter(
        (e) =>
          !allIdsToDelete.includes(e.source) &&
          !allIdsToDelete.includes(e.target),
      );
      setNodes(nodesToKeep);
    } else {
      // 일반 노드(부모 있음) 삭제 시
      const parentId = nodeToDelete.parentNode;
      nodesToKeep = allNodes.filter((n) => n.id !== id) as Node<T>[];

      // 삭제 후 남은 형제 노드들의 위치 재정렬
      const finalNodes = repositionSiblings(nodesToKeep, parentId);
      setNodes(finalNodes);

      edgesToKeep = allEdges.filter(
        (edge) => edge.source !== id && edge.target !== id,
      );
    }

    setEdges(edgesToKeep);
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

  const handleLockToggle = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === id && n.data) {
          return {
            ...n,
            data: { ...n.data, isLocked: !n.data.isLocked },
          };
        }
        return n;
      }),
    );
  };

  return {
    handleDeleteClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
    handleInfoClick,
    handleLockToggle,
  };
}
