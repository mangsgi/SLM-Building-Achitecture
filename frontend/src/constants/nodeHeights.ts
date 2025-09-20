import { type Node } from 'reactflow';
import { nodeRegistry } from '../nodes/components/nodeRegistry';

// -------------------------------------------
// -------------- Node Heights ---------------
// -------------------------------------------

// 노드 높이 계산에 사용되는 상수
export const DEFAULT_NODE_HEIGHT = 43; // 노드가 접혀있을 때의 높이 = 노드 레이블(헤더)[24] + 노드 패딩[8*2] + 테두리[3]
export const FIELD_ROW_HEIGHT = 66; // 속성 필드 한 줄의 높이 = 속성 레이블[24] + 속성 값[30] + 속성 레이블 윗 패딩[4] + 속성 값 아랫 패딩[8]

// 기본 블록 노드 높이 (블록 노드 내 자식 노드가 없을 때의 높이)
export const DEFAULT_BLOCK_NODE_HEIGHT = 90;
// 노드 간 간격 (블록 노드 내 자식 노드 간 간격)
export const NODE_GAP = 10;
// 블록 노드의 시작 y 위치
export const BLOCK_START_Y = 110;

/**
 * 노드의 콘텐츠(필드 개수)에 따라 동적으로 높이를 계산합니다.
 * @param node 높이를 계산할 노드
 * @returns 계산된 노드의 높이 (px)
 */
export const calculateNodeHeight = (node: Node): number => {
  if (!node.type) return DEFAULT_NODE_HEIGHT;
  // getFields 함수를 사용하여 현재 데이터에 따른 필드 목록을 가져옴
  const fields = nodeRegistry.get(node.type)?.getFields(node.data) ?? [];
  const fieldCount = fields.length;
  // console.log('node.type: ', node.type, 'fieldCount: ', fieldCount);
  // 필드 개수에 따라 높이 계산
  const calculatedHeight = DEFAULT_NODE_HEIGHT + fieldCount * FIELD_ROW_HEIGHT;
  return calculatedHeight;
};
