import React from 'react';
import { BaseNodeData } from '../nodes/components/NodeData';
import TestBlock from '../nodes/TestBlock';
import SidebarNodeItem from '../SidebarNodeItem';

interface NodeDefinition {
  // 1. 기본 정보
  type: string;
  label: string;
  description: string;
  category: 'embedding' | 'attention' | 'block' | 'layer';

  // 2. 컴포넌트 관련
  component: React.ComponentType<any>;
  sidebarItem: React.ComponentType<any>;

  // 3. 데이터 관련
  defaultData: BaseNodeData;
  configMapping: {
    [key: string]: string;
  };

  // 4. 노드 관계
  allowedParentTypes?: string[];
  allowedChildTypes?: string[];

  // 5. Info 관련
  infoFields: {
    name: string;
    type: 'number' | 'string' | 'boolean';
    label: string;
    description: string;
  }[];
}

const nodeRegistry: Map<string, NodeDefinition> = new Map([
  [
    'testBlock',
    {
      type: 'testBlock',
      label: 'Test Block',
      description: 'Test block for development and debugging',
      category: 'block',

      component: TestBlock,
      sidebarItem: SidebarNodeItem,

      defaultData: {
        id: '',
        label: 'Test Block',
        inDim: 768,
        outDim: 768,
      },

      configMapping: {
        embDim: 'inDim',
      },

      allowedParentTypes: [],
      allowedChildTypes: [
        'attention',
        'feedForward',
        'normalization',
        'dropout',
      ],

      infoFields: [
        {
          name: 'inDim',
          type: 'number',
          label: 'Input Dimension',
          description: 'Dimension of input tensor',
        },
        {
          name: 'outDim',
          type: 'number',
          label: 'Output Dimension',
          description: 'Dimension of output tensor',
        },
      ],
    },
  ],
]);

// ✅ 노드 정의 가져오기
export const getNodeDefinition = (type: string) => {
  const definition = nodeRegistry.get(type);
  if (!definition) throw new Error(`Unknown node type: ${type}`);
  return definition;
};

// ✅ 모든 노드 타입 가져오기
export const getAllNodeTypes = () => {
  return Array.from(nodeRegistry.keys());
};

// ✅ 카테고리별 노드 타입 가져오기
export const getNodeTypesByCategory = (
  category: NodeDefinition['category'],
) => {
  return Array.from(nodeRegistry.values())
    .filter((def) => def.category === category)
    .map((def) => def.type);
};

// ✅ 노드 컴포넌트 가져오기
export const getNodeComponent = (type: string) => {
  return getNodeDefinition(type).component;
};

// ✅ 노드 데이터 가져오기
export const getNodeDataByType = (
  nodeType: string,
  config: any,
  baseData: BaseNodeData,
) => {
  const definition = getNodeDefinition(nodeType);
  const data = { ...baseData, ...definition.defaultData };

  // Config 매핑 적용
  Object.entries(definition.configMapping).forEach(([configKey, dataKey]) => {
    if (configKey in config) {
      data[dataKey] = config[configKey];
    }
  });

  return data;
};

// ✅ 허용된 부모 노드 타입 가져오기
export const getAllowedParentTypes = (type: string) => {
  return getNodeDefinition(type).allowedParentTypes || [];
};

// ✅ 허용된 자식 노드 타입 가져오기
export const getAllowedChildTypes = (type: string) => {
  return getNodeDefinition(type).allowedChildTypes || [];
};

// ✅ 사이드바 아이템 가져오기
export const getSidebarItems = () => {
  return Array.from(nodeRegistry.values()).map((def) => ({
    type: def.type,
    label: def.label,
    component: def.sidebarItem,
  }));
};
