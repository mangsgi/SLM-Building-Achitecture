import { BaseNodeData, TestBlockData } from './NodeData';
import { FieldConfig } from './FieldRenderer';
import { nodeFieldInfo } from './nodeInfo';

interface NodeDefinition {
  // 1. 기본 정보
  type: string;
  label: string;

  // 2. 데이터 관련
  defaultData: BaseNodeData;

  configMapping?: {
    [key: string]: string;
  };

  // 3. 필드 관련
  getFields: (data: BaseNodeData) => FieldConfig[];
}

export const nodeRegistry: Map<string, NodeDefinition> = new Map([
  [
    'testBlock',
    {
      type: 'testBlock',
      label: 'Test Block',
      defaultData: {
        id: '',
        label: 'Test Block',
        inDim: 768,
        outDim: 768,
      },
      getFields: (data: BaseNodeData) => {
        const typed = data as TestBlockData;
        return [
          {
            type: 'number',
            label: 'Test Type:',
            name: 'testType',
            value: typed.testType?.toString() || '',
            options: ['default', 'custom'],
            info: nodeFieldInfo.testBlock.testType,
          },
        ];
      },
    },
  ],
]);
